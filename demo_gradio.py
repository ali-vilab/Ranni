import argparse
import torch
import einops
import random
import numpy as np
import gradio as gr
import open_clip
from omegaconf import OmegaConf
from transformers import AutoTokenizer
from peft import get_peft_model, LoraConfig, TaskType
from ranni.llama_modeling import LlamaForCausalLM
from ranni.ddim_hacked import DDIMSampler
from ldm.util import instantiate_from_config
from utils import build_llama_prompt, chat, element_template, box_template, seq_to_element, seq_to_element_v2, seq_to_box, box_to_seq, draw_box


parser = argparse.ArgumentParser()
parser.add_argument("--config-path", type=str, default='config/ranni_sdv21_v1.yaml')
args = parser.parse_args()


# global values
global layouts, edit_mask, intermediates
edit_mask = None
intermediates = None
H, W = 768, 768


### txt2panel
# - base llama model
llama_model_root = 'models/llama2_7b_chat'
llama = LlamaForCausalLM.from_pretrained(llama_model_root).cuda()
llama_tokenizer = AutoTokenizer.from_pretrained(llama_model_root)
llama_tokenizer.pad_token_id = (0)

# - lora
peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    inference_mode=False,
    r=64,
    target_modules=["q_proj", "v_proj"],
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
)
llama = get_peft_model(llama, peft_config)
llama = llama.eval().requires_grad_(False).cuda()
lora_weight_ele = torch.load('models/llama2_7b_lora_element.pth', map_location='cpu')   # load an empty lora here
lora_weight_box = torch.load('models/llama2_7b_lora_bbox.pth', map_location='cpu')


### panel2img
config = OmegaConf.load(args.config_path)
model = instantiate_from_config(config.model).cuda()
model.load_state_dict(torch.load('models/ranni_sdv21_v1.pth', map_location='cpu'), strict=False)    # TODO: delete clip loading in other place
ddim_sampler = DDIMSampler(model)


def seed_everything(seed=0):
    seed = seed if seed >= 0 else random.randint(0, 2 ** 32 - 1)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def pack_control(model, ins_masks, ins_texts, control_hint_dim=3):
    ins_texts = torch.cat([model.cond_stage_model(texts, return_global=True).unsqueeze(0) for texts in ins_texts], dim=0)   
    ins_masks = ins_masks.unsqueeze(2).repeat(1, 1, control_hint_dim, 1, 1) # B, N, C, H, W
    return (ins_masks, ins_texts)


def stage1_process(prompt, seed=-1):

    seed_everything(seed)
    device = 'cuda:0'

    # ------ element ----------
    llama.load_state_dict(lora_weight_ele, strict=False)
    ele_prompt = build_llama_prompt(
        message=element_template['prompt_template'].format(prompt),
        system_prompt=element_template['system_prompt_template']
    )
    ele_answer = chat(llama, llama_tokenizer, ele_prompt, device)

    # llama varies 2 different output formats without tuning here
    pred_elements = []
    try:
        pred_elements = seq_to_element(ele_answer)
    except:
        pred_elements = seq_to_element_v2(ele_answer)

    # ------- boxes ----------------
    llama.load_state_dict(lora_weight_box, strict=False)

    # prompt
    elements_str = ','.join(pred_elements)
    box_prompt = build_llama_prompt(
        message=box_template['prompt_template'].format(prompt, elements_str),
        system_prompt=box_template['system_prompt_template'],
    )
    
    # answer
    box_answer = chat(llama, llama_tokenizer, box_prompt, device)
    for line in box_answer.split('\n'):
        if '(' in line and ')' in line:
            box_answer = line
            break
    pred_boxes = seq_to_box(box_answer)

    # mapping to current resolution
    pred_boxes = [
        (box[0], [int(box[1][0] / 768 * W), int(box[1][1] / 768 * H), int(box[1][2] / 768 * W), int(box[1][3] / 768 * H)])
        for box in pred_boxes
    ]
    box_answer = box_to_seq(pred_boxes)
    vis_layout = draw_box(pred_boxes, H=H, W=W)

    global layouts
    layouts = []
    for box in pred_boxes:
        layouts.append({
            'label': box[0].lower(),
            'box': box[1],
        })
    
    layouts.append({
        'label': postfix,
        'box': [W // 2, H // 2, W, H]
    })

    vis_layout = (vis_layout.permute(0, 2, 3, 1) * 127.5 + 127.5).cpu().numpy().clip(0, 255).astype(np.uint8)
    return box_answer, vis_layout


def stage2_process(box_answer, prompt, postfix, seed, n_prompt, guide_scale, steps, control_max_t, control_min_t, panel_control_scale, with_memory, num_samples=1):

    global layouts, edit_mask, intermediates
    seed_everything(seed)

    # conditions
    prompt = prompt + ', ' + postfix
    shape = (4, H // 8, W // 8)

    # panel
    pred_boxes = seq_to_box(box_answer)
    layouts = []
    for box in pred_boxes:
        layouts.append({
            'label': box[0].lower(),
            'box': box[1],
        })
    
    layouts.append({
        'label': postfix,
        'box': [W // 2, H // 2, W, H]
    })

    # pack control
    ins_masks, ins_texts = [], []
    for layout in layouts:
        # label
        ins_texts.append(layout['label'])    

        # box
        mask = torch.zeros((H, W))
        x, y, w, h = layout['box']
        x1, y1 = x - w // 2, y - h // 2
        x2, y2 = x1 + w, y1 + h
        mask[y1: y2, x1: x2] = 1
        ins_masks.append(mask.unsqueeze(0))

    ins_masks = torch.cat(ins_masks, dim=0).unsqueeze(0).repeat(num_samples, 1, 1, 1).cuda()
    ins_texts = [ins_texts] * num_samples

    # cross-attention mask
    B = num_samples
    cross_attn_mask = torch.zeros((B, H, W, model.text_len)).cuda()
    batch_all_tokens = open_clip.tokenize(prompt)

    for b_idx in range(B):
        # remove eos token
        all_tokens = batch_all_tokens[b_idx]
        eos = all_tokens.argmax(dim=-1)
        all_tokens = torch.cat([all_tokens[:eos], all_tokens[eos + 1:]])

        matched_tokens = set()
        for obj_idx in range(ins_masks.size(1)):
            # get phrase token without special tokens
            phrase = ins_texts[b_idx][obj_idx]
            phrase_tokens = open_clip.tokenize(phrase)[0]
            phrase_tokens = phrase_tokens[1: phrase_tokens.argmax(dim=-1)]
            phrase_len = phrase_tokens.size(0)

            # find matching position in global text
            for i in range(len(all_tokens) - phrase_len):
                if (all_tokens[i: i + phrase_len] == phrase_tokens).all(): 
                    # set attn_mask
                    cross_attn_mask[b_idx, :, :, i: i + phrase_len] += ins_masks[b_idx, obj_idx].view(H, W, 1)

                    # remember matched tokens
                    for t in range(i, i + phrase_len):
                        matched_tokens.add(t)
                    break
        
        # set attn of unmatched tokens to all the image
        for t in range(len(all_tokens)):
            if t not in matched_tokens:
                cross_attn_mask[b_idx, :, :, t] = 1
    
    cond = {
        "context": model.get_learned_conditioning([prompt] * num_samples),
        "panel_control": model.panel_conditioner(pack_control(model, ins_masks, ins_texts)),
        "panel_control_scale": panel_control_scale,
        "control_max_t": control_max_t,
        "control_min_t": control_min_t,
        "cross_attn_mask": cross_attn_mask,
    }
    un_cond = {"context": model.get_learned_conditioning([n_prompt] * num_samples)}

    if edit_mask is not None:
        resized_edit_mask = torch.nn.functional.interpolate(edit_mask.view(1, 1, H, W), (H // 8, W // 8), mode='bilinear').cuda()

    samples, intermediates = ddim_sampler.sample(
        steps, num_samples, shape, cond, 
        verbose=False, eta=0.0,
        unconditional_guidance_scale=guide_scale,
        unconditional_conditioning=un_cond,
        log_every_t=1,
        edit_mask=resized_edit_mask if with_memory else None, edit_intermediates=intermediates)

    x_samples = model.decode_first_stage(samples)
    imgs = (x_samples.permute(0, 2, 3, 1) * 127.5 + 127.5).cpu().numpy().clip(0, 255).astype(np.uint8)
    vis_layout = (draw_box(pred_boxes, H=H, W=W, background=imgs[0]).permute(0, 2, 3, 1) * 127.5 + 127.5).cpu().numpy().clip(0, 255).astype(np.uint8)
    imgs = np.concatenate((imgs, vis_layout), axis=0)

    return imgs


def refresh_condition(box_answer, postfix):
    global layouts, edit_mask

    prev_layouts = layouts
    
    # get current layouts
    pred_boxes = seq_to_box(box_answer)
    layouts = []
    for box in pred_boxes:
        layouts.append({
            'label': box[0].lower(),
            'box': box[1],
        })
    
    layouts.append({
        'label': postfix,
        'box': [W // 2, H // 2, W, H]
    })

    # layout diff
    edit_mask = torch.zeros(1, H, W)
    margin = 10
    print(prev_layouts)
    print(layouts)

    # - new position
    for layout in layouts:
        if layout not in prev_layouts:
            x, y, w, h = layout['box']
            edit_mask[:, y - h // 2 - margin: y + h // 2 + margin, x - w // 2 - margin: x + w // 2 + margin] = 1
    
    # - old position
    for playout in prev_layouts:
        if playout not in layouts:
            x, y, w, h = playout['box']
            edit_mask[:, y - h // 2 - margin: y + h // 2 + margin, x - w // 2 - margin: x + w // 2 + margin] = 1
    
    # vis
    vis_layout = draw_box(pred_boxes, H=H, W=W)
    vis_edit_mask = edit_mask.unsqueeze(1).repeat(1, 3, 1, 1)
    vis_imgs = torch.cat([vis_layout, vis_edit_mask], dim=0)

    return (vis_imgs.permute(0, 2, 3, 1) * 127.5 + 127.5).cpu().numpy().clip(0, 255).astype(np.uint8)


block = gr.Blocks().queue()
with block:
    with gr.Row():
        gr.Markdown("## Demo of Ranni")
    with gr.Row():
        with gr.Column():
            prompt = gr.Textbox(label="Prompt")
            postfix = gr.Textbox(label="Postfix (put style-related prompt here)", value='4k image, best quality, extremely detailed')
            seed = gr.Slider(label="Seed", minimum=-1, maximum=2147483647, step=1, value=-1)
            run_button = gr.Button(label="Text-to-Panel", value="Text-to-Panel")
            with gr.Row():
                run_button2 = gr.Button(label="Panel-to-Image", value="Panel-to-Image")
                with_memory = gr.Checkbox(label="with memory", value=False)

            with gr.Accordion("Diffusion Adavanced Options", open=False):
                n_prompt = gr.Textbox(label="Negative Prompt", value='out of frame, lowres, text, error, cropped, worst quality, low quality, jpeg artifacts, ugly, duplicate, morbid, mutilated, out of frame, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, blurry, dehydrated, bad anatomy, bad proportions, extra limbs, cloned face, disfigured, gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, fused fingers, too many fingers, long neck, username, watermark, signature')
                guide_scale = gr.Slider(label="Guide Scale", minimum=0, maximum=20.0, value=7.5, step=0.1)
                steps = gr.Slider(label="Diffusion Steps", minimum=0, maximum=100, value=50, step=1)

            with gr.Accordion("Ranni Adavanced Options", open=False):
                with gr.Row():
                    control_max_t = gr.Slider(label="Control start", minimum=0, maximum=1000, value=1000, step=0)
                    control_min_t = gr.Slider(label="Control stop", minimum=0, maximum=1000, value=600, step=0)
                panel_control_scale = gr.Slider(label="Control scale", minimum=0, maximum=5.0, value=0.6, step=0.1)
        
        with gr.Column():
            mid_result_gallery = gr.Gallery(label='Output', show_label=True, elem_id="gallery").style(grid=1, height='auto')
            box_answer = gr.Textbox(label="Box Answer")
            refresh_button = gr.Button(label="refresh", value="refresh")

        with gr.Column():
            result_gallery = gr.Gallery(label='Output', show_label=True, elem_id="gallery").style(grid=1, height='auto')
    
    examples = [
        ['a black dog and a white cat', 15],
        ['a corgi on top of a chair', 95],
        ['5 red apple on the grass', 555],
        ['a man with red hat and green jacket', 78]
    ]
    gr.Examples(
        examples=examples,
        inputs=[prompt, seed],
    )
            
    run_button.click(
        fn=stage1_process, 
        inputs=[prompt, seed], 
        outputs=[box_answer, mid_result_gallery])
    
    run_button2.click(
        fn=stage2_process, 
        inputs=[box_answer, prompt, postfix, seed, n_prompt, guide_scale, steps, control_max_t, control_min_t, panel_control_scale, with_memory], 
        outputs=[result_gallery])

    refresh_button.click(
        fn=refresh_condition,
        inputs=[box_answer, postfix],
        outputs=[mid_result_gallery]
    )


block.launch(server_name='0.0.0.0')
