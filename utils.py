from transformers import TextIteratorStreamer
import torch
import numpy as np
import random
import cv2
import torchvision.transforms as T
from PIL import Image
import math
import json


def build_llama_prompt(message, system_prompt):
    return f'<s>[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n{message} [/INST]'


def chat(model, tokenizer, prompt, gpu, max_new_tokens=1024, top_p=0.95, top_k=50, temperature=1.0, repetition_penalty=1.0):
    
    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(gpu)
    inputs.pop('token_type_ids')
    streamer = TextIteratorStreamer(tokenizer,
                                    timeout=2.,
                                    skip_prompt=True,
                                    skip_special_tokens=True)
    generate_kwargs = dict(
        inputs,
        streamer=streamer,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        top_p=top_p,
        top_k=top_k,
        temperature=temperature,
        repetition_penalty=repetition_penalty,
        num_beams=1,
    )
    generate_ids = model.generate(**generate_kwargs)

    return ''.join([a for a in streamer])


# ----------- system prompt -------------
description_template = {
    'system_prompt_template': 
        '''As a prompt modifier, your task is to rewrite a text sentence provided by humans to suit image generation models like stable diffusion. Emphasize spatial relationships within the picture. Keep it concise, under 25 words. Example:\n\n'''
        '''Original prompt: fruit, pineapple, bowl, cocktail glass\n'''
        '''New prompt: a bowl of fruit, including a pineapple, surrounded by other fruits and a cocktail glass. The bowl is placed on a table,  and the fruits are arranged in a visually appealing manner.\n\n'''
        '''Original prompt: a wedding\n'''
        '''New prompt: A beautiful bride and groom standing at the altar, surrounded by their smiling friends and family. The church's stained glass windows glow with warm light, casting a colorful and romantic atmosphere.\n\n''',
    
    'prompt_template':
        '''Now show me the new prompt of the original prompt: "{}":'''
}
element_template = {
    'system_prompt_template':  
        '''I will provide you a caption of image, please imagine the image and generate text description of all elements that should be contained in the image. Also show the number of each element.\n'''
        '''Only generate noun phrases indicating visible objects in the image. Include their description words, e.g. a white cat.\n'''
        '''For example:\n\n'''
        '''Caption: Two dogs and three cats playing on the grass, 4K image, best quality\nElements: (dog, 2), (cat, 3), (grass, 1)\n\n'''
        '''Caption: Realistic photo of a group of people walking on the street, cars, tall buildings, a round moon in the black sky\nElements: (person, 7), (street, 1), (car, 3), (round moon, 1), (black sky, 1)\n\n'''
        '''Caption: Draw an image of a basket of green apples on the wooden table, in style of oil painting\nElements: (green apple, 6), (wooden table, 1)\n\n'''
        '''Caption: A black cat and a white dog lying on the sofa, white background\nElements: (black cat, 1), (white dog, 1), (sota, 1), (white background, 1)\n\n'''
        '''Caption: A boy with red hair, white skirt, blue pants and black shoes, 2D anime\nElements: (boy, 1), (red hair, 1), (white skirt, 1), (blue pants, 1), (black shoe, 2)\n\n''',

    'prompt_template':
        '''Now show me the elements of caption "{}" in the above format. Answer shortly. Directly answer the elements. Do not repeat the caption.''',
}

box_template = {
    'system_prompt_template': 'I will provide you a caption of an image and all elements. Your task is to imagine the image and generate the bounding boxes for the provided element in format of [x,y,w,h].\n',
    'prompt_template': '''Now show me the boxes of "Caption: {}\nElements: {}".\n'''
}


# --------- condition convertion ------------

def seq_to_element(seq):

    seq = seq.lower()

    s_pos = [i for i, u in enumerate(seq) if u == '(']
    e_pos = [i for i, u in enumerate(seq) if u == ')']
    num = min(len(s_pos), len(e_pos))

    elements = []
    for s, e in zip(
        s_pos[:num], e_pos[:num]):
        phrase = seq[s+1: e]
        label = phrase.split(',')[0]
        num = int(phrase.split(',')[1])
        for _ in range(num):
            elements.append(label)
    
    return elements


def seq_to_element_v2(seq):

    elements = []
    for line in seq.split('\n'):
        # if not line.startswith('*'):
        #     continue
        try:
            if line.startswith('*'):
                label = line[2:].split(' (')[0]
            else:
                label = line[3:].split(' (')[0]
            num = int(line.split(' (')[1][:-1])
            for _ in range(num):
                elements.append(label)
        except:
            pass
    
    return elements


def seq_to_box(seq):

    s_pos = [i for i, u in enumerate(seq) if u == '(']
    e_pos = [i for i, u in enumerate(seq) if u == ')']
    num = min(len(s_pos), len(e_pos))

    pred_boxes = []
    for s, e in zip(s_pos[:num], e_pos[:num]):
        phrase = seq[s+1: e]
        label = phrase.split(',', 1)[0]
        box = json.loads(phrase.split(',', 1)[1])
        pred_boxes.append((label, box))

    return pred_boxes


def box_to_seq(boxes):
    box_strs = []
    for box in boxes:
        label = box[0]
        x, y, w, h = box[1]
        box_strs.append(
            f'({label},[{int(x)},{int(y)},{int(w)},{int(h)}])'
        )
    
    return ','.join(box_strs)


def draw_box(pred_boxes, H=1024, W=1024, background=None):

    colors = [[255, 0, 0], [0, 255, 170], [85, 0, 255], [170, 0, 255], [255, 0, 85], [170, 255, 0], [255, 0, 255], [0, 255, 85], [0, 255, 0], [255, 0, 170], [255, 170, 0], [0, 170, 255], [85, 255, 0], [0, 85, 255], [255, 255, 0], [0, 255, 255], [255, 85, 0], [0, 0, 255]]    
    
    canvas = np.zeros(shape=(H, W, 3), dtype=np.uint8)

    try:
        for i, layout in enumerate(pred_boxes):
            label = layout[0]
            x, y, w, h = layout[1]
            x1, y1 = x - w // 2, y - h // 2
            x2, y2 = x1 + w, y1 + h
            color = colors[i % len(colors)]

            cv2.rectangle(canvas, (x1, y1), (x2, y2), color=color, thickness=4)
            cv2.rectangle(canvas, (x1, y1), (x2, y1 + 40), color, -1)
            cv2.putText(canvas, label, (x1, y1 + 20), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), thickness=2)
    except:
        pass
    
    if background is not None:
        mask = (canvas.sum(axis=2) == 0).astype(np.uint8)[:, :, None]
        canvas = background * mask + canvas * (1 - mask)

    return T.ToTensor()(Image.fromarray(canvas)).unsqueeze(0) * 2 - 1
