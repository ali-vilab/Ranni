import torch
import torch.nn as nn
from einops import rearrange


def zero_module(module):
    for p in module.parameters():
        p.detach().zero_()
    return module


class PanelConditioner(nn.Module):

    def __init__(self, hint_dim, text_dim, unet_dim):

        super().__init__()
        self.ins_box_mapping = nn.Sequential(
            nn.Conv2d(hint_dim, 16, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(16, 16, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(32, 96, 3, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(96, 96, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(96, 256, 3, stride=2, padding=1),
            nn.SiLU(),
            zero_module(nn.Conv2d(256, unet_dim, 3, padding=1))
        )
        self.ins_text_mapping = nn.Sequential(
            nn.Linear(text_dim, unet_dim),
            nn.SiLU(),
            zero_module(nn.Linear(unet_dim, unet_dim))
        )
    
    def forward(self, control):
        ins_masks, ins_texts = control 

        B, N, C = ins_texts.size()
        
        # box
        ins_masks = rearrange(ins_masks, 'b n c h w -> (b n) c h w')
        ins_2d = self.ins_box_mapping(ins_masks)                # B*N, C', pH, pW

        # text
        ins_texts = rearrange(ins_texts, 'b n c -> (b n) c')
        ins_1d = self.ins_text_mapping(ins_texts)              # B*N, C'

        # merging conditions for each instance
        downsampled_ins_mask = torch.nn.functional.interpolate(
            ins_masks[:, :1], (ins_2d.size(2), ins_2d.size(3)), mode='bilinear', align_corners=True
        )
        instances = ins_2d + ins_1d.unsqueeze(2).unsqueeze(3) * downsampled_ins_mask
        instances = rearrange(instances, '(b n) c h w -> b n c h w', b=B)

        # merging instances
        instances = instances.mean(dim=1)    # B, C', pH, pW

        return instances
        
