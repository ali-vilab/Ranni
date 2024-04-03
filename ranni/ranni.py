import torch
import open_clip
from ldm.models.diffusion.ddpm import LatentDiffusion
from ldm.modules.encoders.modules import FrozenOpenCLIPEmbedder
from ldm.util import instantiate_from_config


class RanniLDM(LatentDiffusion):

    def __init__(self, panel_conditioner_config, panel_key, text_len, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.panel_conditioner = instantiate_from_config(panel_conditioner_config)
        self.panel_key = panel_key
        self.text_len = text_len

    def apply_model(self, x_noisy, t, cond, index=0, prefix='', *args, **kwargs):
        assert isinstance(cond, dict)
        diffusion_model = self.model.diffusion_model
        eps = diffusion_model(x=x_noisy, timesteps=t, **cond)
        return eps
        

class HackedFrozenOpenCLIPEmbedder(FrozenOpenCLIPEmbedder):

    def forward(self, text, return_global=False):
        tokens = open_clip.tokenize(text)
        z = self.encode_with_transformer(tokens.to(self.device))

        # remove eos token
        new_z = []
        for i in range(tokens.size(0)):
            eos = tokens[i].argmax(dim=-1)
            if return_global:
                new_z.append(z[i, eos].unsqueeze(0))
            else:
                new_z.append(torch.cat([z[i, :eos], z[i, eos + 1:]]).unsqueeze(0))

        return torch.cat(new_z, dim=0)
