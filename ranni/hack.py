from ldm.modules.diffusionmodules.openaimodel import AttentionBlock


def _hacked_unet_forward(self, x, timesteps=None, context=None, y=None, panel_control=None, cross_attn_mask=None, **kwargs):
    assert (y is not None) == (
        self.num_classes is not None
    ), "must specify y if and only if the model is class-conditional"
    hs = []
    t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False)
    emb = self.time_embed(t_emb)

    if self.num_classes is not None:
        assert y.shape[0] == x.shape[0]
        emb = emb + self.label_emb(y)

    h = x.type(self.dtype)
    for module in self.input_blocks:
        if isinstance(module, AttentionBlock):
            h = module(h, emb, context, cross_attn_mask)
        else:
            h = module(h, emb, context)

        if len(hs) == 0 and panel_control is not None:
            h += panel_control
        hs.append(h)
    h = self.middle_block(h, emb, context, cross_attn_mask)
    for module in self.output_blocks:
        h = th.cat([h, hs.pop()], dim=1)
        if isinstance(module, AttentionBlock):
            h = module(h, emb, context, cross_attn_mask)
        else:
            h = module(h, emb, context)
    h = h.type(x.dtype)
    if self.predict_codebook_ids:
        return self.id_predictor(h)
    else:
        return self.out(h)
