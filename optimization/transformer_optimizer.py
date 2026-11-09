import torch
from torch.nn import functional as F


def generate_parallel(net, prefix, valid, sampling, eos, perf):
    """Reuse frozen prefix K/V while predicting a complete scale per pass."""
    cfg = sampling.cfg_scale
    prefix = prefix[:, :valid]
    if cfg:
        null = torch.full_like(prefix, net.prefix_pad_id)
        null[:, 0] = eos
        prefix = torch.cat((prefix, null), dim=0)
    keep = prefix != net.prefix_pad_id
    batch = prefix.shape[0]
    vqvae = net.vqvae_local[0]
    vq = vqvae.vq
    stack = net.transformer
    latent = torch.zeros(batch, net.levels[-1], vqvae.embed_dim, device=prefix.device)
    with perf.stage("input_embedding"):
        hidden = stack.wte(vqvae.encoder.wte(prefix))
        if not net.use_rope:
            hidden = hidden + net.wpe(net.wp_prefix[:, :valid])
        level_pos = stack.wle(net.lvl_1L) if net.config.use_level_encoding else 0
        if not net.use_rope:
            level_pos = level_pos + net.wpe(net.wp_1L)
    blocks = stack.blocks
    net._set_kv_caching(True)
    try:
        with perf.stage("transformer_prefix"):
            rope = (net.rope_cos_gen_0, net.rope_sin_gen_0) if net.use_rope else None
            for block in blocks:
                hidden = block(
                    hidden, attn_mask=keep[:, None, None, :], rope_cos_sin=rope
                )
            prefix_kv = [(block.attn.cached_k, block.attn.cached_v) for block in blocks]

        offset = 0
        levels = []
        features = net.build_l0_query_input(batch)
        for i, width in enumerate(net.levels):
            with perf.stage(f"transformer_level_{i}"):
                if net.use_post_upsample_conv:
                    features = net._apply_post_upsample_conv_single(features, i)
                x = stack.wte(features)
                if isinstance(level_pos, torch.Tensor):
                    x = x + level_pos[:, offset : offset + width]
                if net.use_rope:
                    rope = tuple(
                        torch.cat((buffer[:valid], buffer[net.prefix_len :]), dim=0)
                        for buffer in (
                            getattr(net, f"rope_cos_gen_{i}"),
                            getattr(net, f"rope_sin_gen_{i}"),
                        )
                    )
                mask = F.pad(keep, (0, width), value=True)[:, None, None, :]
                for block, (k, v) in zip(blocks, prefix_kv):
                    block.attn.cached_k, block.attn.cached_v = k, v
                    x = block(x, attn_mask=mask, rope_cos_sin=rope)
                logits = net.lm_heads[i](stack.ln_f(x))
            with perf.stage("concept_sampling"):
                if cfg:
                    conditional, null = logits.chunk(2)
                    strength = cfg * i / max(len(net.levels) - 1, 1)
                    logits = (
                        (1 + strength) * conditional - strength * null
                        if i
                        else conditional
                    )
                if sampling.temperature == 0 or (
                    sampling.top_k == 1 and sampling.top_p == 0
                ):
                    ids = logits.argmax(dim=-1)
                else:
                    ids = net._sample_level_raw(
                        logits, sampling.temperature, sampling.top_k, sampling.top_p
                    )
                levels.append(ids)
                if cfg:
                    ids = torch.cat((ids, ids), dim=0)
            with perf.stage("rvq_generation"):
                latent, features = vq.get_next_autoregressive_input(
                    i, latent, vq.embeddings[i](ids)
                )
            offset += width
    finally:
        net._set_kv_caching(False)
    return torch.cat(levels, dim=1), latent[:1] if cfg else latent
