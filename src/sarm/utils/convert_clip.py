# convert_weights.py
import numpy as np
import open_clip
import torch


def get_vitb32_visual(model):
    # Works for both open_clip and OpenAI CLIP variants that open_clip exposes.
    return model.visual


def extract_qkv_and_out_from_mha(attn):
    # attn is torch.nn.MultiheadAttention
    # QKV are packed along dim 0: [3*d, d]
    W = attn.in_proj_weight.detach().cpu().numpy()  # (3d, d)
    b = attn.in_proj_bias.detach().cpu().numpy()  # (3d,)
    Wq, Wk, Wv = np.split(W, 3, axis=0)
    bq, bk, bv = np.split(b, 3, axis=0)

    Wout = attn.out_proj.weight.detach().cpu().numpy()  # (d, d)
    bout = attn.out_proj.bias.detach().cpu().numpy()  # (d,)
    return (Wq, bq), (Wk, bk), (Wv, bv), (Wout, bout)


def to_numpy(t):
    return t.detach().cpu().numpy()


def main():
    # 1) Load ViT-B/32
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="openai", force_quick_gelu=True
    )
    model.eval()

    visual = get_vitb32_visual(model)

    # Sanity: basic shapes
    embed_dim = visual.output_dim  # 512
    width = visual.conv1.out_channels  # 768
    patch_size = visual.conv1.kernel_size[0]  # 32
    grid = visual.grid_size  # 7 (for 224x224)
    layers = len(visual.transformer.resblocks)  # 12
    heads = visual.transformer.width // 64  # 12

    # 2) Collect weights into a flat dict (Equinox-expected names)
    params = {}

    # Patchify conv -> Equinox Conv2d expects (out_c, in_c, kh, kw)
    conv = visual.conv1
    params["patch.weight"] = to_numpy(conv.weight)  # [width, 3, 32, 32]
    # conv1 in CLIP has no bias

    # Class token + positional embedding
    # In CLIP (open_clip), class embedding is .class_embedding, pos embed is .positional_embedding
    params["cls"] = to_numpy(visual.class_embedding)[None, :]  # (1,768)
    params["pos"] = to_numpy(visual.positional_embedding)  # (1+N, 768)

    # Pre-transformer LayerNorm at beginning of encoder
    params["ln_pre.weight"] = to_numpy(visual.ln_pre.weight)
    params["ln_pre.bias"] = to_numpy(visual.ln_pre.bias)

    # Pre-transformer LayerNorm at the end of encoder
    params["ln_post.weight"] = to_numpy(visual.ln_post.weight)
    params["ln_post.bias"] = to_numpy(visual.ln_post.bias)

    # Projection to embed_dim (768 -> 512)
    # In CLIP: visual.proj is (768, 512) in PyTorch shape (in, out)
    params["proj.weight"] = to_numpy(
        visual.proj
    )  # torch Parameter with shape [768, 512]

    # Transformer blocks
    for i, blk in enumerate(visual.transformer.resblocks):
        (Wq, bq), (Wk, bk), (Wv, bv), (Wout, bout) = extract_qkv_and_out_from_mha(
            blk.attn
        )
        base = f"blocks.{i}"
        params[f"{base}.attn.q.weight"] = Wq
        params[f"{base}.attn.q.bias"] = bq
        params[f"{base}.attn.k.weight"] = Wk
        params[f"{base}.attn.k.bias"] = bk
        params[f"{base}.attn.v.weight"] = Wv
        params[f"{base}.attn.v.bias"] = bv
        params[f"{base}.attn.out.weight"] = Wout
        params[f"{base}.attn.out.bias"] = bout
        # LayerNorms
        params[f"{base}.ln1.weight"] = to_numpy(blk.ln_1.weight)
        params[f"{base}.ln1.bias"] = to_numpy(blk.ln_1.bias)
        params[f"{base}.ln2.weight"] = to_numpy(blk.ln_2.weight)
        params[f"{base}.ln2.bias"] = to_numpy(blk.ln_2.bias)

        # MLP: fc1 (Linear), act=QuickGELU, fc2 (Linear)
        params[f"{base}.mlp.fc1.weight"] = to_numpy(blk.mlp.c_fc.weight)
        params[f"{base}.mlp.fc1.bias"] = to_numpy(blk.mlp.c_fc.bias)
        params[f"{base}.mlp.fc2.weight"] = to_numpy(blk.mlp.c_proj.weight)
        params[f"{base}.mlp.fc2.bias"] = to_numpy(blk.mlp.c_proj.bias)

    # 3) Save model meta + weights
    meta = dict(
        embed_dim=embed_dim,
        width=width,
        patch_size=patch_size,
        grid=grid,
        layers=layers,
        heads=heads,
        image_size=224,
    )
    np.savez(
        "checkpoints/vit_b32_openai_weights.npz",
        **params,
        **{f"meta.{k}": v for k, v in meta.items()},
    )
    print("Saved vit_b32_openai_weights.npz")


if __name__ == "__main__":
    main()
