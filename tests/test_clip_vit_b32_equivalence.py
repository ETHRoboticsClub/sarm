# tests/test_clip_vit_b32_equivalence.py
import os

import jax.numpy as jnp
import numpy as np
import open_clip
import pytest
import torch
from PIL import Image, ImageDraw

from sarm.model.clip import ViTB32, load_npz_into_model
from sarm.utils.convert_clip import main as export_vit_b32_weights

WEIGHTS_PATH = "vit_b32_openai_weights.npz"


@pytest.fixture(scope="session")
def pt_model_and_preprocess():
    # Load PyTorch CLIP ViT-B/32 (openai weights) and its official preprocess
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="openai"
    )
    model.eval()
    return model, preprocess


@pytest.fixture(scope="session")
def ensure_weights(pt_model_and_preprocess):
    # Export weights once per session if not already present
    if not os.path.exists(WEIGHTS_PATH):
        export_vit_b32_weights()
    assert os.path.exists(WEIGHTS_PATH), "Failed to export CLIP weights to .npz"
    return WEIGHTS_PATH


def _make_test_images():
    """Create two deterministic 224x224 RGB images as PIL Images."""
    # 1) Solid color with a small contrasting square
    img1 = Image.new("RGB", (224, 224), (220, 30, 30))
    d = ImageDraw.Draw(img1)
    d.rectangle([64, 64, 160, 160], fill=(30, 220, 30))

    # 2) Horizontal gradient
    img2 = Image.new("RGB", (224, 224))
    pixels = img2.load()
    for x in range(224):
        val = int(255 * x / 223)
        for y in range(224):
            pixels[x, y] = (val, 255 - val, (val // 2))
    return [img1, img2]


def _torch_forward_visual(model, imgs_pt):
    with torch.no_grad():
        out = model.visual(imgs_pt)  # Expect (B, 512)
    if (
        not isinstance(out, torch.Tensor)
        or out.ndim != 2
        or out.shape[-1] != model.visual.output_dim
    ):
        raise RuntimeError(
            f"Unexpected visual forward output shape: {getattr(out, 'shape', None)}"
        )
    return out


def test_vit_b32_image_features_match_pytorch(pt_model_and_preprocess, ensure_weights):
    model, preprocess = pt_model_and_preprocess

    # ----- Build (and load weights into) the Equinox model -----
    eq_model = ViTB32(image_size=224, patch_size=32, d=768, layers=12, nheads=12)
    eq_model = load_npz_into_model(eq_model, ensure_weights)

    # ----- Make two test images and preprocess with official transforms -----
    pil_imgs = _make_test_images()
    imgs_pt = torch.stack(
        [preprocess(im.convert("RGB")) for im in pil_imgs], dim=0
    )  # (B,3,224,224)

    # ----- PyTorch path -----
    pt_feat = _torch_forward_visual(model, imgs_pt).cpu().numpy()  # (B,512)

    # ----- Equinox path -----
    imgs_np = imgs_pt.numpy()  # (B,3,224,224) already CLIP-normalized
    imgs_jax = jnp.asarray(imgs_np)
    eq_feat = np.array(eq_model(imgs_jax))  # (B,512)

    # ----- Compare -----
    assert pt_feat.shape == eq_feat.shape == (2, 512)
    max_abs = np.max(np.abs(eq_feat - pt_feat))
    mean_abs = np.mean(np.abs(eq_feat - pt_feat))

    # Tight tolerances; relax very slightly if different BLAS/backends cause tiny drift
    atol = 1e-5
    rtol = 1e-5
    ok = np.allclose(eq_feat, pt_feat, atol=atol, rtol=rtol)
    if not ok:
        # Helpful diagnostics in pytest output
        diff = eq_feat - pt_feat
        print("Max |diff|:", max_abs)
        print("Mean |diff|:", mean_abs)
        # Print a small slice to inspect
        print("eq_feat[0, :8]:", eq_feat[0, :8])
        print("pt_feat[0, :8]:", pt_feat[0, :8])
    assert (
        ok
    ), f"Equinox features differ from PyTorch (max|diff|={max_abs:.3e}, mean|diff|={mean_abs:.3e})"
