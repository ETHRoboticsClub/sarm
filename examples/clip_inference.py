#!/usr/bin/env python3
"""
Example of using the JAX CLIP model with the standalone tokenizer.
This example shows how to use the model WITHOUT any open_clip dependency.
"""

import jax.numpy as jnp
import numpy as np
from PIL import Image

from sarm.model.clip import CLIP, load_clip_npz
from sarm.utils.tokenizer import load_tokenizer


def preprocess_image(image: Image.Image) -> jnp.ndarray:
    """
    Preprocess an image for CLIP (without open_clip dependency).

    Args:
        image: PIL Image

    Returns:
        Preprocessed image tensor (3, 224, 224)
    """
    # Resize to 224x224
    image = image.resize((224, 224), Image.BICUBIC)

    # Convert to numpy array and normalize
    image = np.array(image).astype(np.float32) / 255.0

    # CLIP normalization
    mean = np.array([0.48145466, 0.4578275, 0.40821073])
    std = np.array([0.26862954, 0.26130258, 0.27577711])

    # Normalize each channel
    image = (image - mean) / std

    # Convert from HWC to CHW
    image = np.transpose(image, (2, 0, 1))

    return jnp.array(image)


def main():
    """Run a simple CLIP inference example."""
    print("=" * 80)
    print("JAX CLIP Inference Example (No OpenCLIP Dependency)")
    print("=" * 80)
    print()

    # 1. Load the tokenizer
    print("Loading tokenizer...")
    tokenizer = load_tokenizer()
    print(f"✓ Tokenizer loaded (vocab size: {len(tokenizer.encoder)})")
    print()

    # 2. Load the CLIP model
    print("Loading CLIP model...")
    model = CLIP(
        image_size=224,
        patch_size=32,
        vision_width=768,
        vision_layers=12,
        vision_heads=12,
        context_length=77,
        vocab_size=49408,
        text_width=512,
        text_layers=12,
        text_heads=8,
        embed_dim=512,
    )

    weights_path = "checkpoints/clip_vit_b32_openai.npz"
    model = load_clip_npz(model, weights_path)
    print(f"✓ Model loaded from {weights_path}")
    print()

    # 3. Prepare test data
    print("Preparing test data...")

    # Create a simple test image (red square on gray background)
    img = Image.new("RGB", (224, 224), (128, 128, 128))
    from PIL import ImageDraw

    d = ImageDraw.Draw(img)
    d.rectangle([64, 64, 160, 160], fill=(255, 0, 0))

    # Preprocess image
    image_tensor = preprocess_image(img)

    # Test captions
    texts = [
        "a red square",
        "a blue circle",
        "a gray background",
        "a red square on a gray background",
        "a photograph of nature",
    ]

    print(f"✓ Image: {image_tensor.shape}")
    print(f"✓ Texts: {len(texts)} captions")
    print()

    # 4. Tokenize texts
    print("Tokenizing texts...")
    text_tokens = tokenizer(texts)
    print(f"✓ Tokens shape: {text_tokens.shape}")
    print()

    # 5. Encode image (single image, not batched)
    print("Encoding image...")
    image_features = model.encode_image(image_tensor)
    image_features = image_features / jnp.linalg.norm(image_features)
    print(f"✓ Image features: {image_features.shape}")
    print()

    # 6. Encode texts (one at a time, not batched)
    print("Encoding texts...")
    text_features = []
    for i, text in enumerate(texts):
        tokens = jnp.array(text_tokens[i])
        features = model.encode_text(tokens)
        features = features / jnp.linalg.norm(features)
        text_features.append(features)
    text_features = jnp.stack(text_features)
    print(f"✓ Text features: {text_features.shape}")
    print()

    # 7. Compute similarities
    print("Computing similarities...")
    similarities = image_features @ text_features.T
    similarities = np.array(similarities)

    # Convert to probabilities
    probs = np.exp(similarities * 100) / np.sum(np.exp(similarities * 100))

    print()
    print("Results:")
    print("-" * 80)
    for i, text in enumerate(texts):
        print(f"{text:50s} | Similarity: {similarities[i]:.4f} | Prob: {probs[i]:.2%}")
    print("-" * 80)
    print()

    # Find best match
    best_idx = np.argmax(similarities)
    print(f"Best match: '{texts[best_idx]}' (similarity: {similarities[best_idx]:.4f})")
    print()

    print("✓ Inference complete!")
    print()
    print("Note: This example runs WITHOUT any open_clip dependency!")
    print("Only used for weight conversion initially.")


if __name__ == "__main__":
    main()
