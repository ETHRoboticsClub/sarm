# eqx_clip_vit_b32.py
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np


def quick_gelu(x):
    # Matches CLIP QuickGELU: x * sigmoid(1.702 * x)
    return x * jax.nn.sigmoid(1.702 * x)


class MLP(eqx.Module):
    fc1: eqx.nn.Linear
    fc2: eqx.nn.Linear

    def __init__(self, d, mlp_ratio=4, key=jr.PRNGKey(0)):
        k1, k2 = jr.split(key)
        self.fc1 = eqx.nn.Linear(d, d * mlp_ratio, key=k1, use_bias=True)
        self.fc2 = eqx.nn.Linear(d * mlp_ratio, d, key=k2, use_bias=True)

    def __call__(self, x):
        return self.fc2(quick_gelu(self.fc1(x)))


class SelfAttn(eqx.Module):
    q: eqx.nn.Linear
    k: eqx.nn.Linear
    v: eqx.nn.Linear
    out: eqx.nn.Linear
    nheads: int
    head_dim: int
    scale: float

    def __init__(self, d, nheads, key=jr.PRNGKey(0)):
        kq, kk, kv, ko = jr.split(key, 4)
        self.q = eqx.nn.Linear(d, d, key=kq, use_bias=True)
        self.k = eqx.nn.Linear(d, d, key=kk, use_bias=True)
        self.v = eqx.nn.Linear(d, d, key=kv, use_bias=True)
        self.out = eqx.nn.Linear(d, d, key=ko, use_bias=True)
        self.nheads = nheads
        self.head_dim = d // nheads
        self.scale = 1.0 / jnp.sqrt(self.head_dim)

    def __call__(self, x):
        N, D = x.shape

        def shape_heads(t):
            t = t.reshape(N, self.nheads, self.head_dim)
            return jnp.transpose(t, (1, 0, 2))  # (H, N, Hd)

        q = shape_heads(self.q(x))
        k = shape_heads(self.k(x))
        v = shape_heads(self.v(x))

        attn = jnp.einsum("hnd,hmd->hnm", q, k) * self.scale
        attn = jax.nn.softmax(attn, axis=-1)
        out = jnp.einsum("hnm,hmd->hnd", attn, v)
        out = jnp.transpose(out, (1, 0, 2)).reshape(N, D)
        return self.out(out)


class VmappedLayerNorm(eqx.Module):
    ln: eqx.nn.LayerNorm

    def __init__(self, d, eps=1e-5):
        self.ln = eqx.nn.LayerNorm(d, eps=eps)

    def __call__(self, x):
        return jax.vmap(self.ln)(x)


class Block(eqx.Module):
    ln1: VmappedLayerNorm
    ln2: VmappedLayerNorm
    attn: SelfAttn
    mlp: MLP

    def __init__(self, d, nheads, key=jr.PRNGKey(0)):
        ka, km = jr.split(key)
        self.ln1 = VmappedLayerNorm(d, eps=1e-5)
        self.ln2 = VmappedLayerNorm(d, eps=1e-5)
        self.attn = SelfAttn(d, nheads, key=ka)
        self.mlp = MLP(d, mlp_ratio=4, key=km)

    def __call__(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class ViTB32(eqx.Module):
    patch: eqx.nn.Conv2d
    cls: jnp.ndarray
    pos: jnp.ndarray
    blocks: list
    ln_pre: eqx.nn.LayerNorm
    ln_post: eqx.nn.LayerNorm
    proj: jnp.ndarray  # (768, 512)

    image_size: int
    patch_size: int
    d: int
    nheads: int
    layers: int

    def __init__(
        self,
        image_size=224,
        patch_size=32,
        d=768,
        layers=12,
        nheads=12,
        key=jr.PRNGKey(0),
    ):
        k_conv, k_cls, k_blocks = jr.split(key, 3)
        self.patch = eqx.nn.Conv2d(
            3, d, kernel_size=patch_size, stride=patch_size, use_bias=False, key=k_conv
        )
        n = (image_size // patch_size) ** 2
        self.cls = jr.normal(k_cls, (1, 1, d))
        self.pos = jr.normal(k_cls, (1, n + 1, d)) * 0.01
        self.blocks = [
            Block(d, nheads, key=jr.fold_in(k_blocks, i)) for i in range(layers)
        ]
        self.ln_pre = VmappedLayerNorm(d, eps=1e-5)
        self.ln_post = VmappedLayerNorm(d, eps=1e-5)
        self.proj = jr.normal(k_cls, (d, 512)) * (d**-0.5)

        self.image_size = image_size
        self.patch_size = patch_size
        self.d = d
        self.nheads = nheads
        self.layers = layers

    def __call__(self, img):
        # img: (3, H, W), values already CLIP-normalized
        x = self.patch(img)  # (d, H/ps, W/ps)
        D, Hp, Wp = x.shape
        x = jnp.reshape(x, (D, Hp * Wp))
        x = jnp.transpose(x, (1, 0))  # (N, d)
        x = jnp.concatenate([self.cls, x], axis=0) + self.pos

        x = self.ln_pre(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.ln_post(x)
        cls = x[0, :]  # (d,)
        feat = cls @ self.proj  # (512,)
        return feat


def load_npz_into_model(model: ViTB32, path: str) -> ViTB32:
    data = np.load(path)
    # Patch conv
    model = eqx.tree_at(
        lambda m: m.patch.weight, model, jnp.asarray(data["patch.weight"])
    )

    # Tokens/pos
    model = eqx.tree_at(lambda m: m.cls, model, jnp.asarray(data["cls"]))
    model = eqx.tree_at(lambda m: m.pos, model, jnp.asarray(data["pos"]))

    # Blocks
    def assign_block(b: Block, i: int):
        b = eqx.tree_at(
            lambda x: x.ln1.ln.weight, b, jnp.asarray(data[f"blocks.{i}.ln1.weight"])
        )
        b = eqx.tree_at(
            lambda x: x.ln1.ln.bias, b, jnp.asarray(data[f"blocks.{i}.ln1.bias"])
        )
        b = eqx.tree_at(
            lambda x: x.ln2.ln.weight, b, jnp.asarray(data[f"blocks.{i}.ln2.weight"])
        )
        b = eqx.tree_at(
            lambda x: x.ln2.ln.bias, b, jnp.asarray(data[f"blocks.{i}.ln2.bias"])
        )

        b = eqx.tree_at(
            lambda x: x.attn.q.weight, b, jnp.asarray(data[f"blocks.{i}.attn.q.weight"])
        )
        b = eqx.tree_at(
            lambda x: x.attn.q.bias, b, jnp.asarray(data[f"blocks.{i}.attn.q.bias"])
        )
        b = eqx.tree_at(
            lambda x: x.attn.k.weight, b, jnp.asarray(data[f"blocks.{i}.attn.k.weight"])
        )
        b = eqx.tree_at(
            lambda x: x.attn.k.bias, b, jnp.asarray(data[f"blocks.{i}.attn.k.bias"])
        )
        b = eqx.tree_at(
            lambda x: x.attn.v.weight, b, jnp.asarray(data[f"blocks.{i}.attn.v.weight"])
        )
        b = eqx.tree_at(
            lambda x: x.attn.v.bias, b, jnp.asarray(data[f"blocks.{i}.attn.v.bias"])
        )

        b = eqx.tree_at(
            lambda x: x.attn.out.weight,
            b,
            jnp.asarray(data[f"blocks.{i}.attn.out.weight"]),
        )
        b = eqx.tree_at(
            lambda x: x.attn.out.bias, b, jnp.asarray(data[f"blocks.{i}.attn.out.bias"])
        )

        b = eqx.tree_at(
            lambda x: x.mlp.fc1.weight,
            b,
            jnp.asarray(data[f"blocks.{i}.mlp.fc1.weight"]),
        )
        b = eqx.tree_at(
            lambda x: x.mlp.fc1.bias, b, jnp.asarray(data[f"blocks.{i}.mlp.fc1.bias"])
        )
        b = eqx.tree_at(
            lambda x: x.mlp.fc2.weight,
            b,
            jnp.asarray(data[f"blocks.{i}.mlp.fc2.weight"]),
        )
        b = eqx.tree_at(
            lambda x: x.mlp.fc2.bias, b, jnp.asarray(data[f"blocks.{i}.mlp.fc2.bias"])
        )
        return b

    blocks = [assign_block(b, i) for i, b in enumerate(model.blocks)]
    model = eqx.tree_at(lambda m: m.blocks, model, blocks)

    # Final LN
    model = eqx.tree_at(lambda m: m.ln_pre.ln.weight, model, jnp.asarray(data["ln_pre.weight"]))
    model = eqx.tree_at(lambda m: m.ln_pre.ln.bias, model, jnp.asarray(data["ln_pre.bias"]))
    model = eqx.tree_at(lambda m: m.ln_post.ln.weight, model, jnp.asarray(data["ln_post.weight"]))
    model = eqx.tree_at(lambda m: m.ln_post.ln.bias, model, jnp.asarray(data["ln_post.bias"]))

    # Projection (PyTorch stored as [768,512], Equinox uses the same matmul ordering cls @ proj)
    model = eqx.tree_at(lambda m: m.proj, model, jnp.asarray(data["proj.weight"]))

    return model
