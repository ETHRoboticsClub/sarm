import equinox as eqx
import jax
import jax.random as jr

from sarm.model.clip import Block


class SubtaskTransformer(eqx.Module):

    vis_proj: eqx.nn.Linear
    text_proj: eqx.nn.Linear
    state_proj: eqx.nn.Linear
    blocks: list

    def __init__(
        self,
        d_model=512,
        nheads=8,
        layers=12,
        vis_embed_dim=512,
        text_embed_dim=512,
        state_dim=14,
        num_cameras=1,
        key=jr.PRNGKey(0),
    ):
        k_blocks, k_vis, k_text, k_state = jr.split(key, 4)
        self.vis_proj = eqx.nn.Linear(vis_embed_dim, d_model, key=k_vis)
        self.text_proj = eqx.nn.Linear(text_embed_dim, d_model, key=k_text)
        self.state_proj = eqx.nn.Linear(state_dim, d_model, key=k_state)

        self.blocks = [
            Block(d_model, nheads, key=jr.fold_in(k_blocks, i)) for i in range(layers)
        ]

    def __call__(self, img_features, text_features, state, subtask):
        raise NotImplementedError("Not implemented")
