import einops
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr

from sarm.model.clip import Block


class ProcessTransformer(eqx.Module):

    vis_proj: eqx.nn.Linear
    text_proj: eqx.nn.Linear
    state_proj: eqx.nn.Linear
    fusion_mlp: eqx.nn.Sequential
    blocks: list
    positional_embedding: jnp.ndarray

    def __init__(
        self,
        d_model: int = 512,
        nheads: int = 8,
        layers: int = 12,
        vis_embed_dim: int = 512,
        text_embed_dim: int = 512,
        state_dim: int = 14,
        num_cameras: int = 1,
        key=jr.PRNGKey(0),
    ):
        k_blocks, k_vis, k_text, k_state, k_fusion, k_sparse, k_dense = jr.split(key, 7)
        self.vis_proj = eqx.nn.Linear(vis_embed_dim, d_model, key=k_vis)
        self.text_proj = eqx.nn.Linear(text_embed_dim, d_model, key=k_text)
        self.state_proj = eqx.nn.Linear(state_dim, d_model, key=k_state)
        self.final_proj = {
            "sparse": eqx.nn.Linear(d_model, 1, key=k_sparse),
            "dense": eqx.nn.Linear(d_model, 1, key=k_dense),
        }
        self.fusion_mlp = eqx.nn.Sequential(
            [
                eqx.nn.LayerNorm(
                    (num_cameras + 3) * d_model,
                ),
                eqx.nn.Linear((num_cameras + 3) * d_model, d_model, key=k_fusion),
                jax.nn.relu,
            ],
        )

        self.blocks = [Block(d_model, nheads, key=jr.fold_in(k_blocks, i)) for i in range(layers)]
        self.positional_embedding = jnp.zeros((1, d_model))
        self.num_cameras = num_cameras
        self.d_model = d_model

    def _subtask_encoding(self, subtask: jnp.ndarray):
        """Encode subtask features.


        Args:
            subtask (jnp.ndarray): Subtask features of shape (T, C)

        Returns:
            jnp.ndarray: Subtask features of shape (T, d_model)
        """
        if subtask.shape[-1] == self.vis_proj.out_features:
            return subtask
        elif subtask.shape[-1] > self.vis_proj.out_features:
            return subtask[:, : self.vis_proj.out_features]
        else:
            return jnp.concatenate(
                [
                    subtask,
                    jnp.zeros(
                        (
                            subtask.shape[0],
                            self.d_model - subtask.shape[-1],
                        )
                    ),
                ],
                axis=-1,
            )

    def _build_mask(self, timesteps: int, length: int):
        """Build mask for the subtask transformer.

        Args:
            length (int): Length of the sequence

        Returns:
            jnp.ndarray: Mask of shape ((N+3)*T, (N+3)*T)
        """
        mask_1d = jnp.arange(timesteps) < length
        mask_1d = jnp.where(mask_1d, 0.0, float("-inf"))
        mask_1d = einops.rearrange(mask_1d, "t -> (n t)", n=self.num_cameras + 3)
        mask = mask_1d[None, :] + mask_1d[:, None]  # (N+3)*T, (N+3)*T
        return mask

    def __call__(
        self,
        img_features: jnp.ndarray,
        text_features: jnp.ndarray,
        state: jnp.ndarray,
        subtask: jnp.ndarray,
        length: int,
        schema: str = "sparse",
    ):
        """Forward pass for the subtask transformer.

        Args:
            img_features (jnp.ndarray): Image features of shape (N, T, d_vis)
            text_features (jnp.ndarray): Text features of shape (T, d_text)
            state (jnp.ndarray): State features of shape (T, d_state)
            subtask (jnp.ndarray): Subtask features of shape (T, C)

        Returns:
            jnp.ndarray: Output features of shape (T)
        """
        N, T, D = img_features.shape
        img_features = jax.vmap(jax.vmap(self.vis_proj))(img_features)  # (N, T, d_model)
        text_features = jax.vmap(self.text_proj)(text_features)[None, ...]  # (1, T, d_model)
        state_features = jax.vmap(self.state_proj)(state)[None, ...]  # (1, T, d_model)
        subtask_features = self._subtask_encoding(subtask)[None, ...]  # (1, T, d_model)

        # Combine features
        features = jnp.concatenate(
            [img_features, text_features, state_features, subtask_features], axis=0
        )  # (N + 3, T, d_model)

        features = features.at[:N, 0, :].add(self.positional_embedding)
        features = einops.rearrange(features, "n t d -> (n t) d")  # ((N+3)*T, d_model)

        mask = self._build_mask(length, T)  # ((N+3)*T, (N+3)*T)

        # Apply transformer blocks
        for block in self.blocks:
            features = block(features, mask)

        features = einops.rearrange(features, "(n t) d -> t (n d)")
        features = jax.vmap(self.fusion_mlp)(features)  # (T, d_model)

        features = jax.vmap(self.final_proj[schema])(features).squeeze(-1)  # (T,)

        return jax.vmap(jax.nn.sigmoid)(features)  # (T,)


class StageTransformer(eqx.Module):

    vis_proj: eqx.nn.Linear
    text_proj: eqx.nn.Linear
    state_proj: eqx.nn.Linear
    fusion_mlp: eqx.nn.Sequential
    blocks: list
    positional_embedding: jnp.ndarray

    def __init__(
        self,
        d_model: int = 512,
        nheads: int = 8,
        layers: int = 12,
        vis_embed_dim: int = 512,
        text_embed_dim: int = 512,
        state_dim: int = 14,
        num_cameras: int = 1,
        num_classes_sparse: int = 4,
        num_classes_dense: int = 8,
        key=jr.PRNGKey(0),
    ):
        k_blocks, k_vis, k_text, k_state, k_fusion, k_sparse, k_dense = jr.split(key, 7)
        self.vis_proj = eqx.nn.Linear(vis_embed_dim, d_model, key=k_vis)
        self.text_proj = eqx.nn.Linear(text_embed_dim, d_model, key=k_text)
        self.state_proj = eqx.nn.Linear(state_dim, d_model, key=k_state)
        self.final_proj = {
            "sparse": eqx.nn.Linear(d_model, num_classes_sparse, key=k_sparse),
            "dense": eqx.nn.Linear(d_model, num_classes_dense, key=k_dense),
        }
        self.fusion_mlp = eqx.nn.Sequential(
            [
                eqx.nn.LayerNorm(
                    (num_cameras + 2) * d_model,
                ),
                eqx.nn.Linear((num_cameras + 2) * d_model, d_model, key=k_fusion),
                jax.nn.relu,
            ],
        )

        self.blocks = [Block(d_model, nheads, key=jr.fold_in(k_blocks, i)) for i in range(layers)]
        self.positional_embedding = jnp.zeros((1, d_model))
        self.num_cameras = num_cameras
        self.d_model = d_model

    def _build_mask(self, timesteps: int, length: int):
        """Build mask for the subtask transformer.

        Args:
            length (int): Length of the sequence

        Returns:
            jnp.ndarray: Mask of shape ((N+3)*T, (N+3)*T)
        """
        mask_1d = jnp.arange(timesteps) < length
        mask_1d = jnp.where(mask_1d, 0.0, float("-inf"))
        mask_1d = einops.rearrange(mask_1d, "t -> (n t)", n=self.num_cameras + 2)
        mask = mask_1d[None, :] + mask_1d[:, None]  # (N+3)*T, (N+3)*T
        return mask

    def __call__(
        self,
        img_features: jnp.ndarray,
        text_features: jnp.ndarray,
        state: jnp.ndarray,
        length: int,
        schema: str = "sparse",
    ):
        """Forward pass for the subtask transformer.

        Args:
            img_features (jnp.ndarray): Image features of shape (N, T, d_vis)
            text_features (jnp.ndarray): Text features of shape (T, d_text)
            state (jnp.ndarray): State features of shape (T, d_state)
            subtask (jnp.ndarray): Subtask features of shape (T, C)

        Returns:
            jnp.ndarray: Output features of shape (T)
        """
        N, T, D = img_features.shape
        img_features = jax.vmap(jax.vmap(self.vis_proj))(img_features)  # (N, T, d_model)
        text_features = jax.vmap(self.text_proj)(text_features)[None, ...]  # (1, T, d_model)
        state_features = jax.vmap(self.state_proj)(state)[None, ...]  # (1, T, d_model)

        # Combine features
        features = jnp.concatenate(
            [img_features, text_features, state_features], axis=0
        )  # (N + 2, T, d_model)

        features = features.at[:N, 0, :].add(self.positional_embedding)
        features = einops.rearrange(features, "n t d -> (n t) d")  # ((N+2)*T, d_model)

        mask = self._build_mask(length, T)  # ((N+2)*T, (N+2)*T)

        # Apply transformer blocks
        for block in self.blocks:
            features = block(features, mask)

        features = einops.rearrange(features, "(n t) d -> t (n d)")
        features = jax.vmap(self.fusion_mlp)(features)  # (T, d_model)

        logits = jax.vmap(self.final_proj[schema])(features)  # (T, C)

        return logits  # (T, C)
