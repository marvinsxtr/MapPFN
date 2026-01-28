from functools import partial

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from equinox.nn import RMSNorm
from jaxtyping import Array, Float, PRNGKeyArray

from map_pfn.models.utils import MLP, SinusoidalPosEmb, flash_attention, zero_init_linear


class MMDiTBlock(eqx.Module):
    """Multi-modal DiT transformer block with FiLM conditioning."""

    head_dim: int
    embed_dim: int
    num_heads: int

    # Layer norms
    ln_qkv_n: eqx.nn.LayerNorm
    ln_qkv_c: eqx.nn.LayerNorm
    ln_qkv_t: eqx.nn.LayerNorm
    ln_ff_n: eqx.nn.LayerNorm
    ln_ff_c: eqx.nn.LayerNorm
    ln_ff_t: eqx.nn.LayerNorm

    # QKV projections
    qkv_n: eqx.nn.Linear
    qkv_c: eqx.nn.Linear
    qkv_t: eqx.nn.Linear

    # Output projections
    o_n: eqx.nn.Linear
    o_c: eqx.nn.Linear
    o_t: eqx.nn.Linear

    # FiLM conditioning
    film_an: eqx.nn.Linear
    film_ac: eqx.nn.Linear
    film_at: eqx.nn.Linear
    film_fn: eqx.nn.Linear
    film_fc: eqx.nn.Linear
    film_ft: eqx.nn.Linear

    # RMS layer normalization
    rms_q: RMSNorm | eqx.nn.Identity
    rms_k: RMSNorm | eqx.nn.Identity

    # Feed-forward networks
    ff_n: MLP
    ff_c: MLP
    ff_t: MLP

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        use_rms_norm: bool,
        *,
        key: PRNGKeyArray,
    ) -> None:
        """Initialize MMDitBlock.

        Args:
            embed_dim: Embedding dimensions
            num_heads: Number of attention heads
            use_rms_norm: Whether to use RMS layer normalization
            key: Random key for initialization
        """
        keys = jr.split(key, 15)

        self.head_dim = embed_dim // num_heads
        self.num_heads = num_heads
        self.embed_dim = embed_dim

        self.ln_qkv_n = eqx.nn.LayerNorm(embed_dim)
        self.ln_qkv_c = eqx.nn.LayerNorm(embed_dim)
        self.ln_qkv_t = eqx.nn.LayerNorm(embed_dim)

        self.qkv_n = eqx.nn.Linear(embed_dim, embed_dim * 3, use_bias=False, key=keys[0])
        self.qkv_c = eqx.nn.Linear(embed_dim, embed_dim * 3, use_bias=False, key=keys[1])
        self.qkv_t = eqx.nn.Linear(embed_dim, embed_dim * 3, use_bias=False, key=keys[2])

        self.o_n = eqx.nn.Linear(embed_dim, embed_dim, key=keys[3])
        self.o_c = eqx.nn.Linear(embed_dim, embed_dim, key=keys[4])
        self.o_t = eqx.nn.Linear(embed_dim, embed_dim, key=keys[5])

        self.film_an = zero_init_linear(eqx.nn.Linear(embed_dim, embed_dim * 2, key=keys[6]))
        self.film_ac = zero_init_linear(eqx.nn.Linear(embed_dim, embed_dim * 2, key=keys[7]))
        self.film_at = zero_init_linear(eqx.nn.Linear(embed_dim, embed_dim * 2, key=keys[8]))

        self.film_fn = zero_init_linear(eqx.nn.Linear(embed_dim, embed_dim * 2, key=keys[9]))
        self.film_fc = zero_init_linear(eqx.nn.Linear(embed_dim, embed_dim * 2, key=keys[10]))
        self.film_ft = zero_init_linear(eqx.nn.Linear(embed_dim, embed_dim * 2, key=keys[11]))

        if use_rms_norm:
            self.rms_q = RMSNorm(self.head_dim, eps=1e-8)
            self.rms_k = RMSNorm(self.head_dim, eps=1e-8)
        else:
            self.rms_q = eqx.nn.Identity()
            self.rms_k = eqx.nn.Identity()

        self.ln_ff_n = eqx.nn.LayerNorm(embed_dim)
        self.ln_ff_c = eqx.nn.LayerNorm(embed_dim)
        self.ln_ff_t = eqx.nn.LayerNorm(embed_dim)

        self.ff_n = MLP(embed_dim, embed_dim * 4, embed_dim, key=keys[12])
        self.ff_c = MLP(embed_dim, embed_dim * 4, embed_dim, key=keys[13])
        self.ff_t = MLP(embed_dim, embed_dim * 4, embed_dim, key=keys[14])

    def _heads(self, x: Float[Array, "seq_len embed_dim"]) -> Float[Array, "seq_len num_heads head_dim"]:
        """Reshape for multi-head attention."""
        seq_len, _ = x.shape
        return x.reshape(seq_len, self.num_heads, self.head_dim)

    @eqx.filter_jit
    def __call__(
        self,
        noise: Float[Array, "seq_len_noise embed_dim"],
        cond: Float[Array, "seq_len_cond embed_dim"],
        treatment: Float[Array, "1 embed_dim"],
        t_emb: Float[Array, "1 embed_dim"],
    ) -> tuple[Float[Array, "seq_len_noise embed_dim"], Float[Array, "seq_len_cond embed_dim"]]:
        """Forward pass through MMDitBlock.

        Args:
            noise: Noise input tensor
            cond: Conditioning input tensor
            treatment: Treatment embedding
            t_emb: Time embedding

        Returns:
            Updated noise and conditioning tensors
        """
        seq_len_noise, embed_dim = noise.shape
        seq_len_cond = cond.shape[0]
        seq_len_treatment = treatment.shape[0]

        # Expand time embeddings
        t_n = t_emb.repeat(seq_len_noise, axis=0)
        t_c = t_emb.repeat(seq_len_cond, axis=0)
        t_t = t_emb.repeat(seq_len_treatment, axis=0)

        # FiLM conditioning for attention
        gamma_n, beta_n = jnp.split(jax.vmap(self.film_an)(t_n), 2, axis=-1)
        gamma_c, beta_c = jnp.split(jax.vmap(self.film_ac)(t_c), 2, axis=-1)
        gamma_t, beta_t = jnp.split(jax.vmap(self.film_at)(t_t), 2, axis=-1)

        # Apply FiLM to layer norms
        n = jax.vmap(self.ln_qkv_n)(noise) * (1 + gamma_n) + beta_n
        c = jax.vmap(self.ln_qkv_c)(cond) * (1 + gamma_c) + beta_c
        t = jax.vmap(self.ln_qkv_t)(treatment) * (1 + gamma_t) + beta_t

        # Generate QKV
        qkv_n = jax.vmap(self.qkv_n)(n)
        qkv_c = jax.vmap(self.qkv_c)(c)
        qkv_t = jax.vmap(self.qkv_t)(t)

        q_n, k_n, v_n = jnp.split(qkv_n, 3, axis=-1)
        q_c, k_c, v_c = jnp.split(qkv_c, 3, axis=-1)
        q_t, k_t, v_t = jnp.split(qkv_t, 3, axis=-1)

        # Reshape for multi-head attention
        q = jnp.concatenate([self._heads(q_n), self._heads(q_c), self._heads(q_t)], axis=0)
        k = jnp.concatenate([self._heads(k_n), self._heads(k_c), self._heads(k_t)], axis=0)
        v = jnp.concatenate([self._heads(v_n), self._heads(v_c), self._heads(v_t)], axis=0)

        # Apply RMS normalization to queries and keys
        q = jax.vmap(jax.vmap(self.rms_q))(q)
        k = jax.vmap(jax.vmap(self.rms_k))(k)

        attn_out = flash_attention(q, k, v)

        # Reshape and split attention outputs
        attn_out = attn_out.reshape(seq_len_noise + seq_len_cond + seq_len_treatment, embed_dim)
        a_n, a_c, a_t = jnp.split(attn_out, [seq_len_noise, seq_len_noise + seq_len_cond], axis=0)

        # Apply output projections and residual connections
        noise = noise + jax.vmap(self.o_n)(a_n)
        cond = cond + jax.vmap(self.o_c)(a_c)
        treatment = treatment + jax.vmap(self.o_t)(a_t)

        # FiLM conditioning for feed-forward
        gamma_n2, beta_n2 = jnp.split(jax.vmap(self.film_fn)(t_n), 2, axis=-1)
        gamma_c2, beta_c2 = jnp.split(jax.vmap(self.film_fc)(t_c), 2, axis=-1)
        gamma_t2, beta_t2 = jnp.split(jax.vmap(self.film_ft)(t_t), 2, axis=-1)

        # Apply FiLM to feed-forward layer norms
        n_ff = jax.vmap(self.ln_ff_n)(noise) * (1 + gamma_n2) + beta_n2
        c_ff = jax.vmap(self.ln_ff_c)(cond) * (1 + gamma_c2) + beta_c2
        t_ff = jax.vmap(self.ln_ff_t)(treatment) * (1 + gamma_t2) + beta_t2

        # Feed-forward with residual connections
        noise = noise + jax.vmap(self.ff_n)(n_ff)
        cond = cond + jax.vmap(self.ff_c)(c_ff)
        treatment = treatment + jax.vmap(self.ff_t)(t_ff)

        return noise, cond, treatment


class MMDiT(eqx.Module):
    """Multi-modal DiT transformer."""

    embed_dim: int
    num_reg_tokens: int
    cond_dim: int
    noise_dim: int
    num_blocks: int

    # Learneable parameters
    reg_tok: Float[Array, "num_reg_tokens cond_dim"]

    # Input/output projections
    noise_in: eqx.nn.Linear
    noise_out: eqx.nn.Linear
    cond_in: eqx.nn.Linear
    treatment_in: eqx.nn.Linear

    # Modality encoding
    modality_emb_noise: Float[Array, "seq_len_noise embed_dim"]
    modality_emb_cond: Float[Array, "seq_len_cond embed_dim"]
    modality_emb_treatment: Float[Array, "seq_len_treatment embed_dim"]

    # Time embedding
    time_emb: SinusoidalPosEmb
    time_mlp: MLP

    blocks: MMDiTBlock

    def __init__(
        self,
        embed_dim: int,
        cond_dim: int,
        noise_dim: int,
        num_heads: int,
        num_blocks: int,
        use_rms_norm: bool,
        num_reg_tokens: int,
        *,
        key: PRNGKeyArray,
    ) -> None:
        """Initialize MMDit.

        Args:
            embed_dim: Internal embedding dimension
            cond_dim: Feature dimension of input data
            noise_dim: Noise dimensionality
            num_heads: Number of attention heads
            num_blocks: Number of transformer blocks to stack
            use_rms_norm: Whether to apply RMS normalization to attention Q/K
            num_reg_tokens: Number of register tokens prepended to noise sequences
            num_treatments: Number of unique treatments
            key: Random key
        """
        keys = jr.split(key, 10)

        self.num_reg_tokens = num_reg_tokens
        self.cond_dim = cond_dim
        self.noise_dim = noise_dim
        self.num_blocks = num_blocks
        self.embed_dim = embed_dim

        self.noise_in = eqx.nn.Linear(noise_dim, embed_dim, key=keys[0])
        self.noise_out = eqx.nn.Linear(embed_dim, noise_dim, key=keys[1])
        self.cond_in = eqx.nn.Linear(cond_dim, embed_dim, key=keys[2])
        self.treatment_in = eqx.nn.Linear(cond_dim, embed_dim, key=keys[3])

        self.modality_emb_noise = jr.normal(keys[4], (embed_dim,)) / jnp.sqrt(embed_dim)
        self.modality_emb_cond = jr.normal(keys[5], (embed_dim,)) / jnp.sqrt(embed_dim)
        self.modality_emb_treatment = jr.normal(keys[6], (embed_dim,)) / jnp.sqrt(embed_dim)

        self.reg_tok = jr.normal(keys[7], (num_reg_tokens, noise_dim))

        self.time_emb = SinusoidalPosEmb(embed_dim)
        self.time_mlp = MLP(embed_dim, embed_dim * 4, embed_dim, key=keys[8])

        block_keys = jr.split(keys[9], num_blocks)

        def make_block(key: PRNGKeyArray) -> MMDiTBlock:
            return MMDiTBlock(embed_dim=embed_dim, num_heads=num_heads, use_rms_norm=use_rms_norm, key=key)

        self.blocks = eqx.filter_vmap(make_block)(block_keys)

    @eqx.filter_jit
    def __call__(
        self,
        noise: Float[Array, "seq_len_noise dim"],
        cond: Float[Array, "seq_len_cond cond_dim"],
        treatment: Float[Array, "1 treatment_dim"],
        t: Float[Array, ""],
    ) -> Float[Array, "seq_len_noise dim"]:
        """Forward pass through MMDit.

        Args:
            noise: Noise input tensor
            cond: Conditioning input tensor
            treatment: Treatment
            t: Time values

        Returns:
            Processed noise tensor
        """
        # Add register tokens
        noise = jnp.concatenate([self.reg_tok, noise], axis=0)

        # Project to embedding dimension
        noise = jax.vmap(self.noise_in)(noise) + self.modality_emb_noise
        cond = jax.vmap(self.cond_in)(cond) + self.modality_emb_cond
        treatment = jax.vmap(self.treatment_in)(treatment) + self.modality_emb_treatment

        # Time embedding
        t = jnp.expand_dims(t, axis=0)
        t_pos = jax.vmap(self.time_emb)(t)
        t_emb = jax.vmap(self.time_mlp)(t_pos)

        # Scan over blocks
        dynamic_blocks, static_blocks = eqx.partition(self.blocks, eqx.is_array)

        @partial(jax.checkpoint, policy=jax.checkpoint_policies.dots_with_no_batch_dims_saveable)
        def block_fn(carry: tuple, dynamic_block: MMDiTBlock) -> tuple:
            noise, cond, treatment = carry
            block = eqx.combine(dynamic_block, static_blocks)
            noise, cond, treatment = block(noise, cond, treatment, t_emb)
            return (noise, cond, treatment), None

        (noise, cond, treatment), _ = jax.lax.scan(block_fn, (noise, cond, treatment), dynamic_blocks)

        # Remove register tokens
        noise = noise[self.num_reg_tokens :]

        return jax.vmap(self.noise_out)(noise)
