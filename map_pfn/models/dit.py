"""DiT decoder for the architecture ablation: MMDiT with block weights shared across modalities."""

from functools import partial

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from equinox.nn import RMSNorm
from jaxtyping import Array, Float, PRNGKeyArray

from map_pfn.models.utils import MLP, SinusoidalPosEmb, flash_attention, zero_init_linear


class DiTBlock(eqx.Module):
    """DiT transformer block with FiLM conditioning, shared across the noise, conditioning and treatment tokens."""

    head_dim: int
    embed_dim: int
    num_heads: int

    ln_qkv: eqx.nn.LayerNorm
    ln_ff: eqx.nn.LayerNorm
    qkv: eqx.nn.Linear
    o: eqx.nn.Linear
    film_a: eqx.nn.Linear
    film_f: eqx.nn.Linear
    rms_q: RMSNorm | eqx.nn.Identity
    rms_k: RMSNorm | eqx.nn.Identity
    ff: MLP

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        use_rms_norm: bool,
        *,
        key: PRNGKeyArray,
    ) -> None:
        """Initialize DiTBlock.

        Args:
            embed_dim: Embedding dimensions
            num_heads: Number of attention heads
            use_rms_norm: Whether to use RMS layer normalization
            key: Random key for initialization
        """
        keys = jr.split(key, 5)

        self.head_dim = embed_dim // num_heads
        self.num_heads = num_heads
        self.embed_dim = embed_dim

        self.ln_qkv = eqx.nn.LayerNorm(embed_dim)
        self.ln_ff = eqx.nn.LayerNorm(embed_dim)
        self.qkv = eqx.nn.Linear(embed_dim, embed_dim * 3, use_bias=False, key=keys[0])
        self.o = eqx.nn.Linear(embed_dim, embed_dim, key=keys[1])
        self.film_a = zero_init_linear(eqx.nn.Linear(embed_dim, embed_dim * 2, key=keys[2]))
        self.film_f = zero_init_linear(eqx.nn.Linear(embed_dim, embed_dim * 2, key=keys[3]))

        if use_rms_norm:
            self.rms_q = RMSNorm(self.head_dim, eps=1e-8)
            self.rms_k = RMSNorm(self.head_dim, eps=1e-8)
        else:
            self.rms_q = eqx.nn.Identity()
            self.rms_k = eqx.nn.Identity()

        self.ff = MLP(embed_dim, embed_dim * 4, embed_dim, key=keys[4])

    @eqx.filter_jit
    def __call__(
        self,
        noise: Float[Array, "seq_len_noise embed_dim"],
        cond: Float[Array, "seq_len_cond embed_dim"],
        treatment: Float[Array, "1 embed_dim"],
        t_emb: Float[Array, "1 embed_dim"],
    ) -> tuple[Float[Array, "seq_len_noise embed_dim"], Float[Array, "seq_len_cond embed_dim"]]:
        """Forward pass through DiTBlock.

        Args:
            noise: Noise input tensor
            cond: Conditioning input tensor
            treatment: Treatment embedding
            t_emb: Time embedding

        Returns:
            Updated noise, conditioning and treatment tensors
        """
        sections = [noise.shape[0], noise.shape[0] + cond.shape[0]]
        x = jnp.concatenate([noise, cond, treatment], axis=0)
        seq_len, embed_dim = x.shape

        gamma, beta = jnp.split(self.film_a(t_emb[0]), 2, axis=-1)
        h = jax.vmap(self.ln_qkv)(x) * (1 + gamma) + beta
        qkv = jax.vmap(self.qkv)(h).reshape(seq_len, 3, self.num_heads, self.head_dim)
        q = jax.vmap(jax.vmap(self.rms_q))(qkv[:, 0])
        k = jax.vmap(jax.vmap(self.rms_k))(qkv[:, 1])
        x = x + jax.vmap(self.o)(flash_attention(q, k, qkv[:, 2]).reshape(seq_len, embed_dim))

        gamma, beta = jnp.split(self.film_f(t_emb[0]), 2, axis=-1)
        x = x + jax.vmap(self.ff)(jax.vmap(self.ln_ff)(x) * (1 + gamma) + beta)

        noise, cond, treatment = jnp.split(x, sections, axis=0)
        return noise, cond, treatment


class DiT(eqx.Module):
    """DiT transformer: MMDiT with one set of block weights shared across all modalities."""

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

    blocks: DiTBlock

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
        """Initialize DiT.

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

        def make_block(key: PRNGKeyArray) -> DiTBlock:
            return DiTBlock(embed_dim=embed_dim, num_heads=num_heads, use_rms_norm=use_rms_norm, key=key)

        self.blocks = eqx.filter_vmap(make_block)(block_keys)

    @eqx.filter_jit
    def __call__(
        self,
        noise: Float[Array, "seq_len_noise dim"],
        cond: Float[Array, "seq_len_cond cond_dim"],
        treatment: Float[Array, "1 treatment_dim"],
        t: Float[Array, ""],
    ) -> Float[Array, "seq_len_noise dim"]:
        """Forward pass through DiT.

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
        def block_fn(carry: tuple, dynamic_block: DiTBlock) -> tuple:
            noise, cond, treatment = carry
            block = eqx.combine(dynamic_block, static_blocks)
            noise, cond, treatment = block(noise, cond, treatment, t_emb)
            return (noise, cond, treatment), None

        (noise, cond, treatment), _ = jax.lax.scan(block_fn, (noise, cond, treatment), dynamic_blocks)

        # Remove register tokens
        noise = noise[self.num_reg_tokens :]

        return jax.vmap(self.noise_out)(noise)
