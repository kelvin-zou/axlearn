# Copyright © 2023 Apple Inc.

"""Wrappers for FlashAttention on TPU in JAX with logit bias support."""
import functools
from typing import Optional

import jax
import jax.numpy as jnp
from jax.experimental.pallas.ops.tpu.flash_attention import (
    BlockSizes as FlashBlockSizes,
    flash_attention as tpu_flash_attention, 
)
from jax.experimental.pallas.ops.tpu.splash_attention import (
    BlockSizes as SplashBlockSizes,
    splash_attention_kernel,
    splash_attention_mask,
)
from axlearn.common.utils import Tensor


@functools.partial(
    jax.jit,
    static_argnames=[
        "causal",
        "softmax_scale",
        "block_size",
    ],
)
def flash_attention(
    query: Tensor,  # [batch_size, q_seq_len, num_heads, d_model]
    key: Tensor,  # [batch_size, kv_seq_len, num_heads, d_model]
    value: Tensor,  # [batch_size, kv_seq_len, num_heads, d_model]
    bias: Tensor = None,  # [batch_size, num_heads, q_seq_len, kv_seq_len]
    *,
    causal: bool = False,
    softmax_scale: float = 1.0,
    block_size: int = 128,
    splash_attention: bool = True,
):
    """Wraps JAX's TPU flash-attention, with reshapes and softmax-scaling outside kernel.

    N.B. we apply the softmax scale factor outside of the kernel because:
        1. within-kernel ordering of attention-bias addition and softmax scaling differ to axlearn,
        2. it's more efficient to scale outside the kernel vs. fix order of ops in kernel.

    Args:
        query: The query tensor, of shape [batch_size, target_seq_len, num_heads, head_dim].
        key: The key tensor, of shape [batch_size, source_seq_len, num_heads, head_dim].
        value: The value tensor, of shape [batch_size, source_seq_len, num_heads, head_dim].
        causal: Whether the attention is causal (allows for additional optimizations).
        softmax_scale: A scaling factor applied to the query.
        block_size: The block size for the attention kernel.
        splash_attention: fall back to base flash attention if False.

    Returns:
        The context tensor, of shape [batch_size, q_seq_len, num_heads, head_dim].

    """
    # Apply the softmax scale outside the kernel (see docstring for why).
    if softmax_scale != 1.0:
        query *= softmax_scale
    
    # Cap block size to sequence length.    
    block_size = min(block_size, query.shape[1])

    # Switch num_heads and seq_len axes.
    query = jnp.einsum("btnh->bnth", query)
    key = jnp.einsum("bsnh->bnsh", key)
    value = jnp.einsum("bsnh->bnsh", value)

    # Fall back to flash attention if the query shape is not divisible by 128 or bias is not None.
    splash_attention = splash_attention and query.shape[3] % 128 == 0 and bias is None

    if splash_attention:
        block_sizes = SplashBlockSizes(
            block_q=block_size,
            block_kv_compute=block_size,
            block_kv=block_size,
            block_q_dkv=block_size,
            block_kv_dkv=block_size,
            block_kv_dkv_compute=block_size,
            block_q_dq=block_size,
            block_kv_dq=block_size,
        )

        def wrap_splash_attention(query, key, value):
            # Splash Attention can support multiple mask types,
            # Now we only support causal and full masks, which makes it equivalent to FlashAttention.
            if causal:
                masks = [
                    splash_attention_mask.CausalMask(shape=(query.shape[2], query.shape[2]))
                    for i in range(query.shape[1])
                ]
            else:
                masks = [splash_attention_mask.FullMask() for i in range(query.shape[1]) ]
            multi_head_mask = splash_attention_mask.MultiHeadMask(masks=masks)

            splash_kernel = splash_attention_kernel.make_splash_mha(
                mask=multi_head_mask, head_shards=1, q_seq_shards=1, block_sizes=block_sizes
            )
            return jax.vmap(splash_kernel)(query, key, value)

        context = wrap_splash_attention(query, key, value)

    else:
        block_sizes = FlashBlockSizes(
            block_q=block_size,
            block_k_major=block_size,
            block_k=block_size,
            block_b=1,
            block_q_major_dkv=block_size,
            block_k_major_dkv=block_size,
            block_k_dkv=block_size,
            block_q_dkv=block_size,
            block_k_major_dq=block_size,
            block_k_dq=block_size,
            block_q_dq=block_size,
        )
        context = tpu_flash_attention(
            q=query,
            k=key,
            v=value,
            ab=bias,
            causal=causal,
            # If sm_scale==1.0, the kernel skips applying it.
            sm_scale=1.0,
            block_sizes=block_sizes,
            debug=False,
        )

    # Restore num_heads and seq_len axes.
    return jnp.einsum("bnth->btnh", context)
