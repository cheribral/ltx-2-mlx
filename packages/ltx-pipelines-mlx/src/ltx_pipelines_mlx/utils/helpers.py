"""Pipeline-level helpers matching upstream ltx-pipelines/utils/helpers.py.

These functions form the standard orchestration vocabulary for
building a noised LatentState from conditionings + an optional
initial latent, and for blending denoised model output with the
clean-latent reference per the denoise mask.

They mirror the upstream LTX-2 helpers 1:1, allowing pipeline code
to be read side-by-side with the upstream Python files.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import mlx.core as mx

from ltx_core_mlx.conditioning.types.latent_cond import LatentState, noise_latent_state

if TYPE_CHECKING:
    pass


def post_process_latent(
    denoised: mx.array,
    denoise_mask: mx.array,
    clean: mx.array,
) -> mx.array:
    """Blend denoised model output with clean state based on mask.

    Mirror of upstream ``ltx_pipelines.utils.helpers.post_process_latent``.
    For tokens with ``mask=1`` (generation), the predicted ``denoised``
    is used as-is. For tokens with ``mask=0`` (preserved keyframe /
    reference), ``clean`` is restored.

    Upstream upcasts ``clean`` to fp32 before the blend then casts back
    to ``denoised.dtype`` to reduce bf16 accumulation noise.

    Args:
        denoised: Model x0 prediction, shape ``(B, T, C)``.
        denoise_mask: Mask in ``[0, 1]``, shape ``(B, T, 1)``.
        clean: Clean latent reference, shape ``(B, T, C)``.

    Returns:
        Blended latent, same shape and dtype as ``denoised``.
    """
    out_dtype = denoised.dtype
    blended = denoised * denoise_mask + clean.astype(mx.float32) * (1.0 - denoise_mask)
    return blended.astype(out_dtype)


def state_with_conditionings(
    latent_state: LatentState,
    conditioning_items: list,
    spatial_dims: tuple[int, int, int],
) -> LatentState:
    """Apply a list of conditionings sequentially to a latent state.

    Mirror of upstream ``ltx_pipelines.utils.helpers.state_with_conditionings``.
    Iterates through the conditioning items and applies each one to the
    state. Each conditioning may append tokens (keyframe / reference)
    or modify the denoise mask (temporal region).

    Note: the upstream signature passes ``latent_tools`` as the second
    arg; in MLX our conditionings take ``spatial_dims`` directly via
    their ``apply(state, spatial_dims)`` signature, since we don't have
    the LatentTools abstraction yet.

    Args:
        latent_state: Starting state.
        conditioning_items: List with ``apply(state, spatial_dims)``
            method (e.g. ``VideoConditionByKeyframeIndex``).
        spatial_dims: ``(F, H, W)`` latent spatial dimensions.

    Returns:
        State with all conditionings applied in order.
    """
    for conditioning in conditioning_items:
        latent_state = conditioning.apply(latent_state, spatial_dims)
    return latent_state


def create_noised_state(
    base_shape: tuple[int, ...],
    conditionings: list,
    spatial_dims: tuple[int, int, int],
    positions: mx.array,
    seed: int,
    sigma: float = 1.0,
    initial_latent: mx.array | None = None,
    dtype: mx.Dtype = mx.bfloat16,
) -> LatentState:
    """Build a noised latent state from conditionings + optional initial latent.

    Mirror of upstream ``ltx_pipelines.utils.helpers.create_noised_state``.

    Order of operations (CRITICAL — matches upstream):
        1. Initial state: zero-init OR ``initial_latent`` if provided.
        2. Apply conditionings in order (may append tokens, set mask=0).
        3. Noise: ``GaussianNoiser`` semantics — keyframe tokens
           (mask=0) stay clean, generation tokens (mask=1) get
           ``noise * sigma + clean * (1 - sigma)`` blend.

    Args:
        base_shape: ``(B, N_gen, C)`` shape of the generation tokens
            BEFORE conditioning items append any reference tokens.
        conditionings: List of conditioning items (e.g.
            ``VideoConditionByKeyframeIndex``) to apply.
        spatial_dims: ``(F, H, W)`` latent spatial dims for conditioning.
        positions: ``(1, N_gen, num_axes)`` positions for the
            generation tokens. Conditioning items append their own
            positions for any tokens they introduce.
        seed: Random seed for the noise.
        sigma: Noise scale. ``1.0`` for stage 1 (pure noise on
            generated tokens); ``stage_2_sigmas[0]`` for stage 2
            (partial noise on top of ``initial_latent``).
        initial_latent: Optional starting latent for the generation
            region. ``None`` means start from zeros (typical stage 1).
            Provided as the upscaled stage-1 latent for stage 2.
        dtype: dtype of the resulting state arrays.

    Returns:
        Noised LatentState ready to feed into the denoising loop.
    """
    if initial_latent is None:
        latent = mx.zeros(base_shape, dtype=dtype)
    else:
        latent = initial_latent.astype(dtype)

    state = LatentState(
        latent=latent,
        clean_latent=latent,
        denoise_mask=mx.ones((base_shape[0], base_shape[1], 1), dtype=dtype),
        positions=positions,
    )
    state = state_with_conditionings(state, conditionings, spatial_dims)
    state = noise_latent_state(state, sigma=sigma, seed=seed)
    return state
