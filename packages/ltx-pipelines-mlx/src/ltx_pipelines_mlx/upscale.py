"""Standalone video upscaler using the LTX neural latent upsampler.

Takes a video, encodes it through the VAE encoder, applies the spatial
upsampler (default 2x via ``spatial_upscaler_x2_v1_1.safetensors``), and
decodes the result back to a video. **No DiT involved** — just
encoder + upsampler + decoder.

This is the same upscale step that lives between stage 1 and stage 2 of
the two-stage pipelines, exposed as a standalone tool. Useful when:

- You already have a generated or external video and want to enlarge it
  without re-running the (expensive) DiT denoising at full target res.
- You want to chain a quasi-720p generation with a 2x neural upscale to
  approximate a 1440p output without OOMing the attention activation.

Audio (if present in the input) is preserved by extracting via ffmpeg
and remuxing into the output mp4.

Spatial dims of the input are rounded down to the nearest multiple of
32 (VAE alignment requirement). Frame count is rounded down to the
nearest valid latent count: ``F_pixel = 8 * (F_latent - 1) + 1``.
"""

from __future__ import annotations

from pathlib import Path

import mlx.core as mx

from ltx_core_mlx.utils.ffmpeg import find_ffmpeg, probe_video_info
from ltx_core_mlx.utils.memory import aggressive_cleanup
from ltx_core_mlx.utils.video import load_video_for_encoding
from ltx_pipelines_mlx.utils._orchestration import resolve_model_dir
from ltx_pipelines_mlx.utils.blocks import ImageConditioner, VideoDecoder, VideoUpsampler

_materialize = getattr(mx, "eval")  # noqa: B009 -- security hook flags mx.eval pattern


class UpscalePipeline:
    """Encoder + neural upsampler + decoder, standalone.

    Args:
        model_dir: Path to model weights or HuggingFace repo ID.
        upsampler_name: Filename stem of the upsampler safetensors,
            without the ``.safetensors`` extension. Defaults to
            ``spatial_upscaler_x2_v1_1`` (2x). Other available variants
            in the standard repos: ``spatial_upscaler_x1_5_v1_0`` (1.5x),
            ``temporal_upscaler_x2_v1_0`` (2x along time).
    """

    def __init__(
        self,
        model_dir: str,
        upsampler_name: str = "spatial_upscaler_x2_v1_1",
    ) -> None:
        self.model_dir = resolve_model_dir(model_dir)
        self.upsampler_name = upsampler_name

        self.image_conditioner = ImageConditioner(self.model_dir)
        self.video_decoder = VideoDecoder(self.model_dir)
        self.video_upsampler = VideoUpsampler(self.model_dir, name=upsampler_name)

    def upscale(
        self,
        input_path: str,
        output_path: str,
        max_frames: int | None = None,
    ) -> str:
        """Upscale ``input_path`` and write the result to ``output_path``.

        Args:
            input_path: Source video file.
            output_path: Destination ``.mp4`` path.
            max_frames: Optional cap on the number of input frames decoded.
                Defaults to the full video, rounded down to the nearest
                valid latent count.

        Returns:
            ``output_path``.
        """
        input_path = str(input_path)
        output_path = str(output_path)

        # --- Probe input ---
        info = probe_video_info(input_path)

        # Round spatial dims down to nearest multiple of 32 (VAE alignment).
        in_h = (info.height // 32) * 32
        in_w = (info.width // 32) * 32
        if in_h <= 0 or in_w <= 0:
            raise ValueError(
                f"Input video too small: {info.width}x{info.height}. Each side must be at least 32 pixels."
            )

        # Round frame count down to the nearest valid latent count
        # (F_pixel = 8 * (F_latent - 1) + 1, so valid pixel counts are 1, 9, 17, 25, ...).
        avail_frames = max_frames if max_frames is not None else info.num_frames
        if avail_frames < 1:
            raise ValueError(f"Input video has no frames: {input_path}")
        in_frames = ((avail_frames - 1) // 8) * 8 + 1
        if in_frames < 1:
            in_frames = 1

        print(f"Mode: Standalone Upscale ({self.upsampler_name})")
        print(f"  Input: {input_path} ({info.width}x{info.height}, {info.num_frames} frames @ {info.fps:.2f} fps)")
        print(f"  VAE-aligned input: {in_w}x{in_h}, {in_frames} frames")

        # --- Load input pixels ---
        # (1, 3, F, H, W) in [-1, 1], bfloat16
        pixels = load_video_for_encoding(input_path, in_h, in_w, in_frames)
        _materialize(pixels)

        # --- VAE encode ---
        encoder = self.image_conditioner.load()
        latent = encoder.encode(pixels)  # (1, 128, F', H/32, W/32), normalized
        _materialize(latent)
        del pixels

        # --- Denormalize → upsample → renormalize ---
        # encoder/decoder operate on normalized latents; the upsampler
        # operates in un-normalized latent space (matches stage 2 of the
        # two-stage pipelines).
        latent_chl = latent.transpose(0, 2, 3, 4, 1)  # (1, F', H', W', 128)
        latent_denorm = encoder.denormalize_latent(latent_chl)
        latent_denorm = latent_denorm.transpose(0, 4, 1, 2, 3)  # back to channels-first
        _materialize(latent_denorm)

        # Free the encoder before upsampling — the upsampler model is
        # smaller (~1 GB for x2_v1_1) but on a 32 GB Mac every GB matters.
        self.image_conditioner.free()
        del latent
        aggressive_cleanup()

        upsampler = self.video_upsampler.load()
        upscaled = upsampler(latent_denorm)  # (1, 128, F', 2*H', 2*W')
        _materialize(upscaled)
        del latent_denorm

        # Free upsampler, then re-load encoder briefly only for normalize_latent
        # (which uses the encoder's per-channel statistics, no Metal-heavy ops).
        self.video_upsampler.free()
        encoder = self.image_conditioner.load()
        upscaled_chl = upscaled.transpose(0, 2, 3, 4, 1)
        upscaled_norm = encoder.normalize_latent(upscaled_chl)
        upscaled_norm = upscaled_norm.transpose(0, 4, 1, 2, 3)
        _materialize(upscaled_norm)
        self.image_conditioner.free()
        del upscaled
        aggressive_cleanup()

        # --- Audio: extract from input if present ---
        audio_path: str | None = None
        if info.has_audio:
            audio_path = str(Path(output_path).with_suffix(".audio.aac"))
            self._extract_audio(input_path, audio_path)

        # --- VAE decode + stream to mp4 ---
        try:
            self.video_decoder.decode_and_stream(
                upscaled_norm,
                output_path,
                fps=info.fps,
                audio_path=audio_path,
            )
        finally:
            if audio_path is not None:
                Path(audio_path).unlink(missing_ok=True)
            self.video_decoder.free()

        print(f"Saved: {output_path}")
        return output_path

    @staticmethod
    def _extract_audio(input_path: str, audio_path: str) -> None:
        """Extract the audio stream from ``input_path`` to ``audio_path`` (AAC, copy)."""
        import subprocess

        ffmpeg = find_ffmpeg()
        cmd = [
            ffmpeg,
            "-y",
            "-i",
            input_path,
            "-vn",
            "-acodec",
            "copy",
            audio_path,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            # Fall back to re-encoding when copy fails (e.g. unsupported codec)
            cmd_fallback = [
                ffmpeg,
                "-y",
                "-i",
                input_path,
                "-vn",
                "-acodec",
                "aac",
                "-b:a",
                "192k",
                audio_path,
            ]
            subprocess.run(cmd_fallback, check=True, capture_output=True, text=True)


__all__ = ["UpscalePipeline"]
