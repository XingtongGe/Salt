from pipeline.streaming_training import StreamingTrainingPipeline
from model.dmd import DMD, save_video_as_grid_and_mp4
import torch.nn.functional as F
from typing import Tuple, Optional, Dict, Any
import torch
import torch.distributed as dist
from einops import rearrange
import time
import os


class StreamingDMD(DMD):
    """
    DMD model with streaming training support (LongLive-style).
    Supports chunk-wise generation with KV cache reuse and attention sink.
    """
    def __init__(self, args, device):
        super().__init__(args, device)

        # Streaming training configuration
        self.streaming_training = getattr(args, "streaming_training", False)
        self.streaming_chunk_size = getattr(args, "streaming_chunk_size", 21)
        self.streaming_max_length = getattr(args, "streaming_max_length", 21)
        self.streaming_min_new_frame = getattr(args, "streaming_min_new_frame", 18)
        self.train_first_chunk = getattr(args, "train_first_chunk", True)

        # Streaming state
        self.reset_state()

        if self.streaming_training:
            # Override inference pipeline with streaming pipeline
            self.inference_pipeline = None  # Will be initialized lazily

    def reset_state(self):
        """Reset streaming training state"""
        self.state = {
            "current_length": 0,
            "conditional_info": None,
            "previous_frames": None,  # Store last generated frames for overlap
            "temp_max_length": None,  # Temporary max length for the current sequence
        }

    def _initialize_inference_pipeline(self):
        """
        Lazy initialize the streaming inference pipeline.
        """
        if self.streaming_training:
            self.inference_pipeline = StreamingTrainingPipeline(
                denoising_step_list=self.denoising_step_list,
                scheduler=self.scheduler,
                generator=self.generator,
                num_frame_per_block=self.num_frame_per_block,
                same_step_across_blocks=self.args.same_step_across_blocks,
                last_step_only=self.args.last_step_only,
                context_noise=self.args.context_noise,
                local_attn_size=self.args.model_kwargs.get("local_attn_size", -1) if hasattr(self.args, "model_kwargs") and isinstance(self.args.model_kwargs, dict) else -1,
                slice_last_frames=getattr(self.args, "slice_last_frames", 21),
            )
        else:
            # Fall back to standard pipeline
            super()._initialize_inference_pipeline()

    def _process_first_frame_encoding(self, frames: torch.Tensor) -> torch.Tensor:
        """
        Apply special encoding to the first frame when there's overlap.
        Similar to base.py logic: decode previous frames, take last frame, re-encode.
        """
        total_frames = frames.shape[1]

        if total_frames <= 1:
            return frames

        # Process last 21 frames
        process_frames = min(21, total_frames)

        with torch.no_grad():
            # Decode the frames to be processed into pixels
            frames_to_decode = frames[:, :-(process_frames - 1), ...]
            pixels = self.vae.decode_to_pixel(frames_to_decode)

            # Take the last frame's pixel representation
            last_frame_pixel = pixels[:, -1:, ...].to(self.dtype)
            last_frame_pixel = rearrange(last_frame_pixel, "b t c h w -> b c t h w")

            # Re-encode as image latent
            image_latent = self.vae.encode_to_latent(last_frame_pixel).to(self.dtype)

        remaining_frames = frames[:, -(process_frames - 1):, ...]
        processed_frames = torch.cat([image_latent, remaining_frames], dim=1)

        return processed_frames

    def setup_sequence(
        self,
        conditional_dict: dict,
        unconditional_dict: dict,
        initial_latent: Optional[torch.Tensor] = None,
        temp_max_length: Optional[int] = None,
    ):
        """Set up a new sequence for streaming training"""
        if self.inference_pipeline is None:
            self._initialize_inference_pipeline()

        batch_size = self.args.image_or_video_shape[0] if hasattr(self.args, "image_or_video_shape") else 1

        # Initialize KV cache if needed
        if self.inference_pipeline.kv_cache1 is None:
            self.inference_pipeline._initialize_kv_cache(
                batch_size=batch_size,
                dtype=self.dtype,
                device=self.device
            )

        if self.inference_pipeline.crossattn_cache is None:
            self.inference_pipeline._initialize_crossattn_cache(
                batch_size=batch_size,
                dtype=self.dtype,
                device=self.device
            )

        # Reset state
        self.reset_state()
        self.state["temp_max_length"] = temp_max_length if temp_max_length is not None else self.streaming_max_length

        # Prepare initial sequence
        if initial_latent is not None:
            self.state["current_length"] = initial_latent.shape[1]
            # Initialize cache with initial latent
            timestep = torch.zeros([batch_size, initial_latent.shape[1]], device=self.device, dtype=torch.int64)
            with torch.no_grad():
                self.generator(
                    noisy_image_or_video=initial_latent,
                    conditional_dict=conditional_dict,
                    timestep=timestep,
                    kv_cache=self.inference_pipeline.kv_cache1,
                    crossattn_cache=self.inference_pipeline.crossattn_cache,
                    current_start=0
                )
        else:
            self.state["current_length"] = 0

        # Save conditional information
        self.state["conditional_info"] = {
            "conditional_dict": conditional_dict,
            "unconditional_dict": unconditional_dict,
        }

    def can_generate_more(self) -> bool:
        """Check whether more chunks can be generated"""
        current_length = self.state["current_length"]
        temp_max_length = self.state.get("temp_max_length")
        can_generate = current_length < temp_max_length and (current_length + self.streaming_min_new_frame) <= temp_max_length
        return can_generate

    def generate_next_chunk(self, requires_grad: bool = True) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Generate the next chunk, supporting overlap to ensure temporal continuity.

        Args:
            requires_grad: whether gradients are required

        Returns:
            generated_chunk: the full generated chunk (including overlap frames)
            info: generation info (including timestep, gradient_mask, etc.)
        """
        if not self.can_generate_more():
            raise ValueError("Cannot generate more chunks")

        current_length = self.state["current_length"]
        batch_size = self.args.image_or_video_shape[0] if hasattr(self.args, "image_or_video_shape") else 1

        # Check if previous_frames can be used for overlap
        previous_frames = self.state.get("previous_frames")
        if previous_frames is not None:
            # Randomly select number of new frames (min=min_new_frame, max=chunk_size, step=3)
            max_new_frames = min(self.state["temp_max_length"] - current_length + 1, self.streaming_chunk_size)
            possible_new_frames = list(range(self.streaming_min_new_frame, max_new_frames + 1, 3))

            # Ensure all processes choose the same random value
            if dist.is_initialized():
                if dist.get_rank() == 0:
                    import random
                    selected_idx = random.randint(0, len(possible_new_frames) - 1)
                else:
                    selected_idx = 0
                selected_idx_tensor = torch.tensor(selected_idx, device=self.device, dtype=torch.int32)
                dist.broadcast(selected_idx_tensor, src=0)
                selected_idx = selected_idx_tensor.item()
            else:
                import random
                selected_idx = random.randint(0, len(possible_new_frames) - 1)

            new_frames_to_generate = possible_new_frames[selected_idx]

            # Auto-compute required overlap frames to ensure the final chunk has chunk_size frames
            overlap_frames = self.streaming_chunk_size - new_frames_to_generate
            if overlap_frames > 0 and overlap_frames <= previous_frames.shape[1]:
                overlap_frames_to_use = overlap_frames
            else:
                # If overlap can't be used, generate a full chunk_size without overlap
                overlap_frames_to_use = 0
                new_frames_to_generate = self.streaming_chunk_size
        else:
            overlap_frames_to_use = 0
            new_frames_to_generate = self.streaming_chunk_size

        # Sample noise for new frames
        image_shape = self.args.image_or_video_shape[2:] if hasattr(self.args, "image_or_video_shape") else [16, 60, 104]
        noise_chunk = torch.randn(
            [batch_size, new_frames_to_generate, *image_shape],
            device=self.device,
            dtype=self.dtype
        )

        # Generate new frames
        conditional_dict = self.state["conditional_info"]["conditional_dict"]
        generated_new_frames, denoised_timestep_from, denoised_timestep_to = self.inference_pipeline.generate_chunk_with_cache(
            noise=noise_chunk,
            conditional_dict=conditional_dict,
            current_start_frame=current_length,
            requires_grad=requires_grad,
        )

        # Build the full chunk for loss computation
        if previous_frames is not None:
            full_chunk = torch.cat([previous_frames, generated_new_frames], dim=1)
        else:
            full_chunk = generated_new_frames

        # Update state - save the last chunk_size frames as previous_frames for the next chunk
        frames_to_save = full_chunk.detach().clone()[:, -self.streaming_chunk_size:, ...]

        # Process first-frame encoding (if there is overlap)
        if previous_frames is not None:
            full_chunk = self._process_first_frame_encoding(full_chunk)

        # Create gradient_mask
        if previous_frames is not None:
            # Only newly generated frames require gradients
            gradient_mask = torch.zeros_like(full_chunk, dtype=torch.bool)
            gradient_mask[:, overlap_frames_to_use:overlap_frames_to_use + new_frames_to_generate, ...] = True
        else:
            # For the first chunk, all frames are newly generated (unless train_first_chunk is False)
            if self.train_first_chunk:
                gradient_mask = torch.ones_like(full_chunk, dtype=torch.bool)
            else:
                gradient_mask = torch.zeros_like(full_chunk, dtype=torch.bool)
                # Only train last num_frame_per_block frames
                gradient_mask[:, -self.num_frame_per_block:, ...] = True

        self.state["current_length"] += new_frames_to_generate
        self.state["previous_frames"] = frames_to_save

        # Return info
        info = {
            "denoised_timestep_from": denoised_timestep_from,
            "denoised_timestep_to": denoised_timestep_to,
            "chunk_start_frame": current_length - new_frames_to_generate,
            "chunk_frames": full_chunk.shape[1],
            "new_frames_generated": new_frames_to_generate,
            "current_length": self.state["current_length"],
            "gradient_mask": gradient_mask,
            "overlap_frames_used": overlap_frames_to_use,
        }

        return full_chunk, info

    def _run_generator(
        self,
        image_or_video_shape,
        conditional_dict: dict,
        unconditional_dict: dict = None,
        initial_latent: torch.tensor = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[int], Optional[int]]:
        """
        Run generator with streaming training.
        Generates chunks sequentially until max_length is reached.
        For 5s training, typically generates one chunk of 21 frames.
        """
        if not self.streaming_training:
            # Fall back to standard DMD behavior
            result = super()._run_generator(image_or_video_shape, conditional_dict, initial_latent)
            # Standard DMD returns (pred_image, gradient_mask, d_from, d_to)
            if len(result) == 4:
                return result
            else:
                # Handle old format
                return result[0], result[1] if len(result) > 1 else None, result[2] if len(result) > 2 else None, result[3] if len(result) > 3 else None

        # Setup sequence
        if unconditional_dict is None:
            unconditional_dict = {}
        self.setup_sequence(
            conditional_dict=conditional_dict,
            unconditional_dict=unconditional_dict,
            initial_latent=initial_latent,
            temp_max_length=self.streaming_max_length,
        )

        # For 5s training, typically we only generate one chunk
        # But we support multiple chunks for longer videos
        all_chunks = []
        all_infos = []

        while self.can_generate_more():
            chunk, info = self.generate_next_chunk(requires_grad=True)
            all_chunks.append(chunk)
            all_infos.append(info)

        # Concatenate all chunks
        if len(all_chunks) > 0:
            pred_image = torch.cat(all_chunks, dim=1)
            # Use info from the last chunk
            last_info = all_infos[-1]

            # Combine gradient masks from all chunks
            if len(all_infos) > 1:
                gradient_masks = [info["gradient_mask"] for info in all_infos]
                gradient_mask = torch.cat(gradient_masks, dim=1)
            else:
                gradient_mask = last_info.get("gradient_mask", None)

            return pred_image, gradient_mask, last_info.get("denoised_timestep_from"), last_info.get("denoised_timestep_to")
        else:
            # No chunks generated, fallback to standard
            return super()._run_generator(image_or_video_shape, conditional_dict, initial_latent)

    def generator_loss(
        self,
        image_or_video_shape,
        conditional_dict: dict,
        unconditional_dict: dict,
        clean_latent: torch.Tensor,
        initial_latent: torch.Tensor = None
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute generator loss with streaming training support.
        Note: clean_latent is not used in streaming training (backward simulation is used instead).
        """
        if not self.streaming_training:
            # Fall back to standard DMD behavior
            return super().generator_loss(image_or_video_shape, conditional_dict, unconditional_dict, clean_latent, initial_latent)

        # Generate with streaming
        pred_image, gradient_mask, d_from, d_to = self._run_generator(
            image_or_video_shape=image_or_video_shape,
            conditional_dict=conditional_dict,
            unconditional_dict=unconditional_dict,
            initial_latent=initial_latent
        )

        if pred_image is None:
            # Fallback to standard training
            return super().generator_loss(image_or_video_shape, conditional_dict, unconditional_dict, clean_latent, initial_latent)

        # Compute DMD loss
        dmd_loss, dmd_log_dict, grad = self.compute_distribution_matching_loss(
            image_or_video=pred_image,
            conditional_dict=conditional_dict,
            unconditional_dict=unconditional_dict,
            gradient_mask=gradient_mask,
            denoised_timestep_from=d_from,
            denoised_timestep_to=d_to
        )

        dmd_log_dict["dmd_loss"] = dmd_loss.detach()

        # Logging: Save videos periodically
        with torch.autocast("cuda", dtype=torch.bfloat16):
            self._save_log_videos(
                pred_image=pred_image,
                grad=grad,
                clean_latent=clean_latent,
                d_from=d_from,
                d_to=d_to
            )

        return dmd_loss, dmd_log_dict
