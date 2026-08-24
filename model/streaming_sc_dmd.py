from pipeline.streaming_sc_training import StreamingShortcutInjectedTrainingPipeline
from model.streaming_dmd import StreamingDMD
import torch.nn.functional as F
from typing import Tuple, Optional, Dict, Any
import torch
import torch.distributed as dist
from einops import rearrange
import math


class StreamingSCDMD(StreamingDMD):
    """
    Streaming DMD model with Shortcut Consistency Flow Matching (SCFM) support.
    Combines LongLive-style streaming training with SCFM target guidance.
    """
    def __init__(self, args, device):
        super().__init__(args, device)
        self.consistency_mode = getattr(args, "consistency_mode", "ode")

    def _initialize_inference_pipeline(self):
        """
        Lazy initialize the streaming inference pipeline with SCFM support.
        """
        if self.streaming_training:
            self.inference_pipeline = StreamingShortcutInjectedTrainingPipeline(
                denoising_step_list=self.denoising_step_list,
                scheduler=self.scheduler,
                generator=self.generator,
                num_frame_per_block=self.num_frame_per_block,
                same_step_across_blocks=self.args.same_step_across_blocks,
                last_step_only=self.args.last_step_only,
                context_noise=self.args.context_noise,
                consistency_mode=self.consistency_mode,
                local_attn_size=self.args.model_kwargs.get("local_attn_size", -1) if hasattr(self.args, "model_kwargs") and isinstance(self.args.model_kwargs, dict) else -1,
                slice_last_frames=getattr(self.args, "slice_last_frames", 21),
            )
        else:
            # Fall back to standard pipeline
            super()._initialize_inference_pipeline()

    def generate_next_chunk(self, requires_grad: bool = True, return_scfm_target: bool = False) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Generate the next chunk with optional SCFM target calculation.

        Args:
            requires_grad: whether gradients are required
            return_scfm_target: whether to compute and return SCFM targets

        Returns:
            generated_chunk: the full generated chunk (including overlap frames)
            info: generation info (including timestep, gradient_mask, scfm targets, etc.)
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

        # Generate new frames with optional SCFM target
        conditional_dict = self.state["conditional_info"]["conditional_dict"]
        if return_scfm_target:
            generated_new_frames, denoised_timestep_from, denoised_timestep_to, scfm_pred, scfm_target, scfm_target_x0, scfm_weights = self.inference_pipeline.generate_chunk_with_cache(
                noise=noise_chunk,
                conditional_dict=conditional_dict,
                current_start_frame=current_length,
                requires_grad=requires_grad,
                return_scfm_target=True,
            )
        else:
            generated_new_frames, denoised_timestep_from, denoised_timestep_to = self.inference_pipeline.generate_chunk_with_cache(
                noise=noise_chunk,
                conditional_dict=conditional_dict,
                current_start_frame=current_length,
                requires_grad=requires_grad,
                return_scfm_target=False,
            )
            scfm_pred, scfm_target, scfm_target_x0, scfm_weights = None, None, None, None

        # Build the full chunk for loss computation
        if previous_frames is not None:
            full_chunk = torch.cat([previous_frames, generated_new_frames], dim=1)
            # Also concatenate SCFM tensors if they exist
            if scfm_pred is not None:
                # Pad SCFM tensors with zeros for overlap frames
                scfm_pred_padded = torch.zeros_like(full_chunk)
                scfm_pred_padded[:, overlap_frames_to_use:overlap_frames_to_use + new_frames_to_generate] = scfm_pred
                scfm_pred = scfm_pred_padded

                scfm_target_padded = torch.zeros_like(full_chunk)
                scfm_target_padded[:, overlap_frames_to_use:overlap_frames_to_use + new_frames_to_generate] = scfm_target
                scfm_target = scfm_target_padded

                scfm_target_x0_padded = torch.zeros_like(full_chunk)
                scfm_target_x0_padded[:, overlap_frames_to_use:overlap_frames_to_use + new_frames_to_generate] = scfm_target_x0
                scfm_target_x0 = scfm_target_x0_padded
        else:
            full_chunk = generated_new_frames

        # Update state - save the last chunk_size frames as previous_frames for the next chunk
        frames_to_save = full_chunk.detach().clone()[:, -self.streaming_chunk_size:, ...]

        # Process first-frame encoding (if there is overlap)
        if previous_frames is not None:
            full_chunk = self._process_first_frame_encoding(full_chunk)
            # Also process SCFM tensors if they exist
            if scfm_pred is not None:
                # For overlap frames, we don't have SCFM targets, so keep zeros
                pass

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

        if return_scfm_target and scfm_pred is not None:
            info["scfm_pred"] = scfm_pred
            info["scfm_target"] = scfm_target
            info["scfm_target_x0"] = scfm_target_x0
            info["scfm_weights"] = scfm_weights

        return full_chunk, info

    def _run_generator(
        self,
        image_or_video_shape,
        conditional_dict: dict,
        unconditional_dict: dict = None,
        initial_latent: torch.tensor = None,
        return_scfm_target: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[int], Optional[int], Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Run generator with streaming training and optional SCFM target calculation.
        Generates chunks sequentially until max_length is reached.
        """
        if not self.streaming_training:
            # Fall back to standard DMD behavior (without SCFM)
            result = super()._run_generator(image_or_video_shape, conditional_dict, initial_latent)
            if len(result) == 4:
                return result[0], result[1], result[2], result[3], None, None, None, None
            else:
                return result[0], result[1] if len(result) > 1 else None, result[2] if len(result) > 2 else None, result[3] if len(result) > 3 else None, None, None, None, None

        # Setup sequence
        if unconditional_dict is None:
            unconditional_dict = {}
        self.setup_sequence(
            conditional_dict=conditional_dict,
            unconditional_dict=unconditional_dict,
            initial_latent=initial_latent,
            temp_max_length=self.streaming_max_length,
        )

        # Generate chunks
        all_chunks = []
        all_infos = []

        while self.can_generate_more():
            chunk, info = self.generate_next_chunk(requires_grad=True, return_scfm_target=return_scfm_target)
            all_chunks.append(chunk)
            all_infos.append(info)

        # Concatenate all chunks
        if len(all_chunks) > 0:
            pred_image = torch.cat(all_chunks, dim=1)
            last_info = all_infos[-1]

            # Combine gradient masks from all chunks
            if len(all_infos) > 1:
                gradient_masks = [info["gradient_mask"] for info in all_infos]
                gradient_mask = torch.cat(gradient_masks, dim=1)
            else:
                gradient_mask = last_info.get("gradient_mask", None)

            # Combine SCFM tensors if they exist
            scfm_pred = None
            scfm_target = None
            scfm_target_x0 = None
            scfm_weights = None

            if return_scfm_target and "scfm_pred" in last_info:
                if len(all_infos) > 1:
                    scfm_preds = [info.get("scfm_pred") for info in all_infos if "scfm_pred" in info]
                    scfm_targets = [info.get("scfm_target") for info in all_infos if "scfm_target" in info]
                    scfm_target_x0s = [info.get("scfm_target_x0") for info in all_infos if "scfm_target_x0" in info]
                    if scfm_preds and scfm_preds[0] is not None:
                        scfm_pred = torch.cat(scfm_preds, dim=1)
                        scfm_target = torch.cat(scfm_targets, dim=1)
                        scfm_target_x0 = torch.cat(scfm_target_x0s, dim=1)
                        scfm_weights = last_info.get("scfm_weights", None)
                else:
                    scfm_pred = last_info.get("scfm_pred", None)
                    scfm_target = last_info.get("scfm_target", None)
                    scfm_target_x0 = last_info.get("scfm_target_x0", None)
                    scfm_weights = last_info.get("scfm_weights", None)

            return pred_image, gradient_mask, last_info.get("denoised_timestep_from"), last_info.get("denoised_timestep_to"), scfm_pred, scfm_target, scfm_target_x0, scfm_weights
        else:
            # No chunks generated, fallback to standard
            result = super()._run_generator(image_or_video_shape, conditional_dict, initial_latent)
            if len(result) == 4:
                return result[0], result[1], result[2], result[3], None, None, None, None
            else:
                return result[0], result[1] if len(result) > 1 else None, result[2] if len(result) > 2 else None, result[3] if len(result) > 3 else None, None, None, None, None

    def generator_loss(
        self,
        image_or_video_shape,
        conditional_dict: dict,
        unconditional_dict: dict,
        clean_latent: torch.Tensor,
        initial_latent: torch.Tensor = None
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute generator loss with streaming training and SCFM support.
        """
        if not self.streaming_training:
            # Fall back to standard DMD behavior
            return super().generator_loss(image_or_video_shape, conditional_dict, unconditional_dict, clean_latent, initial_latent)

        lambda_scfm = getattr(self.args, "lambda_scfm", 0.0)
        return_scfm_target = lambda_scfm > 0.0

        # Generate with streaming and optional SCFM target
        pred_image, gradient_mask, d_from, d_to, scfm_pred, scfm_target, scfm_target_x0, scfm_weights = self._run_generator(
            image_or_video_shape=image_or_video_shape,
            conditional_dict=conditional_dict,
            unconditional_dict=unconditional_dict,
            initial_latent=initial_latent,
            return_scfm_target=return_scfm_target
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

        # Compute SCFM loss
        sc_loss = 0.0
        if return_scfm_target and scfm_pred is not None and scfm_target is not None:
            # scfm_pred and scfm_target are velocities
            # Loss = MSE(pred_v, target_v) * weight
            mse = F.mse_loss(scfm_pred.float(), scfm_target.float(), reduction='none')

            if gradient_mask is not None:
                mse = mse * gradient_mask.float()

            # Average over dimensions except batch
            mse = mse.mean(dim=[1, 2, 3, 4])

            # Apply time-dependent weights
            if scfm_weights is not None:
                mse = mse * scfm_weights

            sc_loss = mse.mean()

        # Compute training step-based weight for lambda_scfm
        scfm_warmup_steps = getattr(self.args, "scfm_warmup_steps", 600)
        scfm_warmup_schedule = getattr(self.args, "scfm_warmup_schedule", "cosine")  # "cosine" or "sigmoid"

        if self._current_step is not None and scfm_warmup_steps > 0:
            current_step = float(self._current_step)
            if scfm_warmup_schedule == "cosine":
                progress = min(current_step / scfm_warmup_steps, 1.0)
                scfm_step_weight = 0.5 * (1.0 - math.cos(math.pi * progress))
            elif scfm_warmup_schedule == "sigmoid":
                center = scfm_warmup_steps / 2.0
                scale = scfm_warmup_steps / 6.0
                scfm_step_weight = 1.0 / (1.0 + math.exp(-(current_step - center) / scale))
            else:
                scfm_step_weight = min(current_step / scfm_warmup_steps, 1.0)
        else:
            scfm_step_weight = 1.0

        # Apply step-based weight to lambda_scfm
        effective_lambda_scfm = lambda_scfm * scfm_step_weight
        dmd_log_dict["scfm_step_weight"] = scfm_step_weight
        dmd_log_dict["effective_lambda_scfm"] = effective_lambda_scfm

        total_loss = dmd_loss + effective_lambda_scfm * sc_loss
        dmd_log_dict["scfm_loss"] = effective_lambda_scfm * sc_loss.detach()

        # Logging: Save videos periodically
        with torch.autocast("cuda", dtype=torch.bfloat16):
            # Use ShortcutInjectedDMD's _save_log_videos if SCFM targets are available
            if return_scfm_target and scfm_pred is not None and scfm_target is not None:
                # Import ShortcutInjectedDMD's save method
                from model.sc_dmd import ShortcutInjectedDMD
                ShortcutInjectedDMD._save_log_videos(
                    self,
                    pred_image=pred_image,
                    scfm_pred=scfm_pred,
                    scfm_target=scfm_target,
                    scfm_target_x0=scfm_target_x0,
                    grad=grad,
                    clean_latent=clean_latent,
                    d_from=d_from,
                    d_to=d_to
                )
            else:
                # Fall back to standard DMD logging
                self._save_log_videos(
                    pred_image=pred_image,
                    grad=grad,
                    clean_latent=clean_latent,
                    d_from=d_from,
                    d_to=d_to
                )

        return total_loss, dmd_log_dict
