from pipeline.streaming_training import StreamingTrainingPipeline
from utils.wan_wrapper import WanDiffusionWrapper
from utils.scheduler import SchedulerInterface
from typing import List, Optional, Tuple
import torch
import torch.distributed as dist
import math


class StreamingShortcutInjectedTrainingPipeline(StreamingTrainingPipeline):
    """
    Streaming training pipeline with Shortcut Consistency Flow Matching (SCFM) support.
    Combines streaming chunk-wise generation with SCFM target calculation.
    """
    def __init__(self,
                 denoising_step_list: List[int],
                 scheduler: SchedulerInterface,
                 generator: WanDiffusionWrapper,
                 num_frame_per_block=3,
                 same_step_across_blocks: bool = False,
                 last_step_only: bool = False,
                 context_noise: int = 0,
                 consistency_mode: str = "ode",  # "ode" or "sde"
                 **kwargs):
        super().__init__(
            denoising_step_list=denoising_step_list,
            scheduler=scheduler,
            generator=generator,
            num_frame_per_block=num_frame_per_block,
            same_step_across_blocks=same_step_across_blocks,
            last_step_only=last_step_only,
            context_noise=context_noise,
            **kwargs
        )
        self.consistency_mode = consistency_mode

    def get_velocity(self, x_t, x_0, t):
        """
        Calculate velocity from x_t and x_0.
        Assuming x_t = (1 - sigma_t) * x_0 + sigma_t * x_1
        v = x_1 - x_0 = (x_t - x_0) / sigma_t
        """
        # Avoid division by zero
        sigma_t = t.float() / 1000.0
        sigma_t = sigma_t.clamp(min=1e-5)

        # Reshape sigma_t for broadcasting
        # x_t shape: [B, F, C, H, W]
        # sigma_t shape: [B] or [B, F]
        while sigma_t.ndim < x_t.ndim:
            sigma_t = sigma_t.unsqueeze(-1)

        return (x_t - x_0) / sigma_t

    def generate_chunk_with_cache(
        self,
        noise: torch.Tensor,
        conditional_dict: dict,
        *,
        current_start_frame: int = 0,
        requires_grad: bool = True,
        return_sim_step: bool = False,
        return_scfm_target: bool = False,
    ) -> Tuple[torch.Tensor, Optional[int], Optional[int], Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Chunk generation method with SCFM target support.

        Args:
            noise: noise tensor for a single chunk [batch_size, chunk_frames, C, H, W]
            conditional_dict: dictionary of conditional information
            current_start_frame: start frame index of the chunk in the full sequence
            requires_grad: whether gradients are required
            return_sim_step: whether to return simulation step info
            return_scfm_target: whether to compute and return SCFM targets

        Returns:
            output: generated chunk [batch_size, chunk_frames, C, H, W]
            denoised_timestep_from: starting denoise timestep
            denoised_timestep_to: ending denoise timestep
            scfm_pred: predicted velocity [batch_size, chunk_frames, C, H, W] (if return_scfm_target)
            scfm_target: target velocity [batch_size, chunk_frames, C, H, W] (if return_scfm_target)
            scfm_target_x0: pred_x0 corresponding to scfm_target [batch_size, chunk_frames, C, H, W] (if return_scfm_target)
            scfm_weights: weights for SCFM loss [batch_size] (if return_scfm_target)
        """
        batch_size, chunk_frames, num_channels, height, width = noise.shape
        assert chunk_frames % self.num_frame_per_block == 0
        num_blocks = chunk_frames // self.num_frame_per_block
        all_num_frames = [self.num_frame_per_block] * num_blocks

        # Prepare output tensor
        output = torch.zeros_like(noise)

        # Initialize SCFM containers if needed
        scfm_pred_container = None
        scfm_target_container = None
        scfm_target_x0_container = None
        scfm_weights = None

        if return_scfm_target:
            scfm_pred_container = torch.zeros_like(noise)
            scfm_target_container = torch.zeros_like(noise)
            scfm_target_x0_container = torch.zeros_like(noise)
            scfm_weights = torch.zeros([batch_size], device=noise.device, dtype=noise.dtype)

        # Randomly select denoising steps (synced across ranks)
        num_denoising_steps = len(self.denoising_step_list)
        exit_flags = self.generate_and_sync_list(len(all_num_frames), num_denoising_steps, device=noise.device)

        # Pre-compute next_index for SCFM target calculation (shared across all blocks)
        scfm_next_index = None
        if return_scfm_target and self.same_step_across_blocks:
            exit_index = exit_flags[0]
            if exit_index < num_denoising_steps - 1:
                rank = dist.get_rank() if dist.is_initialized() else 0
                if rank == 0:
                    max_offset = num_denoising_steps - exit_index - 1
                    offset = torch.randint(0, max_offset, (1,), device=noise.device).item()
                    scfm_next_index_tensor = torch.tensor([exit_index + 1 + offset], dtype=torch.long, device=noise.device)
                else:
                    scfm_next_index_tensor = torch.empty((1,), dtype=torch.long, device=noise.device)
                if dist.is_initialized():
                    dist.broadcast(scfm_next_index_tensor, src=0)
                scfm_next_index = scfm_next_index_tensor.item()

        # Determine gradient-enabled range
        if not requires_grad:
            start_gradient_frame_index = chunk_frames  # Out of range: no gradients anywhere
        else:
            start_gradient_frame_index = 0

        local_start_frame = 0
        # Set local_attn_size for the generator model
        if self.local_attn_size != -1:
            self.generator.model.local_attn_size = int(self.local_attn_size)
            self._set_all_modules_max_attention_size(int(self.local_attn_size))

        for block_index, current_num_frames in enumerate(all_num_frames):
            noisy_input = noise[:, local_start_frame:local_start_frame + current_num_frames]

            # Spatial denoising loop
            for step_idx, current_timestep in enumerate(self.denoising_step_list):
                exit_flag = (
                    step_idx == exit_flags[0]
                    if self.same_step_across_blocks
                    else step_idx == exit_flags[block_index]
                )

                timestep = torch.ones(
                    [batch_size, current_num_frames],
                    device=noise.device,
                    dtype=torch.int64
                ) * current_timestep

                if not exit_flag:
                    # Intermediate steps: no gradients
                    with torch.no_grad():
                        if self.consistency_mode == "sde":
                            _, denoised_pred_x0 = self.generator(
                                noisy_image_or_video=noisy_input,
                                conditional_dict=conditional_dict,
                                timestep=timestep,
                                kv_cache=self.kv_cache1,
                                crossattn_cache=self.crossattn_cache,
                                current_start=(current_start_frame + local_start_frame) * self.frame_seq_length,
                            )

                            # Add noise for the next step
                            if step_idx < len(self.denoising_step_list) - 1:
                                next_timestep = self.denoising_step_list[step_idx + 1]
                                noisy_input = self.scheduler.add_noise(
                                    denoised_pred_x0.flatten(0, 1),
                                    torch.randn_like(denoised_pred_x0.flatten(0, 1)),
                                    next_timestep * torch.ones(
                                        [batch_size * current_num_frames], device=noise.device, dtype=torch.long
                                    ),
                                ).unflatten(0, denoised_pred_x0.shape[:2])
                        else:
                            # ODE mode
                            _, denoised_pred_x0 = self.generator(
                                noisy_image_or_video=noisy_input,
                                conditional_dict=conditional_dict,
                                timestep=timestep,
                                kv_cache=self.kv_cache1,
                                crossattn_cache=self.crossattn_cache,
                                current_start=(current_start_frame + local_start_frame) * self.frame_seq_length,
                            )

                            if step_idx < len(self.denoising_step_list) - 1:
                                next_timestep = self.denoising_step_list[step_idx + 1]
                                pred_velocity = self.get_velocity(noisy_input, denoised_pred_x0, timestep)
                                dt = (current_timestep - next_timestep) / 1000.0
                                noisy_input = noisy_input - dt * pred_velocity
                else:
                    # Final step: may require gradients and compute SCFM target
                    enable_grad = local_start_frame >= start_gradient_frame_index
                    context_manager = torch.enable_grad() if enable_grad else torch.no_grad()

                    with context_manager:
                        pred_flow, pred_x0 = self.generator(
                            noisy_image_or_video=noisy_input,
                            conditional_dict=conditional_dict,
                            timestep=timestep,
                            kv_cache=self.kv_cache1,
                            crossattn_cache=self.crossattn_cache,
                            current_start=(current_start_frame + local_start_frame) * self.frame_seq_length,
                        )

                    # Compute SCFM target if requested
                    if return_scfm_target and (step_idx < num_denoising_steps - 1):
                        # === Shortcut Consistency Logic ===
                        # 1. Get current velocity (v1)
                        v1 = pred_flow
                        t_start = current_timestep

                        # 2. Sample next timestep (t_mid)
                        if scfm_next_index is not None:
                            next_index = scfm_next_index
                        else:
                            max_offset = num_denoising_steps - step_idx - 1
                            offset = torch.randint(0, max_offset, (1,), device=noise.device).item()
                            next_index = step_idx + 1 + offset
                        t_mid = self.denoising_step_list[next_index]

                        # 3. Determine t_end (Boundary)
                        mid_idx = num_denoising_steps // 2
                        boundary_step = self.denoising_step_list[mid_idx]

                        if next_index < mid_idx:
                            t_end = boundary_step
                        else:
                            t_end = 0

                        # 4. Step 1: Move to t_mid
                        dt1 = (t_start - t_mid) / 1000.0

                        if self.consistency_mode == "sde":
                            timestep_mid_tensor = torch.ones([batch_size * current_num_frames], device=noise.device, dtype=torch.long) * t_mid
                            noisy_input_mid = self.scheduler.add_noise(
                                pred_x0.detach().flatten(0, 1),
                                torch.randn_like(pred_x0.flatten(0, 1)),
                                timestep_mid_tensor
                            ).unflatten(0, pred_x0.shape[:2])
                        else:
                            # ODE Step to t_mid
                            noisy_input_mid = noisy_input - dt1 * v1.detach()

                        # 5. Step 2: Predict v2 at t_mid
                        timestep_mid_tensor = torch.ones([batch_size, current_num_frames], device=noise.device, dtype=torch.int64) * t_mid
                        with torch.no_grad():
                            v2, v2_x0 = self.generator(
                                noisy_image_or_video=noisy_input_mid,
                                conditional_dict=conditional_dict,
                                timestep=timestep_mid_tensor,
                                kv_cache=self.kv_cache1,
                                crossattn_cache=self.crossattn_cache,
                                current_start=(current_start_frame + local_start_frame) * self.frame_seq_length,
                            )

                        # 6. Calculate Average Velocity Target
                        dt2 = (t_mid - t_end) / 1000.0
                        total_dt = (t_start - t_end) / 1000.0

                        scfm_target = (v1.detach() * dt1 + v2 * dt2) / total_dt

                        # Store results
                        scfm_target_container[:, local_start_frame:local_start_frame + current_num_frames] = scfm_target
                        scfm_pred_container[:, local_start_frame:local_start_frame + current_num_frames] = v1
                        scfm_target_x0_container[:, local_start_frame:local_start_frame + current_num_frames] = v2_x0.detach()

                        # Weighting: Better weight calculation based on noise level
                        sigma_t = t_start / 1000.0
                        w = (1.0 - sigma_t) ** 2
                        if scfm_weights is not None:
                            scfm_weights[:] = w

                    denoised_pred = pred_x0
                    break

            # Record output
            output[:, local_start_frame:local_start_frame + current_num_frames] = denoised_pred

            # Update cache with context noise
            context_timestep = torch.ones_like(timestep) * self.context_noise
            context_noisy = self.scheduler.add_noise(
                denoised_pred.flatten(0, 1),
                torch.randn_like(denoised_pred.flatten(0, 1)),
                context_timestep.flatten(0, 1),
            ).unflatten(0, denoised_pred.shape[:2])

            with torch.no_grad():
                self.generator(
                    noisy_image_or_video=context_noisy,
                    conditional_dict=conditional_dict,
                    timestep=context_timestep,
                    kv_cache=self.kv_cache1,
                    crossattn_cache=self.crossattn_cache,
                    current_start=(current_start_frame + local_start_frame) * self.frame_seq_length,
                )

            local_start_frame += current_num_frames

        # Compute and return timestep information
        if not self.same_step_across_blocks:
            denoised_timestep_from, denoised_timestep_to = None, None
        elif exit_flags[0] == len(self.denoising_step_list) - 1:
            denoised_timestep_to = 0
            denoised_timestep_from = 1000 - torch.argmin(
                (self.scheduler.timesteps.cuda() - self.denoising_step_list[exit_flags[0]].cuda()).abs(), dim=0
            ).item()
        else:
            denoised_timestep_to = 1000 - torch.argmin(
                (self.scheduler.timesteps.cuda() - self.denoising_step_list[exit_flags[0] + 1].cuda()).abs(), dim=0
            ).item()
            denoised_timestep_from = 1000 - torch.argmin(
                (self.scheduler.timesteps.cuda() - self.denoising_step_list[exit_flags[0]].cuda()).abs(), dim=0
            ).item()

        if return_scfm_target:
            return output, denoised_timestep_from, denoised_timestep_to, scfm_pred_container, scfm_target_container, scfm_target_x0_container, scfm_weights

        if return_sim_step:
            return output, denoised_timestep_from, denoised_timestep_to, exit_flags[0] + 1

        return output, denoised_timestep_from, denoised_timestep_to
