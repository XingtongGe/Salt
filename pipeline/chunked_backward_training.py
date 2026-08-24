"""
Chunked backward training pipeline: Phase 1 no-grad forward with per-block cache,
for use with block-wise backward + gradient accumulation (Algorithm 2 style).
Does not modify SelfForcingTrainingPipeline; extends behavior via a new method.
"""
from pipeline.self_forcing_training import SelfForcingTrainingPipeline
from typing import List, Optional, Tuple, Any
import torch
import copy


def _deep_copy_kv_cache(kv_cache: List[dict], crossattn_cache: List[dict]) -> Tuple[List[dict], List[dict]]:
    """Deep copy KV cache and crossattn cache for restoring later (stays on same device)."""
    kv_copy = []
    for blk in kv_cache:
        kv_copy.append({
            "k": blk["k"].clone(),
            "v": blk["v"].clone(),
            "global_end_index": blk["global_end_index"].clone(),
            "local_end_index": blk["local_end_index"].clone(),
        })
    cross_copy = []
    for blk in crossattn_cache:
        cross_copy.append({
            "k": blk["k"].clone(),
            "v": blk["v"].clone(),
            "is_init": blk["is_init"],
        })
    return kv_copy, cross_copy


def _deep_copy_kv_cache_to_cpu(kv_cache: List[dict], crossattn_cache: List[dict]) -> Tuple[List[dict], List[dict]]:
    """Deep copy KV cache and crossattn cache to CPU (for Phase 1 storage; reduces GPU memory)."""
    kv_copy = []
    for blk in kv_cache:
        kv_copy.append({
            "k": blk["k"].clone().cpu(),
            "v": blk["v"].clone().cpu(),
            "global_end_index": blk["global_end_index"].clone().cpu(),
            "local_end_index": blk["local_end_index"].clone().cpu(),
        })
    cross_copy = []
    for blk in crossattn_cache:
        cross_copy.append({
            "k": blk["k"].clone().cpu(),
            "v": blk["v"].clone().cpu(),
            "is_init": blk["is_init"],
        })
    return kv_copy, cross_copy


def _save_kv_cache_indices_only(kv_cache: List[dict], crossattn_cache: List[dict], to_cpu: bool = True
) -> Tuple[List[dict], List[dict]]:
    """Save only indices for self-attn KV (model uses start/end index; buffer content is prefix from Phase 1).
    Crossattn still saved in full. This avoids storing N full k,v tensors."""
    kv_indices = []
    for blk in kv_cache:
        ge = blk["global_end_index"].clone()
        le = blk["local_end_index"].clone()
        if to_cpu:
            ge, le = ge.cpu(), le.cpu()
        kv_indices.append({"global_end_index": ge, "local_end_index": le})
    cross_copy = []
    for blk in crossattn_cache:
        cross_copy.append({
            "k": blk["k"].clone().cpu() if to_cpu else blk["k"].clone(),
            "v": blk["v"].clone().cpu() if to_cpu else blk["v"].clone(),
            "is_init": blk["is_init"],
        })
    return kv_indices, cross_copy


def _restore_kv_cache_from_indices(
    kv_cache: List[dict],
    crossattn_cache: List[dict],
    kv_indices_src: List[dict],
    cross_src: List[dict],
    device: Optional[torch.device] = None,
) -> None:
    """Restore only indices into kv_cache (k,v buffers are unchanged; they already have Phase 1 content).
    Restore full crossattn from cross_src."""
    dst_device = device if device is not None else kv_cache[0]["k"].device
    for i, blk in enumerate(kv_cache):
        blk["global_end_index"].copy_(kv_indices_src[i]["global_end_index"].to(dst_device))
        blk["local_end_index"].copy_(kv_indices_src[i]["local_end_index"].to(dst_device))
    for i, blk in enumerate(crossattn_cache):
        blk["k"].copy_(cross_src[i]["k"].to(dst_device))
        blk["v"].copy_(cross_src[i]["v"].to(dst_device))
        blk["is_init"] = cross_src[i]["is_init"]


def _shrink_kv_cache_to_indices(
    kv_cache: List[dict],
    kv_indices_src: List[dict],
    device: Optional[torch.device] = None,
) -> None:
    """Shrink kv_cache k,v buffers to the valid prefix length (0..local_end_index).
    Frees the rest of the buffer to reduce GPU memory after each block in Phase 2."""
    dst_device = device if device is not None else kv_cache[0]["k"].device
    for i, blk in enumerate(kv_cache):
        le = kv_indices_src[i]["local_end_index"]
        if le.device != dst_device:
            le = le.to(dst_device)
        end_i = le.item()
        if end_i <= 0:
            continue
        # Keep only prefix [0:end_i]; replace buffer so old large buffer can be freed
        new_k = blk["k"][:, :end_i, :, :].clone()
        new_v = blk["v"][:, :end_i, :, :].clone()
        blk["k"] = new_k
        blk["v"] = new_v
        # indices already match (we restored them); no need to change


def _restore_kv_cache(kv_cache: List[dict], crossattn_cache: List[dict],
                      kv_src: List[dict], cross_src: List[dict],
                      device: Optional[torch.device] = None) -> None:
    """Restore kv_cache and crossattn_cache from saved copies (in-place).
    If kv_src/cross_src are on CPU, they are moved to device when copying.
    """
    dst_device = device if device is not None else kv_cache[0]["k"].device
    for i, blk in enumerate(kv_cache):
        blk["k"].copy_(kv_src[i]["k"].to(dst_device))
        blk["v"].copy_(kv_src[i]["v"].to(dst_device))
        blk["global_end_index"].copy_(kv_src[i]["global_end_index"].to(dst_device))
        blk["local_end_index"].copy_(kv_src[i]["local_end_index"].to(dst_device))
    for i, blk in enumerate(crossattn_cache):
        blk["k"].copy_(cross_src[i]["k"].to(dst_device))
        blk["v"].copy_(cross_src[i]["v"].to(dst_device))
        blk["is_init"] = cross_src[i]["is_init"]


class ChunkedBackwardTrainingPipeline(SelfForcingTrainingPipeline):
    """
    Same as SelfForcingTrainingPipeline but adds inference_with_trajectory_and_cache
    for block-wise backward: runs full forward with no_grad and returns per-block
    Xcache (noisy input at exit step) and KV state after each block.
    """

    def inference_with_trajectory_and_cache(
        self,
        noise: torch.Tensor,
        initial_latent: Optional[torch.Tensor] = None,
        **conditional_dict
    ) -> Tuple[
        torch.Tensor,
        List[torch.Tensor],
        List[Tuple[List[dict], List[dict]]],
        Optional[int],
        Optional[int],
        int,
    ]:
        """
        Phase 1: Full forward with no_grad. Cache per-block noisy input at exit step
        and KV state after each block.
        Returns:
            output: [B, N, C, H, W] full pred (all detached)
            Xcache_list: list of length num_blocks, each [B, num_frame_per_block, C, H, W] noisy input at exit step
            KV_after_block_list: list of length num_blocks, each (kv_cache_copy, crossattn_cache_copy) after that block
            denoised_timestep_from, denoised_timestep_to: for DMD loss schedule
            exit_step_index: index in denoising_step_list at which we exit (for re-forward timestep)
        """
        batch_size, num_frames, num_channels, height, width = noise.shape
        if not self.independent_first_frame or (self.independent_first_frame and initial_latent is not None):
            assert num_frames % self.num_frame_per_block == 0
            num_blocks = num_frames // self.num_frame_per_block
        else:
            assert (num_frames - 1) % self.num_frame_per_block == 0
            num_blocks = (num_frames - 1) // self.num_frame_per_block
        num_input_frames = initial_latent.shape[1] if initial_latent is not None else 0
        num_output_frames = num_frames + num_input_frames
        output = torch.zeros(
            [batch_size, num_output_frames, num_channels, height, width],
            device=noise.device,
            dtype=noise.dtype
        )

        self._initialize_kv_cache(batch_size=batch_size, dtype=noise.dtype, device=noise.device)
        self._initialize_crossattn_cache(batch_size=batch_size, dtype=noise.dtype, device=noise.device)

        if self.loop_rope_training:
            initial_start_frame, cyclic_frame_sequence = self.generate_cyclic_start_frame_sequence(
                num_blocks, device=noise.device
            )
            current_start_frame = initial_start_frame
        else:
            current_start_frame = 0
            cyclic_frame_sequence = None
        cyclic_sequence_idx = 0

        if initial_latent is not None:
            timestep = torch.ones([batch_size, 1], device=noise.device, dtype=torch.int64) * 0
            output[:, :1] = initial_latent
            with torch.no_grad():
                self.generator(
                    noisy_image_or_video=initial_latent,
                    conditional_dict=conditional_dict,
                    timestep=timestep * 0,
                    kv_cache=self.kv_cache1,
                    crossattn_cache=self.crossattn_cache,
                    current_start=current_start_frame * self.frame_seq_length
                )
            if self.loop_rope_training and cyclic_frame_sequence is not None:
                cyclic_sequence_idx = (cyclic_sequence_idx + 1) % len(cyclic_frame_sequence)
                current_start_frame = cyclic_frame_sequence[cyclic_sequence_idx]
            else:
                current_start_frame += 1

        all_num_frames = [self.num_frame_per_block] * num_blocks
        if self.independent_first_frame and initial_latent is None:
            all_num_frames = [1] + all_num_frames
        num_denoising_steps = len(self.denoising_step_list)
        exit_flags = self.generate_and_sync_list(len(all_num_frames), num_denoising_steps, device=noise.device)
        exit_step_index = exit_flags[0]

        Xcache_list: List[torch.Tensor] = []
        KV_after_block_list: List[Tuple[List[dict], List[dict]]] = []

        # Full forward with no_grad
        with torch.no_grad():
            for block_index, current_num_frames in enumerate(all_num_frames):
                noisy_input = noise[
                    :, current_start_frame - num_input_frames:current_start_frame + current_num_frames - num_input_frames
                ]

                for index, current_timestep in enumerate(self.denoising_step_list):
                    exit_flag = (index == exit_flags[0]) if self.same_step_across_blocks else (index == exit_flags[block_index])
                    timestep = torch.ones(
                        [batch_size, current_num_frames],
                        device=noise.device,
                        dtype=torch.int64
                    ) * current_timestep

                    if not exit_flag:
                        _, denoised_pred = self.generator(
                            noisy_image_or_video=noisy_input,
                            conditional_dict=conditional_dict,
                            timestep=timestep,
                            kv_cache=self.kv_cache1,
                            crossattn_cache=self.crossattn_cache,
                            current_start=current_start_frame * self.frame_seq_length
                        )
                        next_timestep = self.denoising_step_list[index + 1]
                        noisy_input = self.scheduler.add_noise(
                            denoised_pred.flatten(0, 1),
                            torch.randn_like(denoised_pred.flatten(0, 1)),
                            next_timestep * torch.ones(
                                [batch_size * current_num_frames], device=noise.device, dtype=torch.long
                            )
                        ).unflatten(0, denoised_pred.shape[:2])
                    else:
                        _, denoised_pred = self.generator(
                            noisy_image_or_video=noisy_input,
                            conditional_dict=conditional_dict,
                            timestep=timestep,
                            kv_cache=self.kv_cache1,
                            crossattn_cache=self.crossattn_cache,
                            current_start=current_start_frame * self.frame_seq_length
                        )
                        # Cache noisy input at exit step (for re-forward in Phase 2)
                        Xcache_list.append(noisy_input.clone())
                        break

                output[:, current_start_frame:current_start_frame + current_num_frames] = denoised_pred

                context_timestep = torch.ones_like(timestep) * self.context_noise
                denoised_pred = self.scheduler.add_noise(
                    denoised_pred.flatten(0, 1),
                    torch.randn_like(denoised_pred.flatten(0, 1)),
                    context_timestep * torch.ones(
                        [batch_size * current_num_frames], device=noise.device, dtype=torch.long
                    )
                ).unflatten(0, denoised_pred.shape[:2])
                self.generator(
                    noisy_image_or_video=denoised_pred,
                    conditional_dict=conditional_dict,
                    timestep=context_timestep,
                    kv_cache=self.kv_cache1,
                    crossattn_cache=self.crossattn_cache,
                    current_start=current_start_frame * self.frame_seq_length
                )

                # KV_after_block_list.append(_deep_copy_kv_cache(self.kv_cache1, self.crossattn_cache))

                # Only save indices for self-attn KV (buffer content stays in pipeline; Phase 2 restores indices only).
                KV_after_block_list.append(_save_kv_cache_indices_only(self.kv_cache1, self.crossattn_cache, to_cpu=True))

                if self.loop_rope_training and cyclic_frame_sequence is not None:
                    cyclic_sequence_idx = (cyclic_sequence_idx + 1) % len(cyclic_frame_sequence)
                    current_start_frame = cyclic_frame_sequence[cyclic_sequence_idx]
                else:
                    current_start_frame += current_num_frames

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

        return output, Xcache_list, KV_after_block_list, denoised_timestep_from, denoised_timestep_to, exit_step_index
