from utils.wan_wrapper import WanDiffusionWrapper
from utils.scheduler import SchedulerInterface
from typing import List, Optional
import torch
import torch.distributed as dist
import math

class ShortcutInjectedTrainingPipeline:
    def __init__(self,
                 denoising_step_list: List[int],
                 scheduler: SchedulerInterface,
                 generator: WanDiffusionWrapper,
                 num_frame_per_block=3,
                 independent_first_frame: bool = False,
                 same_step_across_blocks: bool = False,
                 last_step_only: bool = False,
                 num_max_frames: int = 21,
                 context_noise: int = 0,
                 consistency_mode: str = "ode", # "ode" or "sde"
                 loop_rope_training: bool = False,
                 scfm_use_latent_target: bool = True,
                 num_segments: int = 2,
                 use_infinite_attention: bool = False,
                 local_attn_size: int = -1,
                 **kwargs):

        self.scheduler = scheduler
        self.generator = generator
        self.denoising_step_list = denoising_step_list
        if self.denoising_step_list[-1] == 0:
            self.denoising_step_list = self.denoising_step_list[:-1]

        self.num_transformer_blocks = 30
        self.frame_seq_length = 1560
        self.num_frame_per_block = num_frame_per_block
        self.context_noise = context_noise
        self.i2v = False

        self.kv_cache1 = None
        self.kv_cache2 = None
        self.independent_first_frame = independent_first_frame
        self.same_step_across_blocks = same_step_across_blocks
        self.last_step_only = last_step_only
        self.kv_cache_size = num_max_frames * self.frame_seq_length
        self.consistency_mode = consistency_mode
        self.loop_rope_training = loop_rope_training  # Enable loop RoPE training
        # SCFM: True = latent at t_end (Vivix-style), False = average velocity (legacy backup)
        self.scfm_use_latent_target = scfm_use_latent_target
        self.num_segments = num_segments
        self.drift_mode = kwargs.get("drift_mode", "independent")
        self.drift_share_exit_input = kwargs.get("drift_share_exit_input", False)

    def generate_and_sync_list(self, num_blocks, num_denoising_steps, device):
        rank = dist.get_rank() if dist.is_initialized() else 0

        if rank == 0:
            indices = torch.randint(
                low=0,
                high=num_denoising_steps,
                size=(num_blocks,),
                device=device
            )
            if self.last_step_only:
                indices = torch.ones_like(indices) * (num_denoising_steps - 1)
        else:
            indices = torch.empty(num_blocks, dtype=torch.long, device=device)

        dist.broadcast(indices, src=0)
        return indices.tolist()

    def generate_cyclic_start_frame_sequence(self, num_blocks, device):
        """
        Generate a cyclic sequence of start frames for loop RoPE training.
        For 21 frames with num_frame_per_block=3, we have 7 blocks.
        Possible positions: [0, 3, 6, 9, 12, 15, 18]
        Randomly select a starting position and cycle through.
        Example: if start=3, sequence is [3, 6, 9, 12, 15, 18, 0]
        Example: if start=9, sequence is [9, 12, 15, 18, 0, 3, 6]
        """
        rank = dist.get_rank() if dist.is_initialized() else 0

        # Generate all possible start frame positions
        # For 21 frames, num_max_frames=21, we have blocks at: 0, 3, 6, 9, 12, 15, 18
        max_start_frame = self.kv_cache_size // self.frame_seq_length - self.num_frame_per_block  # 21 - 3 = 18
        possible_starts = list(range(0, max_start_frame + 1, self.num_frame_per_block))  # [0, 3, 6, 9, 12, 15, 18]

        if rank == 0:
            # Randomly select a starting position
            start_idx = torch.randint(0, len(possible_starts), (1,), device=device).item()
            start_frame = possible_starts[start_idx]

            # Generate cyclic sequence: start from start_frame and wrap around
            sequence = []
            for i in range(num_blocks):
                seq_idx = (start_idx + i) % len(possible_starts)
                sequence.append(possible_starts[seq_idx])

            start_frame_tensor = torch.tensor(start_frame, dtype=torch.long, device=device)
            sequence_tensor = torch.tensor(sequence, dtype=torch.long, device=device)
        else:
            start_frame_tensor = torch.empty(1, dtype=torch.long, device=device)
            sequence_tensor = torch.empty(num_blocks, dtype=torch.long, device=device)

        # Broadcast to all ranks
        dist.broadcast(start_frame_tensor, src=0)
        dist.broadcast(sequence_tensor, src=0)

        return start_frame_tensor.item(), sequence_tensor.tolist()

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

    @staticmethod
    def _find_segment_noise(next_timestep_val, gen_steps_list, gen_noises):
        """
        For noise_aligned_a: find which generator segment a ref sub-step belongs to
        and return the corresponding noise tensor.
        gen_steps_list is descending (e.g. [1000, 750, 500, 250]).
        A ref sub-step targeting next_timestep_val falls into the segment whose
        lower boundary is the largest gen step <= next_timestep_val.
        """
        nt = next_timestep_val.item() if isinstance(next_timestep_val, torch.Tensor) else next_timestep_val
        for gs in gen_steps_list:
            gs_val = gs.item() if isinstance(gs, torch.Tensor) else gs
            if gs_val <= nt:
                return gen_noises.get(gs_val)
        return None

    def _run_ref_path_for_chunk(
            self,
            noisy_input_ref: torch.Tensor,
            ref_steps_list,
            exit_timestep,
            batch_size: int,
            current_num_frames: int,
            current_start_frame: int,
            kv_cache_ref,
            crossattn_cache_ref,
            conditional_dict: dict,
            noise_device,
            gen_noises: dict = None,
            gen_steps_list = None,
            gen_exit_noisy_input: torch.Tensor = None,
    ):
        """
        Run the high-quality reference path (no grad) for one chunk.
        Denoises using ref_steps_list until reaching exit_timestep,
        then returns the pred_x0 at that timestep and updates kv_cache_ref.

        Noise behavior controlled by self.drift_mode:
        - "independent": fresh noise at every re-noise step (default)
        - "noise_aligned_a": reuse generator's noise within each segment (SDE+ODE)
        - "noise_aligned_b": ODE at intermediate steps, SDE with matched noise at boundary steps

        If gen_exit_noisy_input is provided (drift_share_exit_input=True):
        - ref_pred_x0 (from ref's own chain) is used for cache building
        - An extra forward with gen_exit_noisy_input + kv_cache_ref produces the drift target
        - Returns drift_target instead of ref_pred_x0
        """
        gen_steps_set = None
        if gen_steps_list is not None and self.drift_mode == "noise_aligned_b":
            gen_steps_set = set(
                gs.item() if isinstance(gs, torch.Tensor) else gs
                for gs in gen_steps_list
            )

        with torch.no_grad():
            x = noisy_input_ref
            exit_timestep_tensor = None
            for index, current_timestep in enumerate(ref_steps_list):
                timestep = torch.ones(
                    [batch_size, current_num_frames],
                    device=noise_device, dtype=torch.int64
                ) * current_timestep

                _, denoised_pred_x0 = self.generator(
                    noisy_image_or_video=x,
                    conditional_dict=conditional_dict,
                    timestep=timestep,
                    kv_cache=kv_cache_ref,
                    crossattn_cache=crossattn_cache_ref,
                    current_start=current_start_frame * self.frame_seq_length
                )

                next_timestep_val = ref_steps_list[index + 1] if index + 1 < len(ref_steps_list) else 0

                at_exit = False
                ct = current_timestep.item() if isinstance(current_timestep, torch.Tensor) else current_timestep
                et = exit_timestep.item() if isinstance(exit_timestep, torch.Tensor) else exit_timestep
                if abs(ct - et) < 1e-3:
                    at_exit = True
                elif index + 1 < len(ref_steps_list):
                    nt = next_timestep_val.item() if isinstance(next_timestep_val, torch.Tensor) else next_timestep_val
                    if ct > et and nt < et:
                        at_exit = True

                if at_exit:
                    ref_pred_x0 = denoised_pred_x0
                    exit_timestep_tensor = timestep
                    break

                # Re-noise step: behavior depends on drift_mode
                nt_val = next_timestep_val.item() if isinstance(next_timestep_val, torch.Tensor) else next_timestep_val
                past_exit = nt_val < et if not at_exit else False

                if self.drift_mode == "noise_aligned_a" and gen_noises and not past_exit:
                    segment_noise = self._find_segment_noise(next_timestep_val, gen_steps_list, gen_noises)
                    renoise = segment_noise if segment_noise is not None else torch.randn_like(denoised_pred_x0.flatten(0, 1))
                    x = self.scheduler.add_noise(
                        denoised_pred_x0.flatten(0, 1),
                        renoise,
                        next_timestep_val * torch.ones(
                            [batch_size * current_num_frames], device=noise_device, dtype=torch.long)
                    ).unflatten(0, denoised_pred_x0.shape[:2])

                elif self.drift_mode == "noise_aligned_b" and gen_noises and not past_exit:
                    if gen_steps_set is not None and nt_val in gen_steps_set and nt_val in gen_noises:
                        renoise = gen_noises[nt_val]
                        x = self.scheduler.add_noise(
                            denoised_pred_x0.flatten(0, 1),
                            renoise,
                            next_timestep_val * torch.ones(
                                [batch_size * current_num_frames], device=noise_device, dtype=torch.long)
                        ).unflatten(0, denoised_pred_x0.shape[:2])
                    else:
                        pred_velocity = self.get_velocity(x, denoised_pred_x0, timestep)
                        dt = (current_timestep - next_timestep_val) / 1000.0
                        x = x - dt * pred_velocity

                elif self.consistency_mode == "sde":
                    x = self.scheduler.add_noise(
                        denoised_pred_x0.flatten(0, 1),
                        torch.randn_like(denoised_pred_x0.flatten(0, 1)),
                        next_timestep_val * torch.ones(
                            [batch_size * current_num_frames], device=noise_device, dtype=torch.long)
                    ).unflatten(0, denoised_pred_x0.shape[:2])
                else:
                    pred_velocity = self.get_velocity(x, denoised_pred_x0, timestep)
                    dt = (current_timestep - next_timestep_val) / 1000.0
                    x = x - dt * pred_velocity
            else:
                ref_pred_x0 = denoised_pred_x0

            # Shared-exit-input: extra forward with generator's noisy_input for drift target
            drift_target = ref_pred_x0
            if gen_exit_noisy_input is not None and exit_timestep_tensor is not None:
                saved_ref_indices = [
                    (kv_cache_ref[i]["global_end_index"].clone(),
                     kv_cache_ref[i]["local_end_index"].clone())
                    for i in range(len(kv_cache_ref))
                ]
                _, drift_target = self.generator(
                    noisy_image_or_video=gen_exit_noisy_input,
                    conditional_dict=conditional_dict,
                    timestep=exit_timestep_tensor,
                    kv_cache=kv_cache_ref,
                    crossattn_cache=crossattn_cache_ref,
                    current_start=current_start_frame * self.frame_seq_length
                )
                for i, (g_idx, l_idx) in enumerate(saved_ref_indices):
                    kv_cache_ref[i]["global_end_index"].copy_(g_idx)
                    kv_cache_ref[i]["local_end_index"].copy_(l_idx)

            # Cache the ref result (ref_pred_x0, not drift_target) into kv_cache_ref
            context_timestep = torch.ones(
                [batch_size, current_num_frames],
                device=noise_device, dtype=torch.int64
            ) * self.context_noise
            ref_for_cache = self.scheduler.add_noise(
                ref_pred_x0.flatten(0, 1),
                torch.randn_like(ref_pred_x0.flatten(0, 1)),
                context_timestep.flatten() * torch.ones(
                    [batch_size * current_num_frames], device=noise_device, dtype=torch.long)
            ).unflatten(0, ref_pred_x0.shape[:2])
            self.generator(
                noisy_image_or_video=ref_for_cache,
                conditional_dict=conditional_dict,
                timestep=context_timestep,
                kv_cache=kv_cache_ref,
                crossattn_cache=crossattn_cache_ref,
                current_start=current_start_frame * self.frame_seq_length
            )

        return drift_target

    def _initialize_kv_cache_standalone(self, batch_size, dtype, device):
        """Create and return a standalone KV cache (not stored on self)."""
        kv_cache = []
        for _ in range(self.num_transformer_blocks):
            kv_cache.append({
                "k": torch.zeros([batch_size, self.kv_cache_size, 12, 128], dtype=dtype, device=device),
                "v": torch.zeros([batch_size, self.kv_cache_size, 12, 128], dtype=dtype, device=device),
                "global_end_index": torch.tensor([0], dtype=torch.long, device=device),
                "local_end_index": torch.tensor([0], dtype=torch.long, device=device)
            })
        return kv_cache

    def _initialize_crossattn_cache_standalone(self, batch_size, dtype, device):
        """Create and return a standalone cross-attention cache (not stored on self)."""
        cache = []
        for _ in range(self.num_transformer_blocks):
            cache.append({
                "k": torch.zeros([batch_size, 512, 12, 128], dtype=dtype, device=device),
                "v": torch.zeros([batch_size, 512, 12, 128], dtype=dtype, device=device),
                "is_init": False
            })
        return cache

    def inference_with_trajectory(
            self,
            noise: torch.Tensor,
            initial_latent: Optional[torch.Tensor] = None,
            return_sim_step: bool = False,
            return_scfm_target: bool = False,
            denoising_step_list_override: Optional[List] = None,
            exit_index_override: Optional[int] = None,
            ref_step_list: Optional[List] = None,
            **conditional_dict
    ) -> torch.Tensor:
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

        # Container for SCFM targets (velocity)
        scfm_target_container = None
        scfm_pred_container = None # Store current step predicted velocity
        scfm_target_x0_container = None  # Store pred_x0 corresponding to scfm_target
        scfm_weights = None

        if return_scfm_target:
            # We store velocity targets
            scfm_target_container = torch.zeros_like(output)
            scfm_pred_container = torch.zeros_like(output)
            scfm_target_x0_container = torch.zeros_like(output)  # Store pred_x0 for scfm_target
            scfm_weights = torch.zeros(batch_size, device=noise.device)

        # Step 1: Initialize KV cache
        self._initialize_kv_cache(batch_size, dtype=noise.dtype, device=noise.device)
        self._initialize_crossattn_cache(batch_size, dtype=noise.dtype, device=noise.device)

        # Drift control: initialize ref path KV cache if ref_step_list provided
        kv_cache_ref = None
        crossattn_cache_ref = None
        drift_target_container = None
        if ref_step_list is not None:
            kv_cache_ref = self._initialize_kv_cache_standalone(batch_size, dtype=noise.dtype, device=noise.device)
            crossattn_cache_ref = self._initialize_crossattn_cache_standalone(batch_size, dtype=noise.dtype, device=noise.device)
            drift_target_container = torch.zeros(
                [batch_size, num_output_frames, num_channels, height, width],
                device=noise.device, dtype=noise.dtype
            )
            if hasattr(ref_step_list, "tolist"):
                ref_step_list = ref_step_list.tolist()

        # Step 2: Generate cyclic start frame sequence for loop RoPE training (if enabled)
        if self.loop_rope_training:
            initial_start_frame, cyclic_frame_sequence = self.generate_cyclic_start_frame_sequence(
                num_blocks, device=noise.device
            )
            current_start_frame = initial_start_frame
        else:
            current_start_frame = 0
            cyclic_frame_sequence = None

        # Track which position in the cyclic sequence we're at
        cyclic_sequence_idx = 0  # Index into cyclic_frame_sequence

        # Step 3: Cache context feature
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
                if kv_cache_ref is not None:
                    self.generator(
                        noisy_image_or_video=initial_latent,
                        conditional_dict=conditional_dict,
                        timestep=timestep * 0,
                        kv_cache=kv_cache_ref,
                        crossattn_cache=crossattn_cache_ref,
                        current_start=current_start_frame * self.frame_seq_length
                    )
            # Move to next position in cyclic sequence
            if self.loop_rope_training and cyclic_frame_sequence is not None:
                cyclic_sequence_idx = (cyclic_sequence_idx + 1) % len(cyclic_frame_sequence)
                current_start_frame = cyclic_frame_sequence[cyclic_sequence_idx]
            else:
                current_start_frame += 1

        # Step 4: Temporal denoising loop
        all_num_frames = [self.num_frame_per_block] * num_blocks
        if self.independent_first_frame and initial_latent is None:
            all_num_frames = [1] + all_num_frames

        steps_list = denoising_step_list_override if denoising_step_list_override is not None else self.denoising_step_list
        if hasattr(steps_list, "tolist"):
            steps_list = steps_list.tolist()
        num_denoising_steps = len(steps_list)
        if exit_index_override is not None:
            exit_flags = [exit_index_override] * len(all_num_frames)
        else:
            exit_flags = self.generate_and_sync_list(len(all_num_frames), num_denoising_steps, device=noise.device)
        start_gradient_frame_index = num_output_frames - 21

        # Pre-compute next_index and t_end/t_mid for SCFM (shared across all blocks when same_step_across_blocks)
        scfm_next_index = None
        scfm_t_end = None
        scfm_t_mid = None
        if return_scfm_target and self.same_step_across_blocks:
            exit_index = exit_flags[0]
            if exit_index < num_denoising_steps - 1:
                rank = dist.get_rank() if dist.is_initialized() else 0
                L = num_denoising_steps
                t_start = steps_list[exit_index]
                if self.scfm_use_latent_target:
                    if rank == 0:
                        boundary = [steps_list[0]]
                        for i in range(1, self.num_segments):
                            boundary.append(steps_list[i * L // self.num_segments])
                        boundary.append(0)
                        # Ensure at least one step exists between t_end and t_start
                        # by requiring t_end < next_step (where next_step < t_start)
                        next_step = steps_list[exit_index + 1]
                        valid = [b for b in boundary if b < next_step]
                        if valid:
                            valid_sorted = sorted(valid, reverse=True)
                            top2 = valid_sorted[:min(2, len(valid_sorted))]
                            scfm_t_end = top2[torch.randint(len(top2), (1,), device=noise.device).item()]
                        else:
                            scfm_t_end = boundary[-1]
                        valid_mid_indices = [i for i in range(L) if scfm_t_end < steps_list[i] < t_start]
                        if valid_mid_indices:
                            mid_choice = valid_mid_indices[torch.randint(len(valid_mid_indices), (1,), device=noise.device).item()]
                            scfm_t_mid = steps_list[mid_choice]
                        else:
                            scfm_t_mid = scfm_t_end
                        # steps_list 元素是 tensor，boundary[-1] 可能是 Python 0，先转成 int 再建 tensor
                        _te = scfm_t_end.item() if isinstance(scfm_t_end, torch.Tensor) else scfm_t_end
                        _tm = scfm_t_mid.item() if isinstance(scfm_t_mid, torch.Tensor) else scfm_t_mid
                        scfm_te_tensor = torch.tensor([_te, _tm], dtype=torch.long, device=noise.device)
                    else:
                        scfm_te_tensor = torch.empty((2,), dtype=torch.long, device=noise.device)
                    if dist.is_initialized():
                        dist.broadcast(scfm_te_tensor, src=0)
                    scfm_t_end, scfm_t_mid = scfm_te_tensor[0].item(), scfm_te_tensor[1].item()
                else: # 老采样方案
                    if rank == 0:
                        max_offset = num_denoising_steps - exit_index - 1
                        offset = torch.randint(0, max_offset, (1,), device=noise.device).item()
                        scfm_next_index_tensor = torch.tensor([exit_index + 1 + offset], dtype=torch.long, device=noise.device)
                    else:
                        scfm_next_index_tensor = torch.empty((1,), dtype=torch.long, device=noise.device)
                    if dist.is_initialized():
                        dist.broadcast(scfm_next_index_tensor, src=0)
                    scfm_next_index = scfm_next_index_tensor.item()
                    _tm = steps_list[scfm_next_index]
                    mid_idx = num_denoising_steps // 2
                    boundary_step = steps_list[mid_idx]
                    scfm_t_mid = _tm.item() if isinstance(_tm, torch.Tensor) else _tm
                    scfm_t_end = (boundary_step.item() if isinstance(boundary_step, torch.Tensor) else boundary_step) if scfm_next_index < mid_idx else 0
                print('exit_index: ', exit_index)
                if scfm_next_index is not None:
                    print('scfm_next_index: ', scfm_next_index)
                print('scfm_t_end: ', scfm_t_end, ' scfm_t_mid: ', scfm_t_mid)

        for block_index, current_num_frames in enumerate(all_num_frames):
            noisy_input = noise[:, current_start_frame - num_input_frames:current_start_frame + current_num_frames - num_input_frames]
            gen_noises = {}

            # Step 3.1: Spatial denoising loop
            for index, current_timestep in enumerate(steps_list):
                if self.same_step_across_blocks:
                    exit_flag = (index == exit_flags[0])
                else:
                    exit_flag = (index == exit_flags[block_index])

                timestep = torch.ones(
                    [batch_size, current_num_frames],
                    device=noise.device,
                    dtype=torch.int64) * current_timestep

                if not exit_flag:
                    with torch.no_grad():
                        _, denoised_pred_x0 = self.generator(
                            noisy_image_or_video=noisy_input,
                            conditional_dict=conditional_dict,
                            timestep=timestep,
                            kv_cache=self.kv_cache1,
                            crossattn_cache=self.crossattn_cache,
                            current_start=current_start_frame * self.frame_seq_length
                        )

                        next_timestep_val = steps_list[index + 1]

                        if self.consistency_mode == "sde":
                            renoise = torch.randn_like(denoised_pred_x0.flatten(0, 1))
                            if kv_cache_ref is not None and self.drift_mode in ("noise_aligned_a", "noise_aligned_b"):
                                gen_noises[next_timestep_val] = renoise
                            noisy_input = self.scheduler.add_noise(
                                denoised_pred_x0.flatten(0, 1),
                                renoise,
                                next_timestep_val * torch.ones(
                                    [batch_size * current_num_frames], device=noise.device, dtype=torch.long)
                            ).unflatten(0, denoised_pred_x0.shape[:2])
                        else:
                            pred_velocity = self.get_velocity(noisy_input, denoised_pred_x0, timestep)
                            dt = (current_timestep - next_timestep_val) / 1000.0
                            noisy_input = noisy_input - dt * pred_velocity

                else:
                    # === Training Step ===
                    # 1. Predict current step (Gradient required)
                    # We need both flow (velocity) and x0
                    pred_flow, pred_x0 = self.generator(
                        noisy_image_or_video=noisy_input,
                        conditional_dict=conditional_dict,
                        timestep=timestep,
                        kv_cache=self.kv_cache1,
                        crossattn_cache=self.crossattn_cache,
                        current_start=current_start_frame * self.frame_seq_length
                    )

                    if return_scfm_target and (index < num_denoising_steps - 1):
                        # === Shortcut Consistency Logic (aligned with Vivix get_scfm_target) ===
                        v1 = pred_flow
                        t_start = current_timestep
                        L = num_denoising_steps

                        # Use shared t_end/t_mid when pre-computed (same across chunks); else sample per chunk
                        if scfm_t_end is not None and scfm_t_mid is not None:
                            t_end, t_mid = scfm_t_end, scfm_t_mid
                        elif self.scfm_use_latent_target:

                            # 这段只有在same_step_across_blocks为False时才会执行

                            # --- Boundary: same as Vivix (align with 2/4-step inference) ---
                            boundary = [steps_list[0]]
                            for i in range(1, self.num_segments):
                                boundary_idx = i * L // self.num_segments
                                boundary.append(steps_list[boundary_idx])
                            boundary.append(0)
                            # Ensure at least one step exists between t_end and t_start
                            # by requiring t_end < next_step (where next_step < t_start)
                            next_step = steps_list[index + 1]
                            valid = [b for b in boundary if b < next_step]
                            if valid:
                                valid_sorted = sorted(valid, reverse=True)
                                top2 = valid_sorted[:min(2, len(valid_sorted))]
                                t_end = top2[torch.randint(len(top2), (1,), device=noise.device).item()]
                            else:
                                t_end = boundary[-1]
                            valid_mid_indices = [i for i in range(L) if t_end < steps_list[i] < t_start]
                            if valid_mid_indices:
                                mid_choice = valid_mid_indices[torch.randint(len(valid_mid_indices), (1,), device=noise.device).item()]
                                t_mid = steps_list[mid_choice]
                            else:
                                t_mid = t_end
                        else:
                            # --- Legacy backup: velocity target, next_index -> t_mid, mid or 0 -> t_end ---
                            if scfm_next_index is not None:
                                next_index = scfm_next_index
                            else:
                                max_offset = num_denoising_steps - index - 1
                                offset = torch.randint(0, max_offset, (1,), device=noise.device).item()
                                next_index = index + 1 + offset
                            t_mid = steps_list[next_index]
                            mid_idx = num_denoising_steps // 2
                            boundary_step = steps_list[mid_idx]
                            t_end = boundary_step if next_index < mid_idx else 0

                        dt1 = (t_start - t_mid) / 1000.0
                        dt2 = (t_mid - t_end) / 1000.0
                        total_dt = (t_start - t_end) / 1000.0
                        noisy_input_mid = noisy_input - dt1 * v1.detach()

                        timestep_mid_tensor = torch.ones([batch_size, current_num_frames], device=noise.device, dtype=torch.int64) * t_mid
                        with torch.no_grad():
                            v2, v2_x0 = self.generator(
                                noisy_image_or_video=noisy_input_mid,
                                conditional_dict=conditional_dict,
                                timestep=timestep_mid_tensor,
                                kv_cache=self.kv_cache1,
                                crossattn_cache=self.crossattn_cache,
                                current_start=current_start_frame * self.frame_seq_length
                            )

                        if self.scfm_use_latent_target:
                            # Target = latent at t_end (teacher: x_mid - dt2 * v2)
                            # Student pred = latent at t_end (one-step: x_start - total_dt * v1)
                            scfm_target = noisy_input_mid - dt2 * v2
                            scfm_pred_t_end = noisy_input - total_dt * v1
                            scfm_target_container[:, current_start_frame:current_start_frame + current_num_frames] = scfm_target
                            scfm_pred_container[:, current_start_frame:current_start_frame + current_num_frames] = scfm_pred_t_end
                        else:
                            # Legacy: target = average velocity, pred = v1
                            scfm_target = (v1.detach() * dt1 + v2 * dt2) / total_dt
                            scfm_target_container[:, current_start_frame:current_start_frame + current_num_frames] = scfm_target
                            scfm_pred_container[:, current_start_frame:current_start_frame + current_num_frames] = v1

                        scfm_target_x0_container[:, current_start_frame:current_start_frame + current_num_frames] = v2_x0.detach()
                        if scfm_weights is not None:
                            scfm_weights[:] = 1 - t_start / 1000.0

                    denoised_pred = pred_x0
                    break

            # Step 3.1b: Run reference path for drift control (if enabled)
            if ref_step_list is not None and kv_cache_ref is not None:
                noisy_input_ref = noise[:, current_start_frame - num_input_frames:current_start_frame + current_num_frames - num_input_frames]
                exit_timestep = steps_list[exit_flags[0]] if self.same_step_across_blocks else steps_list[exit_flags[block_index]]
                drift_target = self._run_ref_path_for_chunk(
                    noisy_input_ref=noisy_input_ref,
                    ref_steps_list=ref_step_list,
                    exit_timestep=exit_timestep,
                    batch_size=batch_size,
                    current_num_frames=current_num_frames,
                    current_start_frame=current_start_frame,
                    kv_cache_ref=kv_cache_ref,
                    crossattn_cache_ref=crossattn_cache_ref,
                    conditional_dict=conditional_dict,
                    noise_device=noise.device,
                    gen_noises=gen_noises,
                    gen_steps_list=steps_list,
                    gen_exit_noisy_input=noisy_input.detach() if self.drift_share_exit_input else None,
                )
                drift_target_container[:, current_start_frame:current_start_frame + current_num_frames] = drift_target.detach()

            # Step 3.2: record the model's output (x0)
            output[:, current_start_frame:current_start_frame + current_num_frames] = denoised_pred

            # Step 3.3: rerun with timestep zero to update the cache
            context_timestep = torch.ones_like(timestep) * self.context_noise
            denoised_pred = self.scheduler.add_noise(
                denoised_pred.flatten(0, 1),
                torch.randn_like(denoised_pred.flatten(0, 1)),
                context_timestep.flatten() * torch.ones(
                    [batch_size * current_num_frames], device=noise.device, dtype=torch.long)
            ).unflatten(0, denoised_pred.shape[:2])
            with torch.no_grad():
                self.generator(
                    noisy_image_or_video=denoised_pred,
                    conditional_dict=conditional_dict,
                    timestep=context_timestep,
                    kv_cache=self.kv_cache1,
                    crossattn_cache=self.crossattn_cache,
                    current_start=current_start_frame * self.frame_seq_length
                )

            # Step 3.4: update the start and end frame indices
            if self.loop_rope_training and cyclic_frame_sequence is not None:
                # Use the cyclic sequence for the next block
                cyclic_sequence_idx = (cyclic_sequence_idx + 1) % len(cyclic_frame_sequence)
                current_start_frame = cyclic_frame_sequence[cyclic_sequence_idx]
            else:
                # Normal behavior: increment by current_num_frames
                current_start_frame += current_num_frames

        # Step 3.5: Return (use float for argmin so warped values e.g. 748.2 are not truncated to 748; match self_forcing_training)
        def _step_value_for_argmin(s):
            if isinstance(s, torch.Tensor):
                return s.to(device=noise.device, dtype=torch.float32)
            return torch.tensor(s, device=noise.device, dtype=torch.float32)
        if not self.same_step_across_blocks:
            denoised_timestep_from, denoised_timestep_to = None, None
        elif exit_flags[0] == len(steps_list) - 1:
            denoised_timestep_to = 0
            denoised_timestep_from = 1000 - torch.argmin(
                (self.scheduler.timesteps.cuda() - _step_value_for_argmin(steps_list[exit_flags[0]])).abs(), dim=0).item()
        else:
            denoised_timestep_to = 1000 - torch.argmin(
                (self.scheduler.timesteps.cuda() - _step_value_for_argmin(steps_list[exit_flags[0] + 1])).abs(), dim=0).item()
            denoised_timestep_from = 1000 - torch.argmin(
                (self.scheduler.timesteps.cuda() - _step_value_for_argmin(steps_list[exit_flags[0]])).abs(), dim=0).item()

        if return_scfm_target:
            return output, scfm_pred_container, scfm_target_container, scfm_target_x0_container, scfm_weights, denoised_timestep_from, denoised_timestep_to, drift_target_container

        return output, denoised_timestep_from, denoised_timestep_to, drift_target_container

    def _initialize_kv_cache(self, batch_size, dtype, device):
        kv_cache1 = []
        for _ in range(self.num_transformer_blocks):
            kv_cache1.append({
                "k": torch.zeros([batch_size, self.kv_cache_size, 12, 128], dtype=dtype, device=device),
                "v": torch.zeros([batch_size, self.kv_cache_size, 12, 128], dtype=dtype, device=device),
                "global_end_index": torch.tensor([0], dtype=torch.long, device=device),
                "local_end_index": torch.tensor([0], dtype=torch.long, device=device)
            })
        self.kv_cache1 = kv_cache1

    def _initialize_crossattn_cache(self, batch_size, dtype, device):
        crossattn_cache = []
        for _ in range(self.num_transformer_blocks):
            crossattn_cache.append({
                "k": torch.zeros([batch_size, 512, 12, 128], dtype=dtype, device=device),
                "v": torch.zeros([batch_size, 512, 12, 128], dtype=dtype, device=device),
                "is_init": False
            })
        self.crossattn_cache = crossattn_cache
