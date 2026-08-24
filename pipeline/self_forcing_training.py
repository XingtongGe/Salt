from utils.wan_wrapper import WanDiffusionWrapper
from utils.scheduler import SchedulerInterface
from typing import List, Optional
import torch
import torch.distributed as dist


class SelfForcingTrainingPipeline:
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
                 loop_rope_training: bool = False,
                 **kwargs):
        super().__init__()
        self.scheduler = scheduler
        self.generator = generator
        self.denoising_step_list = denoising_step_list
        if self.denoising_step_list[-1] == 0:
            self.denoising_step_list = self.denoising_step_list[:-1]  # remove the zero timestep for inference

        # Wan specific hyperparameters
        self.num_transformer_blocks = 30
        self.frame_seq_length = 1560
        self.num_frame_per_block = num_frame_per_block
        self.context_noise = context_noise
        self.i2v = False

        self.kv_cache1 = None
        self.kv_cache2 = None
        self.independent_first_frame = independent_first_frame # 默认是False
        self.same_step_across_blocks = same_step_across_blocks # True
        self.last_step_only = last_step_only # False
        self.kv_cache_size = num_max_frames * self.frame_seq_length # 存了所有21帧
        self.loop_rope_training = loop_rope_training  # Enable loop RoPE training

    def generate_and_sync_list(self, num_blocks, num_denoising_steps, device):
        rank = dist.get_rank() if dist.is_initialized() else 0

        if rank == 0:
            # Generate random indices
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

        dist.broadcast(indices, src=0)  # Broadcast the random indices to all ranks
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

    def inference_with_trajectory(
            self,
            noise: torch.Tensor,
            initial_latent: Optional[torch.Tensor] = None,
            return_sim_step: bool = False,
            **conditional_dict
    ) -> torch.Tensor:
        batch_size, num_frames, num_channels, height, width = noise.shape
        if not self.independent_first_frame or (self.independent_first_frame and initial_latent is not None):
            # If the first frame is independent and the first frame is provided, then the number of frames in the
            # noise should still be a multiple of num_frame_per_block
            assert num_frames % self.num_frame_per_block == 0
            num_blocks = num_frames // self.num_frame_per_block
        else:
            # Using a [1, 4, 4, 4, 4, 4, ...] model to generate a video without image conditioning
            assert (num_frames - 1) % self.num_frame_per_block == 0
            num_blocks = (num_frames - 1) // self.num_frame_per_block
        num_input_frames = initial_latent.shape[1] if initial_latent is not None else 0
        num_output_frames = num_frames + num_input_frames  # add the initial latent frames
        output = torch.zeros(
            [batch_size, num_output_frames, num_channels, height, width],
            device=noise.device,
            dtype=noise.dtype
        )

        # Step 1: Initialize KV cache to all zeros
        self._initialize_kv_cache(
            batch_size=batch_size, dtype=noise.dtype, device=noise.device
        )
        self._initialize_crossattn_cache(
            batch_size=batch_size, dtype=noise.dtype, device=noise.device
        )

        # Step 2: Generate cyclic start frame sequence for loop RoPE training (if enabled)
        # We need num_blocks which was calculated above
        if self.loop_rope_training:
            initial_start_frame, cyclic_frame_sequence = self.generate_cyclic_start_frame_sequence(
                num_blocks, device=noise.device
            )
            current_start_frame = initial_start_frame
        else:
            current_start_frame = 0
            cyclic_frame_sequence = None

        # Step 3: Cache context feature
        # initial_latent 指的是长视频训练的时候，generator在先前生成的帧
        # Track which position in the cyclic sequence we're at
        cyclic_sequence_idx = 0  # Index into cyclic_frame_sequence

        if initial_latent is not None:
            timestep = torch.ones([batch_size, 1], device=noise.device, dtype=torch.int64) * 0
            # Assume num_input_frames is 1 + self.num_frame_per_block * num_input_blocks
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
        num_denoising_steps = len(self.denoising_step_list)
        exit_flags = self.generate_and_sync_list(len(all_num_frames), num_denoising_steps, device=noise.device)
        start_gradient_frame_index = num_output_frames - 21

        # for block_index in range(num_blocks):
        for block_index, current_num_frames in enumerate(all_num_frames):
            noisy_input = noise[
                :, current_start_frame - num_input_frames:current_start_frame + current_num_frames - num_input_frames]

            # Step 3.1: Spatial denoising loop
            for index, current_timestep in enumerate(self.denoising_step_list):
                if self.same_step_across_blocks: # 这个默认也是true，就特么多余写这个exit flag列表
                    exit_flag = (index == exit_flags[0])
                else:
                    exit_flag = (index == exit_flags[block_index])  # Only backprop at the randomly selected timestep (consistent across all ranks)
                timestep = torch.ones(
                    [batch_size, current_num_frames],
                    device=noise.device,
                    dtype=torch.int64) * current_timestep

                if not exit_flag:
                    with torch.no_grad():
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
                                [batch_size * current_num_frames], device=noise.device, dtype=torch.long)
                        ).unflatten(0, denoised_pred.shape[:2])
                else:
                    # for getting real output
                    # with torch.set_grad_enabled(current_start_frame >= start_gradient_frame_index):
                    if current_start_frame < start_gradient_frame_index: # 只训5s的时候，start_gradient_frame_index是0
                        with torch.no_grad():
                            _, denoised_pred = self.generator(
                                noisy_image_or_video=noisy_input,
                                conditional_dict=conditional_dict,
                                timestep=timestep,
                                kv_cache=self.kv_cache1,
                                crossattn_cache=self.crossattn_cache,
                                current_start=current_start_frame * self.frame_seq_length
                            )
                    else:
                        _, denoised_pred = self.generator(
                            noisy_image_or_video=noisy_input,
                            conditional_dict=conditional_dict,
                            timestep=timestep,
                            kv_cache=self.kv_cache1,
                            crossattn_cache=self.crossattn_cache,
                            current_start=current_start_frame * self.frame_seq_length
                        )
                    break

            # Step 3.2: record the model's output
            output[:, current_start_frame:current_start_frame + current_num_frames] = denoised_pred

            # Step 3.3: rerun with timestep zero to update the cache
            context_timestep = torch.ones_like(timestep) * self.context_noise # 0
            # add context noise
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

        # Step 3.5: Return the denoised timestep
        # 看起来应该是在找当前这个timestep在scheduler中的索引
        if not self.same_step_across_blocks:
            denoised_timestep_from, denoised_timestep_to = None, None
        elif exit_flags[0] == len(self.denoising_step_list) - 1:
            denoised_timestep_to = 0
            denoised_timestep_from = 1000 - torch.argmin(
                (self.scheduler.timesteps.cuda() - self.denoising_step_list[exit_flags[0]].cuda()).abs(), dim=0).item()
        else:
            denoised_timestep_to = 1000 - torch.argmin(
                (self.scheduler.timesteps.cuda() - self.denoising_step_list[exit_flags[0] + 1].cuda()).abs(), dim=0).item()
            denoised_timestep_from = 1000 - torch.argmin(
                (self.scheduler.timesteps.cuda() - self.denoising_step_list[exit_flags[0]].cuda()).abs(), dim=0).item()

        if return_sim_step:
            return output, denoised_timestep_from, denoised_timestep_to, exit_flags[0] + 1

        return output, denoised_timestep_from, denoised_timestep_to

    def inference_from_latent_to_x0(
            self,
            initial_latent: torch.Tensor,
            start_step_index: int,
            **conditional_dict
    ) -> torch.Tensor:
        """
        Denoise from x_mid (latent at anchor timestep) to x0 chunk by chunk.
        RoPE starts from 0 and increases normally per chunk.
        Used for MMD: teacher gives x_mid, generator finishes to x0_hat.

        Args:
            initial_latent: [B, F, C, H, W] latent at anchor timestep (x_mid)
            start_step_index: index into denoising_step_list to start from (e.g. 2 for steps [500, 250])
        Returns:
            output: [B, F, C, H, W] denoised x0
        """
        batch_size, num_frames, num_channels, height, width = initial_latent.shape
        device = initial_latent.device
        dtype = initial_latent.dtype
        assert num_frames % self.num_frame_per_block == 0
        num_blocks = num_frames // self.num_frame_per_block

        step_list = self.denoising_step_list
        if isinstance(step_list, torch.Tensor):
            step_list = step_list[start_step_index:].cpu().tolist()
        else:
            step_list = list(step_list)[start_step_index:]
        if not step_list:
            return initial_latent
        print('step_list: ', step_list)

        self._initialize_kv_cache(batch_size=batch_size, dtype=dtype, device=device)
        self._initialize_crossattn_cache(batch_size=batch_size, dtype=dtype, device=device)

        output = torch.zeros_like(initial_latent)
        current_start_frame = 0

        for block_index in range(num_blocks):
            current_num_frames = self.num_frame_per_block
            chunk_slice = slice(
                current_start_frame,
                current_start_frame + current_num_frames
            )
            print('chunk_slice: ', chunk_slice)
            noisy_input = initial_latent[:, chunk_slice].clone()

            for i, current_timestep in enumerate(step_list):

                t_val = current_timestep.item() if isinstance(current_timestep, torch.Tensor) else current_timestep
                timestep = torch.ones(
                    [batch_size, current_num_frames],
                    device=device,
                    dtype=torch.int64
                ) * t_val

                with torch.no_grad():
                    _, denoised_pred = self.generator(
                        noisy_image_or_video=noisy_input,
                        conditional_dict=conditional_dict,
                        timestep=timestep,
                        kv_cache=self.kv_cache1,
                        crossattn_cache=self.crossattn_cache,
                        current_start=current_start_frame * self.frame_seq_length,
                        cache_read_only=True
                    )

                if i < len(step_list) - 1:
                    next_timestep = step_list[i + 1]
                    print('current_timestep, next_timestep: ', current_timestep, next_timestep)
                    next_val = next_timestep.item() if isinstance(next_timestep, torch.Tensor) else next_timestep
                    noisy_input = self.scheduler.add_noise(
                        denoised_pred.flatten(0, 1),
                        torch.randn_like(denoised_pred.flatten(0, 1), device=device, dtype=dtype),
                        next_val * torch.ones(
                            [batch_size * current_num_frames], device=device, dtype=torch.long
                        )
                    ).unflatten(0, denoised_pred.shape[:2])
                else:
                    output[:, chunk_slice] = denoised_pred

            # context_timestep = torch.ones_like(timestep) * self.context_noise
            # denoised_pred = self.scheduler.add_noise(
            #     denoised_pred.flatten(0, 1),
            #     torch.randn_like(denoised_pred.flatten(0, 1), device=device, dtype=dtype),
            #     context_timestep.flatten() * torch.ones(
            #         [batch_size * current_num_frames], device=device, dtype=torch.long
            #     )
            # ).unflatten(0, denoised_pred.shape[:2])
            # with torch.no_grad():
            #     self.generator(
            #         noisy_image_or_video=denoised_pred,
            #         conditional_dict=conditional_dict,
            #         timestep=context_timestep,
            #         kv_cache=self.kv_cache1,
            #         crossattn_cache=self.crossattn_cache,
            #         current_start=current_start_frame * self.frame_seq_length
            #     )

            current_start_frame += current_num_frames

        return output

    def _initialize_kv_cache(self, batch_size, dtype, device):
        """
        Initialize a Per-GPU KV cache for the Wan model.
        """
        kv_cache1 = []

        for _ in range(self.num_transformer_blocks):
            kv_cache1.append({
                "k": torch.zeros([batch_size, self.kv_cache_size, 12, 128], dtype=dtype, device=device),
                "v": torch.zeros([batch_size, self.kv_cache_size, 12, 128], dtype=dtype, device=device),
                "global_end_index": torch.tensor([0], dtype=torch.long, device=device),
                "local_end_index": torch.tensor([0], dtype=torch.long, device=device)
            })

        self.kv_cache1 = kv_cache1  # always store the clean cache

    def _initialize_crossattn_cache(self, batch_size, dtype, device):
        """
        Initialize a Per-GPU cross-attention cache for the Wan model.
        """
        crossattn_cache = []

        for _ in range(self.num_transformer_blocks):
            crossattn_cache.append({
                "k": torch.zeros([batch_size, 512, 12, 128], dtype=dtype, device=device),
                "v": torch.zeros([batch_size, 512, 12, 128], dtype=dtype, device=device),
                "is_init": False
            })
        self.crossattn_cache = crossattn_cache
