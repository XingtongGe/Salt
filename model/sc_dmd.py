from pipeline.sc_training import ShortcutInjectedTrainingPipeline
from pipeline.longlive_sc_training import LongLiveShortcutInjectedTrainingPipeline
from model.dmd import DMD
import torch.nn.functional as F
from typing import Tuple, Optional
import torch
import torch.distributed as dist
from einops import rearrange
import os
import imageio
import numpy as np
import math
import random


def save_video_as_grid_and_mp4(
    video_batch: torch.Tensor, save_path: str, fps: int = 5, args=None, key=None
):
    """
    Save a batch of videos as MP4 files.
    Args:
        video_batch: Tensor of shape [B, T, C, H, W] in range [0, 1]
        save_path: Directory to save videos
        fps: Frames per second for video
        args: Optional args object
        key: Optional key for naming
    """
    os.makedirs(save_path, exist_ok=True)

    for i, vid in enumerate(video_batch):
        gif_frames = []
        for frame in vid:
            frame = rearrange(frame, "c h w -> h w c")
            frame = (255.0 * frame).cpu().numpy().astype(np.uint8)
            gif_frames.append(frame)
        now_save_path = os.path.join(save_path, f"{i:06d}.mp4")
        with imageio.get_writer(now_save_path, fps=fps) as writer:
            for frame in gif_frames:
                writer.append_data(frame)


def _warp_step_list_if_needed(step_list, scheduler, warp: bool, device):
    """Convert raw step list to warped (scheduler timesteps) if warp_denoising_step."""
    t = torch.tensor(step_list, dtype=torch.long, device=device)
    if not warp:
        return t
    timesteps = torch.cat((scheduler.timesteps.cpu(), torch.tensor([0], dtype=torch.float32)))
    idx = (1000 - t).cpu()
    return timesteps[idx].to(device)


class ShortcutInjectedDMD(DMD):
    # Default 8/4/2 step lists for mixed training (raw step values)
    DEFAULT_SCFM_STEP_LISTS_8 = [1000, 875, 750, 625, 500, 375, 250, 125]
    DEFAULT_SCFM_STEP_LISTS_4 = [1000, 750, 500, 250]
    DEFAULT_SCFM_STEP_LISTS_2 = [1000, 500]
    DEFAULT_SCFM_STEP_PROBS = [0.4, 0.4, 0.2]

    def __init__(self, args, device):
        super().__init__(args, device)
        self.consistency_mode = getattr(args, "consistency_mode", "ode")
        self.drift_mode = getattr(args, "drift_mode", "independent")
        self._log_step_counter = 0  # Internal counter for logging
        self._logdir = None
        self._current_step = None
        # Mixed step-list training: 8-step / 4-step / 2-step with random list + exit_index per run
        self.scfm_mixed_step_lists = getattr(args, "scfm_mixed_step_lists", False)
        if self.scfm_mixed_step_lists:
            raw_8 = getattr(args, "scfm_step_lists_8", self.DEFAULT_SCFM_STEP_LISTS_8)
            raw_4 = getattr(args, "scfm_step_lists_4", self.DEFAULT_SCFM_STEP_LISTS_4)
            raw_2 = getattr(args, "scfm_step_lists_2", self.DEFAULT_SCFM_STEP_LISTS_2)
            warp = getattr(args, "warp_denoising_step", False)
            self._scfm_step_lists_raw = [raw_8, raw_4, raw_2]
            self._scfm_step_lists = [
                _warp_step_list_if_needed(raw_8, self.scheduler, warp, device),
                _warp_step_list_if_needed(raw_4, self.scheduler, warp, device),
                _warp_step_list_if_needed(raw_2, self.scheduler, warp, device),
            ]
            probs = getattr(args, "scfm_step_probs", self.DEFAULT_SCFM_STEP_PROBS)
            self._scfm_step_probs = probs
        else:
            self._scfm_step_lists = None
            self._scfm_step_probs = None

    def set_logging_params(self, step: int = None, logdir: str = None):
        """Set logging parameters (step and logdir) for video saving."""
        if step is not None:
            self._current_step = step
        if logdir is not None:
            self._logdir = logdir

    def _save_log_videos(
        self,
        pred_image: torch.Tensor,
        scfm_pred: Optional[torch.Tensor] = None,
        scfm_target: Optional[torch.Tensor] = None,
        scfm_target_x0: Optional[torch.Tensor] = None,
        grad: Optional[torch.Tensor] = None,
        clean_latent: Optional[torch.Tensor] = None,
        d_from: Optional[int] = None,
        d_to: Optional[int] = None,
        drift_target: Optional[torch.Tensor] = None
    ):
        """
        Save videos for logging purposes.
        Args:
            pred_image: Generated image/video latent [B, T, C, H, W]
            scfm_pred: Shortcut consistency predicted velocity [B, T, C, H, W] (optional)
            scfm_target: Shortcut consistency target velocity [B, T, C, H, W] (optional)
            scfm_target_x0: pred_x0 corresponding to scfm_target [B, T, C, H, W] (optional)
            grad: DMD gradient [B, T, C, H, W] (optional)
            clean_latent: Clean training data latent [B, T, C, H, W] (optional)
            d_from: Denoised timestep from (optional)
            d_to: Denoised timestep to (optional)
        """
        # Check if we should save logs
        is_main_process = dist.get_rank() == 0 if dist.is_initialized() else True
        if not is_main_process:
            return

        # Get logging parameters
        current_step = self._current_step if self._current_step is not None else self._log_step_counter
        logdir = self._logdir if self._logdir is not None else getattr(self.args, "logdir", None)

        # Check if we should save (every 20 steps, similar to reference code)
        # Reference: if current_iter % 20 in [0, 5]
        log_interval = getattr(self.args, "log_video_interval", 20)
        if current_step % log_interval not in [0, 5]:
            self._log_step_counter += 1
            return

        if logdir is None:
            self._log_step_counter += 1
            return

        # Need at least pred_image and grad to save (grad always present from DMD)
        if grad is None:
            self._log_step_counter += 1
            return

        batch_id = 0  # Save first sample in batch

        # Build items to show
        if scfm_pred is not None and scfm_target is not None and scfm_target_x0 is not None:
            _list = [
                ('pred_image', pred_image.detach()),
                ('scfm_pred', scfm_pred.detach()),
                ('scfm_target', scfm_target_x0.detach()),
                ('grad', grad.detach()),
            ]
        else:
            _list = [('pred_image', pred_image.detach()), ('grad', grad.detach())]

        if drift_target is not None:
            _list.append(('drift_target', drift_target.detach()))

        res = []
        for _k, item in _list:
            with torch.no_grad():
                samples_z = item.contiguous()[batch_id:batch_id+1]
                samples_x = self.vae.decode_to_pixel(samples_z)
                temp = torch.clamp((samples_x + 1.0) / 2.0, min=0.0, max=1.0).cpu()
            res.append(temp)

        has_drift = drift_target is not None
        if has_drift and len(res) == 5:
            # 2x3 grid: pred_image | scfm_pred | drift_target
            #            scfm_target | grad     | (black pad)
            pad = torch.zeros_like(res[0])
            row1 = torch.cat((res[0], res[1], res[4]), dim=-1)
            row2 = torch.cat((res[2], res[3], pad), dim=-1)
            video_grid = torch.cat((row1, row2), dim=-2)
        elif has_drift and len(res) == 3:
            # 1x3 grid: pred_image | grad | drift_target
            video_grid = torch.cat((res[0], res[1], res[2]), dim=-1)
        elif len(res) == 4:
            # 2x2 grid: pred_image, scfm_pred, scfm_target, grad
            row1 = torch.cat((res[0], res[1]), dim=-1)
            row2 = torch.cat((res[2], res[3]), dim=-1)
            video_grid = torch.cat((row1, row2), dim=-2)
        else:
            # 1x2 grid: pred_image | grad (no SCFM)
            video_grid = torch.cat((res[0], res[1]), dim=-1)

        # Prepare save path
        timestep_str = f'T{d_from:.1f}-{d_to:.1f}' if d_from is not None and d_to is not None else ''
        save_dir = os.path.join(logdir, f"{current_step:06d}{'-' + timestep_str if timestep_str else ''}")

        # Get fps from config or use default
        fps = getattr(self.args, "sampling_fps", 16)

        # Save video
        save_video_as_grid_and_mp4(
            video_grid,
            save_dir,
            fps=fps,
            args=self.args,
            key="sc_dmd"
        )

        # except Exception as e:
        #     # Don't crash training if logging fails
        #     if dist.get_rank() == 0 if dist.is_initialized() else True:
        #         print(f"Warning: Failed to save log videos: {e}")

        self._log_step_counter += 1

    def compute_distribution_matching_loss(
        self,
        image_or_video: torch.Tensor,
        conditional_dict: dict,
        unconditional_dict: dict,
        gradient_mask: Optional[torch.Tensor] = None,
        denoised_timestep_from: Optional[int] = None,
        denoised_timestep_to: Optional[int] = None
    ) -> Tuple[torch.Tensor, dict, torch.Tensor, torch.Tensor]:
        """
        Compute the DMD loss (eq 7 in https://arxiv.org/abs/2311.18828).
        Modified version that also returns the gradient and pred_real_image (for MMD).
        Input:
            - image_or_video: a tensor with shape [B, F, C, H, W] where the number of frame is 1 for images.
            - conditional_dict: a dictionary containing the conditional information (e.g. text embeddings, image embeddings).
            - unconditional_dict: a dictionary containing the unconditional information (e.g. null/negative text embeddings, null/negative image embeddings).
            - gradient_mask: a boolean tensor with the same shape as image_or_video indicating which pixels to compute loss .
        Output:
            - dmd_loss: a scalar tensor representing the DMD loss.
            - dmd_log_dict: a dictionary containing the intermediate tensors for logging.
            - grad: the computed gradient tensor for logging.
            - pred_real_image: real score prediction (for MMD when mmd_weight > 0).
        """
        original_latent = image_or_video

        batch_size, num_frame = image_or_video.shape[:2]

        with torch.no_grad():
            # Step 1: Randomly sample timestep based on the given schedule and corresponding noise
            min_timestep = denoised_timestep_to if self.ts_schedule and denoised_timestep_to is not None else self.min_score_timestep
            max_timestep = denoised_timestep_from if self.ts_schedule_max and denoised_timestep_from is not None else self.num_train_timestep
            timestep = self._get_timestep(
                min_timestep,
                max_timestep,
                batch_size,
                num_frame,
                self.num_frame_per_block,
                uniform_timestep=True
            )

            # TODO:should we change it to `timestep = self.scheduler.timesteps[timestep]`?
            if self.timestep_shift > 1:
                timestep = self.timestep_shift * \
                    (timestep / 1000) / \
                    (1 + (self.timestep_shift - 1) * (timestep / 1000)) * 1000
            timestep = timestep.clamp(self.min_step, self.max_step)

            noise = torch.randn_like(image_or_video)
            noisy_latent = self.scheduler.add_noise(
                image_or_video.flatten(0, 1),
                noise.flatten(0, 1),
                timestep.flatten(0, 1)
            ).detach().unflatten(0, (batch_size, num_frame))

            # Step 2: Compute the KL grad
            grad, dmd_log_dict, pred_real_image = self._compute_kl_grad(
                noisy_image_or_video=noisy_latent,
                estimated_clean_image_or_video=original_latent,
                timestep=timestep,
                conditional_dict=conditional_dict,
                unconditional_dict=unconditional_dict
            )

        if gradient_mask is not None:
            dmd_loss = 0.5 * F.mse_loss(original_latent.double(
            )[gradient_mask], (original_latent.double() - grad.double()).detach()[gradient_mask], reduction="mean")
        else:
            dmd_loss = 0.5 * F.mse_loss(original_latent.double(
            ), (original_latent.double() - grad.double()).detach(), reduction="mean")

        return dmd_loss, dmd_log_dict, grad, pred_real_image

    def _initialize_inference_pipeline(self):
        """
        Lazy initialize the inference pipeline.
        - use_infinite_attention: ShortcutInjectedTrainingPipeline with infinity-rope cache sizing
        - LongLive (local_attn_size + sink_size): LongLiveShortcutInjectedTrainingPipeline
        - Otherwise: ShortcutInjectedTrainingPipeline (default)
        """
        model_kwargs = getattr(self.args, "model_kwargs", {}) or {}
        local_attn_size = model_kwargs.get("local_attn_size", -1)
        sink_size = model_kwargs.get("sink_size", 0)
        use_infinite_attention = model_kwargs.get("use_infinite_attention", False)
        slice_last_frames = getattr(self.args, "slice_last_frames", 21)
        num_training_frames = getattr(self.args, "num_training_frames", self.num_training_frames)
        use_longlive = (local_attn_size != -1) and (sink_size != 0) and (not use_infinite_attention)

        common = dict(
            denoising_step_list=self.denoising_step_list,
            scheduler=self.scheduler,
            generator=self.generator,
            num_frame_per_block=self.num_frame_per_block,
            independent_first_frame=self.args.independent_first_frame,
            same_step_across_blocks=self.args.same_step_across_blocks,
            last_step_only=self.args.last_step_only,
            num_max_frames=self.num_training_frames,
            context_noise=self.args.context_noise,
            consistency_mode=self.consistency_mode,
            loop_rope_training=getattr(self.args, "loop_rope_training", False),
            scfm_use_latent_target=getattr(self.args, "scfm_use_latent_target", True),
            num_segments=getattr(self.args, "num_segments", 2),
            drift_mode=self.drift_mode,
            drift_share_exit_input=getattr(self.args, "drift_share_exit_input", False),
        )
        if use_infinite_attention:
            self.inference_pipeline = ShortcutInjectedTrainingPipeline(
                **common,
                use_infinite_attention=True,
                local_attn_size=local_attn_size,
            )
        elif use_longlive:
            self.inference_pipeline = LongLiveShortcutInjectedTrainingPipeline(
                **common,
                local_attn_size=local_attn_size,
                sink_size=sink_size,
                slice_last_frames=slice_last_frames,
                num_training_frames=num_training_frames,
            )
        else:
            self.inference_pipeline = ShortcutInjectedTrainingPipeline(**common)

    def _consistency_backward_simulation(
        self,
        noise: torch.Tensor,
        return_scfm_target: bool = False,
        denoising_step_list_override=None,
        exit_index_override: Optional[int] = None,
        ref_step_list=None,
        **conditional_dict: dict
    ) -> torch.Tensor:
        if self.inference_pipeline is None:
            self._initialize_inference_pipeline()

        return self.inference_pipeline.inference_with_trajectory(
            noise=noise,
            return_scfm_target=return_scfm_target,
            denoising_step_list_override=denoising_step_list_override,
            exit_index_override=exit_index_override,
            ref_step_list=ref_step_list,
            **conditional_dict,
        )

    def _run_generator(
        self,
        image_or_video_shape,
        conditional_dict: dict,
        initial_latent: torch.tensor = None,
        return_scfm_target: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Step 1: Sample noise and backward simulate the generator's input
        assert getattr(self.args, "backward_simulation", True), "Backward simulation needs to be enabled"
        if initial_latent is not None:
            conditional_dict["initial_latent"] = initial_latent
        if self.args.i2v:
            noise_shape = [image_or_video_shape[0], image_or_video_shape[1] - 1, *image_or_video_shape[2:]]
        else:
            noise_shape = image_or_video_shape.copy()

        # Generate frames
        min_num_frames = 20 if self.args.independent_first_frame else 21
        max_num_frames = self.num_training_frames - 1 if self.args.independent_first_frame else self.num_training_frames
        assert max_num_frames % self.num_frame_per_block == 0
        assert min_num_frames % self.num_frame_per_block == 0
        max_num_blocks = max_num_frames // self.num_frame_per_block
        min_num_blocks = min_num_frames // self.num_frame_per_block
        num_generated_blocks = torch.randint(min_num_blocks, max_num_blocks + 1, (1,), device=self.device)
        dist.broadcast(num_generated_blocks, src=0)
        num_generated_blocks = num_generated_blocks.item()
        num_generated_frames = num_generated_blocks * self.num_frame_per_block
        if self.args.independent_first_frame and initial_latent is None:
            num_generated_frames += 1
            min_num_frames += 1
        noise_shape[1] = num_generated_frames

        # Mixed step-list: sample one of 8/4/2 step lists and exit_index; SCFM only when 8-step
        denoising_step_list_override = None
        exit_index_override = None
        effective_return_scfm = return_scfm_target
        if self.scfm_mixed_step_lists:
            rank = dist.get_rank() if dist.is_initialized() else 0
            if rank == 0:
                list_idx = torch.multinomial(
                    torch.tensor(self._scfm_step_probs, dtype=torch.float32, device=self.device), 1
                ).item()
                chosen_list = self._scfm_step_lists[list_idx]
                L = len(chosen_list) if hasattr(chosen_list, "__len__") else chosen_list.shape[0]
                exit_idx = torch.randint(0, L, (1,), device=self.device).item()
                mix_tensor = torch.tensor([list_idx, exit_idx], dtype=torch.long, device=self.device)
            else:
                mix_tensor = torch.empty(2, dtype=torch.long, device=self.device)
            if dist.is_initialized():
                dist.broadcast(mix_tensor, src=0)
            list_idx = mix_tensor[0].item()
            exit_index_override = mix_tensor[1].item()
            denoising_step_list_override = self._scfm_step_lists[list_idx]
            effective_return_scfm = return_scfm_target and (list_idx == 0)

        # Drift control: determine ref_step_list based on current step list
        # Activate ref path when drift loss OR ref-based MMD is enabled
        lambda_drift = getattr(self.args, "lambda_drift", 0.0)
        mmd_use_ref = getattr(self.args, "mmd_use_ref", False)
        mmd_weight = getattr(self.args, "mmd_weight", 0.0)
        mmd_start_step = getattr(self.args, "mmd_start_step", 0)
        mmd_active = mmd_weight > 0.0 and (
            mmd_start_step <= 0 or self._current_step is None or self._current_step >= mmd_start_step
        )
        need_ref_path = (lambda_drift > 0.0) or (mmd_use_ref and mmd_active)
        ref_step_list = None
        if need_ref_path and self.scfm_mixed_step_lists and denoising_step_list_override is not None and exit_index_override != 0:
            # list_idx: 0=8-step, 1=4-step, 2=2-step
            if list_idx == 1:
                # rank = dist.get_rank() if dist.is_initialized() else 0
                # if rank == 0:
                #     use_8_step_ref = torch.tensor([random.choice([0, 1])], dtype=torch.long, device=self.device)
                # else:
                #     use_8_step_ref = torch.empty(1, dtype=torch.long, device=self.device)
                # if dist.is_initialized():
                #     dist.broadcast(use_8_step_ref, src=0)
                # # use_8_step_ref = random.choice([True, False])
                # if use_8_step_ref:
                #     ref_step_list = self._scfm_step_lists[0]
                # else:
                #     ref_step_list = None
                ref_step_list = None
            elif list_idx == 2:
                use_4_step_ref = True
                # rank = dist.get_rank() if dist.is_initialized() else 0
                # if rank == 0:
                #     use_4_step_ref = torch.tensor([random.choice([0, 1])], dtype=torch.long, device=self.device)
                # else:
                #     use_4_step_ref = torch.empty(1, dtype=torch.long, device=self.device)
                # if dist.is_initialized():
                #     dist.broadcast(use_4_step_ref, src=0)
                # use_8_step_ref = random.choice([True, False])
                if use_4_step_ref:
                    ref_step_list = self._scfm_step_lists[1]
                else:
                    ref_step_list = None

                # rank = dist.get_rank() if dist.is_initialized() else 0
                # if rank == 0:
                #     drift_ref_choice = torch.tensor([random.choice([0, 1])], dtype=torch.long, device=self.device)
                # else:
                #     drift_ref_choice = torch.empty(1, dtype=torch.long, device=self.device)
                # if dist.is_initialized():
                #     dist.broadcast(drift_ref_choice, src=0)
                # ref_step_list = self._scfm_step_lists[drift_ref_choice.item()]
            # list_idx == 0 (8-step): no drift target needed

        # Call pipeline
        if effective_return_scfm:
            pred_image, scfm_pred, scfm_target, scfm_target_x0, scfm_weights, d_from, d_to, drift_target = self._consistency_backward_simulation(
                noise=torch.randn(noise_shape, device=self.device, dtype=self.dtype),
                return_scfm_target=True,
                denoising_step_list_override=denoising_step_list_override,
                exit_index_override=exit_index_override,
                ref_step_list=ref_step_list,
                **conditional_dict,
            )
        else:
            pred_image, d_from, d_to, drift_target = self._consistency_backward_simulation(
                noise=torch.randn(noise_shape, device=self.device, dtype=self.dtype),
                return_scfm_target=False,
                denoising_step_list_override=denoising_step_list_override,
                exit_index_override=exit_index_override,
                ref_step_list=ref_step_list,
                **conditional_dict,
            )
            scfm_pred, scfm_target, scfm_target_x0, scfm_weights = None, None, None, None

        # Slice last 21 frames logic (same as base)
        if pred_image.shape[1] > 21:
            with torch.no_grad():
                latent_to_decode = pred_image[:, :-20, ...]
                pixels = self.vae.decode_to_pixel(latent_to_decode)
                frame = pixels[:, -1:, ...].to(self.dtype)
                frame = rearrange(frame, "b t c h w -> b c t h w")
                image_latent = self.vae.encode_to_latent(frame).to(self.dtype)
            pred_image_last_21 = torch.cat([image_latent, pred_image[:, -20:, ...]], dim=1)

            # Also slice SCFM tensors if they exist
            if scfm_pred is not None:
                scfm_pred = torch.cat([torch.zeros_like(image_latent), scfm_pred[:, -20:, ...]], dim=1)
                scfm_target = torch.cat([torch.zeros_like(image_latent), scfm_target[:, -20:, ...]], dim=1)
                if scfm_target_x0 is not None:
                    scfm_target_x0 = torch.cat([torch.zeros_like(image_latent), scfm_target_x0[:, -20:, ...]], dim=1)
            if drift_target is not None:
                drift_target = torch.cat([torch.zeros_like(image_latent), drift_target[:, -20:, ...]], dim=1)
        else:
            pred_image_last_21 = pred_image

        if num_generated_frames != min_num_frames:
            gradient_mask = torch.ones_like(pred_image_last_21, dtype=torch.bool)
            if self.args.independent_first_frame:
                gradient_mask[:, :1] = False
            else:
                gradient_mask[:, :self.num_frame_per_block] = False
        else:
            gradient_mask = None

        pred_image_last_21 = pred_image_last_21.to(self.dtype)

        return pred_image_last_21, scfm_pred, scfm_target, scfm_target_x0, scfm_weights, gradient_mask, d_from, d_to, drift_target

    def generator_loss(
        self,
        image_or_video_shape,
        conditional_dict: dict,
        unconditional_dict: dict,
        clean_latent: torch.Tensor,
        initial_latent: torch.Tensor = None
    ) -> Tuple[torch.Tensor, dict]:

        lambda_scfm = getattr(self.args, "lambda_scfm", 0.0)
        scfm_disable_step = getattr(self.args, "scfm_disable_step", None)
        scfm_disabled = (
            scfm_disable_step is not None
            and self._current_step is not None
            and self._current_step >= scfm_disable_step
        )
        return_scfm_target = (lambda_scfm > 0.0) and not scfm_disabled

        pred_image, scfm_pred, scfm_target, scfm_target_x0, scfm_weights, gradient_mask, d_from, d_to, drift_target = self._run_generator(
            image_or_video_shape=image_or_video_shape,
            conditional_dict=conditional_dict,
            initial_latent=initial_latent,
            return_scfm_target=return_scfm_target
        )

        # 1. DMD Loss
        dmd_loss, dmd_log_dict, grad, pred_real_image = self.compute_distribution_matching_loss(
            image_or_video=pred_image,
            conditional_dict=conditional_dict,
            unconditional_dict=unconditional_dict,
            gradient_mask=gradient_mask,
            denoised_timestep_from=d_from,
            denoised_timestep_to=d_to
        )

        # Add DMD loss to log dict
        dmd_log_dict["dmd_loss"] = dmd_loss.detach()

        # 1.5 MMD guidance loss (optional)
        # mmd_use_ref=True: use drift ref_x0 as real (no teacher forward needed)
        # mmd_use_ref=False (default): use teacher multi-step Euler as real
        mmd_weight = getattr(self.args, "mmd_weight", 0.0)
        mmd_exit_timestep_min = getattr(self.args, "mmd_exit_timestep_min", 0)
        mmd_start_step = getattr(self.args, "mmd_start_step", 0)
        if mmd_start_step > 0 and self._current_step is not None and self._current_step < mmd_start_step:
            mmd_weight = 0.0
        if mmd_weight > 0.0 and (d_from is not None and d_from > mmd_exit_timestep_min):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                mmd_result = self._compute_mmd_guidance_loss_teacher_only(
                    pred_image=pred_image,
                    pred_real_image=pred_real_image,
                    conditional_dict=conditional_dict,
                    unconditional_dict=unconditional_dict,
                    gradient_mask=gradient_mask,
                    ref_image=drift_target,
                )
            if mmd_result is not None:
                mmd_loss, mmd_log_dict = mmd_result
                if mmd_loss is not None:
                    dmd_log_dict.update(mmd_log_dict)
                    dmd_log_dict["mmd_loss"] = mmd_loss.detach()
                    dmd_loss = dmd_loss + mmd_weight * mmd_loss

        # 2. Shortcut Consistency Loss
        sc_loss = 0.0
        if return_scfm_target and scfm_pred is not None and scfm_target is not None:
            # scfm_use_latent_target=True (default): scfm_pred = student latent at t_end, scfm_target = teacher latent at t_end
            # scfm_use_latent_target=False (legacy): scfm_pred = v1, scfm_target = average velocity
            mse = F.mse_loss(scfm_pred.float(), scfm_target.float(), reduction='none')

            if gradient_mask is not None:
                mse = mse * gradient_mask.float()

            # Average over dimensions except batch
            mse = mse.mean(dim=[1, 2, 3, 4])

            # Apply time-dependent weights
            if scfm_weights is not None:
                mse = mse * scfm_weights

            sc_loss = mse.mean()
            # dmd_log_dict["scfm_loss"] = sc_loss.detach()

        # 3. Compute training step-based weight for lambda_scfm
        # In early training, generator quality is poor, so we downweight scfm loss
        # Weight grows from 0 (step=0) to 1 (step=warmup_steps) using sigmoid or cosine schedule
        # scfm_warmup_steps = getattr(self.args, "scfm_warmup_steps", 0)
        # scfm_warmup_schedule = getattr(self.args, "scfm_warmup_schedule", "cosine")  # "cosine" or "sigmoid"

        # if self._current_step is not None and scfm_warmup_steps > 0:
        #     current_step = float(self._current_step)
        #     if scfm_warmup_schedule == "cosine":
        #         # Cosine schedule: smooth transition from 0 to 1
        #         # cos(π * (1 - progress)) maps [0, 1] -> [1, -1], then (1 - cos) / 2 -> [0, 1]
        #         progress = min(current_step / scfm_warmup_steps, 1.0)
        #         scfm_step_weight = 0.5 * (1.0 - math.cos(math.pi * progress))
        #     elif scfm_warmup_schedule == "sigmoid":
        #         # Sigmoid schedule: S-shaped curve
        #         # Center at warmup_steps/2, scale to make it transition smoothly
        #         # sigmoid((x - center) / scale) where center=warmup_steps/2, scale=warmup_steps/6
        #         center = scfm_warmup_steps / 2.0
        #         scale = scfm_warmup_steps / 6.0  # Controls the steepness
        #         scfm_step_weight = 1.0 / (1.0 + math.exp(-(current_step - center) / scale))
        #     else:
        #         # Linear schedule (fallback)
        #         scfm_step_weight = min(current_step / scfm_warmup_steps, 1.0)
        # else:
            # If no current_step info or warmup disabled, use full weight
        scfm_step_weight = 1.0

        # Apply step-based weight to lambda_scfm; force 0 after scfm_disable_step
        if scfm_disabled:
            effective_lambda_scfm = 0.0
        else:
            effective_lambda_scfm = lambda_scfm * scfm_step_weight
        dmd_log_dict["scfm_step_weight"] = scfm_step_weight
        dmd_log_dict["effective_lambda_scfm"] = effective_lambda_scfm

        # 4. Drift control loss: MSE between student pred_x0 and higher-step ref pred_x0
        drift_loss = 0.0
        lambda_drift = getattr(self.args, "lambda_drift", 0.0)
        if lambda_drift > 0.0 and drift_target is not None:
            drift_mse = F.mse_loss(pred_image.float(), drift_target.float(), reduction='none')
            if gradient_mask is not None:
                drift_mse = drift_mse * gradient_mask.float()
            drift_loss = drift_mse.mean()
            dmd_log_dict["drift_loss_raw"] = drift_loss.detach()

        total_loss = dmd_loss + effective_lambda_scfm * sc_loss + lambda_drift * drift_loss

        dmd_log_dict["scfm_loss"] = effective_lambda_scfm * sc_loss
        if isinstance(drift_loss, torch.Tensor):
            dmd_log_dict["drift_loss"] = (lambda_drift * drift_loss).detach()

        # Logging: Save videos periodically
        with torch.autocast("cuda", dtype=torch.bfloat16):
            self._save_log_videos(
                pred_image=pred_image,
                scfm_pred=scfm_pred,
                scfm_target=scfm_target,
                scfm_target_x0=scfm_target_x0 if return_scfm_target else None,
                grad=grad,
                clean_latent=clean_latent,
                d_from=d_from,
                d_to=d_to,
                drift_target=drift_target
            )

        return total_loss, dmd_log_dict

    def critic_loss(
        self,
        image_or_video_shape,
        conditional_dict: dict,
        unconditional_dict: dict,
        clean_latent: torch.Tensor,
        initial_latent: torch.Tensor = None
    ) -> Tuple[torch.Tensor, dict]:
        """
        Generate image/videos from noise and train the critic with generated samples.
        Overridden to handle the different return values of _run_generator in SC-DMD.

        Generator vs critic _run_generator:
        - Both use the same denoising_step_list (or, when scfm_mixed_step_lists, a random 8/4/2 list + exit_index).
        - Critic always passes return_scfm_target=False, so no SCFM target is computed (only generated samples + d_from/d_to).
        - Generator passes return_scfm_target=(lambda_scfm>0 and not scfm_disabled); when mixed, SCFM is only computed when 8-step list is chosen.
        """
        # Step 1: Run generator on backward simulated noisy input
        # Note: We disable SC target calculation here as we only need the generated samples for the critic
        with torch.no_grad():
            generated_image, _, _, _, _, _, denoised_timestep_from, denoised_timestep_to, _ = self._run_generator(
                image_or_video_shape=image_or_video_shape,
                conditional_dict=conditional_dict,
                initial_latent=initial_latent,
                return_scfm_target=False
            )

        # Step 2: Compute the fake prediction
        min_timestep = denoised_timestep_to if self.ts_schedule and denoised_timestep_to is not None else self.min_score_timestep
        max_timestep = denoised_timestep_from if self.ts_schedule_max and denoised_timestep_from is not None else self.num_train_timestep
        critic_timestep = self._get_timestep(
            min_timestep,
            max_timestep,
            image_or_video_shape[0],
            image_or_video_shape[1],
            self.num_frame_per_block,
            uniform_timestep=True
        )

        if self.timestep_shift > 1:
            critic_timestep = self.timestep_shift * \
                (critic_timestep / 1000) / (1 + (self.timestep_shift - 1) * (critic_timestep / 1000)) * 1000

        critic_timestep = critic_timestep.clamp(self.min_step, self.max_step)

        critic_noise = torch.randn_like(generated_image)
        noisy_generated_image = self.scheduler.add_noise(
            generated_image.flatten(0, 1),
            critic_noise.flatten(0, 1),
            critic_timestep.flatten(0, 1)
        ).unflatten(0, image_or_video_shape[:2])

        _, pred_fake_image = self.fake_score(
            noisy_image_or_video=noisy_generated_image,
            conditional_dict=conditional_dict,
            timestep=critic_timestep
        )

        # Step 3: Compute the denoising loss for the fake critic
        if self.args.denoising_loss_type == "flow":
            from utils.wan_wrapper import WanDiffusionWrapper
            flow_pred = WanDiffusionWrapper._convert_x0_to_flow_pred(
                scheduler=self.scheduler,
                x0_pred=pred_fake_image.flatten(0, 1),
                xt=noisy_generated_image.flatten(0, 1),
                timestep=critic_timestep.flatten(0, 1)
            )
            pred_fake_noise = None
        else:
            flow_pred = None
            pred_fake_noise = self.scheduler.convert_x0_to_noise(
                x0=pred_fake_image.flatten(0, 1),
                xt=noisy_generated_image.flatten(0, 1),
                timestep=critic_timestep.flatten(0, 1)
            ).unflatten(0, image_or_video_shape[:2])

        denoising_loss = self.denoising_loss_func(
            x=generated_image.flatten(0, 1),
            x_pred=pred_fake_image.flatten(0, 1),
            noise=critic_noise.flatten(0, 1),
            noise_pred=pred_fake_noise,
            alphas_cumprod=self.scheduler.alphas_cumprod,
            timestep=critic_timestep.flatten(0, 1),
            flow_pred=flow_pred
        )

        # Step 5: Debugging Log
        critic_log_dict = {
            "critic_timestep": critic_timestep.detach()
        }

        return denoising_loss, critic_log_dict
