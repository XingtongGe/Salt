from pipeline import SelfForcingTrainingPipeline
import torch.nn.functional as F
from typing import Optional, Tuple
import torch
import torch.distributed as dist
from einops import rearrange
import os
import imageio
import numpy as np

from model.base import SelfForcingModel


def _mmd2_pairwise(x, y, sigma=100, kernel='rbf', c=0.001, eps=1e-5):
    """
    Pairwise-kernel MMD^2 for small N (rbf/laplace/energy).
    x, y: [B, N, D]
    """
    xx = torch.bmm(x, x.transpose(1, 2))
    yy = torch.bmm(y, y.transpose(1, 2))
    xy = torch.bmm(x, y.transpose(1, 2))

    rx = torch.diagonal(xx, dim1=1, dim2=2).unsqueeze(1).expand_as(xx)
    ry = torch.diagonal(yy, dim1=1, dim2=2).unsqueeze(1).expand_as(yy)

    dxx = rx.transpose(1, 2) + rx - 2.0 * xx
    dyy = ry.transpose(1, 2) + ry - 2.0 * yy
    dxy = rx.transpose(1, 2) + ry - 2.0 * xy

    if kernel in ["rbf", "laplace"]:
        if kernel == "laplace":
            alpha = 1 / sigma
            dxx = dxx.sqrt().clamp(min=eps)
            dxy = dxy.sqrt().clamp(min=eps)
            dyy = dyy.sqrt().clamp(min=eps)
        elif kernel == 'rbf':
            alpha = 1 / (2 * sigma**2)

        k_xx = torch.exp(-alpha * dxx)
        k_xy = torch.exp(-alpha * dxy)
        k_yy = torch.exp(-alpha * dyy)

        n = x.shape[1]
        xx_sum = (k_xx.sum(dim=(1, 2)) - n) / (n * (n - 1))
        yy_sum = (k_yy.sum(dim=(1, 2)) - n) / (n * (n - 1))
        xy_sum = k_xy.sum(dim=(1, 2)) / (n * n)
        mmd2 = xx_sum + yy_sum - 2.0 * xy_sum

    elif kernel == 'energy':
        n = x.shape[1]
        diag_mask = torch.eye(n, device=x.device, dtype=torch.bool).unsqueeze(0)

        dxx_safe = dxx.clamp(min=eps * eps)
        dyy_safe = dyy.clamp(min=eps * eps)
        dxy_safe = dxy.clamp(min=eps * eps)

        k_xx = dxx_safe.sqrt()
        k_yy = dyy_safe.sqrt()
        k_xy = dxy_safe.sqrt()

        k_xx = k_xx.masked_fill(diag_mask, 0.0)
        k_yy = k_yy.masked_fill(diag_mask, 0.0)

        xx_sum = k_xx.sum(dim=(1, 2)) / (n * (n - 1))
        yy_sum = k_yy.sum(dim=(1, 2)) / (n * (n - 1))
        xy_sum = k_xy.mean(dim=(1, 2))
        mmd2 = 2.0 * xy_sum - xx_sum + yy_sum

    return mmd2


def mmd2_loss(
    x,
    y,
    sigma=100,
    kernel='linear',
    do_pdm_v2=False,
    c=0.001,
    eps=1e-5,
    num_frames=None,
):
    """
    Compute MMD (Maximum Mean Discrepancy) loss between two feature sets.

    Args:
        x: Feature tensor [B, N, D]
        y: Feature tensor [B, N, D]
        sigma: Kernel parameter for RBF/Laplace kernels
        kernel: Kernel type ('linear', 'rbf', 'laplace', 'energy',
                'temporal_rbf', 'temporal_laplace', 'temporal_energy')
                'temporal_*' variants first average spatial tokens per frame,
                then compute pairwise-kernel MMD on frame-level features (N=T).
        do_pdm_v2: Whether to use PDM v2 format
        c: Small constant for numerical stability
        eps: Epsilon for clamping
        num_frames: Number of temporal frames (required for temporal_* kernels)

    Returns:
        mmd2: MMD^2 value
    """
    assert x.ndim == 3

    if do_pdm_v2:
        x = x.flatten(0, 1).unsqueeze(0)
        y = y.flatten(0, 1).unsqueeze(0)

    x = x.float()
    y = y.float()

    if kernel.startswith('temporal_'):
        assert num_frames is not None, "num_frames required for temporal_* kernels"
        B, N, D = x.shape
        S = N // num_frames
        # [B, N, D] -> [B, T, S, D] -> spatial avg -> [B, T, D]
        x = x.view(B, num_frames, S, D).mean(dim=2)
        y = y.view(B, num_frames, S, D).mean(dim=2)
        base_kernel = kernel.replace('temporal_', '')
        return _mmd2_pairwise(x, y, sigma=sigma, kernel=base_kernel, c=c, eps=eps).mean()

    if kernel in ['rbf', 'energy', "laplace"]:
        return _mmd2_pairwise(x, y, sigma=sigma, kernel=kernel, c=c, eps=eps).mean()

    elif kernel == 'linear':
        dxy = (x.mean(dim=1) - y.mean(dim=1)) ** 2
        mmd2 = (dxy + c ** 2).sqrt().clamp(min=eps) - c

    else:
        raise ValueError(f"Unsupported PDM kernel: {kernel}")

    return mmd2.mean()


def spatial_only_trd_loss(
    real_feature: torch.Tensor,
    fake_feature: torch.Tensor,
    num_frames: int,
    margin: float = 0.1,
    first_chunk_only: bool = False,
    num_frame_per_block: int = 1,
) -> torch.Tensor:
    """
    Spatial-only Token Relation Distillation: align per-frame token relation
    (Gram / similarity matrix) of fake to real. Optionally only on first chunk.

    Args:
        real_feature: [B, N, D] ref (multi-step) features, N = num_frames * S
        fake_feature: [B, N, D] student (few-step) features
        num_frames: F
        margin: ReLU(|sim - target_sim| - margin).mean()
        first_chunk_only: if True, only use frames 0..num_frame_per_block-1
        num_frame_per_block: used when first_chunk_only is True

    Returns:
        scalar loss
    """
    B, N, D = real_feature.shape
    S = N // num_frames
    assert num_frames * S == N
    real_feature = real_feature.float().view(B, num_frames, S, D)
    fake_feature = fake_feature.float().view(B, num_frames, S, D)

    if first_chunk_only:
        n_use = min(num_frame_per_block, num_frames)
        real_feature = real_feature[:, :n_use]
        fake_feature = fake_feature[:, :n_use]
        num_frames_use = n_use
    else:
        num_frames_use = num_frames

    real_feature = torch.nn.functional.normalize(real_feature, dim=-1)
    fake_feature = torch.nn.functional.normalize(fake_feature, dim=-1)
    # [B, F, S, D] -> per frame: [B, S, D] @ [B, D, S] -> [B, S, S]
    losses = []
    for f in range(num_frames_use):
        r = real_feature[:, f]
        g = fake_feature[:, f]
        real_sim = torch.bmm(r, r.transpose(1, 2))
        fake_sim = torch.bmm(g, g.transpose(1, 2))
        diff = (fake_sim - real_sim).abs() - margin
        losses.append(torch.nn.functional.relu(diff).mean())
    return torch.stack(losses).mean()


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


class DMD(SelfForcingModel):
    def __init__(self, args, device):
        """
        Initialize the DMD (Distribution Matching Distillation) module.
        This class is self-contained and compute generator and fake score losses
        in the forward pass.
        """
        super().__init__(args, device)
        self.num_frame_per_block = getattr(args, "num_frame_per_block", 1)
        self.same_step_across_blocks = getattr(args, "same_step_across_blocks", True)
        self.num_training_frames = getattr(args, "num_training_frames", 21)
        self._log_step_counter = 0  # Internal counter for logging
        self._logdir = None
        self._current_step = None

        if self.num_frame_per_block > 1:
            self.generator.model.num_frame_per_block = self.num_frame_per_block

        self.independent_first_frame = getattr(args, "independent_first_frame", False)
        if self.independent_first_frame:
            self.generator.model.independent_first_frame = True
        if args.gradient_checkpointing:
            self.generator.enable_gradient_checkpointing()
            self.fake_score.enable_gradient_checkpointing()
            self.real_score.enable_gradient_checkpointing()

        # this will be init later with fsdp-wrapped modules
        self.inference_pipeline: SelfForcingTrainingPipeline = None

        # Step 2: Initialize all dmd hyperparameters
        self.num_train_timestep = args.num_train_timestep
        self.min_step = int(0.02 * self.num_train_timestep)
        self.max_step = int(0.98 * self.num_train_timestep)
        if hasattr(args, "real_guidance_scale"):
            self.real_guidance_scale = args.real_guidance_scale
            self.fake_guidance_scale = args.fake_guidance_scale
        else:
            self.real_guidance_scale = args.guidance_scale
            self.fake_guidance_scale = 0.0
        self.timestep_shift = getattr(args, "timestep_shift", 1.0)
        self.ts_schedule = getattr(args, "ts_schedule", True)
        print('ts_schedule', self.ts_schedule)
        self.ts_schedule_max = getattr(args, "ts_schedule_max", False)
        self.min_score_timestep = getattr(args, "min_score_timestep", 0)

        if getattr(self.scheduler, "alphas_cumprod", None) is not None:
            self.scheduler.alphas_cumprod = self.scheduler.alphas_cumprod.to(device)
        else:
            self.scheduler.alphas_cumprod = None

    def set_logging_params(self, step: int = None, logdir: str = None):
        """Set logging parameters (step and logdir) for video saving."""
        if step is not None:
            self._current_step = step
        if logdir is not None:
            self._logdir = logdir

    def _save_log_videos(
        self,
        pred_image: torch.Tensor,
        grad: Optional[torch.Tensor] = None,
        clean_latent: Optional[torch.Tensor] = None,
        d_from: Optional[int] = None,
        d_to: Optional[int] = None,
        pred_real_image: Optional[torch.Tensor] = None,
        pred_real_image_for_mmd: Optional[torch.Tensor] = None
    ):
        """
        Save videos for logging purposes.
        Args:
            pred_image: Generated image/video latent [B, T, C, H, W]
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

        # Check if we have required items (pred_image is required, grad is optional)
        if grad is None:
            # If grad is not available, only save pred_image
            self._log_step_counter += 1
            return

        # try:
        batch_id = 0  # Save first sample in batch

        # Prepare data dictionary with pred_image and grad
        _list = {
            'pred_image': pred_image.detach(),
            'grad': grad.detach()
        }
        if clean_latent is not None:
            _list['clean_latent'] = clean_latent.detach()
        if pred_real_image is not None:
            _list['pred_real_image'] = pred_real_image.detach()
        if pred_real_image_for_mmd is not None:
            _list['pred_real_image_for_mmd'] = pred_real_image_for_mmd.detach()

        # Decode latents to pixel space
        res = []
        for k, item in _list.items():
            with torch.no_grad():
                samples_z = item.contiguous()[batch_id:batch_id+1]
                # VAE decode: latent [B, T, C, H, W] -> pixel [B, T, C, H, W]
                # VAE can accept bfloat16, decode_to_pixel returns [B, T, C, H, W] format
                samples_x = self.vae.decode_to_pixel(samples_z)
                # Normalize to [0, 1]
                temp = torch.clamp((samples_x + 1.0) / 2.0, min=0.0, max=1.0).cpu()
            res.append(temp)

        # Create grid layout based on number of videos
        if len(res) == 2:
            # Stack horizontally: pred_image and grad
            video_grid = torch.cat(res, dim=-1)  # [B, T, C, H, 2*W]
        elif len(res) == 3:
            # Create 2x2 grid: pred_image, grad, clean_latent, (empty or duplicate)
            row1 = torch.cat((res[0], res[1]), dim=-1)  # [B, T, C, H, 2*W]
            row2 = torch.cat((res[2], res[2]), dim=-1)  # Duplicate clean_latent
            video_grid = torch.cat((row1, row2), dim=-2)  # [B, T, C, 2*H, 2*W]
        elif len(res) == 4:
            # Create 2x2 grid: pred_image, grad, pred_real_image, pred_real_image_for_mmd
            row1 = torch.cat((res[0], res[1]), dim=-1)  # [B, T, C, H, 2*W]
            row2 = torch.cat((res[2], res[3]), dim=-1)  # [B, T, C, H, 2*W]
            video_grid = torch.cat((row1, row2), dim=-2)  # [B, T, C, 2*H, 2*W]
        elif len(res) == 5:
            # Create 2x3 grid (pad last column): pred_image, grad, pred_real_image, pred_real_image_for_mmd, clean_latent
            row1 = torch.cat((res[0], res[1], res[2]), dim=-1)  # [B, T, C, H, 3*W]
            row2 = torch.cat((res[3], res[4], res[4]), dim=-1)  # [B, T, C, H, 3*W] (duplicate clean_latent)
            video_grid = torch.cat((row1, row2), dim=-2)  # [B, T, C, 2*H, 3*W]
        else:
            video_grid = res[0]

        # Prepare save path
        timestep_str = f'T{d_from:.1f}-{d_to:.1f}' if d_from is not None and d_to is not None else ''
        save_dir = os.path.join(logdir, f"{current_step:06d}{'-' + timestep_str if timestep_str else ''}")

        # Get fps from config or use default
        fps = getattr(self.args, "sampling_fps", 16)

        # Save video
        print('video_grid', video_grid.shape)

        save_video_as_grid_and_mp4(
            video_grid,
            save_dir,
            fps=fps,
            args=self.args,
            key="dmd"
        )

        # except Exception as e:
        #     # Don't crash training if logging fails
        #     if dist.get_rank() == 0 if dist.is_initialized() else True:
        #         print(f"Warning: Failed to save log videos: {e}")

        self._log_step_counter += 1



    def _compute_kl_grad(
        self, noisy_image_or_video: torch.Tensor,
        estimated_clean_image_or_video: torch.Tensor,
        timestep: torch.Tensor,
        conditional_dict: dict, unconditional_dict: dict,
        normalization: bool = True
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute the KL grad (eq 7 in https://arxiv.org/abs/2311.18828).
        Input:
            - noisy_image_or_video: a tensor with shape [B, F, C, H, W] where the number of frame is 1 for images.
            - estimated_clean_image_or_video: a tensor with shape [B, F, C, H, W] representing the estimated clean image or video.
            - timestep: a tensor with shape [B, F] containing the randomly generated timestep.
            - conditional_dict: a dictionary containing the conditional information (e.g. text embeddings, image embeddings).
            - unconditional_dict: a dictionary containing the unconditional information (e.g. null/negative text embeddings, null/negative image embeddings).
            - normalization: a boolean indicating whether to normalize the gradient.
        Output:
            - kl_grad: a tensor representing the KL grad.
            - kl_log_dict: a dictionary containing the intermediate tensors for logging.
        """
        # Step 1: Compute the fake score
        _, pred_fake_image_cond = torch.utils.checkpoint.checkpoint(
            self.fake_score,
            noisy_image_or_video,
            conditional_dict,
            timestep,
            use_reentrant=False
        )

        if self.fake_guidance_scale != 0.0:
            _, pred_fake_image_uncond = torch.utils.checkpoint.checkpoint(
                self.fake_score,
                noisy_image_or_video,
                unconditional_dict,
                timestep,
                use_reentrant=False
            )
            pred_fake_image = pred_fake_image_cond + (
                pred_fake_image_cond - pred_fake_image_uncond
            ) * self.fake_guidance_scale
        else:
            pred_fake_image = pred_fake_image_cond

        # Step 2: Compute the real score
        # We compute the conditional and unconditional prediction
        # and add them together to achieve cfg (https://arxiv.org/abs/2207.12598)
        _, pred_real_image_cond = torch.utils.checkpoint.checkpoint(
            self.real_score,
            noisy_image_or_video,
            conditional_dict,
            timestep,
            use_reentrant=False
        )

        _, pred_real_image_uncond = torch.utils.checkpoint.checkpoint(
            self.real_score,
            noisy_image_or_video,
            unconditional_dict,
            timestep,
            use_reentrant=False
        )

        pred_real_image = pred_real_image_cond + (
            pred_real_image_cond - pred_real_image_uncond
        ) * self.real_guidance_scale

        # Step 3: Compute the DMD gradient (DMD paper eq. 7).
        grad = (pred_fake_image - pred_real_image)

        # TODO: Change the normalizer for causal teacher
        if normalization:
            # Step 4: Gradient normalization (DMD paper eq. 8).
            p_real = (estimated_clean_image_or_video - pred_real_image)
            normalizer = torch.abs(p_real).mean(dim=[1, 2, 3, 4], keepdim=True)
            grad = grad / normalizer
        grad = torch.nan_to_num(grad)

        return grad, {
            "dmdtrain_gradient_norm": torch.mean(torch.abs(grad)).detach(),
            "timestep": timestep.detach()
        }, pred_real_image

    def _compute_mmd_guidance_loss_teacher_only_backup(
        self,
        pred_image: torch.Tensor,
        pred_real_image: torch.Tensor,
        conditional_dict: dict,
        unconditional_dict: dict,
        gradient_mask: Optional[torch.Tensor] = None
    ) -> Tuple[Optional[torch.Tensor], dict]:
        """
        [BACKUP] Original MMD: real sample = teacher single-step CFG prediction on noisy_sample.
        Kept for reference / fallback.
        """
        try:
            batch_size, num_frames = pred_image.shape[:2]
            device = pred_image.device
            dtype = pred_image.dtype
            mmd_guidance_timestep_min = getattr(self.args, "mmd_guidance_timestep_min", 100)
            mmd_guidance_timestep_max = getattr(self.args, "mmd_guidance_timestep_max", 600)
            mmd_timestep = self._get_timestep(
                mmd_guidance_timestep_min,
                mmd_guidance_timestep_max,
                batch_size,
                num_frames,
                self.num_frame_per_block,
                uniform_timestep=True
            )
            if self.timestep_shift > 1:
                mmd_timestep = self.timestep_shift * \
                    (mmd_timestep / 1000.0) / \
                    (1 + (self.timestep_shift - 1) * (mmd_timestep / 1000.0)) * 1000.0
            mmd_timestep = mmd_timestep.clamp(self.min_step, self.max_step).long()
            noise_for_guidance = torch.randn_like(pred_image)
            noisy_sample = self.scheduler.add_noise(
                pred_image.flatten(0, 1),
                noise_for_guidance.flatten(0, 1),
                mmd_timestep.flatten(0, 1)
            ).unflatten(0, (batch_size, num_frames))
            cfg_scale = torch.ones(batch_size, device=device) * 3.5
            with torch.no_grad():
                real_output_cond = torch.utils.checkpoint.checkpoint(
                    self.real_score, noisy_sample, conditional_dict, mmd_timestep, use_reentrant=False
                )
                if len(real_output_cond) >= 2:
                    _, pred_real_cond = real_output_cond[:2]
                    del real_output_cond
                else:
                    del noisy_sample
                    return None
                real_output_uncond = torch.utils.checkpoint.checkpoint(
                    self.real_score, noisy_sample, unconditional_dict, mmd_timestep, use_reentrant=False
                )
                if len(real_output_uncond) >= 2:
                    _, pred_real_uncond = real_output_uncond[:2]
                    del real_output_uncond, noisy_sample
                else:
                    del pred_real_cond, noisy_sample
                    return None
                cfg_scale_expanded = cfg_scale.view(batch_size, 1, 1, 1, 1)
                pred_real_image_for_mmd = pred_real_uncond + cfg_scale_expanded * (pred_real_cond - pred_real_uncond)
                del pred_real_cond, pred_real_uncond, noise_for_guidance
                torch.cuda.empty_cache()
            mmd_feature_timestep_min = getattr(self.args, "mmd_feature_timestep_min", 100)
            mmd_feature_timestep_max = getattr(self.args, "mmd_feature_timestep_max", 300)
            mmd_feature_timestep = self._get_timestep(
                mmd_feature_timestep_min, mmd_feature_timestep_max,
                batch_size, num_frames, self.num_frame_per_block, uniform_timestep=True
            )
            if self.timestep_shift > 1:
                mmd_feature_timestep = self.timestep_shift * \
                    (mmd_feature_timestep / 1000.0) / \
                    (1 + (self.timestep_shift - 1) * (mmd_feature_timestep / 1000.0)) * 1000.0
            mmd_feature_timestep = mmd_feature_timestep.clamp(self.min_step, self.max_step).long()
            shared_noise = torch.randn_like(pred_image).detach()
            mmd_feature_block = getattr(self.args, "mmd_feature_block", 20)
            with torch.no_grad():
                noisy_real_for_mmd = self.scheduler.add_noise(
                    pred_real_image_for_mmd.flatten(0, 1), shared_noise.flatten(0, 1),
                    mmd_feature_timestep.flatten(0, 1)
                ).unflatten(0, (batch_size, num_frames))
                real_output = torch.utils.checkpoint.checkpoint(
                    self.real_score, noisy_real_for_mmd, conditional_dict, mmd_feature_timestep,
                    mmd_features=mmd_feature_block, use_reentrant=False
                )
                if len(real_output) == 3:
                    _, _, real_feature_mmd = real_output
                    del real_output, noisy_real_for_mmd
                else:
                    del noisy_real_for_mmd
                    return None
            noisy_fake_for_mmd = self.scheduler.add_noise(
                pred_image.flatten(0, 1), shared_noise.flatten(0, 1),
                mmd_feature_timestep.flatten(0, 1)
            ).unflatten(0, (batch_size, num_frames))
            fake_output = torch.utils.checkpoint.checkpoint(
                self.real_score, noisy_fake_for_mmd, conditional_dict, mmd_feature_timestep,
                mmd_features=mmd_feature_block, use_reentrant=False
            )
            if len(fake_output) == 3:
                _, _, fake_feature_mmd = fake_output
            else:
                del noisy_fake_for_mmd
                return None
            mmd_kernel = getattr(self.args, "mmd_kernel", "linear")
            mmd_sigma = getattr(self.args, "mmd_sigma", 100)
            mmd_loss = mmd2_loss(real_feature_mmd.detach(), fake_feature_mmd, sigma=mmd_sigma, kernel=mmd_kernel, num_frames=num_frames)
            if gradient_mask is not None:
                mmd_loss = mmd_loss * gradient_mask.float().mean()
            log_dict = {
                "mmd_loss": mmd_loss.detach(),
                "mmd_feature_timestep": mmd_feature_timestep.float().mean().detach(),
                "pred_real_image_for_mmd": pred_real_image_for_mmd.detach(),
            }
            return mmd_loss, log_dict
        except Exception as e:
            if dist.get_rank() == 0 if dist.is_initialized() else True:
                print(f"Warning: MMD guidance loss (teacher_only) failed: {e}")
            return None

    def _compute_mmd_guidance_loss_teacher_euler(
        self,
        pred_image: torch.Tensor,
        pred_real_image: torch.Tensor,
        conditional_dict: dict,
        unconditional_dict: dict,
        gradient_mask: Optional[torch.Tensor] = None
    ) -> Tuple[Optional[torch.Tensor], dict]:
        """
        [BACKUP2] MMD with teacher multi-step Euler: add noise at mmd_timestep (e.g. 300), then teacher
        takes 3 Euler steps (300 -> 200 -> 100 -> 0) to get pred_real_image_for_mmd. All in no_grad.
        Step timesteps are in unshifted space; shift is applied for model/scheduler.
        """
        try:
            batch_size, num_frames = pred_image.shape[:2]
            device = pred_image.device
            dtype = pred_image.dtype
            # Start timestep: fixed 300 (config mmd_guidance_timestep_min/max = 300)
            mmd_guidance_timestep_min = getattr(self.args, "mmd_guidance_timestep_min", 300)
            mmd_guidance_timestep_max = getattr(self.args, "mmd_guidance_timestep_max", 301)
            mmd_timestep = self._get_timestep(
                mmd_guidance_timestep_min,
                mmd_guidance_timestep_max,
                batch_size,
                num_frames,
                self.num_frame_per_block,
                uniform_timestep=True
            )
            if self.timestep_shift > 1:
                mmd_timestep = self.timestep_shift * (mmd_timestep / 1000.0) / (
                    1 + (self.timestep_shift - 1) * (mmd_timestep / 1000.0)) * 1000.0
            mmd_timestep = mmd_timestep.clamp(self.min_step, self.max_step).long()
            noise_for_guidance = torch.randn_like(pred_image)
            noisy_sample = self.scheduler.add_noise(
                pred_image.flatten(0, 1),
                noise_for_guidance.flatten(0, 1),
                mmd_timestep.flatten(0, 1)
            ).unflatten(0, (batch_size, num_frames))
            # Teacher step list (unshifted): 300, 200, 100 -> then step to 0
            step_list_unshifted = getattr(
                self.args, "mmd_teacher_step_timesteps", [300, 200, 100]
            )
            if not isinstance(step_list_unshifted, (list, tuple)):
                step_list_unshifted = [300, 200, 100]
            step_list_unshifted = list(step_list_unshifted)[:3]
            # Build shifted timesteps for model and sigma lookup
            def shift_t(t):
                t = float(t)
                if self.timestep_shift <= 1:
                    return t
                return self.timestep_shift * (t / 1000.0) / (
                    1 + (self.timestep_shift - 1) * (t / 1000.0)) * 1000.0
            step_t_shifted = [shift_t(t) for t in step_list_unshifted]
            sigmas = self.scheduler.sigmas.to(device)
            timesteps = self.scheduler.timesteps.to(device)
            def sigma_at_t(t_val):
                tid = torch.argmin(
                    (timesteps - torch.tensor(t_val, device=device, dtype=timesteps.dtype)).abs()
                ).item()
                return sigmas[tid].to(dtype).reshape(1, 1, 1, 1)
            cfg_scale = getattr(self.args, "mmd_cfg_scale", 3.5)
            cfg_scale = torch.ones(batch_size, device=device) * cfg_scale
            cfg_scale_expanded = cfg_scale.view(batch_size, 1, 1, 1, 1)
            x = noisy_sample
            with torch.no_grad():
                for i in range(len(step_t_shifted)):
                    t_cur = step_t_shifted[i]
                    t_next = step_t_shifted[i + 1] if i + 1 < len(step_t_shifted) else 0.0
                    timestep_cur = torch.full(
                        (batch_size, num_frames), int(round(t_cur)), device=device, dtype=torch.long
                    )
                    out_cond = self.real_score(x, conditional_dict, timestep_cur)
                    if len(out_cond) < 2:
                        return None
                    flow_cond, _ = out_cond[:2]
                    out_uncond = self.real_score(x, unconditional_dict, timestep_cur)
                    if len(out_uncond) < 2:
                        return None
                    flow_uncond, _ = out_uncond[:2]
                    flow = flow_uncond + cfg_scale_expanded * (flow_cond - flow_uncond)
                    sigma_cur = sigma_at_t(t_cur)
                    sigma_next = sigma_at_t(t_next)
                    # Euler: x_next = x + flow * (sigma_next - sigma_cur)
                    x = x + flow * (sigma_next - sigma_cur)
                pred_real_image_for_mmd = x
                del x, flow, flow_cond, flow_uncond, noise_for_guidance
                torch.cuda.empty_cache()
            mmd_feature_timestep_min = getattr(self.args, "mmd_feature_timestep_min", 100)
            mmd_feature_timestep_max = getattr(self.args, "mmd_feature_timestep_max", 300)
            mmd_feature_timestep = self._get_timestep(
                mmd_feature_timestep_min, mmd_feature_timestep_max,
                batch_size, num_frames, self.num_frame_per_block, uniform_timestep=True
            )
            if self.timestep_shift > 1:
                mmd_feature_timestep = self.timestep_shift * (mmd_feature_timestep / 1000.0) / (
                    1 + (self.timestep_shift - 1) * (mmd_feature_timestep / 1000.0)) * 1000.0
            mmd_feature_timestep = mmd_feature_timestep.clamp(self.min_step, self.max_step).long()
            shared_noise = torch.randn_like(pred_image).detach()
            mmd_feature_block = getattr(self.args, "mmd_feature_block", 20)
            with torch.no_grad():
                noisy_real_for_mmd = self.scheduler.add_noise(
                    pred_real_image_for_mmd.flatten(0, 1), shared_noise.flatten(0, 1),
                    mmd_feature_timestep.flatten(0, 1)
                ).unflatten(0, (batch_size, num_frames))
                real_output = self.real_score(
                    noisy_real_for_mmd, conditional_dict, mmd_feature_timestep,
                    mmd_features=mmd_feature_block
                )
                if len(real_output) != 3:
                    return None
                _, _, real_feature_mmd = real_output
                del real_output, noisy_real_for_mmd
            noisy_fake_for_mmd = self.scheduler.add_noise(
                pred_image.flatten(0, 1), shared_noise.flatten(0, 1),
                mmd_feature_timestep.flatten(0, 1)
            ).unflatten(0, (batch_size, num_frames))
            fake_output = self.real_score(
                noisy_fake_for_mmd, conditional_dict, mmd_feature_timestep,
                mmd_features=mmd_feature_block
            )
            if len(fake_output) != 3:
                del noisy_fake_for_mmd
                return None
            _, _, fake_feature_mmd = fake_output
            mmd_kernel = getattr(self.args, "mmd_kernel", "linear")
            mmd_sigma = getattr(self.args, "mmd_sigma", 100)

            is_rank0 = (not dist.is_initialized()) or dist.get_rank() == 0
            if is_rank0 and not getattr(self, '_mmd_dist_logged', False):
                with torch.no_grad():
                    _n_sample = min(1024, real_feature_mmd.shape[1])
                    _rx = real_feature_mmd[:1, :_n_sample].float()
                    _fx = fake_feature_mmd[:1, :_n_sample].float()
                    _rr = torch.cdist(_rx, _rx).view(-1)
                    _ff = torch.cdist(_fx, _fx).view(-1)
                    _rf = torch.cdist(_rx, _fx).view(-1)
                    print(f"[MMD dist] feature shape: {list(real_feature_mmd.shape)}, "
                          f"real-real median={_rr.median():.1f} mean={_rr.mean():.1f}, "
                          f"fake-fake median={_ff.median():.1f} mean={_ff.mean():.1f}, "
                          f"real-fake median={_rf.median():.1f} mean={_rf.mean():.1f}")
                    B, N, D = real_feature_mmd.shape
                    S = N // num_frames
                    _rx_t = real_feature_mmd[:1].view(1, num_frames, S, D).mean(dim=2).float()
                    _fx_t = fake_feature_mmd[:1].view(1, num_frames, S, D).mean(dim=2).float()
                    _rr_t = torch.cdist(_rx_t, _rx_t).view(-1)
                    _ff_t = torch.cdist(_fx_t, _fx_t).view(-1)
                    _rf_t = torch.cdist(_rx_t, _fx_t).view(-1)
                    print(f"[MMD dist temporal] frame features [{num_frames}, {D}], "
                          f"real-real median={_rr_t.median():.1f} mean={_rr_t.mean():.1f}, "
                          f"fake-fake median={_ff_t.median():.1f} mean={_ff_t.mean():.1f}, "
                          f"real-fake median={_rf_t.median():.1f} mean={_rf_t.mean():.1f}")
                self._mmd_dist_logged = True

            mmd_loss = mmd2_loss(real_feature_mmd.detach(), fake_feature_mmd, sigma=mmd_sigma, kernel=mmd_kernel, num_frames=num_frames)
            if gradient_mask is not None:
                mmd_loss = mmd_loss * gradient_mask.float().mean()
            log_dict = {
                "mmd_loss": mmd_loss.detach(),
                "mmd_feature_timestep": mmd_feature_timestep.float().mean().detach(),
                "pred_real_image_for_mmd": pred_real_image_for_mmd.detach(),
            }
            return mmd_loss, log_dict
        except Exception as e:
            if dist.get_rank() == 0 if dist.is_initialized() else True:
                print(f"Warning: MMD guidance loss (teacher_only) failed: {e}")
            return None

    def _compute_mmd_guidance_loss_teacher_only(
        self,
        pred_image: torch.Tensor,
        pred_real_image: torch.Tensor,
        conditional_dict: dict,
        unconditional_dict: dict,
        gradient_mask: Optional[torch.Tensor] = None,
        ref_image: torch.Tensor = None,
    ) -> Tuple[Optional[torch.Tensor], dict]:
        """
        MMD guidance loss with configurable "real" source.
        - If ref_image is provided and mmd_use_ref=True: use ref_x0 as real (no teacher needed)
        - Otherwise: fall back to teacher multi-step Euler (backup2)
        """
        mmd_use_ref = getattr(self.args, "mmd_use_ref", False)
        if mmd_use_ref:
            if ref_image is None:
                return None
            return self._compute_mmd_guidance_loss_ref(
                pred_image=pred_image,
                ref_image=ref_image,
                conditional_dict=conditional_dict,
                gradient_mask=gradient_mask,
            )
        return self._compute_mmd_guidance_loss_teacher_euler(
            pred_image=pred_image,
            pred_real_image=pred_real_image,
            conditional_dict=conditional_dict,
            unconditional_dict=unconditional_dict,
            gradient_mask=gradient_mask,
        )

    def _compute_mmd_guidance_loss_ref(
        self,
        pred_image: torch.Tensor,
        ref_image: torch.Tensor,
        conditional_dict: dict,
        gradient_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[Optional[torch.Tensor], dict]:
        """
        MMD loss using ref_x0 (from drift ref path) as "real" and pred_x0 as "fake".
        Both are noised at a shared timestep, features extracted via real_score, then kernel MMD.
        No teacher forward passes needed.
        """
        try:
            batch_size, num_frames = pred_image.shape[:2]
            device = pred_image.device

            mmd_feature_timestep_min = getattr(self.args, "mmd_feature_timestep_min", 100)
            mmd_feature_timestep_max = getattr(self.args, "mmd_feature_timestep_max", 300)
            mmd_feature_timestep = self._get_timestep(
                mmd_feature_timestep_min, mmd_feature_timestep_max,
                batch_size, num_frames, self.num_frame_per_block, uniform_timestep=True
            )
            if self.timestep_shift > 1:
                mmd_feature_timestep = self.timestep_shift * (mmd_feature_timestep / 1000.0) / (
                    1 + (self.timestep_shift - 1) * (mmd_feature_timestep / 1000.0)) * 1000.0
            mmd_feature_timestep = mmd_feature_timestep.clamp(self.min_step, self.max_step).long()

            shared_noise = torch.randn_like(pred_image).detach()
            mmd_feature_block = getattr(self.args, "mmd_feature_block", 20)

            with torch.no_grad():
                noisy_real = self.scheduler.add_noise(
                    ref_image.flatten(0, 1), shared_noise.flatten(0, 1),
                    mmd_feature_timestep.flatten(0, 1)
                ).unflatten(0, (batch_size, num_frames))
                real_output = self.real_score(
                    noisy_real, conditional_dict, mmd_feature_timestep,
                    mmd_features=mmd_feature_block
                )
                if len(real_output) != 3:
                    return None
                _, _, real_feature_mmd = real_output
                del real_output, noisy_real

            noisy_fake = self.scheduler.add_noise(
                pred_image.flatten(0, 1), shared_noise.flatten(0, 1),
                mmd_feature_timestep.flatten(0, 1)
            ).unflatten(0, (batch_size, num_frames))
            fake_output = self.real_score(
                noisy_fake, conditional_dict, mmd_feature_timestep,
                mmd_features=mmd_feature_block
            )
            if len(fake_output) != 3:
                del noisy_fake
                return None
            _, _, fake_feature_mmd = fake_output

            ref_align_loss = getattr(self.args, "ref_align_loss", "mmd")
            if ref_align_loss == "trd":
                trd_margin = getattr(self.args, "trd_margin", 0.1)
                trd_first_chunk_only = getattr(self.args, "trd_first_chunk_only", False)
                align_loss = spatial_only_trd_loss(
                    real_feature_mmd.detach(),
                    fake_feature_mmd,
                    num_frames=num_frames,
                    margin=trd_margin,
                    first_chunk_only=trd_first_chunk_only,
                    num_frame_per_block=self.num_frame_per_block,
                )
                if gradient_mask is not None:
                    align_loss = align_loss * gradient_mask.float().mean()
                log_dict = {
                    "mmd_loss": align_loss.detach(),
                    "mmd_feature_timestep": mmd_feature_timestep.float().mean().detach(),
                }
                return align_loss, log_dict

            mmd_kernel = getattr(self.args, "mmd_kernel", "linear")
            mmd_sigma = getattr(self.args, "mmd_sigma", 100)

            is_rank0 = (not dist.is_initialized()) or dist.get_rank() == 0
            if is_rank0 and not getattr(self, '_mmd_ref_dist_logged', False):
                with torch.no_grad():
                    _n_sample = min(1024, real_feature_mmd.shape[1])
                    _rx = real_feature_mmd[:1, :_n_sample].float()
                    _fx = fake_feature_mmd[:1, :_n_sample].float()
                    _rf = torch.cdist(_rx, _fx).view(-1)
                    print(f"[MMD-ref dist] feature shape: {list(real_feature_mmd.shape)}, "
                          f"ref-gen median={_rf.median():.1f} mean={_rf.mean():.1f}")
                self._mmd_ref_dist_logged = True

            mmd_loss = mmd2_loss(real_feature_mmd.detach(), fake_feature_mmd, sigma=mmd_sigma, kernel=mmd_kernel, num_frames=num_frames)
            if gradient_mask is not None:
                mmd_loss = mmd_loss * gradient_mask.float().mean()
            log_dict = {
                "mmd_loss": mmd_loss.detach(),
                "mmd_feature_timestep": mmd_feature_timestep.float().mean().detach(),
            }
            return mmd_loss, log_dict
        except Exception as e:
            if dist.get_rank() == 0 if dist.is_initialized() else True:
                print(f"Warning: MMD guidance loss (ref) failed: {e}")
            return None

    def _compute_mmd_guidance_loss(
        self,
        pred_image: torch.Tensor,
        pred_real_image: torch.Tensor,
        conditional_dict: dict,
        unconditional_dict: dict,
        gradient_mask: Optional[torch.Tensor] = None
    ) -> Tuple[Optional[torch.Tensor], dict]:
        """
        MMD with teacher-as-start: teacher one-step to anchor, then generator chunk-by-chunk to x0.
        Real reference = generator(x_mid -> x0), so details come from student, prior from teacher.
        """
        # try:
        batch_size, num_frames = pred_image.shape[:2]
        device = pred_image.device
        dtype = pred_image.dtype

        # 1) Sample mmd_timestep in [300, 600] (configurable)
        mmd_guidance_timestep_min = getattr(self.args, "mmd_guidance_timestep_min", 300)
        mmd_guidance_timestep_max = getattr(self.args, "mmd_guidance_timestep_max", 600)
        mmd_timestep = self._get_timestep(
            mmd_guidance_timestep_min,
            mmd_guidance_timestep_max,
            batch_size,
            num_frames,
            self.num_frame_per_block,
            uniform_timestep=True
        )
        if self.timestep_shift > 1:
            mmd_timestep = self.timestep_shift * (mmd_timestep / 1000.0) / (
                1 + (self.timestep_shift - 1) * (mmd_timestep / 1000.0)) * 1000.0
        mmd_timestep = mmd_timestep.clamp(self.min_step, self.max_step).long()

        noise_for_guidance = torch.randn_like(pred_image)
        noisy_sample = self.scheduler.add_noise(
            pred_image.flatten(0, 1),
            noise_for_guidance.flatten(0, 1),
            mmd_timestep.flatten(0, 1)
        ).unflatten(0, (batch_size, num_frames))

        # 2) Anchor timestep (shifted) for flow integration
        mmd_t_val = mmd_timestep[0, 0].item() if mmd_timestep.dim() > 0 else int(mmd_timestep.item())
        # anchor_raw = 500 if mmd_t_val > 520 else 250
        anchor_raw = 250
        anchor_index = 2 if anchor_raw == 500 else 3  # [1000, 750, 500, 250]
        anchor_shifted = float(anchor_raw)
        if self.timestep_shift > 1:
            anchor_shifted = self.timestep_shift * (anchor_shifted / 1000.0) / (
                1 + (self.timestep_shift - 1) * (anchor_shifted / 1000.0)) * 1000.0
        # min_s = int(self.min_step) if isinstance(self.min_step, int) else self.min_step.item()
        # max_s = int(self.max_step) if isinstance(self.max_step, int) else self.max_step.item()
        # anchor_shifted = max(min_s, min(max_s, int(round(anchor_shifted))))
        print('anchor_shifted: ', anchor_shifted)
        anchor_t = torch.full(
            (batch_size * num_frames,), anchor_shifted, device=device, dtype=torch.long
        )

        # 3) Teacher flow at (noisy_sample, mmd_t); then flow formula: x_mid = x_t + (sigma_anchor - sigma_t) * flow
        cfg_scale = torch.ones(batch_size, device=device) * 3.5
        with torch.no_grad():
            real_output_cond = torch.utils.checkpoint.checkpoint(
                self.real_score, noisy_sample, conditional_dict, mmd_timestep, use_reentrant=False
            )
            if len(real_output_cond) < 2:
                del noisy_sample
                return None
            flow_cond, _ = real_output_cond[:2]
            del real_output_cond
            real_output_uncond = torch.utils.checkpoint.checkpoint(
                self.real_score, noisy_sample, unconditional_dict, mmd_timestep, use_reentrant=False
            )
            if len(real_output_uncond) < 2:
                del flow_cond, noisy_sample
                return None
            flow_uncond, _ = real_output_uncond[:2]
            del real_output_uncond
            cfg_scale_expanded = cfg_scale.view(batch_size, 1, 1, 1, 1)
            flow_cfg = flow_uncond + cfg_scale_expanded * (flow_cond - flow_uncond)
            del flow_cond, flow_uncond
            flat_flow = flow_cfg.flatten(0, 1)
            flat_noisy = noisy_sample.flatten(0, 1)
            flat_t = mmd_timestep.flatten(0, 1)
            del flow_cfg, noisy_sample
            # sigma lookup: x_s = x_t + (sigma_s - sigma_t) * flow (flow matching ODE)
            sigmas = self.scheduler.sigmas.to(device)
            timesteps = self.scheduler.timesteps.to(device)
            timestep_id_t = torch.argmin(
                (timesteps.unsqueeze(0) - flat_t.unsqueeze(1).to(timesteps.dtype)).abs(), dim=1
            )
            sigma_t = sigmas[timestep_id_t].to(dtype).reshape(-1, 1, 1, 1)
            timestep_id_anchor = torch.argmin(
                (timesteps.unsqueeze(0) - anchor_t.unsqueeze(1).to(timesteps.dtype)).abs(), dim=1
            )
            sigma_anchor = sigmas[timestep_id_anchor].to(dtype).reshape(-1, 1, 1, 1)
            x_mid = (flat_noisy + (sigma_anchor - sigma_t) * flat_flow).unflatten(0, (batch_size, num_frames))
            print('sigma_anchor, sigma_t: ', sigma_anchor.mean().item(), sigma_t.mean().item())
            pred_x0_teacher = (flat_noisy - sigma_t * flat_flow).unflatten(0, (batch_size, num_frames))
            del flat_flow, flat_noisy, flat_t, sigma_t, sigma_anchor, noise_for_guidance
        torch.cuda.empty_cache()

        # 4) Generator chunk-by-chunk from x_mid to x0 -> pred_real_image_for_mmd
        # if self.inference_pipeline is None:
            # self._initialize_inference_pipeline()
        with torch.no_grad():
            pred_real_image_for_mmd = self.inference_pipeline.inference_from_latent_to_x0(
                x_mid, start_step_index=anchor_index, **conditional_dict
            )

        # 5) MMD feature extraction and loss (same as before)
        mmd_feature_timestep_min = getattr(self.args, "mmd_feature_timestep_min", 100)
        mmd_feature_timestep_max = getattr(self.args, "mmd_feature_timestep_max", 300)
        mmd_feature_timestep = self._get_timestep(
            mmd_feature_timestep_min, mmd_feature_timestep_max,
            batch_size, num_frames, self.num_frame_per_block, uniform_timestep=True
        )
        if self.timestep_shift > 1:
            mmd_feature_timestep = self.timestep_shift * (mmd_feature_timestep / 1000.0) / (
                1 + (self.timestep_shift - 1) * (mmd_feature_timestep / 1000.0)) * 1000.0
        mmd_feature_timestep = mmd_feature_timestep.clamp(self.min_step, self.max_step).long()
        shared_noise = torch.randn_like(pred_image).detach()
        mmd_feature_block = getattr(self.args, "mmd_feature_block", 20)

        with torch.no_grad():
            noisy_real_for_mmd = self.scheduler.add_noise(
                pred_real_image_for_mmd.flatten(0, 1), shared_noise.flatten(0, 1),
                mmd_feature_timestep.flatten(0, 1)
            ).unflatten(0, (batch_size, num_frames))
            real_output = torch.utils.checkpoint.checkpoint(
                self.real_score, noisy_real_for_mmd, conditional_dict, mmd_feature_timestep,
                mmd_features=mmd_feature_block, use_reentrant=False
            )
            if len(real_output) != 3:
                del noisy_real_for_mmd
                return None
            _, _, real_feature_mmd = real_output
            del real_output, noisy_real_for_mmd

        noisy_fake_for_mmd = self.scheduler.add_noise(
            pred_image.flatten(0, 1), shared_noise.flatten(0, 1),
            mmd_feature_timestep.flatten(0, 1)
        ).unflatten(0, (batch_size, num_frames))
        fake_output = torch.utils.checkpoint.checkpoint(
            self.real_score, noisy_fake_for_mmd, conditional_dict, mmd_feature_timestep,
            mmd_features=mmd_feature_block, use_reentrant=False
        )
        if len(fake_output) != 3:
            del noisy_fake_for_mmd
            return None
        _, _, fake_feature_mmd = fake_output

        mmd_kernel = getattr(self.args, "mmd_kernel", "linear")
        mmd_sigma = getattr(self.args, "mmd_sigma", 100)
        mmd_loss = mmd2_loss(
            real_feature_mmd.detach(), fake_feature_mmd,
            sigma=mmd_sigma, kernel=mmd_kernel, num_frames=num_frames
        )
        if gradient_mask is not None:
            mmd_loss = mmd_loss * gradient_mask.float().mean()
        log_dict = {
            "mmd_loss": mmd_loss.detach(),
            "mmd_feature_timestep": mmd_feature_timestep.float().mean().detach(),
            "x_mid": x_mid.detach(),
            "pred_x0_teacher": pred_x0_teacher.detach(),
            "pred_real_image_for_mmd": pred_real_image_for_mmd.detach(),
        }
        return mmd_loss, log_dict
        # except Exception as e:
        #     if dist.get_rank() == 0 if dist.is_initialized() else True:
        #         print(f"Warning: MMD guidance loss computation failed: {e}")
        #     return None

    def compute_distribution_matching_loss(
        self,
        image_or_video: torch.Tensor,
        conditional_dict: dict,
        unconditional_dict: dict,
        gradient_mask: Optional[torch.Tensor] = None,
        denoised_timestep_from: int = 0,
        denoised_timestep_to: int = 0
    ) -> Tuple[torch.Tensor, dict, torch.Tensor]:
        """
        Compute the DMD loss (eq 7 in https://arxiv.org/abs/2311.18828).
        Input:
            - image_or_video: a tensor with shape [B, F, C, H, W] where the number of frame is 1 for images.
            - conditional_dict: a dictionary containing the conditional information (e.g. text embeddings, image embeddings).
            - unconditional_dict: a dictionary containing the unconditional information (e.g. null/negative text embeddings, null/negative image embeddings).
            - gradient_mask: a boolean tensor with the same shape as image_or_video indicating which pixels to compute loss .
        Output:
            - dmd_loss: a scalar tensor representing the DMD loss.
            - dmd_log_dict: a dictionary containing the intermediate tensors for logging.
            - grad: the computed gradient tensor for logging.
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

        # Add DMD loss to log dict
        dmd_log_dict["dmd_loss"] = dmd_loss.detach()

        return dmd_loss, dmd_log_dict, grad, pred_real_image

    def generator_loss(
        self,
        image_or_video_shape,
        conditional_dict: dict,
        unconditional_dict: dict,
        clean_latent: torch.Tensor,
        initial_latent: torch.Tensor = None
    ) -> Tuple[torch.Tensor, dict]:
        """
        Generate image/videos from noise and compute the DMD loss.
        The noisy input to the generator is backward simulated.
        This removes the need of any datasets during distillation.
        See Sec 4.5 of the DMD2 paper (https://arxiv.org/abs/2405.14867) for details.
        Input:
            - image_or_video_shape: a list containing the shape of the image or video [B, F, C, H, W].
            - conditional_dict: a dictionary containing the conditional information (e.g. text embeddings, image embeddings).
            - unconditional_dict: a dictionary containing the unconditional information (e.g. null/negative text embeddings, null/negative image embeddings).
            - clean_latent: a tensor containing the clean latents [B, F, C, H, W]. Need to be passed when no backward simulation is used.
        Output:
            - loss: a scalar tensor representing the generator loss.
            - generator_log_dict: a dictionary containing the intermediate tensors for logging.
        """
        # Step 1: Unroll generator to obtain fake videos
        pred_image, gradient_mask, denoised_timestep_from, denoised_timestep_to = self._run_generator(
            image_or_video_shape=image_or_video_shape,
            conditional_dict=conditional_dict,
            initial_latent=initial_latent
        )

        # Step 2: Compute the DMD loss
        dmd_loss, dmd_log_dict, grad, pred_real_image = self.compute_distribution_matching_loss(
            image_or_video=pred_image,
            conditional_dict=conditional_dict,
            unconditional_dict=unconditional_dict,
            gradient_mask=gradient_mask,
            denoised_timestep_from=denoised_timestep_from,
            denoised_timestep_to=denoised_timestep_to
        )
        torch.cuda.empty_cache()

        # Step 3: Compute MMD guidance loss (optional)
        mmd_loss = None
        mmd_log_dict = None
        mmd_weight = getattr(self.args, "mmd_weight", 0.0)
        if mmd_weight > 0.0:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                mmd_result = self._compute_mmd_guidance_loss_teacher_only(
                    pred_image=pred_image,
                    pred_real_image=pred_real_image,
                    conditional_dict=conditional_dict,
                    unconditional_dict=unconditional_dict,
                    gradient_mask=gradient_mask
                )
                torch.cuda.empty_cache()
            if mmd_result is not None:
                mmd_loss, mmd_log_dict = mmd_result
                print('mmd_loss', mmd_loss.mean())
                if mmd_loss is not None:
                    dmd_log_dict.update(mmd_log_dict)
                    dmd_log_dict["mmd_loss"] = mmd_loss.detach()
                    dmd_loss = dmd_loss + mmd_weight * mmd_loss

        # Logging: Save videos periodically
        with torch.autocast("cuda", dtype=torch.bfloat16):
            self._save_log_videos(
                pred_image=pred_image,
                grad=grad,
                clean_latent=clean_latent,
                d_from=denoised_timestep_from,
                d_to=denoised_timestep_to,
                pred_real_image=mmd_log_dict["pred_x0_teacher"] if (mmd_log_dict is not None and "pred_x0_teacher" in mmd_log_dict) else None,
                pred_real_image_for_mmd=mmd_log_dict["pred_real_image_for_mmd"] if (mmd_log_dict is not None and "pred_real_image_for_mmd" in mmd_log_dict) else None
            )

        torch.cuda.empty_cache()
        return dmd_loss, dmd_log_dict

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
        The noisy input to the generator is backward simulated.
        This removes the need of any datasets during distillation.
        See Sec 4.5 of the DMD2 paper (https://arxiv.org/abs/2405.14867) for details.
        Input:
            - image_or_video_shape: a list containing the shape of the image or video [B, F, C, H, W].
            - conditional_dict: a dictionary containing the conditional information (e.g. text embeddings, image embeddings).
            - unconditional_dict: a dictionary containing the unconditional information (e.g. null/negative text embeddings, null/negative image embeddings).
            - clean_latent: a tensor containing the clean latents [B, F, C, H, W]. Need to be passed when no backward simulation is used.
        Output:
            - loss: a scalar tensor representing the generator loss.
            - critic_log_dict: a dictionary containing the intermediate tensors for logging.
        """

        # Step 1: Run generator on backward simulated noisy input
        with torch.no_grad():
            generated_image, _, denoised_timestep_from, denoised_timestep_to = self._run_generator(
                image_or_video_shape=image_or_video_shape,
                conditional_dict=conditional_dict,
                initial_latent=initial_latent
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

        _, pred_fake_image = torch.utils.checkpoint.checkpoint(
            self.fake_score,
            noisy_generated_image,
            conditional_dict,
            critic_timestep,
            use_reentrant=False
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