"""
Chunked backward DMD: block-wise backward + gradient accumulation to save memory.
Uses Phase 1 no-grad forward with per-block cache, then Phase 2 per-block
re-forward + DMD loss + backward. Does not modify model/dmd.py.
"""
from pipeline.chunked_backward_training import (
    ChunkedBackwardTrainingPipeline,
    _restore_kv_cache_from_indices,
    _shrink_kv_cache_to_indices,
    _deep_copy_kv_cache,
)
from model.dmd import DMD
import torch.nn.functional as F
from typing import Optional, Tuple, List
import torch
import torch.distributed as dist
from einops import rearrange


class ChunkedBackwardDMD(DMD):
    """
    DMD with block-wise backward and gradient accumulation (Algorithm 2 style).
    Phase 1: Full forward with no_grad, cache per-block Xcache and KV.
    Phase 2: For each block i (reverse order), restore KV, re-forward block i with grad,
    build full Xi (concat detached Xθ for j≠i and grad output for i), compute DMD loss, backward.
    """

    def _initialize_inference_pipeline(self):
        self.inference_pipeline = ChunkedBackwardTrainingPipeline(
            denoising_step_list=self.denoising_step_list,
            scheduler=self.scheduler,
            generator=self.generator,
            num_frame_per_block=self.num_frame_per_block,
            independent_first_frame=self.args.independent_first_frame,
            same_step_across_blocks=self.args.same_step_across_blocks,
            last_step_only=self.args.last_step_only,
            num_max_frames=self.num_training_frames,
            context_noise=self.args.context_noise,
            loop_rope_training=getattr(self.args, "loop_rope_training", False),
        )

    def generator_loss(
        self,
        image_or_video_shape,
        conditional_dict: dict,
        unconditional_dict: dict,
        clean_latent: torch.Tensor,
        initial_latent: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Phase 1: No-grad forward, cache Xcache and KV per block.
        Phase 2: For each block (reverse), restore KV, re-forward that block with grad,
        build full 5s Xi, compute DMD loss, backward (gradients accumulate).
        Return detached mean loss for logging; trainer's backward() is no-op.
        """
        if self.inference_pipeline is None:
            self._initialize_inference_pipeline()

        # Same noise shape as base: 21 frames (or num_training_frames)
        min_num_frames = 20 if self.args.independent_first_frame else 21
        max_num_frames = (
            self.num_training_frames - 1
            if self.args.independent_first_frame
            else self.num_training_frames
        )
        assert max_num_frames % self.num_frame_per_block == 0
        assert min_num_frames % self.num_frame_per_block == 0
        max_num_blocks = max_num_frames // self.num_frame_per_block
        min_num_blocks = min_num_frames // self.num_frame_per_block
        num_generated_blocks = torch.randint(
            min_num_blocks, max_num_blocks + 1, (1,), device=self.device
        )
        dist.broadcast(num_generated_blocks, src=0)
        num_generated_blocks = num_generated_blocks.item()
        num_generated_frames = num_generated_blocks * self.num_frame_per_block
        if self.args.independent_first_frame and initial_latent is None:
            num_generated_frames += 1
        noise_shape = list(image_or_video_shape)
        noise_shape[1] = num_generated_frames
        noise = torch.randn(noise_shape, device=self.device, dtype=self.dtype)

        if initial_latent is not None:
            conditional_dict = dict(conditional_dict)
            conditional_dict["initial_latent"] = initial_latent

        # Phase 1: full forward with no_grad, get output and per-block caches
        output, Xcache_list, KV_after_block_list, d_from, d_to, exit_step_index = (
            self.inference_pipeline.inference_with_trajectory_and_cache(
                noise=noise, **conditional_dict
            )
        )

        # Slice to last 21 frames and first-frame re-encode (same as base)
        num_blocks_total = len(Xcache_list)
        if output.shape[1] > 21:
            with torch.no_grad():
                latent_to_decode = output[:, :-20, ...]
                pixels = self.vae.decode_to_pixel(latent_to_decode)
                frame = pixels[:, -1:, ...].to(self.dtype)
                frame = rearrange(frame, "b t c h w -> b c t h w")
                image_latent = self.vae.encode_to_latent(frame).to(self.dtype)
            output_last_21 = torch.cat([image_latent, output[:, -20:, ...]], dim=1)
            num_blocks_last_21 = 7
            Xcache_list_use = Xcache_list[-num_blocks_last_21:]
        else:
            output_last_21 = output
            num_blocks_last_21 = output.shape[1] // self.num_frame_per_block
            Xcache_list_use = Xcache_list

        gradient_mask = None
        if num_generated_frames != min_num_frames:
            gradient_mask = torch.ones_like(output_last_21, dtype=torch.bool)
            if self.args.independent_first_frame:
                gradient_mask[:, :1] = False
            else:
                gradient_mask[:, : self.num_frame_per_block] = False

        Xθ = output_last_21.detach()
        exit_timestep = self.inference_pipeline.denoising_step_list[exit_step_index]
        device = output_last_21.device
        dtype = output_last_21.dtype
        batch_size = output_last_21.shape[0]
        nf = self.num_frame_per_block

        # DMD once on full sequence (like Vivix): get grad_full, no backward here.
        _, dmd_log_dict, grad_full, pred_real_image = self.compute_distribution_matching_loss(
            image_or_video=output_last_21,
            conditional_dict=conditional_dict,
            unconditional_dict=unconditional_dict,
            gradient_mask=gradient_mask,
            denoised_timestep_from=d_from,
            denoised_timestep_to=d_to,
        )

        # MMD: full-sequence gradient (Option 2). Compute MMD loss on full 5s, get mmd_grad_full via
        # autograd, then add per-block slices in Phase 2. Use autocast bfloat16 + optional checkpoint to reduce OOM.
        mmd_grad_full = None
        mmd_weight = getattr(self.args, "mmd_weight", 0.0)
        mmd_use_checkpoint = getattr(self.args, "mmd_checkpoint_full_path", True)
        if mmd_weight > 0.0 and pred_real_image is not None:
            # Same values as output_last_21, no clone: shared storage, detach so grad won't flow to generator.
            pred_for_mmd_temp = output_last_21.detach().clone() #.requires_grad_(True)
            pred_for_mmd = torch.randn_like(pred_for_mmd_temp).to(output_last_21.device).requires_grad_(True)
            # del pred_for_mmd_temp
            pred_real_image_temp = pred_real_image.detach()
            pred_real_image = torch.randn_like(pred_real_image_temp).to(output_last_21.device).requires_grad_(False)


            def _mmd_loss_fn(pred_img):
                """Returns only MMD loss for checkpoint; backward will re-run this to get gradients."""
                r = self._compute_mmd_guidance_loss(
                    pred_image=pred_img,
                    pred_real_image=pred_real_image,
                    conditional_dict=conditional_dict,
                    unconditional_dict=unconditional_dict,
                    gradient_mask=gradient_mask,
                )
                if r is None:
                    return pred_img.new_zeros(1).squeeze(0)
                loss, _ = r
                return loss if loss is not None else pred_img.new_zeros(1).squeeze(0)

            with torch.autocast("cuda", dtype=torch.bfloat16):
                # with torch.enable_grad():
                    if mmd_use_checkpoint:
                        mmd_loss = torch.utils.checkpoint.checkpoint(
                            _mmd_loss_fn,
                            pred_for_mmd,
                            use_reentrant=False,
                        )
                    else:
                        mmd_result = self._compute_mmd_guidance_loss(
                            pred_image=pred_for_mmd,
                            pred_real_image=pred_real_image,
                            conditional_dict=conditional_dict,
                            unconditional_dict=unconditional_dict,
                            gradient_mask=gradient_mask,
                        )
                        mmd_loss = mmd_result[0] if mmd_result and mmd_result[0] is not None else pred_for_mmd.new_zeros(1).squeeze(0)
                        if mmd_result is not None and mmd_result[1] is not None:
                            dmd_log_dict.update(mmd_result[1])

            if mmd_loss is not None and mmd_loss.requires_grad:
                # with torch.autocast("cuda", dtype=torch.bfloat16):
                mmd_loss = mmd_loss.float()
                print('mmd_loss', mmd_loss.shape, mmd_loss.mean())
                mmd_loss.backward()
                if pred_for_mmd.grad is not None:
                    mmd_grad_full = pred_for_mmd.grad.detach().float().clone()
                    print('mmd_grad_full', mmd_grad_full.shape, mmd_grad_full.mean())
                dmd_log_dict["mmd_loss"] = mmd_loss.detach().float()
                if mmd_use_checkpoint and "mmd_feature_timestep" not in dmd_log_dict:
                    dmd_log_dict["mmd_feature_timestep"] = torch.tensor(0.0, device=output_last_21.device)
            del pred_for_mmd
            torch.cuda.empty_cache()

        total_loss = 0.0

        # Phase 2: block-wise backward (reverse order); each block only does re-forward + MSE(pred_block, pred_block - grad_block).
        # After each block we pop the KV we used (Algorithm 2 line 37) so GPU doesn't hold all N copies until the end.
        for block_idx in range(num_blocks_last_21 - 1, -1, -1):
            pop_idx = None  # index to pop after this block (free memory)
            if num_blocks_total > num_blocks_last_21:
                restore_idx = num_blocks_total - num_blocks_last_21 - 1 + block_idx
                kv_indices_src, cross_src = KV_after_block_list[restore_idx]
                pop_idx = restore_idx
                _restore_kv_cache_from_indices(
                    self.inference_pipeline.kv_cache1,
                    self.inference_pipeline.crossattn_cache,
                    kv_indices_src,
                    cross_src,
                    device=device,
                )
            else:
                if block_idx > 0:
                    kv_indices_src, cross_src = KV_after_block_list[block_idx - 1]
                    pop_idx = block_idx - 1
                    _restore_kv_cache_from_indices(
                        self.inference_pipeline.kv_cache1,
                        self.inference_pipeline.crossattn_cache,
                        kv_indices_src,
                        cross_src,
                        device=device,
                    )
                else:
                    self.inference_pipeline._initialize_kv_cache(
                        batch_size=batch_size, dtype=dtype, device=device
                    )
                    self.inference_pipeline._initialize_crossattn_cache(
                        batch_size=batch_size, dtype=dtype, device=device
                    )

            # Pass a deep copy of KV cache to the generator so the computation graph
            # references this copy; _restore_kv_cache in the next iteration won't
            # overwrite it, avoiding "storage of size 0" during backward.
            kv_for_fwd, cross_for_fwd = _deep_copy_kv_cache(
                self.inference_pipeline.kv_cache1,
                self.inference_pipeline.crossattn_cache,
            )
            x_cache_i = Xcache_list_use[block_idx]
            timestep_i = (
                torch.ones(
                    [batch_size, nf],
                    device=device,
                    dtype=torch.int64,
                )
                * exit_timestep
            )
            with torch.enable_grad():
                _, x0_i = self.generator(
                    noisy_image_or_video=x_cache_i,
                    conditional_dict=conditional_dict,
                    timestep=timestep_i,
                    kv_cache=kv_for_fwd,
                    crossattn_cache=cross_for_fwd,
                    current_start=block_idx * nf * self.inference_pipeline.frame_seq_length,
                )

            grad_block = grad_full[:, block_idx * nf : (block_idx + 1) * nf].clone()
            if mmd_grad_full is not None:
                grad_block = grad_block + mmd_weight * mmd_grad_full[
                    :, block_idx * nf : (block_idx + 1) * nf
                ]
            if gradient_mask is not None:
                mask_block = gradient_mask[:, block_idx * nf : (block_idx + 1) * nf]
                dmd_loss_i = 0.5 * F.mse_loss(
                    x0_i.double()[mask_block],
                    (x0_i.double() - grad_block.double()).detach()[mask_block],
                    reduction="mean",
                )
            else:
                dmd_loss_i = 0.5 * F.mse_loss(
                    x0_i.double(),
                    (x0_i.double() - grad_block.double()).detach(),
                    reduction="mean",
                )
            (dmd_loss_i / num_blocks_last_21).backward()
            total_loss = total_loss + dmd_loss_i.detach()

            # Shrink model's real KV buffer to current valid prefix (0..end_index), then pop saved entry.
            if pop_idx is not None:
                _shrink_kv_cache_to_indices(
                    self.inference_pipeline.kv_cache1,
                    kv_indices_src,
                    device=device,
                )
                KV_after_block_list.pop(pop_idx)

        total_loss = total_loss / num_blocks_last_21
        dmd_log_dict["dmd_loss"] = total_loss

        # Logging: save videos once (use full output_last_21 and grad_full from one-time DMD)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            self._save_log_videos(
                pred_image=output_last_21.detach(),
                grad=grad_full.detach(),
                clean_latent=clean_latent,
                d_from=d_from,
                d_to=d_to,
                pred_real_image=pred_real_image.detach() if pred_real_image is not None else None,
                pred_real_image_for_mmd=None,
            )

        return total_loss.detach(), dmd_log_dict
