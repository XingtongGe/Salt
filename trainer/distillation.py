import gc
import logging

from utils.dataset import ShardingLMDBDataset, cycle
from utils.dataset import TextDataset
from utils.distributed import EMA_FSDP, fsdp_wrap, fsdp_state_dict, launch_distributed_job
from utils.misc import (
    set_seed,
    merge_dict_list
)
import torch.distributed as dist
from omegaconf import OmegaConf
from model import CausVid, DMD, SiD, ShortcutInjectedDMD, StreamingDMD, StreamingSCDMD, ChunkedBackwardDMD
from pipeline import CausalInferencePipeline
import torch
from tensorboardX import SummaryWriter
import wandb
import time
import os
import imageio
import numpy as np


class Trainer:
    def __init__(self, config):
        self.config = config
        self.step = 0

        # Step 1: Initialize the distributed training environment (rank, seed, dtype, logging etc.)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        launch_distributed_job()
        global_rank = dist.get_rank()
        self.world_size = dist.get_world_size()

        self.dtype = torch.bfloat16 if config.mixed_precision else torch.float32
        self.device = torch.cuda.current_device()
        self.is_main_process = global_rank == 0
        self.causal = config.causal
        self.disable_wandb = config.disable_wandb

        # use a random seed for the training
        if config.seed == 0:
            random_seed = torch.randint(0, 10000000, (1,), device=self.device)
            dist.broadcast(random_seed, src=0)
            config.seed = random_seed.item()

        set_seed(config.seed + global_rank)

        if self.is_main_process and not self.disable_wandb:
            wandb.login(host=config.wandb_host, key=config.wandb_key)
            wandb.init(
                config=OmegaConf.to_container(config, resolve=True),
                name=config.config_name,
                mode="online",
                entity=config.wandb_entity,
                project=config.wandb_project,
                dir=config.wandb_save_dir
            )

        self.output_path = config.logdir
        self.writer = SummaryWriter(log_dir=os.path.join(config.logdir, "tensorboard")) if self.is_main_process else None

        # Step 2: Initialize the model and optimizer
        print('config.distribution_loss: ', config.distribution_loss)

        if config.distribution_loss == "causvid":
            self.model = CausVid(config, device=self.device)
        elif config.distribution_loss == "dmd":
            self.model = DMD(config, device=self.device)
        elif config.distribution_loss == "sc_dmd":
            self.model = ShortcutInjectedDMD(config, device=self.device)
        elif config.distribution_loss == "streaming_dmd":
            self.model = StreamingDMD(config, device=self.device)
        elif config.distribution_loss == "streaming_sc_dmd":
            self.model = StreamingSCDMD(config, device=self.device)
        elif config.distribution_loss == "chunked_backward_dmd":
            self.model = ChunkedBackwardDMD(config, device=self.device)
        elif config.distribution_loss == "sid":
            self.model = SiD(config, device=self.device)
        else:
            raise ValueError("Invalid distribution matching loss")

        # Save pretrained model state_dicts to CPU
        self.fake_score_state_dict_cpu = self.model.fake_score.state_dict()

        self.model.generator = fsdp_wrap(
            self.model.generator,
            sharding_strategy=config.sharding_strategy,
            mixed_precision=config.mixed_precision,
            wrap_strategy=config.generator_fsdp_wrap_strategy
        )

        self.model.real_score = fsdp_wrap(
            self.model.real_score,
            sharding_strategy=config.sharding_strategy,
            mixed_precision=config.mixed_precision,
            wrap_strategy=config.real_score_fsdp_wrap_strategy
        )

        self.model.fake_score = fsdp_wrap(
            self.model.fake_score,
            sharding_strategy=config.sharding_strategy,
            mixed_precision=config.mixed_precision,
            wrap_strategy=config.fake_score_fsdp_wrap_strategy
        )

        self.model.text_encoder = fsdp_wrap(
            self.model.text_encoder,
            sharding_strategy=config.sharding_strategy,
            mixed_precision=config.mixed_precision,
            wrap_strategy=config.text_encoder_fsdp_wrap_strategy,
            cpu_offload=getattr(config, "text_encoder_cpu_offload", False)
        )

        if not config.no_visualize or config.load_raw_video:
            self.model.vae = self.model.vae.to(
                device=self.device, dtype=torch.bfloat16 if config.mixed_precision else torch.float32)

        self.generator_optimizer = torch.optim.AdamW(
            [param for param in self.model.generator.parameters()
             if param.requires_grad],
            lr=config.lr,
            betas=(config.beta1, config.beta2),
            weight_decay=config.weight_decay
        )

        self.critic_optimizer = torch.optim.AdamW(
            [param for param in self.model.fake_score.parameters()
             if param.requires_grad],
            lr=config.lr_critic if hasattr(config, "lr_critic") else config.lr,
            betas=(config.beta1_critic, config.beta2_critic),
            weight_decay=config.weight_decay
        )

        # Step 3: Initialize the dataloader
        if self.config.i2v:
            dataset = ShardingLMDBDataset(config.data_path, max_pair=int(1e8))
        else:
            dataset = TextDataset(config.data_path)
        sampler = torch.utils.data.distributed.DistributedSampler(
            dataset, shuffle=True, drop_last=True)
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=config.batch_size,
            sampler=sampler,
            num_workers=8)

        if dist.get_rank() == 0:
            print("DATASET SIZE %d" % len(dataset))
        self.dataloader = cycle(dataloader)

        ##############################################################################################################
        # 6. Set up EMA parameter containers
        rename_param = (
            lambda name: name.replace("_fsdp_wrapped_module.", "")
            .replace("_checkpoint_wrapped_module.", "")
            .replace("_orig_mod.", "")
        )
        self.name_to_trainable_params = {}
        for n, p in self.model.generator.named_parameters():
            if not p.requires_grad:
                continue

            renamed_n = rename_param(n)
            self.name_to_trainable_params[renamed_n] = p
        ema_weight = config.ema_weight
        self.generator_ema = None
        if (ema_weight is not None) and (ema_weight > 0.0):
            print(f"Setting up EMA with weight {ema_weight}")
            self.generator_ema = EMA_FSDP(self.model.generator, decay=ema_weight)

        ##############################################################################################################
        # 7. (If resuming) Load the model and optimizer, lr_scheduler, ema's statedicts
        if getattr(config, "generator_ckpt", False):
            ckpt_key = getattr(config, "generator_ckpt_key", "generator")
            print(f"Loading pretrained generator from {config.generator_ckpt} (key={ckpt_key})")
            state_dict = torch.load(config.generator_ckpt, map_location="cpu")
            if ckpt_key in state_dict:
                state_dict = state_dict[ckpt_key]
            elif "generator" in state_dict:
                state_dict = state_dict["generator"]
            elif "model" in state_dict:
                state_dict = state_dict["model"]

            if ckpt_key == "generator_ema":
                # EMA shadow dict uses raw param names from module.named_parameters(),
                # must load via summon_full_params to match FSDP's internal layout
                from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
                with FSDP.summon_full_params(self.model.generator, writeback=True):
                    for n, p in self.model.generator.module.named_parameters():
                        if n in state_dict:
                            p.data.copy_(state_dict[n].to(dtype=p.dtype, device=p.device))
                        else:
                            print(f"WARNING: EMA key '{n}' not found in checkpoint, skipping")
            else:
                self.model.generator.load_state_dict(
                    state_dict, strict=True
                )

        ##############################################################################################################

        # Let's delete EMA params for early steps to save some computes at training and inference
        if self.step < config.ema_start_step:
            self.generator_ema = None

        self.max_grad_norm_generator = getattr(config, "max_grad_norm_generator", 10.0)
        self.max_grad_norm_critic = getattr(config, "max_grad_norm_critic", 10.0)
        self.previous_time = None

    def save(self):
        print("Start gathering distributed model states...")
        generator_state_dict = fsdp_state_dict(
            self.model.generator)
        critic_state_dict = fsdp_state_dict(
            self.model.fake_score)

        if self.config.ema_start_step < self.step:
            state_dict = {
                "generator": generator_state_dict,
                "critic": critic_state_dict,
                "generator_ema": self.generator_ema.state_dict(),
            }
        else:
            state_dict = {
                "generator": generator_state_dict,
                "critic": critic_state_dict,
            }

        if self.is_main_process:
            os.makedirs(os.path.join(self.output_path,
                        f"checkpoint_model_{self.step:06d}"), exist_ok=True)
            torch.save(state_dict, os.path.join(self.output_path,
                       f"checkpoint_model_{self.step:06d}", "model.pt"))
            print("Model saved to", os.path.join(self.output_path,
                  f"checkpoint_model_{self.step:06d}", "model.pt"))

    def fwdbwd_one_step(self, batch, train_generator):
        self.model.eval()  # prevent any randomness (e.g. dropout)

        if self.step % 20 == 0:
            torch.cuda.empty_cache()

        # Step 1: Get the next batch of text prompts
        text_prompts = batch["prompts"]
        if self.config.i2v:
            clean_latent = None
            image_latent = batch["ode_latent"][:, -1][:, 0:1, ].to(
                device=self.device, dtype=self.dtype)
        else:
            clean_latent = None
            image_latent = None

        batch_size = len(text_prompts)
        image_or_video_shape = list(self.config.image_or_video_shape)
        image_or_video_shape[0] = batch_size

        # Step 2: Extract the conditional infos
        with torch.no_grad():
            conditional_dict = self.model.text_encoder(
                text_prompts=text_prompts)

            if not getattr(self, "unconditional_dict", None):
                unconditional_dict = self.model.text_encoder(
                    text_prompts=[self.config.negative_prompt] * batch_size)
                unconditional_dict = {k: v.detach()
                                      for k, v in unconditional_dict.items()}
                self.unconditional_dict = unconditional_dict  # cache the unconditional_dict
            else:
                unconditional_dict = self.unconditional_dict

        # Step 3: Store gradients for the generator (if training the generator)
        if train_generator:
            # Set logging parameters for SC-DMD if applicable
            if hasattr(self.model, 'set_logging_params'):
                self.model.set_logging_params(step=self.step, logdir=self.output_path)

            generator_loss, generator_log_dict = self.model.generator_loss(
                image_or_video_shape=image_or_video_shape,
                conditional_dict=conditional_dict,
                unconditional_dict=unconditional_dict,
                clean_latent=clean_latent,
                initial_latent=image_latent if self.config.i2v else None
            )

            # ChunkedBackwardDMD already calls .backward() per block inside generator_loss
            if self.config.distribution_loss != "chunked_backward_dmd":
                generator_loss.backward()
            generator_grad_norm = self.model.generator.clip_grad_norm_(
                self.max_grad_norm_generator)

            generator_log_dict.update({"generator_loss": generator_loss,
                                       "generator_grad_norm": generator_grad_norm})

            return generator_log_dict
        else:
            generator_log_dict = {}

        # Step 4: Store gradients for the critic (if training the critic)
        critic_loss, critic_log_dict = self.model.critic_loss(
            image_or_video_shape=image_or_video_shape,
            conditional_dict=conditional_dict,
            unconditional_dict=unconditional_dict,
            clean_latent=clean_latent,
            initial_latent=image_latent if self.config.i2v else None
        )

        critic_loss.backward()
        critic_grad_norm = self.model.fake_score.clip_grad_norm_(
            self.max_grad_norm_critic)

        critic_log_dict.update({"critic_loss": critic_loss,
                                "critic_grad_norm": critic_grad_norm})

        return critic_log_dict

    def run_inference_test(self, test_prompts=None):
        """
        Run inference test with current model.
        Args:
            test_prompts: List of text prompts for testing. If None, use default prompts.
        Note: In distributed training, all processes must participate in FSDP model forward pass.
        """
        if test_prompts is None:
            # Default test prompts
            test_prompts = [
                "A beautiful sunset over the ocean with waves gently crashing on the shore.",
                "A cat playing with a ball of yarn in a cozy living room.",
                "A futuristic cityscape with flying cars and neon lights at night.",
                "a bunch of houses that are on a hillside.",
                "A handheld camera following a dog running through a park.",
                "A skeleton wearing a flower hat and sunglasses dances in the wild at sunset.",
                "A kangaroo wearing boxing gloves, sparring with a punching bag in a gym.",
                "A woman applying bright red lipstick in front of a mirror.",
                "A fat rabbit wearing a purple robe walking through a fantasy landscape.",
                "A plump, fluffy rabbit donning a voluminous purple robe walks gracefully through a vibrant fantasy landscape. The rabbit has large, expressive eyes and a gentle, curious expression. Its fur is soft and thick, and the robe drapes elegantly over its body. The landscape features rolling hills covered in lush green grass, colorful wildflowers, and towering magical trees with shimmering leaves. In the distance, there are sparkling waterfalls and mystical castles. The scene is bathed in warm, golden sunlight. Medium shot, focusing on the rabbit's walk through the picturesque environment.",
                "A person drinking coffee in a cafe.",
                "A cozy, warm café setting with soft ambient lighting and wooden furnishings. A young adult, casually dressed in a sweater and jeans, sits at a small round table. They hold a steaming cup of coffee in their hand, taking a sip while looking pensively out the window. The café is moderately busy with other patrons engaged in conversations. The background showcases various coffee drinks and pastries displayed on a counter. The person’s expression is relaxed and content. Medium shot focusing on the person’s face and the coffee cup, capturing the intimate atmosphere of the café.",
                "An astronaut flying in space, zoom out.",
                "Astronaut floating in space with a helmet visor reflecting Earth below. The astronaut, wearing a full spacesuit with the American flag on the shoulder, is performing a spacewalk, arms extended as if in motion. The background shows the vastness of space with stars twinkling and Earth in the distance. The scene begins with a close-up of the astronaut and gradually zooms out to reveal the enormity of space surrounding them. Wide shot, showcasing the astronaut against the backdrop of the universe.",
                "A boat sailing leisurely along the Seine River with the Eiffel Tower in background.",
                "A serene, picturesque scene of a small wooden boat gently gliding along the Seine River in Paris, France. The boat is rowed leisurely by a middle-aged man in a casual striped shirt and khaki pants, who rows smoothly with rhythmic strokes. The Eiffel Tower stands majestically in the background, partially visible through the misty morning air. The riverbank is lined with lush green trees and quaint buildings, reflecting off the calm waters. The overall atmosphere is peaceful and tranquil, capturing the essence of a lazy summer day. Wide shot, static camera.",
            ]

        # Distribute tasks across processes (similar to reference code)
        if dist.is_initialized():
            world_size = dist.get_world_size()
            rank = dist.get_rank()
            print('world_size, rank: ', world_size, rank)
            # Distribute prompts across processes
            if world_size > len(test_prompts):
                test_prompts = test_prompts * (world_size // len(test_prompts) + 1)
                test_prompts = test_prompts[:world_size]
            else:
                test_prompts = test_prompts[:world_size]
            rank_prompts = test_prompts[rank:len(test_prompts):world_size]
            print('rank_prompts: ', rank_prompts)
            # Ensure each rank has at least one task to avoid empty inference
            if len(rank_prompts) == 0:
                # If this rank has no tasks, assign the first prompt as fallback
                rank_prompts = [test_prompts[0]] if len(test_prompts) > 0 else []
                print(f'[Rank {rank}] Warning: No tasks assigned, using fallback prompt')
        else:
            rank_prompts = test_prompts
            rank = 0

        # Sync all processes before inference
        if dist.is_initialized():
            dist.barrier()

        # try:
        # Set model to eval mode (all processes)
        self.model.eval()

        # Create inference pipeline using current model components
        generator = self.model.generator
        text_encoder = self.model.text_encoder
        vae = self.model.vae

        # Create a temporary config-like object for pipeline initialization
        class InferenceConfig:
            def __init__(self, config, denoising_step_list):
                self.denoising_step_list = denoising_step_list
                self.warp_denoising_step = getattr(config, "warp_denoising_step", False)
                self.num_frame_per_block = getattr(config, "num_frame_per_block", 1)
                self.independent_first_frame = getattr(config, "independent_first_frame", False)
                self.model_kwargs = getattr(config, "model_kwargs", {})
                self.context_noise = getattr(config, "context_noise", 0)

        # When scfm_mixed_step_lists: run inference for 4-step and 2-step lists separately and save to different dirs
        if getattr(self.config, "scfm_mixed_step_lists", False):
            inference_step_configs = [
                ("_4step", getattr(self.config, "scfm_step_lists_4", [1000, 750, 500, 250])),
                ("_2step", getattr(self.config, "scfm_step_lists_2", [1000, 500])),
            ]
        else:
            default_list = getattr(self.config, "denoising_step_list", [1000, 750, 500, 250])
            inference_step_configs = [("", default_list)]

        num_output_frames = getattr(self.config, "num_training_frames", 21)
        num_channels = getattr(self.config, "image_or_video_shape", [1, 21, 16, 60, 104])[2]
        height = getattr(self.config, "image_or_video_shape", [1, 21, 16, 60, 104])[3]
        width = getattr(self.config, "image_or_video_shape", [1, 21, 16, 60, 104])[4]

        save_dir = os.path.join(self.output_path, f"inference_test_step_{self.step:06d}")
        os.makedirs(save_dir, exist_ok=True)
        file_suffix = ""  # for single-list mode: rank0_test_00.mp4

        for suffix, denoising_step_list in inference_step_configs:
            if suffix:
                file_suffix = suffix  # e.g. _4step -> rank0_test_00_4step.mp4
            inference_config = InferenceConfig(self.config, denoising_step_list)
            pipeline = CausalInferencePipeline(
                args=inference_config,
                device=self.device,
                generator=generator,
                text_encoder=text_encoder,
                vae=vae
            )
            if rank == 0:
                print("inference_test denoising_step_list (%s): " % (suffix or "default"), pipeline.denoising_step_list)

            for i, prompt in enumerate(rank_prompts):
                if dist.is_initialized():
                    dist.barrier()
                noise = torch.randn(
                    [1, num_output_frames, num_channels, height, width],
                    device=self.device,
                    dtype=self.dtype
                )
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                    video = pipeline.inference(
                        noise=noise,
                        text_prompts=[prompt],
                        return_latents=False
                    )
                if dist.is_initialized():
                    dist.barrier()

                video_array = video[0].cpu().numpy()
                video_array = (video_array * 255.0).astype(np.uint8)
                video_array = np.transpose(video_array, (0, 2, 3, 1))
                save_path = os.path.join(save_dir, f"rank{rank}_test_{i:02d}{file_suffix}.mp4")
                with imageio.get_writer(save_path, fps=16) as writer:
                    for frame in video_array:
                        writer.append_data(frame)
                print(f"[Rank {rank}] Saved video to {save_path}")

        print(f"[Rank {rank}] Saved test videos to {save_dir}")

        # except Exception as e:
        #     print(f"[Inference Test] Error during inference test: {e}")
        #     import traceback
        #     traceback.print_exc()

    def generate_video(self, pipeline, prompts, image=None):
        batch_size = len(prompts)
        if image is not None:
            image = image.squeeze(0).unsqueeze(0).unsqueeze(2).to(device="cuda", dtype=torch.bfloat16)

            # Encode the input image as the first latent
            initial_latent = pipeline.vae.encode_to_latent(image).to(device="cuda", dtype=torch.bfloat16)
            initial_latent = initial_latent.repeat(batch_size, 1, 1, 1, 1)
            sampled_noise = torch.randn(
                [batch_size, self.model.num_training_frames - 1, 16, 60, 104],
                device="cuda",
                dtype=self.dtype
            )
        else:
            initial_latent = None
            sampled_noise = torch.randn(
                [batch_size, self.model.num_training_frames, 16, 60, 104],
                device="cuda",
                dtype=self.dtype
            )

        video, _ = pipeline.inference(
            noise=sampled_noise,
            text_prompts=prompts,
            return_latents=True,
            initial_latent=initial_latent
        )
        current_video = video.permute(0, 1, 3, 4, 2).cpu().numpy() * 255.0
        return current_video

    def train(self):
        start_step = self.step

        while True:
            TRAIN_GENERATOR = self.step % self.config.dfake_gen_update_ratio == 0

            # Train the generator
            if TRAIN_GENERATOR:
                self.generator_optimizer.zero_grad(set_to_none=True)
                extras_list = []
                batch = next(self.dataloader)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                    extra = self.fwdbwd_one_step(batch, True)
                extras_list.append(extra)
                generator_log_dict = merge_dict_list(extras_list)
                # print('generator_log_dict', generator_log_dict.keys())
                self.generator_optimizer.step()
                if self.generator_ema is not None:
                    self.generator_ema.update(self.model.generator)

            # Train the critic
            self.critic_optimizer.zero_grad(set_to_none=True)
            extras_list = []
            batch = next(self.dataloader)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                extra = self.fwdbwd_one_step(batch, False)
            extras_list.append(extra)
            critic_log_dict = merge_dict_list(extras_list)
            self.critic_optimizer.step()

            # Increment the step since we finished gradient update


            # Run inference test every 100 iterations
            # if self.step % 100 == 0 and self.is_main_process:
            if self.step % 100 == 0:
                torch.cuda.empty_cache()
                with torch.no_grad():
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                        self.run_inference_test()
                torch.cuda.empty_cache()

            # === Dynamic Timestep Switching for SC-DMD ===
            if self.config.distribution_loss == "sc_dmd":
                warmup_steps = getattr(self.config, "warmup_steps", -1)
                if warmup_steps > 0 and self.step == warmup_steps:
                    target_steps = getattr(self.config, "target_denoising_step_list", None)
                    if target_steps is not None:
                        if self.is_main_process:
                            old_steps = self.model.denoising_step_list.cpu().tolist() if hasattr(self.model, 'denoising_step_list') else "N/A"
                            print(f"[SC-DMD] Switching denoising steps from {old_steps} to {target_steps} at step {self.step}")

                        # Update model's denoising_step_list
                        # Follow the same initialization logic as base.py
                        self.model.denoising_step_list = torch.tensor(target_steps, dtype=torch.long, device=self.model.device)
                        if self.config.warp_denoising_step:
                            # Create timesteps on the same device as denoising_step_list
                            timesteps = torch.cat((self.model.scheduler.timesteps.to(self.model.device), torch.tensor([0], dtype=torch.float32, device=self.model.device)))
                            indices = (1000 - self.model.denoising_step_list).long()
                            self.model.denoising_step_list = timesteps[indices]

                        # Reset inference_pipeline so it will be reinitialized with new denoising_step_list
                        if hasattr(self.model, 'inference_pipeline'):
                            self.model.inference_pipeline = None
                        # Switch num_segments to target (e.g. 4 -> 2 for 4-step)
                        target_num_seg = getattr(self.config, 'target_num_segments', None)
                        if target_num_seg is not None:
                            self.config.num_segments = target_num_seg
                            if self.is_main_process:
                                print(f"[SC-DMD] num_segments switched to {target_num_seg}")
                        if self.is_main_process:
                            new_steps = self.model.denoising_step_list.cpu().tolist()
                            print(f"[SC-DMD] Successfully switched to {new_steps}")

                    # Switch num_training_frames (e.g. 21 -> 42 for 5s -> 5-10s)
                    target_num_frames = getattr(self.config, 'target_num_training_frames', None)
                    if target_num_frames is not None:
                        old_frames = self.model.num_training_frames
                        self.model.num_training_frames = target_num_frames
                        self.config.num_training_frames = target_num_frames
                        if hasattr(self.model, 'inference_pipeline'):
                            self.model.inference_pipeline = None
                        if self.is_main_process:
                            print(f"[SC-DMD] num_training_frames switched from {old_frames} to {target_num_frames}")

                    # Switch mmd_weight (e.g. 0 -> 0.05 to enable MMD after warmup)
                    target_mmd_weight = getattr(self.config, 'target_mmd_weight', None)
                    if target_mmd_weight is not None:
                        old_mmd = getattr(self.model.args, 'mmd_weight', 0.0)
                        self.model.args.mmd_weight = target_mmd_weight
                        self.config.mmd_weight = target_mmd_weight
                        if self.is_main_process:
                            print(f"[SC-DMD] mmd_weight switched from {old_mmd} to {target_mmd_weight}")
            # ============================================

            # Create EMA params (if not already created)
            if (self.step >= self.config.ema_start_step) and \
                    (self.generator_ema is None) and (self.config.ema_weight > 0):
                self.generator_ema = EMA_FSDP(self.model.generator, decay=self.config.ema_weight)

            # Save the model
            if (not self.config.no_save) and (self.step - start_step) > 0 and self.step % self.config.log_iters == 0:
                # torch.cuda.empty_cache()
                self.save()
                # torch.cuda.empty_cache()

            # Logging
            if self.is_main_process:
                wandb_loss_dict = {}
                if TRAIN_GENERATOR:
                    wandb_loss_dict.update(
                        {
                            "generator_loss": generator_log_dict["generator_loss"].mean().item(),
                            "generator_grad_norm": generator_log_dict["generator_grad_norm"].mean().item(),
                            "dmdtrain_gradient_norm": generator_log_dict["dmdtrain_gradient_norm"].mean().item()
                        }
                    )

                wandb_loss_dict.update(
                    {
                        "critic_loss": critic_log_dict["critic_loss"].mean().item(),
                        "critic_grad_norm": critic_log_dict["critic_grad_norm"].mean().item()
                    }
                )

                # Print training info every 5 iterations
                if self.step % 5 == 0:
                    # Get learning rates
                    gen_lr = self.generator_optimizer.param_groups[0]['lr']
                    crit_lr = self.critic_optimizer.param_groups[0]['lr']

                    # Prepare print message
                    info_parts = [
                        f"Step: {self.step}",
                        f"Gen_LR: {gen_lr:.2e}",
                        f"Crit_LR: {crit_lr:.2e}",
                        f"Gen_Loss: {generator_log_dict['generator_loss'].mean().item():.4f}" if TRAIN_GENERATOR else "",
                        f"Crit_Loss: {critic_log_dict['critic_loss'].mean().item():.4f}",
                    ]

                    # Add DMD loss and SCFM loss if available
                    if TRAIN_GENERATOR:
                        # Get DMD loss directly from log dict
                        if "dmd_loss" in generator_log_dict:
                            dmd_loss_val = generator_log_dict["dmd_loss"]
                            if isinstance(dmd_loss_val, torch.Tensor):
                                dmd_loss_val = dmd_loss_val.mean().item() if dmd_loss_val.numel() > 1 else dmd_loss_val.item()
                            else:
                                dmd_loss_val = float(dmd_loss_val)
                            info_parts.append(f"DMD_Loss: {dmd_loss_val:.4f}")

                        # Get SCFM loss directly from log dict
                        if "scfm_loss" in generator_log_dict:
                            scfm_loss_val = generator_log_dict["scfm_loss"]
                            if isinstance(scfm_loss_val, torch.Tensor):
                                scfm_loss_val = scfm_loss_val.mean().item() if scfm_loss_val.numel() > 1 else scfm_loss_val.item()
                            else:
                                scfm_loss_val = float(scfm_loss_val)
                            info_parts.append(f"SCFM_Loss: {scfm_loss_val:.4f}")
                        if "drift_loss" in generator_log_dict:
                            drift_loss_val = generator_log_dict["drift_loss"]
                            if isinstance(drift_loss_val, torch.Tensor):
                                drift_loss_val = drift_loss_val.mean().item() if drift_loss_val.numel() > 1 else drift_loss_val.item()
                            else:
                                drift_loss_val = float(drift_loss_val)
                            if drift_loss_val > 0:
                                info_parts.append(f"Drift_Loss: {drift_loss_val:.4f}")

                        # Add gradient norms
                        if "generator_grad_norm" in generator_log_dict:
                            info_parts.append(f"Gen_GradNorm: {generator_log_dict['generator_grad_norm'].mean().item():.4f}")
                        if "dmdtrain_gradient_norm" in generator_log_dict:
                            info_parts.append(f"DMD_GradNorm: {generator_log_dict['dmdtrain_gradient_norm'].mean().item():.4f}")

                    # Add critic gradient norm
                    if "critic_grad_norm" in critic_log_dict:
                        info_parts.append(f"Crit_GradNorm: {critic_log_dict['critic_grad_norm'].mean().item():.4f}")

                    # Filter out empty strings and print
                    info_parts = [p for p in info_parts if p]
                    print(" | ".join(info_parts))

                if not self.disable_wandb:
                    wandb.log(wandb_loss_dict, step=self.step)

                # TensorBoard: DMD loss, SCFM loss, and main losses
                if self.writer is not None:
                    if TRAIN_GENERATOR:
                        if "dmd_loss" in generator_log_dict:
                            v = generator_log_dict["dmd_loss"]
                            v = v.mean().item() if isinstance(v, torch.Tensor) and v.numel() > 1 else (v.item() if isinstance(v, torch.Tensor) else float(v))
                            self.writer.add_scalar("train/dmd_loss", v, self.step)
                        if "scfm_loss" in generator_log_dict:
                            v = generator_log_dict["scfm_loss"]
                            v = v.mean().item() if isinstance(v, torch.Tensor) and v.numel() > 1 else (v.item() if isinstance(v, torch.Tensor) else float(v))
                            self.writer.add_scalar("train/scfm_loss", v, self.step)
                        if "mmd_loss" in generator_log_dict:
                            v = generator_log_dict["mmd_loss"]
                            v = v.mean().item() if isinstance(v, torch.Tensor) and v.numel() > 1 else (v.item() if isinstance(v, torch.Tensor) else float(v))
                            self.writer.add_scalar("train/mmd_loss", v, self.step)
                        if "drift_loss" in generator_log_dict:
                            v = generator_log_dict["drift_loss"]
                            v = v.mean().item() if isinstance(v, torch.Tensor) and v.numel() > 1 else (v.item() if isinstance(v, torch.Tensor) else float(v))
                            self.writer.add_scalar("train/drift_loss", v, self.step)
                        if "drift_loss_raw" in generator_log_dict:
                            v = generator_log_dict["drift_loss_raw"]
                            v = v.mean().item() if isinstance(v, torch.Tensor) and v.numel() > 1 else (v.item() if isinstance(v, torch.Tensor) else float(v))
                            self.writer.add_scalar("train/drift_loss_raw", v, self.step)
                        if "generator_loss" in generator_log_dict:
                            self.writer.add_scalar("train/generator_loss", generator_log_dict["generator_loss"].mean().item(), self.step)
                    self.writer.add_scalar("train/critic_loss", critic_log_dict["critic_loss"].mean().item(), self.step)

            if self.step % self.config.gc_interval == 0:
                if dist.get_rank() == 0:
                    logging.info("DistGarbageCollector: Running GC.")
                gc.collect()
                torch.cuda.empty_cache()

            if self.is_main_process:
                current_time = time.time()
                if self.previous_time is None:
                    self.previous_time = current_time
                else:
                    if not self.disable_wandb:
                        wandb.log({"per iteration time": current_time - self.previous_time}, step=self.step)
                    self.previous_time = current_time

            self.step += 1
