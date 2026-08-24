"""
LongLive-style training pipeline: explicitly sets local_attn_size and sink_size
on the generator before inference, and uses LongLive KV cache size.
Use this pipeline when training with LongLive init (e.g. 5s init with local + sink).
"""
from typing import List, Optional
from .self_forcing_training import SelfForcingTrainingPipeline


class LongLiveTrainingPipeline(SelfForcingTrainingPipeline):
    """
    Same as SelfForcingTrainingPipeline but with LongLive-specific behavior:
    - Accepts local_attn_size, sink_size, slice_last_frames, num_training_frames.
    - KV cache size = min(local_attn_size + slice_last_frames, num_training_frames) * frame_seq_length.
    - Before temporal denoising, sets generator.model.local_attn_size, each block's
      self_attn.sink_size, and _set_all_modules_max_attention_size.
    """

    def __init__(self,
                 denoising_step_list: List[int],
                 scheduler,
                 generator,
                 num_frame_per_block=3,
                 independent_first_frame: bool = False,
                 same_step_across_blocks: bool = False,
                 last_step_only: bool = False,
                 num_max_frames: int = 21,
                 context_noise: int = 0,
                 loop_rope_training: bool = False,
                 *,
                 local_attn_size=-1,
                 sink_size=0,
                 slice_last_frames: int = 21,
                 num_training_frames: Optional[int] = None,
                 **kwargs):
        # Pass through to parent without LongLive kwargs so parent uses default kv_cache_size
        super().__init__(
            denoising_step_list=denoising_step_list,
            scheduler=scheduler,
            generator=generator,
            num_frame_per_block=num_frame_per_block,
            independent_first_frame=independent_first_frame,
            same_step_across_blocks=same_step_across_blocks,
            last_step_only=last_step_only,
            num_max_frames=num_max_frames,
            context_noise=context_noise,
            loop_rope_training=loop_rope_training,
            **kwargs,
        )
        num_training_frames = num_training_frames if num_training_frames is not None else num_max_frames
        self.local_attn_size = local_attn_size
        self.sink_size = sink_size
        if not isinstance(self.local_attn_size, int) and hasattr(self.local_attn_size, "__iter__"):
            self.local_attn_size = list(self.local_attn_size)
        # LongLive KV cache size
        if isinstance(self.local_attn_size, (list, tuple)):
            base = int(max(self.local_attn_size)) if len(self.local_attn_size) > 0 else -1
            kv_frames = min(base + slice_last_frames, num_training_frames) if base >= 0 else num_max_frames
        else:
            base = int(self.local_attn_size)
            kv_frames = min(base + slice_last_frames, num_training_frames) if base >= 0 else num_max_frames
        self.kv_cache_size = kv_frames * self.frame_seq_length

    def _set_all_modules_max_attention_size(self, local_attn_size_value: int):
        """
        Set a unified upper bound for all submodules that contain the max_attention_size attribute.
        local_attn_size_value == -1 indicates global attention (use Wan's default token limit 32760).
        Otherwise set to local_attn_size_value * frame_seq_length.
        """
        if isinstance(local_attn_size_value, (list, tuple)):
            raise ValueError("_set_all_modules_max_attention_size expects an int, got list/tuple.")

        if int(local_attn_size_value) == -1:
            target_size = 32760
        else:
            target_size = int(local_attn_size_value) * self.frame_seq_length

        if hasattr(self.generator.model, "max_attention_size"):
            setattr(self.generator.model, "max_attention_size", target_size)
        for name, module in self.generator.model.named_modules():
            if hasattr(module, "max_attention_size"):
                try:
                    setattr(module, "max_attention_size", target_size)
                except Exception:
                    pass

    def inference_with_trajectory(self, noise, initial_latent=None, return_sim_step=False, **conditional_dict):
        # LongLive: set local_attn_size and sink on generator before temporal denoising
        local_attn_value = (
            int(max(self.local_attn_size))
            if isinstance(self.local_attn_size, (list, tuple)) and len(self.local_attn_size) > 0
            else int(self.local_attn_size)
        )
        if local_attn_value != -1:
            self.generator.model.local_attn_size = local_attn_value
            self._set_all_modules_max_attention_size(local_attn_value)
        if self.sink_size and hasattr(self.generator.model, "blocks"):
            for block in self.generator.model.blocks:
                if hasattr(block, "self_attn") and hasattr(block.self_attn, "sink_size"):
                    block.self_attn.sink_size = int(self.sink_size)

        return super().inference_with_trajectory(
            noise=noise,
            initial_latent=initial_latent,
            return_sim_step=return_sim_step,
            **conditional_dict,
        )
