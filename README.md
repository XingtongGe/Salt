# Salt🧂: Self-Consistent Distribution Matching with Cache-Aware Training for Fast Video Generation

> Add some *Salt* 🧂 to your video generation distillation.

Xingtong Ge<sup>1,2</sup>, Yi Zhang<sup>2</sup>, Yushi Huang<sup>1</sup>,
Dailan He<sup>2</sup>, Xiahong Wang<sup>2</sup>, Bingqi Ma<sup>2</sup>,
Guanglu Song<sup>2</sup>, Yu Liu<sup>2</sup>, Jun Zhang<sup>1</sup>

<sup>1</sup> Hong Kong University of Science and Technology<br>
<sup>2</sup> Vivix Group Limited

[![arXiv](https://img.shields.io/badge/arXiv-2604.03118-b31b1b.svg)](https://arxiv.org/abs/2604.03118)
[![PDF](https://img.shields.io/badge/Paper-PDF-red.svg)](https://arxiv.org/pdf/2604.03118)
[![Project Page](https://img.shields.io/badge/Project-Page-green.svg)](https://xingtongge.github.io/Salt/)
[![Hugging Face](https://img.shields.io/badge/Models%20%26%20Prompts-Hugging%20Face-yellow.svg)](https://huggingface.co/domiso/Salt)

## Abstract

Distilling video generation models to extremely low inference budgets (e.g., 2-4 NFEs) is crucial for real-time deployment, yet remains challenging. Trajectory-style consistency distillation often becomes conservative under complex video dynamics, yielding over-smoothed appearance and weak motion. Distribution matching distillation (DMD) can recover sharp, mode-seeking samples, but its local training signals do not explicitly regularize how denoising updates compose across timesteps, making composed rollouts prone to drift. To overcome this challenge, we propose Self-Consistent Distribution Matching Distillation (SC-DMD), which explicitly regularizes the endpoint-consistent composition of consecutive denoising updates. For real-time autoregressive video generation, we further treat the KV cache as a quality-parameterized condition and propose cache-distribution-aware training. This training scheme applies SC-DMD over multi-step rollouts and introduces a cache-conditioned feature alignment objective that steers low-quality outputs toward high-quality references. Across extensive experiments on both non-autoregressive backbones (e.g., Wan 2.1) and autoregressive real-time paradigms (e.g., Self Forcing, Causal Forcing, and LongLive), Salt consistently improves low-NFE video generation quality while remaining compatible with diverse KV-cache memory mechanisms.

## Code Release

Salt improves few-step video generation with two complementary components:

1. **SC-DMD** regularizes the consistency of direct and composed denoising
   updates.
2. **Cache-conditioned alignment** uses high-quality cache trajectories as
   references. This release provides the spatial-only TRD alignment used by
   the selected autoregressive recipes.

The implementation supports Self Forcing, Causal Forcing, and LongLive
initializations. The canonical mixed-step distribution samples 8-, 4-, and
2-step trajectories with probabilities **0.4 / 0.4 / 0.2**.

## Selected Results

### Text-to-video generation on VBench

**Diffusion models**

| Model | NFE | Total | Quality | Semantic |
| --- | ---: | ---: | ---: | ---: |
| rCM | 4 | 82.73 | 83.65 | **79.04** |
| DMD | 4 | 82.78 | 84.39 | 76.36 |
| Salt (SC-DMD) | 4 | **83.19** | **84.42** | **78.30** |

**Autoregressive models**

| Model | NFE | Total | Quality | Semantic |
| --- | ---: | ---: | ---: | ---: |
| Self Forcing | 4 | 84.20 | 84.74 | **82.05** |
| Salt + Self Forcing | 4 | **84.47** | **85.27** | 81.28 |
| LongLive | 4 | 84.40 | 85.12 | 81.53 |
| Salt + LongLive | 4 | **84.93** | **85.41** | **83.00** |
| Causal Forcing | 4 | 84.62 | 85.41 | 81.47 |
| Salt + Causal Forcing | 4 | **85.08** | **85.96** | **81.59** |
| Salt + Causal Forcing | 2 | **84.80** | **85.63** | **81.49** |

### Image-to-video generation on VBench-I2V

| Method | NFE | I2V Score | Quality | Background Consistency | Motion Smoothness | Dynamic Degree | Imaging Quality | Temporal Flicker |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| PCM | 8 | 93.63 | 78.52 | 97.34 | 98.24 | 30.98 | 70.42 | 97.67 |
| DMD | 4 | 93.09 | 78.89 | 92.79 | 97.99 | 58.46 | 70.35 | 95.21 |
| LightX2V | 4 | 93.50 | 80.92 | 95.87 | 97.89 | 60.33 | 71.67 | 96.30 |
| Salt (SC-DMD) | 4 | **93.90** | 80.86 | **95.97** | **98.37** | 52.85 | **72.16** | **97.41** |
| Salt-alpha | 4 | 93.88 | **81.71** | 95.46 | 98.30 | **68.13** | 72.08 | 96.48 |

### Long-horizon autoregressive generation on VBench-Long

| Backbone | Total | Quality | Semantic |
| --- | ---: | ---: | ---: |
| Causal Forcing | 78.11 | 82.57 | 60.25 |
| Salt + Causal Forcing | **78.28** | 82.15 | **62.77** |
| LongLive | 79.03 | 82.82 | 63.88 |
| Salt + LongLive | **79.27** | **82.90** | **64.74** |

## Qualitative Results

The figure below shows additional qualitative comparisons with the Causal Forcing baseline. Salt better preserves subject identity, object geometry, scene composition, and motion smoothness across challenging examples.

![Qualitative comparison with Causal Forcing](assets/vis_supp.png)

*Figure: Qualitative comparisons with the Causal Forcing baseline. Salt better preserves subject identity, object geometry, scene composition, and temporal coherence across challenging examples, including the umbrella, trombone, reading-girl, and grape cases.*

## Repository layout

```text
configs/
  baseline/             original Self/Causal/LongLive training recipes
  mixed_sc_dmd/         mixed-step SC-DMD recipes
  mixed_sc_dmd_trd/     mixed-step SC-DMD + TRD recipes
  inference/            released-checkpoint inference recipes
model/                  DMD, SC-DMD, TRD, and streaming model losses
pipeline/               Self Forcing, Causal Forcing, and LongLive rollouts
trainer/                distributed distillation training loop
utils/                  data, FSDP, scheduler, and Wan wrappers
wan/                    modified Wan2.1 and causal/LongLive backbones
```

See [configs/README.md](configs/README.md) for the recipe matrix and the
paper-to-code map.

## Requirements

The code targets Linux, Python 3.10, PyTorch 2.4+, and NVIDIA GPUs. Training
uses FSDP and was developed for multi-GPU H100/A100 environments. Inference can
run on fewer GPUs, subject to model size and available memory.

```bash
conda create -n salt python=3.10 -y
conda activate salt

pip install -r requirements.txt
# Recommended on supported NVIDIA GPUs:
pip install flash-attn --no-build-isolation
pip install -e .
```

Optional Wan prompt-extension and USP dependencies are listed in
`requirements-optional.txt`.

## Model and prompt preparation

Download the Wan2.1 backbones into the paths used by the public configs:

```bash
huggingface-cli download Wan-AI/Wan2.1-T2V-1.3B \
  --local-dir wan_models/Wan2.1-T2V-1.3B
huggingface-cli download Wan-AI/Wan2.1-T2V-14B \
  --local-dir wan_models/Wan2.1-T2V-14B
```

Download the released Salt checkpoints and the training prompts together. Run
this command from the repository root so all relative paths match the configs:

```bash
hf download domiso/Salt --local-dir .
```

This creates:

```text
checkpoints/
  salt_cf.pt               Salt + Causal Forcing inference checkpoint
  salt_ll.pt               Salt + LongLive inference checkpoint
prompts/
  vidprom_filtered_extended.txt
```

Checkpoint files are intentionally not stored in Git. Training is data-free
with respect to videos. The included text file contains one training prompt
per line; replace `data_path` if using another prompt collection.

Training from the three initialization families additionally requires
`checkpoints/ode_init.pt`, `checkpoints/causal_ode.pt`, or the optional original
`checkpoints/self_forcing_dmd.pt`, depending on the selected recipe. Obtain
these initialization weights from their Self Forcing/Causal Forcing sources or
update `generator_ckpt` to your local equivalents.

## Training

The examples below use a single 8-GPU node. Adjust `--nproc_per_node`, FSDP
settings, and `total_batch_size` for your environment.

### Mixed-step SC-DMD

```bash
# Self Forcing initialization
torchrun --nproc_per_node=8 train.py \
  --config_path configs/mixed_sc_dmd/self_forcing.yaml \
  --logdir logs/sc_dmd_self_forcing --disable-wandb

# Causal Forcing initialization
torchrun --nproc_per_node=8 train.py \
  --config_path configs/mixed_sc_dmd/causal_forcing.yaml \
  --logdir logs/sc_dmd_causal_forcing --disable-wandb

# LongLive initialization
torchrun --nproc_per_node=8 train.py \
  --config_path configs/mixed_sc_dmd/longlive.yaml \
  --logdir logs/sc_dmd_longlive --disable-wandb
```

### Mixed-step SC-DMD + TRD alignment

```bash
CONFIG=configs/mixed_sc_dmd_trd/self_forcing.yaml \
  bash scripts/train.sh

CONFIG=configs/mixed_sc_dmd_trd/causal_forcing.yaml \
  bash scripts/train.sh

CONFIG=configs/mixed_sc_dmd_trd/longlive.yaml \
  bash scripts/train.sh
```

All six mixed-step recipes use:

```yaml
scfm_step_lists_8: [1000, 875, 750, 625, 500, 375, 250, 125]
scfm_step_lists_4: [1000, 750, 500, 250]
scfm_step_lists_2: [1000, 500]
scfm_step_probs: [0.4, 0.4, 0.2]  # 8-step / 4-step / 2-step
```

## Inference

Salt + Causal Forcing uses the EMA generator, while the selected Salt +
LongLive checkpoint uses the regular generator:

```bash
python inference.py \
  --config_path configs/inference/salt_causal_forcing.yaml \
  --checkpoint_path checkpoints/salt_cf.pt \
  --data_path prompts/example_prompts.txt \
  --output_folder outputs/salt_cf \
  --use_ema

python inference.py \
  --config_path configs/inference/salt_longlive.yaml \
  --checkpoint_path checkpoints/salt_ll.pt \
  --data_path prompts/example_prompts.txt \
  --output_folder outputs/salt_ll
```

## Acknowledgements

This codebase builds on
[Self Forcing](https://github.com/guandeh17/Self-Forcing),
[CausVid](https://github.com/tianweiy/CausVid), and
[Wan2.1](https://github.com/Wan-Video/Wan2.1). Please also cite the original
projects when using their components.

## License

The repository contains components under more than one license. Most code is
provided under Apache-2.0, while
`wan/modules/causal_model_longlive.py` is marked CC-BY-NC-SA-4.0. See
[THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md) before redistribution or
commercial use. Model weights are governed by their respective upstream
licenses.

## Citation

```bibtex
@article{ge2026salt,
  title={Salt: Self-consistent distribution matching with cache-aware training for fast video generation},
  author={Ge, Xingtong and Zhang, Yi and Huang, Yushi and He, Dailan and Wang, Xiahong and Ma, Bingqi and Song, Guanglu and Liu, Yu and Zhang, Jun},
  journal={arXiv preprint arXiv:2604.03118},
  year={2026}
}
```
