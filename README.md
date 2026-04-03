# Salt. 🧂

### Salt: Self-Consistent Distribution Matching with Cache-Aware Training for Fast Video Generation

> Add some *Salt* 🧂 to your video generation distillation

Xingtong Ge<sup>1,2</sup>, Yi Zhang<sup>2</sup>, Yushi Huang<sup>1</sup>, Dailan He<sup>2</sup>, Xiahong Wang<sup>2</sup>, Bingqi Ma<sup>2</sup>, Guanglu Song<sup>2</sup>, Yu Liu<sup>2</sup>, Jun Zhang<sup>1</sup>

<sup>1</sup> Hong Kong University of Science and Technology  
<sup>2</sup> Vivix Group Limited

## Abstract

Distilling video generation models to extremely low inference budgets (e.g., 2-4 NFEs) is crucial for real-time deployment, yet remains challenging. Trajectory-style consistency distillation often becomes conservative under complex video dynamics, yielding over-smoothed appearance and weak motion. Distribution matching distillation (DMD) can recover sharp, mode-seeking samples, but its local training signals do not explicitly regularize how denoising updates compose across timesteps, making composed rollouts prone to drift. To overcome this challenge, we propose Self-Consistent Distribution Matching Distillation (SC-DMD), which explicitly regularizes the endpoint-consistent composition of consecutive denoising updates. For real-time autoregressive video generation, we further treat the KV cache as a quality-parameterized condition and propose cache-distribution-aware training. This training scheme applies SC-DMD over multi-step rollouts and introduces a cache-conditioned feature alignment objective that steers low-quality outputs toward high-quality references. Across extensive experiments on both non-autoregressive backbones (e.g., Wan 2.1) and autoregressive real-time paradigms (e.g., Self Forcing, Causal Forcing, and LongLive), Salt consistently improves low-NFE video generation quality while remaining compatible with diverse KV-cache memory mechanisms.

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
