# Third-party code and licenses

This repository is derived from and incorporates code from several projects.
Retain upstream copyright notices when redistributing modified files.

| Component | Location | Source | License |
| --- | --- | --- | --- |
| Self Forcing / CausVid-derived training code | `model/`, `pipeline/`, `trainer/`, `utils/` | https://github.com/guandeh17/Self-Forcing and https://github.com/tianweiy/CausVid | Apache-2.0 according to the upstream repositories |
| Wan2.1 | `wan/` except the item below | https://github.com/Wan-Video/Wan2.1 | Apache-2.0 according to upstream |
| LongLive causal backbone | `wan/modules/causal_model_longlive.py` | Adopted from Self Forcing as stated in the file header | [CC-BY-NC-SA-4.0](LICENSES/CC-BY-NC-SA-4.0.txt) (file-level SPDX marker) |
| FramePack memory helper | `demo_utils/memory.py` | https://github.com/lllyasviel/FramePack | Apache-2.0 according to the file header |

The root `LICENSE` contains Apache-2.0 for the portions of this repository that
the Salt authors are entitled to license under those terms. It does not
override file-level or upstream licenses. In particular, the LongLive file's
NonCommercial and ShareAlike conditions may affect redistribution and use of a
build that includes that component.

Wan, Self Forcing, Causal Forcing, and Salt model weights are separate works
and may have terms different from the source code. Consult each checkpoint's
source before redistribution or commercial use.
