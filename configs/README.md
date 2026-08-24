# Configuration matrix

| Recipe | Self Forcing | Causal Forcing | LongLive |
| --- | --- | --- | --- |
| Original baseline | `baseline/self_forcing.yaml` | `baseline/causal_forcing.yaml` | `baseline/longlive.yaml` |
| Mixed-step SC-DMD | `mixed_sc_dmd/self_forcing.yaml` | `mixed_sc_dmd/causal_forcing.yaml` | `mixed_sc_dmd/longlive.yaml` |
| Mixed-step SC-DMD + TRD | `mixed_sc_dmd_trd/self_forcing.yaml` | `mixed_sc_dmd_trd/causal_forcing.yaml` | `mixed_sc_dmd_trd/longlive.yaml` |

The mixed-step probability order is 8-step / 4-step / 2-step, and every
canonical mixed recipe uses `[0.4, 0.4, 0.2]`.

LongLive is selected by `model_kwargs.local_attn_size: 12` together with
`sink_size: 3`; there is no separate `longlive: true` flag. Causal Forcing is
selected through the causal ODE initialization checkpoint.

TRD recipes set `ref_align_loss: trd`, `mmd_use_ref: true`, and
`mmd_weight: 0.1`. The `mmd_*` names are retained for compatibility with the
current implementation even when the selected alignment loss is TRD.

## Paper-to-code map

- Mixed-step sampling and SC-DMD: `model/sc_dmd.py`
- Direct/composed endpoints and SCFM targets: `pipeline/sc_training.py`
- Spatial-only TRD alignment: `model/dmd.py`
- Self Forcing rollout: `pipeline/self_forcing_training.py`
- LongLive SC-DMD rollout: `pipeline/longlive_sc_training.py`
- Original LongLive streaming rollout: `model/streaming_dmd.py` and
  `pipeline/streaming_training.py`
- Trainer/model dispatch: `trainer/distillation.py`

All model, checkpoint, and prompt paths are repository-relative. Checkpoint
weights and large prompt collections are not included in Git.
