# TransAdapter Baseline

We use the original [TransAdapter](https://github.com/enesdoruk/TransAdapter)
implementation directly, no custom code was written for this baseline. This
folder only contains the exact commands used to run it on all 6 source-target
combinations of the wood chip moisture dataset.

## Setup

1. Clone the original repo:
   ```bash
   git clone https://github.com/enesdoruk/TransAdapter.git
   ```
2. Follow their setup instructions (dependencies, pretrained Swin backbone
   checkpoint, etc.) — not repeated here since it's already documented there.
3. Prepare `source_list.txt`, `target_list.txt`, and `test_list_target.txt`
   for each of the 6 source-target combinations, following the format their
   repo expects.

## Running

All 6 runs use identical hyperparameters, only `--dataset` and `--output_dir`
change per combination. See `run_transadapter_commands.txt` in this folder
for the exact command used for each of the 6 combinations (1→2, 2→1, 1→3,
3→1, 2→3, 3→2).

Run each command from inside the cloned `TransAdapter` repo.

## Notes

- No code from the original repo is duplicated here, only the commands
  needed to reproduce our exact runs.
- Results reported in the paper for this baseline came directly from these
  runs.
