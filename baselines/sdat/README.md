# SDAT (CDAN + SDAT) Baseline

Scripts for running the CDAN w/ SDAT baseline on all 6 source-target combinations
of the wood chip moisture dataset. These follow the original method from
[val-iisc/SDAT](https://github.com/val-iisc/SDAT) (`examples/cdan_sdat.py`),
adapted to load our wood chip data instead of Office-Home.

## Setup

1. Clone the original SDAT repo:
   ```bash
   git clone https://github.com/val-iisc/SDAT.git
   ```
2. Copy all 6 scripts in this folder into `SDAT/examples/`. They rely on
   SDAT's own modules (`dalib`, `common`), so they must run from inside that folder.
3. Make sure the shared dataset files (`dataset_1_2.py`, `dataset_2_1.py`, etc.)
   from this repo's `/datasets` folder are present, they're imported automatically
   via a relative path, so no manual copying is needed as long as the overall
   repo structure is kept intact.

## Files

| Script | Source → Target |
|---|---|
| `train_cdan_sdat_woodchip_1_2.py` | 1 → 2 |
| `train_cdan_sdat_woodchip_2_1.py` | 2 → 1 |
| `train_cdan_sdat_woodchip_1_3.py` | 1 → 3 |
| `train_cdan_sdat_woodchip_3_1.py` | 3 → 1 |
| `train_cdan_sdat_woodchip_2_3.py` | 2 → 3 |
| `train_cdan_sdat_woodchip_3_2.py` | 3 → 2 |

## Running

From inside `SDAT/examples/`:
```bash
python train_cdan_sdat_woodchip_1_2.py
```

Each script trains for 10 epochs, saves the best checkpoint (by source
validation accuracy), then runs full evaluation on source and target,
producing a classification report, confusion matrix, prediction visualization
grids, and saved `.npy` arrays of predictions/labels. Output is written to a
`SAVE_DIR` set near the top of each script, update this path to your own
local output folder before running.

## Notes

- All 6 scripts share the same hyperparameters (ResNet-50 backbone, SAM
  optimizer with `rho=0.02`, CDAN adversarial loss); only the dataset and
  output folder differ per combination.
- These scripts are ours; the underlying SDAT method and library code are
  from the original authors and are not included here, see their repo above.
