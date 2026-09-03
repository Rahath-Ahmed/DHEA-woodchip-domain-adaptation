# DHEA: Dual-Head Encoder Adaptation for Wood Chip Moisture Assessment

This repository contains the official code for our paper, "DHEA (Dual-Head
Encoder Adaptation): A Dual-Encoder Domain Adaptation Network for Wood Chip
Moisture Classification." DHEA is an unsupervised domain adaptation (UDA)
method that predicts wood chip moisture class (Dry, Medium, Wet) on an
unlabeled target domain, along with UDA Score, a label-free metric for
selecting the best training checkpoint without access to target labels.

## Abstract

Wood chips are raw material in industries such as pelleting mills, bio-refineries, and paper mills, where moisture content affects cost and quality. The standard method, oven-drying, takes hours, too slow for real-time decisions. Studies have shown machine learning models can estimate moisture from images faster, but a model trained on one source performs poorly on another due to differences in lighting, shape, and texture, a problem called domain shift. Labeling every new source is impractical, since each label needs oven-drying. Unsupervised domain adaptation (UDA) addresses this, letting a model trained on a labeled source work on an unlabeled one. However, adversarial UDA methods discard domain-specific information instead of using it, a gap this study addresses. We propose DHEA (Dual-Head
Encoder Adaptation), a dual-encoder network that predicts wood chip moisture classes in an unseen target domain without sacrificing accuracy. DHEA uses a Common Encoder, which suppresses domain information to classify moisture using domain-invarient features, and a Hetero Encoder, which captures that discarded signal, easing pressure on the Common Encoder. Since target labels are never available in UDA, there is no way to pick the best checkpoint; we propose UDA Score, a label-free checkpoint-selection proxy. DHEA beats eight baselines across six source-target pairs, reaching 74.8% target accuracy, nearly 13 points above the best baseline. UDA Score picks
a checkpoint within 1.75% of the best result, closer than three other proxies. These results suggest preserving domain-specific information, rather than discarding it, is a promising direction for domain adaptation.

## Repository Structure

```
.
├── dhea/               DHEA model and training code, one notebook per source-target combination (1_2, 2_1, 1_3, 3_1, 2_3, 3_2)
├── baselines/
│   ├── source_only/    Source-only baseline (no adaptation), all 6 combinations
│   ├── adapt_library/  DANN, ADDA, CDAN, MDD, DeepCORAL baselines, built on the Adapt library, all 6 combinations each                   
│   ├── sdat/           CDAN+SDAT baseline, all 6 combinations. Depends on the original SDAT repository (see baselines/sdat/README.md)
│   └── transadapter/   Commands used to run the original TransAdapter implementation on all 6 combinations (see
│                       baselines/transadapter/README.md)
├── proxy_metrics/      Comparison of UDA Score against other checkpoint selection proxies
├── ablation/           Ablation study on DHEA's loss components
├── datasets/           Dataset loading code, shared across all experiments
├── LICENSE
└── README.md
```

Source-target combinations are numbered 1, 2, and 3, corresponding to the
three wood chip data sources used in the paper. For example, `1_2` refers to
source domain 1 adapted to target domain 2.

## Requirements

- Python 3.8+
- PyTorch
- torchvision
- numpy, scikit-learn, matplotlib, seaborn

Baseline code additionally depends on the
[Adapt library](https://github.com/adapt-python/adapt) (DANN, ADDA, CDAN,
MDD, DeepCORAL), and on external repositories for SDAT and TransAdapter, see
the README files inside their respective folders for setup instructions.

## Dataset

The wood chip moisture dataset used in this work is not included in this
repository but can be made available from the corresponding author upon reasonable request.

## Running the Code

Each notebook or script under `dhea/` and `baselines/` corresponds to one
source-target combination and can be run independently. All experiments
share the same three-class label set (Dry, Medium, Wet) and the same data
splits, defined in `datasets/`.

## Results

DHEA is evaluated against eight baselines (source-only, DANN, ADDA, CDAN,
CDAN+SDAT, MDD, DeepCORAL, TransAdapter) across all six source-target
combinations. DHEA achieves an average target accuracy of 74.8%, an
improvement of nearly 13 points over the best-performing baseline. UDA
Score, our proposed checkpoint selection method, selects a checkpoint
within 1.75% of the best achievable target accuracy, without using any
target labels.

Full per-combination results, ablation results, and proxy metric
comparisons are reported in the paper.

## Citation

If you use this code, please cite our paper:

```
[Add citation once the paper is published, e.g. BibTeX entry]
```

## License

This project is released under the MIT License, see the LICENSE file for
details.

## Acknowledgements

This work was supported by the Louisiana Board of Regents.

## Contact

For questions about this code, please open an issue on this repository or
contact [add your email address].
