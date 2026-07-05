# traffic_sign

Training-free, visibility-aware routing of classical operators for traffic
sign recognition under degraded visual conditions (**VA-Adaptive**).

This repository accompanies the paper *"Visibility-aware adaptive routing of
classical operators for traffic sign recognition under degraded visual
conditions"* (under review). It contains the implementation of VA-Adaptive,
the evaluation and analysis scripts, and the per-image outputs that reproduce
every table and figure in the paper.

## Method in one paragraph

For each input image, VA-Adaptive computes three inexpensive visibility
statistics (mean brightness, standard-deviation contrast, and Canny edge
density) and routes the image to one of four classical actions, namely gamma
correction, contrast-limited adaptive histogram equalization (CLAHE), linear
contrast stretch, or passthrough, according to a priority-ordered rule whose
four thresholds are calibrated once on clean training data. The recognition
network is frozen and the routing module contains no learnable parameters, so
no part of the procedure is trained on degraded data.

## Repository layout

```
traffic_sign/
├── src/                       code
│   ├── enhance.py             the VA-Adaptive routing rule and classical operators
│   ├── model.py               the CompactCNN classifier
│   ├── train_backbone_ablation.py
│   ├── evaluate_backbone_ablation.py
│   ├── F_master_sweep_cache.py    CURE-TSR prediction cache (classical methods)
│   ├── G_synth_router_data.py     synthetic GTSRB features for the learned routers
│   ├── H_learned_router.py        logistic-regression and MLP selectors
│   ├── I_gtsdb_eval.py            GTSDB cross-dataset transfer
│   ├── J_local_deep_eval.py       AdaIR and HVI-CIDNet on real CURE-TSR
│   ├── K_merge_results.py         merges all runs into the final CSV/JSON products
│   ├── L_timing_enhance_only.py   enhancement-stage latency
│   ├── M_significance_addenda.py  GTSDB and learned-router significance
│   ├── N_backbone_significance.py backbone-ablation significance
│   ├── make_figures.py            Figures 1, 4, 5, 6, 7, 11
│   └── make_fig10.py              Figure 10 (deep-model qualitative panels)
├── outputs_revision/          per-image outputs that reproduce the tables and figures
│   ├── merged_per_image.csv
│   ├── cells_per_method.csv
│   ├── extended_results.json
│   ├── L_timing_enhance_only.results.json
│   ├── M_significance_addenda.results.json
│   └── N_backbone_significance.results.json
├── results/
│   ├── thresholds.json            the four calibrated routing thresholds
│   ├── backbone_ablation.csv
│   └── backbone_ablation.json
├── models/                    small classifier weights (see note below)
├── README.md
└── LICENSE
```

## Requirements

Python 3.10 or later, with:

```
pip install numpy opencv-python torch torchvision matplotlib scikit-learn safetensors
```

The experiments in the paper were run on CPU (Intel Core Ultra 7 258V) with
PyTorch 2.5.1 and OpenCV 4.13. A GPU is not required.

## Datasets (download separately; not included here)

The three benchmarks are publicly available from their authors and are not
redistributed in this repository.

- **GTSRB** (training and synthetic-degradation evaluation): https://benchmark.ini.rub.de/gtsrb_news.html
- **GTSDB** (cross-dataset transfer): https://benchmark.ini.rub.de/gtsdb_news.html
- **CURE-TSR** (real-degradation evaluation): https://github.com/olivesgatech/CURE-TSR

Please verify the current download locations on the authors' pages.

## Model weights

- The small **classifier** weights used in the paper (CompactCNN, and the
  ShuffleNetV2 and MobileNetV2 backbones for the ablation) are included under
  `models/`.
- The **learned restoration** weights are large and are not redistributed here.
  Download them from the official repositories and place them under `models/`:
  **AdaIR** and **HVI-CIDNet** from their respective author releases. The
  scripts read these with the authors' released checkpoints and apply them
  zero-shot, with no retraining.

## Reproducing the results

### Figures (no dataset or model needed)

`make_figures.py` regenerates Figures 1, 4, 5, 6, 7, and 11 directly from the
files in `outputs_revision/`. Edit `PROJECT_ROOT` at the top of the script if
your checkout is not at the default path, then:

```
cd src
python make_figures.py            # all six
python make_figures.py 5 11       # a subset, by figure number
```

Figure 10 re-runs the two deep models on real CURE-TSR crops and therefore
needs the CURE-TSR dataset and the AdaIR / HVI-CIDNet weights:

```
python make_fig10.py --cidnet-weight <path-to-cidnet-weights>
```

### Tables (no dataset or model needed)

The per-image outputs in `outputs_revision/` reproduce every table:
`merged_per_image.csv` holds the per-image predictions of the aligned methods,
`cells_per_method.csv` the per-cell accuracies, and `extended_results.json`
the aggregated metrics, oracles, and parameter counts. The paired confidence
intervals and exact McNemar tests are recomputed by `M_significance_addenda.py`
and `N_backbone_significance.py`, which emit the corresponding `results.json`
files.

### Full pipeline from scratch (datasets and weights required)

With the datasets and weights in place, the raw results are produced by running
the numbered scripts in order: `F` (classical methods on CURE-TSR), `G` and `H`
(learned routers), `I` (GTSDB), `J` (AdaIR and HVI-CIDNet), then `K` to merge
everything into the products under `outputs_revision/`, and `L`, `M`, `N` for
latency and significance. The backbone ablation is produced by
`train_backbone_ablation.py` followed by `evaluate_backbone_ablation.py`.

Note: the numbered generation scripts (`F`–`N`) contain machine-specific paths
to the local dataset and weight locations; adjust the paths at the top of each
script to your environment. The two figure scripts use a single configurable
`PROJECT_ROOT` variable.

## Citation

If you use this code, please cite the paper. A BibTeX entry will be added here
upon publication.

## License

Released under the MIT License. See `LICENSE`.
