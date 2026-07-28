# Measuring the Capability Boundary of Image Restoration for Traffic Sign Recognition

Code, evaluation scripts and result files for the manuscript

> Xiang Ji, "Measuring the Capability Boundary of Image Restoration for
> Traffic Sign Recognition," submitted to *IEEE Transactions on Intelligent
> Transportation Systems*.

The article asks three questions in order: which degradations of a traffic sign
crop are recoverable at all, whether one restoration mechanism can recover them
all, and whether the resulting recovery is worth what it costs. Every number,
figure and table it prints can be recomputed from the files in this repository.

---

## Where to put this repository

The scripts resolve their inputs against a fixed project root:

```python
PROJECT_ROOT = Path(r"D:\Project\traffic_sign")
```

**Clone or unpack this repository to `D:\Project\traffic_sign`.** With the
repository at that path every command below runs as written.

If you cannot use that path, nothing needs editing: the path is only a default.
Every script takes its inputs as command-line arguments, so an explicit path
overrides it. For example

```
python FIG01_collapse_and_boundary.py --merged /your/path/merged_per_image.csv ^
                                      --dcp    /your/path/dcp_cure.csv ^
                                      --outdir /your/path/figures
```

Run any script with `--help` to see what it accepts.

---

## Requirements

Python 3.11 or later.

```
pip install -r requirements.txt
```

`torch` and `torchvision` are needed only to run the classifier or the learned
restorers. Redrawing the figures and rebuilding the tables from the result files
already in this repository needs neither.

---

## Layout

```
src/                 experiment, evaluation and figure scripts
outputs_revision/    the result files every figure and table is computed from
  fig06_samples/     the eighteen crops Fig. 6 displays, plus manifest.json
  figures/           the nine figures as .pdf, .png and .svg
models/              mbnetv3_baseline.pth   the frozen classifier
                     params_learned.json    parameter counts for Table I
```

Two directories are **not** in this repository and have to be obtained
separately, both because of their size and because they are not ours to
redistribute: the datasets, and the weights of the five learned restorers. See
*Data and third-party weights* below.

---

## Reproducing the article without re-running the experiments

The result files in `outputs_revision/` are the ones the article was written
from. With them, the nine figures, the three tables and every number in
Sections IV and V can be rebuilt in a few minutes, without the datasets and
without the third-party weights.

```
cd src

python FIG01_collapse_and_boundary.py
python FIG02_noise_injection_oracle.py
python FIG03_capability_boundary.py
python FIG04_operator_selection_oracle.py
python FIG05_rule_and_its_worth.py
python FIG06_what_the_classifier_sees.py
python FIG07_accuracy_cost.py
python FIG08_cascade.py
python FIG09_noise_robustness.py

python TABLES_main.py --params-json ../models/params_learned.json
```

Figures are written to `outputs_revision/figures/`. Set the environment variable
`FIGPDF=1` to also emit PDF; the default is PNG and SVG.

`TABLES_main.py` prints Tables I, II and III to the console. It also writes
`_table1.json` and `_tabledata.json`, which fed the manuscript generator used
while the article was still in Word; the article is now typeset in LaTeX and
nothing reads those two files. The console output is the part to read.

`--params-json` supplies the parameter counts of the five learned restorers, so
that Table I can be built without their weight files. Drop the flag if you have
downloaded the weights and want the counts read from them directly.

### Checking the numbers

```
cd src
python check_numbers.py ../outputs_revision/ZA_deep_vs_injection.json ^
                        ../outputs_revision/merged_per_image.csv ^
                        ../outputs_revision/dcp_cure.csv
```

This recomputes, from the data alone, everything Sections IV and V state: the
forty-eight validated cells, the 432 tests in the Holm family, the seventeen
cleared combinations and their distribution over the twelve challenges, the four
gain figures for the challenges that clear nothing, all twelve rows of Table II,
the three operator-selection ceilings and the twelve per-challenge headroom bars
of Fig. 4. Each value is compared against what the article prints. The script
exits non-zero if any of them differs.

---

## Running the pipeline from scratch

This needs the datasets and, for the learned restorers, their weights.

```
cd src

python train_baseline.py                 # classifier, clean GTSRB only
python K_merge_results.py                # -> merged_per_image.csv
python Q_dcp_branch.py                   # -> dcp_cure.csv
python O_eval_three_locally.py           # the five learned restorers, same crops
python Z_injection_definitive.py         # -> Z_all12_per_image.csv
python ZA_deep_vs_injection.py           # -> ZA_deep_vs_injection.json
python U_noise_selector.py               # -> U_noise_selector.csv
python N_timing_stable.py                # -> N_timing_stable.results.json
python M_multiseed_cascade.py            # -> M_multiseed_cascade.json
python EXPORT_fig06_samples.py           # -> fig06_samples/
```

Then the figure and table commands above.

`N_timing_stable.py` measures latency and its results depend on the machine. The
figures in the article were timed on one CPU thread with a batch of one; each
crop was timed five times and the fastest reading kept, twenty-five times for
front ends under five milliseconds. Reproducing the latency column on other
hardware will give other numbers; the accuracy figures will not change.

`EXPORT_fig06_samples.py` chooses the six crops of Fig. 6 by rule from the merged
file rather than from a written-down list, re-classifies every crop it exports,
and aborts if any of them fails to reproduce the prediction already on record.
The eighteen crops it produced are already in `outputs_revision/fig06_samples/`,
so Fig. 6 can be drawn without running it.

---

## Data and third-party weights

**CURE-TSR.** Twelve controlled degradations at five severities applied to
traffic sign crops. Available from the authors of the dataset at
<https://github.com/olivesgatech/CURE-TSR>. `src/download_cure_tsr.py` fetches
it; place it under `datasets/CURE-TSR/`. All experiments use the real subset,
pooling the `Real_Train` and `Real_Test` splits as the published protocol does.

**GTSRB.** Used only to train the classifier. Place it under `data/gtsrb/`.

**The five learned restorers.** Their weights belong to their authors and are
not redistributed here. Download them from the official releases:

| model | source |
|---|---|
| PromptIR | <https://github.com/va1shn9v/PromptIR> |
| AdaIR | <https://github.com/c-yn/AdaIR> |
| CIDNet | <https://github.com/Fediory/HVI-CIDNet> |
| Zero-DCE | <https://github.com/Li-Chongyi/Zero-DCE> |
| FFA-Net | <https://github.com/zhilin007/FFA-Net> |

Place them in `models/`. The file names the scripts expect are in
`src/revision_utils.py` and in each script's own `--*-weight` default.

---

## Notes

**The classifier's file name.** `models/mbnetv3_baseline.pth` holds the compact
convolutional network described in Section III-B: five 3x3 convolutions in three
blocks of widths 32, 64 and 128, 145,291 trainable parameters. It is defined in
`src/model.py` as `CompactCNN`. The file name dates from an earlier version of
this project and does **not** mean MobileNetV3. The weights are frozen for every
experiment in the article; no CURE-TSR crop took part in training.

**Averaging.** A front end's degraded average is the mean over the sixty
(challenge, severity) cells of the accuracy within each cell, not the mean over
crops. Cells hold different numbers of crops and a mean over crops would let the
larger cells decide the answer.

**Identifying a crop.** Pooling `Real_Train` and `Real_Test` makes a filename
non-unique: 21,716 filenames occur in both, and the two occurrences are
different crops. Every record therefore carries an occurrence index, and the
pair of filename and occurrence is the key used throughout.

---

## Citation

```bibtex
@article{ji2026capability,
  author  = {Ji, Xiang},
  title   = {Measuring the Capability Boundary of Image Restoration
             for Traffic Sign Recognition},
  journal = {IEEE Transactions on Intelligent Transportation Systems},
  note    = {Submitted},
  year    = {2026}
}
```

## License

The code in this repository is released under the MIT License; see `LICENSE`.
The datasets and the weights of the five learned restorers are covered by their
own licences and are not redistributed here.
