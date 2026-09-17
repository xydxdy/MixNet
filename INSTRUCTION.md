# Homework — Build and Benchmark Your Own MI-EEG Pipeline

**Dataset:** BCI Competition IV-2a (BCIC2a) · **Setting:** subject-dependent · **Task:** 2-class motor imagery (left hand vs right hand)

You will design your own EEG preprocessing pipeline and your own neural
network, plug both into this repository, and benchmark them against **MixNet**
under exactly the same protocol — same 9 subjects, same 5 folds, same test
session, same metrics.

---

## 1. Learning objectives

By the end of this assignment you should be able to:

1. Implement a feature-extraction pipeline and a deep model against a fixed
   experimental API instead of a one-off notebook.
2. Run a controlled comparison against a strong published baseline and report
   it honestly, including where your method loses.
3. Analyse a result instead of only reporting it: read the per-subject table
   against the paired significance test, and form a hypothesis for the pattern you
   see.

---

## 2. The dataset

| Property | Value |
|---|---|
| Subjects | 9 (`A01`–`A09`) |
| Sessions | `T` (training) and `E` (evaluation), recorded on different days |
| Classes used | 2 — left hand (`0`), right hand (`1`) |
| Trials | 144 per session per subject (72 per class) |
| Channels | 20 pre-selected motor-cortex channels (see `mixnet/preprocessing/config.py`) |
| Original rate | 250 Hz, downsampled to 100 Hz by the provided loader |
| MI window | 4 S (2 s → 6 s after the cue) |

**Subject-dependent** means each subject gets their own model. For every
subject, session `T` is split into train/validation with 5-fold stratified
cross-validation, and session `E` is the **untouched test set** of every fold.
Never train on session `E`.

---

## 3. What is given vs what you write

| File | Status |
|---|---|
| `mixnet/preprocessing/BCIC2a/HW_prep.py` | **YOU WRITE** — 2 functions |
| `mixnet/models/HW_Net.py` | **YOU WRITE** — 2 functions |
| `experiments/configs/HW_Net.py` | **YOU TUNE** — shapes + hyper-parameters |
| `experiments/prep_HW.py` | given — runs your pipeline (NOTE: add your own parameters) |
| `experiments/run_HW_Net.py` | given — 5-fold training/eval driver |
| `experiments/benchmark.py` | given — aggregation + statistics, no edits needed |
| everything under `mixnet/` else | given — do not modify |

Each file you edit starts with a header block explaining the contract it must
satisfy. Read those headers first — they answer most questions.

---

## 4. Setup

> **Run every command below from the `experiments/` directory** unless stated
> otherwise. The scripts resolve `datasets/` and `logs/` relative to the
> current working directory, and both are git-ignored.

### 4.1 Prerequisites

- Python **3.8.10** (the package declares `>=3.8, <=3.10.4`)
- An NVIDIA GPU with CUDA — `mixnet/models/base.py` reads GPU memory during
  evaluation, so a GPU is effectively required
- `git`

### 4.2 Fork and clone

Fork `https://github.com/Max-Phairot-A/MixNet` to your own GitHub account so
you can commit your work, then:

```bash
git clone https://github.com/<your-github-username>/MixNet.git
cd MixNet
git switch homework
```

Work on that branch.

### 4.3 Create the environment

The reference environment is the TensorFlow 2.7.0 GPU image:

```bash
docker pull tensorflow/tensorflow:2.7.0-gpu
docker run -ti --gpus all --name mixnet_container -v $(pwd):/workspace \
    docker.io/tensorflow/tensorflow:2.7.0-gpu bash
cd /workspace
```

Or with conda, if you already have a working CUDA setup:

```bash
conda create -n mixnet python=3.8.10 -y
conda activate mixnet
pip install tensorflow-gpu==2.7.0
```

### 4.4 Install the package — read this carefully

```bash
pip install -r requirements.txt
pip install -e .         
```

> ### Use `pip install -e .`
>
> `pip install -e .` installs in *editable* mode: it links to your working
> copy, so your edits take effect immediately. If you must use `pip install .`,
> you have to re-run it after **every** change under `mixnet/`.

Verify the install points at your clone:

```bash
python -c "import mixnet; print(mixnet.__file__)"
# should print <your clone>/mixnet/__init__.py — NOT a site-packages path
```

### 4.5 Download the data

```bash
cd experiments
python download_datasets.py --dataset 'BCIC2a'
```

This writes `experiments/datasets/BCIC2a/raw/A0{1..9}{T,E}.mat`. If the
download fails, fetch the files manually from
<http://bnci-horizon-2020.eu/database/data-sets> (dataset 001-2014) and drop
them into that folder.

---

## 5. Reproduce the MixNet baseline first

Do this **before** writing any code. It confirms your environment works and
gives you the number you have to beat.

```bash
cd experiments

# MixNet's own preprocessing (spectral-spatial signals), subject-dependent only
python prep_spectral_spatial_signals.py --dataset 'BCIC2a' --setting 'dependent'

# MixNet with the paper's optimal BCIC2a subject-dependent hyper-parameters
python run_MixNet.py --model_name 'MixNet' --dataset 'BCIC2a' \
    --train_type 'subject_dependent' --data_type 'spectral_spatial_signals' \
    --adaptive_gradient True --policy 'HistoricalTangentSlope' \
    --log_dir 'logs' --num_class 2 --GPU 0 \
    --margin 1.0 --n_component 2 --warmup 7
```

Record the resulting accuracy and F1 with:

```bash
python benchmark.py
```

**Deliverable checkpoint:** paste the MixNet summary table into your report.
That is your baseline; every later claim is relative to it.

---

## 6. Task 1 — your preprocessing pipeline

**File:** `mixnet/preprocessing/BCIC2a/HW_prep.py`

Implement two functions:

| Function | Called with | Must return |
|---|---|---|
| `fit_transform(X, y, pick_smp_freq, **kwargs)` | the training fold only | `(X_out, state)` |
| `transform(X, state, pick_smp_freq, **kwargs)` | validation and test folds | `X_out` |

You receive `X` of shape `(n_trials, 20, 400)` — 20 channels, 4 s at 100 Hz —
and labels `y ∈ {0, 1}`. You return features of shape `(n_trials, ...)`; the
trailing shape is yours to choose.

### The rule that carries the most marks

**Anything that learns from data must be fitted inside `fit_transform` on the
training fold only, and merely applied inside `transform`.** CSP filters, PCA,
a mean/std scaler, a whitening matrix, a channel-selection criterion — all of
these leak if you fit them on the full dataset before splitting. Leakage
inflates accuracy and makes your comparison against MixNet meaningless.

Stateless operations (band-pass filtering, cropping, resampling, log-variance)
are safe to apply to each split independently.

### Approaches you may take

- filter bank + CSP (`from mixnet.preprocessing import FBCSP`)
- band-power / log-variance features per channel per band
- Riemannian covariance + tangent-space projection
- STFT or wavelet time-frequency maps
- band-pass + per-channel z-scoring (a simple, respectable baseline)

Helpers already in the repo: `mixnet.utils.butter_bandpass_filter`,
`resampling`, `psd_welch`; `mixnet.preprocessing.FBCSP`,
`SpectralSpatialMapping`.

NOTE: You can also add your own helper functions in this file or in
`mixnet/preprocessing/BCIC2a/HW_prep_helpers.py` if you want to keep this file clean.

### Run it

```bash
cd experiments
python prep_HW.py
```

Output lands in
`experiments/datasets/BCIC2a/HW_prep/2_class/subject_dependent/`.
Expect 6 `.npy` files per fold for a subject (30 files per subject) — 270 files in total for 9 subjects.

---

## 7. Task 2 — your model

**File:** `mixnet/models/HW_Net.py`

Implement `_config()` (hyper-parameters) and `build()` (the architecture).
The training loops (`train_step` / `val_step` / `test_step` / `pred_step`) are
already written for you. Note that you may rewrite these steps if you go multi-task 
or use other loss functions. See MixNet for reference.

### Contract

- **Input:** one tensor of shape `self.input_shape` (no batch axis)
- **Output:** exactly **one** tensor of shape `(batch, num_class)` ending in
  `layers.Activation('softmax')` — multi-output models take a different code
  path in `mixnet/models/base.py` and will not work
- **Name:** `name=self.model_name`; never `'MixNet'` or `'MIN2Net'`, because
  `base.py` branches on those names
- Keep the `load_weights` block, or `evaluate()` will silently score an
  untrained network

### Get the plumbing working first

Before designing anything clever, paste this into `build()` and run one
subject end to end:

```python
input1  = layers.Input(shape=self.input_shape)
x       = layers.Flatten()(input1)
x       = layers.Dense(self.num_class)(x)
softmax = layers.Activation('softmax', name='softmax')(x)
model   = Model(inputs=input1, outputs=softmax, name=self.model_name)
```

Then replace it. `mixnet/models/EEGNet.py` and `DeepConvNet.py` are two
complete worked examples of the same contract.

### If you go multi-task, the step functions are no longer free

The `train_step` / `val_step` / `test_step` / `pred_step` methods given to you
are written for a **single-task** model: one output, one loss
(cross-entropy). MixNet is multi-task — it optimises reconstruction (MSE) +
triplet + cross-entropy together with adaptive loss weights — and you may do
the same, but then **you have to rewrite all four steps**, plus three other
things that must change with them:

| What | Change |
|---|---|
| `build()` | return a **list** of outputs, classifier softmax **last** |
| `configs/HW_Net.py` | one entry per task, same order, in `loss`, `loss_names` and `loss_weights`; keep the name `'crossentropy'` for the classification loss or class balancing breaks |
| the four steps | unpack every output, compute every loss, combine with the incoming `loss_weights`, log one `*_<name>_loss` per task |
| `evaluate()` | **override it inside `HW_Net`** — `base.py` only unpacks multi-output models named `MixNet`/`MIN2Net`; every other model hits `np.argmax(test_pred, axis=1)`, which is garbage for a list of outputs |

Do **not** rename your model to `MixNet*` to sneak into that branch, and do
not edit `base.py`. Overriding `evaluate()` in `HW_Net.py` is allowed — it is
one of your two files.

`mixnet/models/MixNet.py` is the reference implementation — read its steps
side by side with the single-task ones in your file.

### Match the shapes

Nothing derives the shapes for you — you set them yourself in
`experiments/configs/HW_Net.py`, and the two must agree:

- `data_params.data_format` — how `DataLoader` reshapes what Task 1 saved
- `model_params.input_shape` — what your `build()` receives, i.e. the
  reshaped tensor **without** the leading trial axis

See `mixnet.utils.DataLoader._change_data_format` for details.

### Run it

```bash
cd experiments
python run_HW_Net.py
```

Results land in `experiments/logs/HW_Net/subject_dependent_2_classes_BCIC2a/`:

| File | Contents |
|---|---|
| `S001_all_results.csv` | one row per fold — `test_acc`, `f1-score`, timing, memory |
| `S001_prediction_results.npy` | `y_true` / `y_pred` per fold |
| `S001_fold01_out_weights.h5` | best checkpoint of that fold |
| `S001_fold01_out_log.csv` | per-epoch training curve |

### Tuning

Tune in `experiments/configs/HW_Net.py` (`lr`, `batch_size`,
`dropout_rate`, `es_patience`, ...). Every key of `model_params` is forwarded to
your model and re-applied as `self.<key>`, so you can sweep hyper-parameters
without touching model code.

**Select on validation, never on test.** Change the `log_path` suffix for each
configuration so runs do not overwrite each other — `benchmark.py` discovers
every directory under `logs/` automatically.

---

## 8. Task 3 — benchmark against MixNet

```bash
cd experiments
python benchmark.py
```

With no arguments it finds every run under `logs/`, averages the 5 folds within
each subject, reports mean ± std across the 9 subjects, and runs a paired
comparison between a `MixNet` run and an `HW_Net` run. To be explicit:

```bash
python benchmark.py --baseline MixNet --candidate HW_Net --metric test_acc
```

It writes `benchmark_per_subject.csv` and `benchmark_summary.csv` and prints
markdown tables you can paste straight into your report.

### How to read the output

- **Per-subject table** — the honest picture. BCIC2a has large between-subject
  variance; a method can win on average while losing on 4 of 9 subjects.
- **Paired t-test / Wilcoxon** — is the difference more than noise? With only
  9 subjects the power is low, so *never* report a p-value without the
  per-subject table beside it.
- **`candidate wins on k/9 subjects`** — often more informative than the mean.

### You are not required to beat MixNet

MixNet is a published method with tuned hyper-parameters. A careful, honest,
well-analysed comparison that loses scores better than a win you cannot
explain or that came from a leak. What is graded is the quality of the
experiment and the analysis.

---

## 9. Deliverables

Push to your homework branch and submit its URL.

1. **Code** — your `HW_prep.py`, `HW_Net.py`, `configs/HW_Net.py`
2. **Results** — `benchmark_per_subject.csv`, `benchmark_summary.csv`, and the
   `S*_all_results.csv` files for your runs (these are git-ignored by default;
   copy them into a `results/` folder at the repo root and commit that)
3. **`REPORT`** (3–5 A4 pages) at the repo root:
   - **Preprocessing** — what you built and *why*; how you prevented leakage
   - **Model** — architecture diagram or layer table, parameter count, design rationale
   - **Results** — the per-subject and summary tables, plus the paired test
   - **Analysis** — where you beat MixNet, where you lose, and your hypothesis why;
     which subjects were hardest and what they have in common
   - **Reproduction** — the exact commands you ran

---

## 10. Rules

1. Do not modify anything under `mixnet/` other than your two files.
2. Do not train on session `E`, and do not select hyper-parameters on it.
3. Keep `k_folds=5` and all 9 subjects, so the comparison stays paired.
4. Fit every data-driven transform on the training fold only.
5. Cite any external code or paper you draw on.
6. Report what you actually ran. Reproducibility is more important than your model's performance!

---

## 11. Grading

| Criterion | Weight |
|---|---|
| Preprocessing — correct, leak-free, justified | 25% |
| Model — valid contract, sound design, justified | 25% |
| Benchmark — correct protocol, paired statistics, complete tables | 20% |
| Analysis — depth of interpretation | 20% |
| Reproducibility — the grader can re-run your commands and get your numbers | 10% |

---

## 12. Suggested schedule

| Week | Goal |
|---|---|
| 1 | Setup, data download, MixNet baseline reproduced and recorded |
| 1-2 | Task 1 — preprocessing running for all 9 subjects |
| 1-2 | Task 2 — model training end to end, first tuning pass |
| 3 | Task 3 — benchmark, `REPORT` |

Good luck — and start the baseline run early.