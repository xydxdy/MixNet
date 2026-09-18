# Homework — Build and Benchmark Your Own MI-EEG Pipeline

In this assignment, you will build your own motor imagery EEG (MI-EEG) classification pipeline. You will design both an EEG preprocessing/feature-extraction pipeline and a neural network, integrate them into the existing repository, and compare your method with **MixNet**.

To make the comparison fair, your method and MixNet must be evaluated under exactly the same experimental protocol: the same 9 subjects, the same 5-fold cross-validation procedure, the same test session, and the same evaluation metrics.

---

## 1. Learning Objectives

By the end of this assignment, you should be able to:

1. **Build a reusable EEG pipeline.**  
   Implement your preprocessing or feature-extraction method and neural network within the repository's existing experimental framework.

2. **Run a controlled benchmark.**  
   Compare your proposed method with a strong published baseline under the same experimental conditions, and report the results transparently, including cases where your method performs worse.

3. **Interpret your results.**  
   Go beyond reporting average performance. Examine the per-subject results together with the paired statistical test, identify patterns in the results, and propose reasonable explanations for what you observe.

---

## 2. Dataset and Experimental Setting

**Dataset:** BCI Competition IV-2a (`BCIC2a`)  
**Setting:** Subject-dependent  
**Task:** Two-class motor imagery classification — left hand vs. right hand

| Property | Value |
|---|---|
| Subjects | 9 (`A01`–`A09`) |
| Sessions | `T` (training) and `E` (evaluation), recorded on different days |
| Classes used | 2 — left hand (`0`) and right hand (`1`) |
| Trials | 144 per session per subject (72 per class) |
| Channels | 20 pre-selected motor-cortex channels (see [`mixnet/preprocessing/config.py`](mixnet/preprocessing/config.py)) |
| Original sampling rate | 250 Hz, downsampled to 100 Hz by the provided loader (optional) |
| MI window | 4 s, `2 s → 6 s` after the cue (optional) |

In the **subject-dependent** setting, each subject has a separate model. For each subject, session `T` is divided into training and validation sets using 5-fold stratified cross-validation. Session `E` is used as the **untouched test set** for every fold.

**Never train on session `E`, and never use it for hyperparameter selection.**

---

## 3. What Is Provided and What You Need to Implement

| File | Status |
|---|---|
| [`mixnet/preprocessing/BCIC2a/HW_prep.py`](mixnet/preprocessing/BCIC2a/HW_prep.py) | **YOU WRITE** — 2 required functions |
| [`mixnet/models/HW_Net.py`](mixnet/models/HW_Net.py) | **YOU WRITE** — model implementation |
| [`experiments/configs/HW_Net.py`](experiments/configs/HW_Net.py) | **YOU TUNE** — shapes and hyperparameters |
| [`experiments/prep_HW.py`](experiments/prep_HW.py) | Provided — runs your preprocessing pipeline; add your own parameters if needed |
| [`experiments/run_HW_Net.py`](experiments/run_HW_Net.py) | Provided — 5-fold training and evaluation driver |
| [`experiments/benchmark.py`](experiments/benchmark.py) | Provided — aggregation and statistical comparison; no edits needed |
| Everything else under [`mixnet/`](mixnet) | Provided — do not modify |

Each file that you are expected to edit begins with a header describing the interface it must satisfy. Read those headers carefully before writing code.

---

## 4. Setup

> **Run all commands below from the [`experiments/`](experiments) directory unless stated otherwise.**
>
> The scripts resolve `datasets/` and `logs/` relative to the current working directory, and both directories are git-ignored.

### 4.1 Prerequisites

You will need:

- Python **3.8.10**  
  The package declares support for `>=3.8, <=3.10.4`.
- An NVIDIA GPU with CUDA.
- `git`

### 4.2 Create the Environment

The reference environment uses the TensorFlow 2.7.0 GPU Docker image:

```bash
docker pull tensorflow/tensorflow:2.7.0-gpu

docker run -ti --gpus all --name mixnet_container \
    -v $(pwd):/workspace \
    docker.io/tensorflow/tensorflow:2.7.0-gpu bash

cd /workspace
```

Alternatively, if you already have a working CUDA setup, you may use Conda:

```bash
conda create -n mixnet python=3.8.10 -y
conda activate mixnet
pip install tensorflow-gpu==2.7.0
```

### 4.3 Fork and Clone the Repository

Fork the following repository to your own GitHub account:

`https://github.com/xydxdy/MixNet`

When creating the fork, make sure to **uncheck** the option:

> **Copy the DEFAULT branch only**

Then clone your fork and switch to the homework branch:

```bash
git clone https://github.com/<your-github-username>/MixNet.git
cd MixNet
git switch homework
```

Work on the `homework` branch throughout the assignment.

### 4.4 Install the Package

Install the dependencies and then install the repository in editable mode:

```bash
pip install -r requirements.txt
pip install -e .
```

> ### Use `pip install -e .`
>
> Editable mode links Python directly to your working copy of the repository, so changes under [`mixnet/`](mixnet) take effect immediately.
>
> If you instead use `pip install .`, you will need to reinstall the package after every change.

Verify that Python is importing `mixnet` from your clone:

```bash
python -c "import mixnet; print(mixnet.__file__)"
```

You should see something similar to:

```text
/workspace/MixNet/mixnet/__init__.py
```

It should **not** point to a `site-packages` directory.

### 4.5 Download the Dataset

From the repository root:

```bash
cd experiments
python download_datasets.py --dataset 'BCIC2a'
```

The dataset should be downloaded to:

```text
experiments/datasets/BCIC2a/raw/A0{1..9}{T,E}.mat
```

If the automatic download fails, download dataset `001-2014` manually from:

`http://bnci-horizon-2020.eu/database/data-sets`

and place the files in the same directory.

---

## 5. Reproduce the MixNet Baseline First

Before implementing your own method, reproduce the MixNet baseline.

This step confirms that your environment is working correctly and gives you a reference result for the later comparison.

Run:

```bash
cd experiments

# MixNet preprocessing: spectral-spatial signals, subject-dependent setting
python prep_spectral_spatial_signals.py \
    --dataset 'BCIC2a' \
    --setting 'dependent'

# MixNet using the paper's BCIC2a subject-dependent hyperparameters
python run_MixNet.py \
    --model_name 'MixNet' \
    --dataset 'BCIC2a' \
    --train_type 'subject_dependent' \
    --data_type 'spectral_spatial_signals' \
    --adaptive_gradient True \
    --policy 'HistoricalTangentSlope' \
    --log_dir 'logs' \
    --num_class 2 \
    --GPU 0 \
    --margin 1.0 \
    --n_component 2 \
    --warmup 7
```

Then summarize the results using:

```bash
python benchmark.py
```

### Baseline Checkpoint

Include the MixNet summary table in your report.

This is your baseline, and all later comparisons should be made relative to this result.

---

## 6. Task 1 — Build Your Preprocessing Pipeline

**File:** [`mixnet/preprocessing/BCIC2a/HW_prep.py`](mixnet/preprocessing/BCIC2a/HW_prep.py)

Implement the following two functions:

| Function | Input | Required output |
|---|---|---|
| `fit_transform(X, y, pick_smp_freq, **kwargs)` | Training fold only | `(X_out, state)` |
| `transform(X, state, pick_smp_freq, **kwargs)` | Validation and test folds | `X_out` |

The input EEG has shape:

```text
(n_trials, 20, 400)
```

where:

- `20` = EEG channels
- `400` = 4 seconds sampled at 100 Hz

The labels are:

```text
y ∈ {0, 1}
```

Your output can have any trailing feature shape:

```text
(n_trials, ...)
```

as long as your model is designed to accept it.

### The Most Important Rule: Prevent Data Leakage

**Any operation that learns parameters from the data must be fitted using the training fold only.**

Examples include:

- CSP filters
- PCA
- normalization using dataset-level mean and standard deviation
- whitening
- channel selection
- learned feature selection

These operations must be fitted inside `fit_transform()` using only the current training fold. The resulting fitted parameters should be stored in `state` and then applied to validation and test data inside `transform()`.

Fitting any of these operations on the full dataset before cross-validation introduces data leakage and invalidates the comparison with MixNet.

Stateless operations may be applied independently to each split. Examples include:

- band-pass filtering
- cropping
- resampling
- log-variance computation

### Possible Approaches

You are free to design your own pipeline. Possible directions include:

- Filter bank + CSP using [`mixnet.preprocessing.FBCSP`](mixnet/preprocessing/FBCSP.py)
- Band-power or log-variance features for each channel and frequency band
- Riemannian covariance features with tangent-space projection
- STFT or wavelet time-frequency maps
- Band-pass filtering followed by per-channel normalization

Useful functions and classes already available in the repository include:

```python
mixnet.utils.butter_bandpass_filter
mixnet.utils.resampling
mixnet.utils.psd_welch
mixnet.preprocessing.FBCSP
mixnet.preprocessing.SpectralSpatialMapping
```

You may also define additional helper functions directly in:

```text
mixnet/preprocessing/BCIC2a/HW_prep.py
```

or place them in:

```text
mixnet/preprocessing/BCIC2a/HW_prep_helpers.py
```

if you prefer to keep the main file clean.

### Run the Preprocessing Pipeline

```bash
cd experiments
python prep_HW.py
```

The processed data will be saved to:

```text
experiments/datasets/BCIC2a/HW_prep/2_class/subject_dependent/
```

For each subject, you should expect:

- 6 `.npy` files per fold
- 5 folds
- 30 files per subject
- 270 files total across 9 subjects

---

## 7. Task 2 — Build Your Neural Network

**File:** [`mixnet/models/HW_Net.py`](mixnet/models/HW_Net.py)

Implement your model architecture and any model-specific configuration required by the provided framework.

The training and evaluation pipeline already provides:

- `train_step`
- `val_step`
- `test_step`
- `pred_step`

For a standard single-task classifier, you can use these directly.

### Model Contract

Your model must satisfy the following interface:

- **Input:** one tensor with shape `self.input_shape`, excluding the batch dimension
- **Output:** one tensor with shape:

```text
(batch, num_class)
```

- The final layer must be:

```python
layers.Activation('softmax')
```

- The model name must be:

```python
name=self.model_name
```

Do **not** name your model `MixNet` or `MIN2Net`, because [`mixnet/models/base.py`](mixnet/models/base.py) contains special branches for those model names.

Keep the provided `load_weights` block. Otherwise, `evaluate()` may silently evaluate an untrained model.

### First Make Sure the Pipeline Works

Before building a more complex architecture, use the following minimal model in `build()` and run one subject end to end:

```python
input1 = layers.Input(shape=self.input_shape)

x = layers.Flatten()(input1)
x = layers.Dense(self.num_class)(x)

softmax = layers.Activation(
    'softmax',
    name='softmax'
)(x)

model = Model(
    inputs=input1,
    outputs=softmax,
    name=self.model_name
)
```

Once the full pipeline works, replace this model with your actual architecture.

You can use the following files as examples:

```text
mixnet/models/EEGNet.py
mixnet/models/DeepConvNet.py
```

Both follow the same model contract.

### If You Use a Multi-Task Model

The provided `train_step`, `val_step`, `test_step`, and `pred_step` functions are designed for a **single-task classification model** with one output and one cross-entropy loss.

You may implement a multi-task architecture, similar to MixNet, but then you are responsible for updating all related components.

If you use multiple outputs, the following changes are required:

| Component | Required change |
|---|---|
| `build()` | Return a **list of outputs**, with the classifier softmax output **last** |
| [`configs/HW_Net.py`](experiments/configs/HW_Net.py) | Add one entry per task, in the same order, to `loss`, `loss_names`, and `loss_weights` |
| Classification loss name | Keep the name `'crossentropy'`, otherwise class balancing will not work correctly |
| Step functions | Rewrite `train_step`, `val_step`, `test_step`, and `pred_step` to unpack all outputs and compute all losses |
| Loss logging | Log one `*_<name>_loss` value for each task |
| `evaluate()` | Override it inside [`HW_Net.py`](mixnet/models/HW_Net.py) |

The `evaluate()` override is necessary because [`base.py`](mixnet/models/base.py) only handles multi-output predictions automatically for models named `MixNet` or `MIN2Net`.

Do **not** rename your model to `MixNet` to enter this code path, and do not edit [`base.py`](mixnet/models/base.py).

Use:

```text
mixnet/models/MixNet.py
```

as the reference implementation for multi-task training.

### Match the Input Shapes

Your preprocessing output and model input must agree.

Configure both of the following in:

```text
experiments/configs/HW_Net.py
```

- `data_params.data_format`
- `model_params.input_shape`

`data_params.data_format` determines how `DataLoader` reshapes the data saved by Task 1.

`model_params.input_shape` specifies the shape received by your model, excluding the leading trial/batch dimension.

See:

```python
mixnet.utils.DataLoader._change_data_format
```

for details.

### Run Your Model

```bash
cd experiments
python run_HW_Net.py
```

Results will be saved under:

```text
experiments/logs/HW_Net/subject_dependent_2_classes_BCIC2a/
```

Typical output files include:

| File | Contents |
|---|---|
| `S001_all_results.csv` | One row per fold containing `test_acc`, `f1-score`, timing, memory usage, etc. |
| `S001_prediction_results.npy` | `y_true` and `y_pred` for each fold |
| `S001_fold01_out_weights.h5` | Best checkpoint for the fold |
| `S001_fold01_out_log.csv` | Per-epoch training log |

### Hyperparameter Tuning

Tune your model through:

```text
experiments/configs/HW_Net.py
```

Examples include:

- `lr`
- `batch_size`
- `dropout_rate`
- `es_patience`

Every entry in `model_params` is passed to the model and assigned as:

```python
self.<key>
```

This makes it possible to change hyperparameters without editing the model code.

**Select hyperparameters using validation performance only. Never tune using the test session.**

Use a different `log_path` suffix for each configuration so that runs do not overwrite one another. `benchmark.py` automatically discovers experiment directories under `logs/`.

---

## 8. Task 3 — Benchmark Against MixNet

Run:

```bash
cd experiments
python benchmark.py
```

With no additional arguments, the script:

1. Finds runs under `logs/`
2. Averages the 5 folds within each subject
3. Reports mean ± standard deviation across the 9 subjects
4. Performs a paired comparison between MixNet and your `HW_Net`

You can also specify the comparison explicitly:

```bash
python benchmark.py \
    --baseline MixNet \
    --candidate HW_Net \
    --metric test_acc
```

The script produces:

```text
benchmark_per_subject.csv
benchmark_summary.csv
```

It also prints Markdown tables that you can paste directly into your report.

### How to Interpret the Results

#### Per-subject performance

Do not rely only on the overall average.

BCIC2a contains substantial between-subject variability. A model may achieve a higher overall mean while still performing worse than the baseline for several subjects.

Examine which subjects improve and which subjects do not.

#### Paired statistical test

Use the paired t-test or Wilcoxon result together with the per-subject table.

There are only 9 subjects, so statistical power is limited. A p-value should therefore never be reported without also showing the underlying subject-level results.

#### Number of subject-level wins

The output:

```text
candidate wins on k/9 subjects
```

can help show whether the improvement is consistent across subjects or driven by only a few cases.

### You Are Not Required to Beat MixNet

MixNet is a published model with tuned hyperparameters.

A careful, reproducible experiment that performs worse than MixNet but is properly analysed is more valuable than an unexplained improvement caused by data leakage or an unfair comparison.

The quality of your experimental design and analysis matters more than simply obtaining the highest score.

---

## 9. Deliverables

Push your work to your `homework` branch and submit the repository URL.

### 1. Code

Submit:

```text
mixnet/preprocessing/BCIC2a/HW_prep.py
mixnet/models/HW_Net.py
experiments/configs/HW_Net.py
```

### 2. Results

Submit:

```text
benchmark_per_subject.csv
benchmark_summary.csv
S*_all_results.csv
```

The experiment outputs are git-ignored by default.

Copy the required result files into a directory such as:

```text
results/
```

at the repository root and commit them.

### 3. Report

Submit a **3–5 page A4 report** at the repository root.

Your report should contain the following sections:

#### Preprocessing

Describe:

- what preprocessing or feature extraction you implemented
- why you chose it
- which parts were data-driven
- how you prevented data leakage

#### Model

Include:

- your model architecture
- an architecture diagram or layer table
- number of trainable parameters
- the reasoning behind the design

#### Results

Include:

- MixNet baseline results
- your method's results
- per-subject comparison
- overall summary
- paired statistical test

#### Analysis

Discuss:

- where your method performs better than MixNet
- where it performs worse
- whether the differences are consistent across subjects
- which subjects are most difficult
- possible explanations for the observed pattern

Your explanation should be supported by the results rather than being purely speculative.

#### Reproduction

Provide the exact commands needed to reproduce:

1. preprocessing
2. training
3. evaluation
4. benchmarking

---

## 10. Rules

1. Do not modify files under [`mixnet/`](mixnet) other than the files specifically assigned to you.
2. Do not train on session `E`.
3. Do not use session `E` for model selection or hyperparameter tuning.
4. Keep `k_folds=5`.
5. Use all 9 subjects so that the comparison with MixNet remains paired.
6. Fit every data-driven preprocessing step using the training fold only.
7. Cite any external code, repository, or paper that you use.
8. Report the experiments you actually ran.
9. Reproducibility is more important than obtaining the highest performance.

---

## 11. Grading

| Criterion | Weight |
|---|---:|
| Preprocessing — correct, leak-free, and well justified | 30% |
| Model — valid implementation, sound design, and clear rationale | 30% |
| Report - quality of results presentation, comparison with MixNet, statistical analysis, and interpretation of findings | 30% |
| Reproducibility — the grader can rerun your commands and reproduce your results | 10% |

---

## 12. Suggested Schedule

| Week | Goal |
|---|---|
| Week 1 | Complete setup, download the dataset, reproduce MixNet, and record the baseline |
| Weeks 1–2 | Task 1 — implement and run preprocessing for all 9 subjects |
| Weeks 1–2 | Task 2 — train your model end to end and perform initial tuning |
| Week 3 | Task 3 — run the final benchmark and complete the report |

Start the MixNet baseline early so that you have enough time to confirm the environment and fix any setup issues before developing your own method.

Good luck!