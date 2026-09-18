"""
=============================================================================
 HOMEWORK PART 1 -- BUILD YOUR OWN PREPROCESSING PIPELINE  (BCIC2a)
=============================================================================

Your job in this file is to turn *raw* BCI Competition IV-2a MI-EEG trials
into the feature tensors that your model in `mixnet/models/HW_Net.py`
will be trained on.

WHAT IS ALREADY DONE FOR YOU (do not change unless you know why):
  * reading + cropping the raw .mat files              -> `_load_bcic2a()`
  * the 5-fold subject-dependent cross-validation loop -> `subject_dependent_setting()`
  * saving the .npy files with the exact file names    -> `_save_data_with_valset()`
    that `mixnet.utils.DataLoader` expects

WHAT YOU MUST IMPLEMENT (the two `TODO` functions below):
  1. `fit_transform()` -- learn any data-driven parameters on the TRAINING
     fold only, and return the transformed training features.
  2. `transform()`     -- apply the *already learned* parameters to the
     validation / test folds.

-----------------------------------------------------------------------------
THE GOLDEN RULE OF THIS ASSIGNMENT
-----------------------------------------------------------------------------
 Anything that *LEARNS* from data (CSP filters, PCA, a mean/std scaler, a
 whitening matrix, a channel-selection criterion, ...) must be fitted on the
 TRAINING fold ONLY, then merely *applied* to validation and test data.
 Fitting on all the data first and splitting afterwards is data leakage: it
 silently inflates your accuracy and makes the benchmark against MixNet
 meaningless. Marks are deducted for leakage.

 Stateless operations (band-pass filtering, cropping, resampling, log of the
 variance, ...) are safe to apply to every split independently.
-----------------------------------------------------------------------------

Data you receive in `fit_transform` / `transform`
    X : np.ndarray, shape (n_trials, n_channels, n_samples)
        n_channels = len(sel_chs)              (20 by default, see config.py)
        n_samples  = MI_len * pick_smp_freq    (4 s * 100 Hz = 400 by default)
    y : np.ndarray, shape (n_trials,)  with values in {0, 1}
        0 = left hand, 1 = right hand

Data you must return
    X_out : np.ndarray, shape (n_trials, ...anything you like...)
        Keep the trial axis first. Whatever trailing shape you choose decides
        the `input_shape` of your model: `DataLoader` reshapes it according to
        `data_format`, and the result minus the trial axis is what your
        network receives. The driver below prints the shape after every fold
        -- note it down, you need it in `experiments/configs/HW_Net.py`. See
        the data_format table in that file.
"""

import os

import numpy as np
from sklearn.model_selection import StratifiedKFold

from mixnet.preprocessing.BCIC2a import raw
from mixnet.preprocessing.config import CONSTANT
from mixnet.utils import butter_bandpass_filter

CONSTANT = CONSTANT['BCIC2a']
raw_path = CONSTANT['raw_path']
n_subjs = CONSTANT['n_subjs']
orig_smp_freq = CONSTANT['orig_smp_freq']
MI_len = CONSTANT['MI']['len']

# Name of the folder your features are written to. It must match the
# `--data_type` argument of run_HW_Net.py. Keep it as is.
DATA_TYPE = 'HW_prep'


# =============================================================================
#  TODO 1 / 2 -- fit on the training fold and transform it
#
#  NOTE: you are allowed to add any extra arguments to `fit_transform()` and `transform()`
#  but you must NOT change the signature of `subject_dependent_setting()`.
# =============================================================================
def fit_transform(X, y, pick_smp_freq, lowcut=8.0, highcut=30.0, order=5,
                  **kwargs):
    """Learn the pipeline parameters on the TRAINING fold and transform it.

    Called once per (subject, fold) with the training split only.

    Pipeline (very simple, on purpose):
      1. Band-pass filter each trial to the mu+beta band (8-30 Hz), where
         most of the motor-imagery ERD/ERS signal lives. This is STATELESS,
         so it is safe to apply identically to every split.
      2. Per-channel z-score normalization, using the mean/std computed on
         this training fold only. This IS data-driven, so the mean/std are
         computed here and stored in `state` for `transform()` to reuse.

    Args:
        X (np.ndarray): (n_trials, n_channels, n_samples) raw cropped signals.
        y (np.ndarray): (n_trials,) labels in {0, 1}. Not needed by this
            simple pipeline; kept in the signature to match the contract.
        pick_smp_freq (int): sampling rate of `X` in Hz (100 by default).
        lowcut (float): low edge of the band-pass filter, in Hz.
        highcut (float): high edge of the band-pass filter, in Hz.
        order (int): Butterworth filter order.
        **kwargs: forwarded from `experiments/prep_HW.py`; unused here.

    Returns:
        X_out (np.ndarray): (n_trials, n_channels, n_samples) filtered and
            per-channel normalized training features.
        state (dict): the per-channel mean/std and the filter settings, so
            `transform()` can reproduce exactly the same transformation.
    """
    X_filt = butter_bandpass_filter(X, lowcut, highcut, pick_smp_freq, order)

    # one mean/std per channel, pooled over trials and time -> shape (1, C, 1)
    mean = X_filt.mean(axis=(0, 2), keepdims=True)
    std = X_filt.std(axis=(0, 2), keepdims=True) + 1e-8

    X_out = (X_filt - mean) / std
    state = {
        'mean': mean,
        'std': std,
    }
    return X_out.astype(np.float32), state


# =============================================================================
#  TODO 2 / 2 -- apply the learned pipeline to validation / test folds
#
#  NOTE: you are allowed to add any extra arguments to `fit_transform()` and `transform()`
#  but you must NOT change the signature of `subject_dependent_setting()`.
# =============================================================================
def transform(X, state, pick_smp_freq, lowcut=8.0, highcut=30.0, order=5, **kwargs):
    """Apply the pipeline learned by `fit_transform()` to unseen trials.

    Called twice per (subject, fold): once for the validation split and once
    for the test split.
    NOTE: It does NOT re-fit anything -- it only reads from `state`, so the
    training-fold mean/std never leak into validation/test.

    Args:
        X (np.ndarray): (n_trials, n_channels, n_samples) raw cropped signals.
        state (dict): the object returned by `fit_transform()`.
        pick_smp_freq (int): sampling rate of `X` in Hz.
        **kwargs: same extra hyper-parameters as `fit_transform()`.

    Returns:
        X_out (np.ndarray): (n_trials, n_channels, n_samples) features with
            the SAME trailing shape as the output of `fit_transform()`.
    """
    X_filt = butter_bandpass_filter(
        X, lowcut, highcut, pick_smp_freq, order)
    X_out = (X_filt - state['mean']) / state['std']
    return X_out.astype(np.float32)


# =============================================================================
#  Provided for you -- the subject-dependent 5-fold CV driver
#  
#  NOTE: you are allowed to add any extra arguments to `fit_transform()`, 
# `transform()`, and `subject_dependent_setting()`
#  but you must NOT change the signature of `subject_dependent_setting()`.
# =============================================================================
def subject_dependent_setting(k_folds, pick_smp_freq, save_path, num_class=2,
                              sel_chs=None, subjects=None, **kwargs):
    """Run the whole pipeline for every subject in the subject-dependent setting.

    Subject-dependent means: for each of the 9 subjects we train, validate and
    test on that subject alone. Session T is split into train/validation with
    stratified k-fold CV; session E is the untouched test set of every fold.

    Args:
        k_folds (int): number of CV folds (5 to match the MixNet baseline).
        pick_smp_freq (int): target sampling rate in Hz (100 in the baseline).
        save_path (str): dataset root, e.g. 'datasets'.
        num_class (int): 2 = left hand vs right hand.
        sel_chs (list[str] | None): channel names; None = all 20 from config.py.
        subjects (list[int] | None): 1-based subject ids; None = all 9.
        **kwargs: forwarded verbatim to `fit_transform()` / `transform()`.
    """
    sel_chs = CONSTANT['sel_chs'] if sel_chs is None else sel_chs
    save_path = os.path.join(
        save_path, 'BCIC2a', DATA_TYPE, '{}_class'.format(num_class),
        'subject_dependent')
    os.makedirs(save_path, exist_ok=True)

    id_chosen_chs = raw.chanel_selection(sel_chs)
    subjects = range(1, n_subjs + 1) if subjects is None else subjects

    for subject in subjects:
        X_tr, y_tr, X_te, y_te = _load_bcic2a(
            subject, pick_smp_freq, num_class, id_chosen_chs)

        if X_tr.ndim != 3:
            raise Exception('Dimension Error, must have 3 dimensions, '
                            'found {}'.format(X_tr.shape))

        skf = StratifiedKFold(n_splits=k_folds, random_state=42, shuffle=True)
        for fold, (train_index, val_index) in enumerate(skf.split(X_tr, y_tr)):
            print('SUBJECT:', subject, 'FOLD:', fold + 1,
                  'TRAIN:', len(train_index), 'VALIDATION:', len(val_index))

            X_tr_cv, X_val_cv = X_tr[train_index], X_tr[val_index]
            y_tr_cv, y_val_cv = y_tr[train_index], y_tr[val_index]

            # ---- your pipeline: fit on train only, then apply everywhere ----
            X_tr_out, state = fit_transform(
                X_tr_cv, y_tr_cv, pick_smp_freq=pick_smp_freq, **kwargs)
            X_val_out = transform(
                X_val_cv, state, pick_smp_freq=pick_smp_freq, **kwargs)
            X_te_out = transform(
                X_te, state, pick_smp_freq=pick_smp_freq, **kwargs)
            # -----------------------------------------------------------------

            _check_shapes(X_tr_out, X_val_out, X_te_out,
                          y_tr_cv, y_val_cv, y_te)
            print('Feature shapes -- train {}, val {}, test {}'.format(
                X_tr_out.shape, X_val_out.shape, X_te_out.shape))

            save_name = 'S{:03d}_fold{:03d}'.format(subject, fold + 1)
            _save_data_with_valset(save_path, save_name,
                                   X_tr_out, y_tr_cv,
                                   X_val_out, y_val_cv,
                                   X_te_out, y_te)
            print('The preprocessing of subject {} from fold {} is DONE!!!'
                  .format(subject, fold + 1))


def _load_bcic2a(subject, pick_smp_freq, num_class, id_chosen_chs):
    """Read session T (train) and session E (test), crop to the MI window."""
    start = CONSTANT['MI']['start']   # 2 s after the cue
    stop = CONSTANT['MI']['stop']     # 6 s after the cue
    return raw.load_crop_data(PATH=raw_path, subject=subject,
                              start=start, stop=stop,
                              new_smp_freq=pick_smp_freq,
                              num_class=num_class,
                              id_chosen_chs=id_chosen_chs)


def _check_shapes(X_tr, X_val, X_te, y_tr, y_val, y_te):
    """Fail early and loudly on the mistakes that cost the most debugging time."""
    for name, X, y in [('train', X_tr, y_tr), ('val', X_val, y_val),
                       ('test', X_te, y_te)]:
        X = np.asarray(X)
        if X.shape[0] != len(y):
            raise Exception(
                'Trial-axis mismatch on the {} split: X has {} trials but y '
                'has {}. Your transform must keep the trial axis first and '
                'must not drop or reorder trials.'.format(
                    name, X.shape[0], len(y)))
        if not np.all(np.isfinite(X)):
            raise Exception(
                'The {} split contains NaN or Inf. This usually means an '
                'unstable filter (check the order and the band edges against '
                'the Nyquist frequency) or a division by a zero variance.'
                .format(name))
    if X_tr.shape[1:] != X_val.shape[1:] or X_tr.shape[1:] != X_te.shape[1:]:
        raise Exception(
            'Feature shapes differ across splits: train {}, val {}, test {}. '
            'transform() must produce the same trailing shape as '
            'fit_transform().'.format(
                X_tr.shape[1:], X_val.shape[1:], X_te.shape[1:]))


def _save_data_with_valset(save_path, NAME, X_train, y_train,
                           X_val, y_val, X_test, y_test):
    """Write the six .npy files under the names `DataLoader` looks for."""
    np.save(save_path + '/X_train_' + NAME + '.npy', X_train)
    np.save(save_path + '/X_val_' + NAME + '.npy', X_val)
    np.save(save_path + '/X_test_' + NAME + '.npy', X_test)
    np.save(save_path + '/y_train_' + NAME + '.npy', y_train)
    np.save(save_path + '/y_val_' + NAME + '.npy', y_val)
    np.save(save_path + '/y_test_' + NAME + '.npy', y_test)
    print('save DONE')
