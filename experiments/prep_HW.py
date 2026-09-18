"""
=============================================================================
 HOMEWORK -- RUN YOUR PREPROCESSING PIPELINE  (Part 1)
=============================================================================

Reads the raw BCIC2a .mat files and writes your features to
    datasets/BCIC2a/HW_prep/2_class/subject_dependent/

Usage
    python prep_HW.py

Prerequisite
    python download_datasets.py --dataset 'BCIC2a'
"""
import mixnet.preprocessing as prep

prep.BCIC2a.HW_prep.subject_dependent_setting(
    k_folds=5,
    pick_smp_freq=100,
    save_path='datasets',
    num_class=2,
    # ---------------------------------------------------------------------------
    #  Hyper-parameters of the band-pass + per-channel z-score pipeline in
    #  mixnet/preprocessing/BCIC2a/HW_prep.py
    # ---------------------------------------------------------------------------
    lowcut=8.0,
    highcut=30.0,
    order=5,
    )

