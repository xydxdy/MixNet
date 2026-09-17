"""
=============================================================================
 HOMEWORK -- TRAIN AND EVALUATE YOUR MODEL  (Part 2)
=============================================================================

Subject-dependent 5-fold cross-validation on BCIC2a

Usage
    # one subject
    python run_HW_Net.py --subjects 1

    # the full experiment: 9 subjects x 5 folds
    python run_HW_Net.py

    # on a specific GPU
    python run_HW_Net.py --GPU 0

Prerequisite
    python download_datasets.py --dataset 'BCIC2a'
    python prep_HW.py

Results land in
    logs/HW_Net/subject_dependent_2_classes_BCIC2a/
        S001_all_results.csv          <- one row per fold (test_acc, f1-score, ...)
        S001_prediction_results.npy   <- y_true / y_pred per fold
        S001_fold01_out_weights.h5    <- best checkpoint of that fold
        S001_fold01_out_log.csv       <- training log
"""

import argparse
import os

import numpy as np
import tensorflow as tf

from mixnet.utils import write_log, DataLoader, str2bool
from configs import exp_config


def main(subject):
    # create an object of DataLoader
    loader = DataLoader(subject=subject, **config.data_params)

    results = []
    for fold in range(1, config.data_params.n_folds + 1):

        prefix_log = 'S{:03d}_fold{:02d}'.format(subject, fold)
        model = config.model(prefix_log=prefix_log, **config.model_params)

        # load dataset
        X_train, y_train = loader.load_train_set(fold=fold)
        X_val, y_val = loader.load_val_set(fold=fold)
        X_test, y_test = loader.load_test_set(fold=fold)
        print("Check type of MI classes: ", np.unique(y_train))

        model.fit(X_train, y_train, X_val, y_val)
        Y, evaluation = model.evaluate(X_test, y_test)

        # logging
        csv_file = config.log_path + '/S{:03d}_all_results.csv'.format(subject)
        if fold == 1:
            write_log(csv_file, data=evaluation.keys(), mode='w')
        write_log(csv_file, data=evaluation.values(), mode='a')
        results.append(Y)
        tf.keras.backend.clear_session()

    # writing results
    np.save(config.log_path + '/S{:03d}_prediction_results.npy'.format(subject),
            results)
    print('------------------------- S{:03d} Done--------------------------'
          .format(subject))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='HW_Net',
                        help='name of the config file in configs/')
    parser.add_argument('--dataset', type=str, default='BCIC2a',
                        help='dataset name')
    parser.add_argument('--train_type', type=str, default='subject_dependent',
                        help='this homework uses subject_dependent')
    parser.add_argument('--data_type', type=str, default='HW_prep',
                        help='must match HW_prep.DATA_TYPE')
    parser.add_argument('--num_class', type=int, default=2,
                        help='number of classes')
    parser.add_argument('--loss_weights', nargs='+', default=None, type=float,
                        help='loss weights, one per loss')
    parser.add_argument('--log_dir', type=str, default='logs',
                        help='path to save logs')
    parser.add_argument('--subjects', nargs='+', default=None, type=int,
                        help='subject ids; two values = an inclusive range; '
                             'default = all subjects')
    parser.add_argument('--GPU', type=str, default='0', help='GPU ID')
    args = parser.parse_args()

    # Specify GPU used
    os.environ['CUDA_VISIBLE_DEVICES'] = args.GPU
    print(tf.config.experimental.list_physical_devices('GPU'))

    exp_setup = {'dataset': args.dataset,
                 'train_type': args.train_type,
                 'data_type': args.data_type,
                 'num_class': args.num_class,
                 'loss_weights': args.loss_weights,
                 'log_dir': args.log_dir}
    config = exp_config.get_params(args.model_name, **exp_setup)
    print(config)
    for directory in [config.log_path]:
        if not os.path.exists(directory):
            os.makedirs(directory)

    if args.subjects == None:  # loop to train all subjects
        for subject in range(1, config.data_params.n_subjects + 1):
            main(subject)
    elif len(args.subjects) == 2:
        for subject in range(args.subjects[0], args.subjects[1] + 1):
            main(subject)
    else:
        for subject in args.subjects:
            main(subject)
