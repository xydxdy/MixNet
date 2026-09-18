"""
=============================================================================
 EXPERIMENT CONFIG FOR HW_Net  (BCIC2a, subject-dependent)
=============================================================================

`experiments/run_HW_Net.py` loads this file through
`configs/exp_config.get_params()` and uses the returned dotdict to build the
model, the DataLoader and the log directory.

You will come back to this file constantly -- it is where you tune the
training model without touching model code, because every key of
`model_params` is forwarded to `HW_Net.__init__` and re-applied as
`self.<key>` at the end of `HW_Net._config()`.

-----------------------------------------------------------------------------
 THE ONE THING THAT BREAKS FIRST: SHAPES
-----------------------------------------------------------------------------
mixnet.utils.DataLoader is the class that loads your data and passes it 
to your model. 

Every array your Part 1 pipeline saved is passed through
`mixnet.utils.DataLoader._change_data_format()` on its way to your model --
`load_train_set()`, `load_val_set()` and `load_test_set()` all call it. It is
about twenty lines; open it and read it alongside this section.

The letters are axis names: N = trials, C = channels, T = time, D = depth
(the singleton axis a Conv2D needs), S = sub-bands/freqs, H/W = height/width
of a 2-D spatial map.

The below table shows the five valid values for `data_format` and what they do to 
the shape of your data. The result is what your model will see, so make sure it 
matches the input shape you defined in `HW_Net.__init__()`.

  data_format | what the method does                      | result
  ------------+-------------------------------------------+--------------
  'NCTD'      | reshape(n, C, T, 1)                       | (n, C, T, 1)
  'NDCT'      | reshape(n, 1, C, T)                       | (n, 1, C, T)
  'NTCD'      | reshape(n, C, T, 1), then swapaxes(1, 3)  | (n, 1, T, C)
  'NSHWD'     | reshape(n, S, H, W, 1)                    | (n, S, H, W, 1)
   None       | nothing at all                            | (n, C, T)

Anything other than these five values raises:
    Value Error: data_format requires None, 'NCTD', 'NDCT', 'NTCD' or 'NSHWD'
    
Note: You can add your custom data_format to the list above, but you must also implement it in
`mixnet.utils.DataLoader._change_data_format()`
-----------------------------------------------------------------------------
"""

from mixnet.models import *
from mixnet.loss import *
from mixnet.utils import dotdict
from tensorflow.keras.optimizers import Adam


# =============================================================================
#  TODO -- keep in sync with your Part 1 pipeline
# =============================================================================
def get_params(dataset, train_type, data_type, num_class, loss_weights=None,
               log_dir='logs', **kwargs):
    """Return a dotdict of parameters for `run_HW_Net.py`."""

    model_name = 'HW_Net'
    n_subjects = 9 if dataset == 'BCIC2a' else 0

    # HW_prep.py keeps the trailing shape of the raw signal, (20, 400):
    # 20 channels, 4 s at 100 Hz. 'NDCT' turns that into (n, 1, 20, 400),
    # which is what HW_Net.input_shape below expects.
    time_points = 400
    input_shape = (1, 20, time_points)

    # one weight per task, in the SAME order as loss / loss_names below:
    # [reconstruction (mse), classification (crossentropy)]
    loss_weights = [1., 1.] if loss_weights is None else loss_weights

    log_path = '{}/{}/{}_{}_classes_{}'.format(
        log_dir, model_name, train_type, str(num_class), dataset)

    # subject-dependent hyper-parameters (small dataset -> small batch size)
    factor = 0.5
    es_patience = 20
    lr = 0.01
    min_lr = 0.01
    batch_size = 10
    patience = 5
    epochs = 200
    min_epochs = 0
    dropout_rate = 0.5

    params = dotdict({
        'model': HW_Net,
        'model_params': dotdict({
            'model_name': model_name,
            'input_shape': input_shape,
            'latent_dim': 32,
            'class_balancing': True,
            'f1_average': 'macro',
            'num_class': num_class,
            # multi-task: one entry per task, classifier ('crossentropy')
            # kept under that exact name so class balancing can find it.
            'loss': [MeanSquaredError(), SparseCategoricalCrossentropy()],
            'loss_names': ['mse', 'crossentropy'],
            'loss_weights': loss_weights,
            'epochs': epochs,
            'batch_size': batch_size,
            'dropout_rate': dropout_rate,
            'optimizer': Adam(beta_1=0.9, beta_2=0.999, epsilon=1e-08),
            'lr': lr,
            'min_lr': min_lr,
            'factor': factor,
            'patience': patience,
            'es_patience': es_patience,
            'min_epochs': min_epochs,
            'verbose': 1,
            'log_path': log_path,
            'data_format': 'channels_first',
        }),

        'data_params': dotdict({
            'dataset': dataset,
            'train_type': train_type,
            'data_format': 'NDCT',
            'data_type': data_type,
            'num_class': num_class,
            'dataset_path': 'datasets',
            'n_subjects': n_subjects,
            'n_folds': 5,
            'load_path': 'datasets/{}/{}/{}_class/'.format(
                dataset, data_type, num_class),
        }),

        'log_path': log_path,
    })

    return params
