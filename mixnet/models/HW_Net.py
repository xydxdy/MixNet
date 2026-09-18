"""
=============================================================================
 HOMEWORK PART 2 -- BUILD YOUR OWN MODEL  (HW_Net)
=============================================================================

`HW_Net` plugs into the exact same training machinery as MixNet, EEGNet
and DeepConvNet, so the benchmark in Part 3 is a fair comparison:
same folds, same subjects, same early stopping, same metrics.

THIS FILE IS THE "MULTI-TASK" EXAMPLE: it implements the optional path
described in "IF YOU GO MULTI-TASK" below with the smallest multi-task
model that still makes sense -- a shared encoder with two heads:
  * a decoder that reconstructs the (filtered, normalized) input -- MSE
  * a classifier that predicts left/right hand from the same latent -- CE
No triplet/metric-learning term and no adaptive loss-weight blending: those
are MixNet-specific extras, not requirements of "multi-task".

WHAT IS ALREADY DONE FOR YOU (do not change unless you know why):
  * inheritance from `BaseModel`, which gives you `.fit()` and `.predict()`
    for free (`.evaluate()` is overridden below -- see point 4).

WHAT THIS FILE IMPLEMENTS:
  1. `_config()` -- architecture hyper-parameters.
  2. `build()`   -- return a `tf.keras.models.Model`.
  3. `train_step` / `val_step` / `test_step` / `pred_step` -- rewritten for
     TWO outputs / TWO losses (see point 3 below).
  4. `evaluate()` -- overridden because `mixnet/models/base.py` only
     unpacks multi-output predictions for models named 'MixNet'/'MIN2Net'.

-----------------------------------------------------------------------------
 NOTE: IF YOU GO MULTI-TASK (read this before you start)
-----------------------------------------------------------------------------
 MixNet itself is multi-task: it optimises reconstruction (MSE) + deep metric
 learning (triplet) + classification (cross-entropy) at once, with adaptive
 loss weights. You are allowed to do the same, but the single-task contract
 no longer holds and FOUR things must change together:

 1. `build()` returns a LIST of outputs, with the classifier softmax LAST:
        Model(inputs=input1, outputs=[decoder_out, latent, softmax], ...)

 2. The config (`experiments/configs/HW_Net.py`) must declare one entry per
    task, in the SAME order, in all three lists:
        'loss':         [MeanSquaredError(), triplet_loss(margin=1.0),
                         SparseCategoricalCrossentropy()],
        'loss_names':   ['mse', 'triplet', 'crossentropy'],
        'loss_weights': [1.0, 1.0, 1.0],
    Keep the name 'crossentropy' for the classification loss -- `Trainer`
    looks it up by that name to apply class balancing.

    This example uses only two tasks:
        'loss':         [MeanSquaredError(), SparseCategoricalCrossentropy()],
        'loss_names':   ['mse', 'crossentropy'],
        'loss_weights': [1.0, 1.0],

 3. `train_step` / `val_step` / `test_step` / `pred_step` below must compute
    every loss, combine them with the incoming `loss_weights`, and log one
    `*_<loss_name>_loss` entry per task. `mixnet/models/MixNet.py` is the
    reference implementation -- read its steps side by side with this file.
    The shape of a multi-task step is:

        xr, z, y_logis = self.model(x, training=True)
        mse_loss   = self.loss.mse(x, xr)
        ce_loss    = self.loss.crossentropy(y, y_logis)
        losses     = [mse_loss, ce_loss]                # same order as loss_names
        train_loss = tf.reduce_sum(loss_weights * losses)
        ...
        return logs, (xr, z, y_logis)                  # outputs, classifier last

    The `loss_weights` argument is not decoration: `Trainer` passes the
    adaptive weights in through it every batch. Ignore it and adaptive
    gradient blending silently does nothing.

 4. `evaluate()` / `predict()` in `mixnet/models/base.py` unpack multiple
    outputs ONLY for models whose name starts with 'MixNet' or 'MIN2Net';
    every other model takes the single-output branch, `np.argmax(test_pred,
    axis=1)`, which produces garbage for a list of outputs. Do NOT rename
    your model to sneak into that branch, and do NOT edit `base.py`.
    Override `evaluate()` (and `predict()` if you use it) inside THIS class
    instead -- it is one of the two files you are allowed to change. Copy the
    MixNet branch of `base.py` as your starting point, remembering that
    `Trainer.testing()` hands you a tuple with the classifier output last.

 Multi-task is not required and earns no marks by itself. It earns marks only
 if your report shows what each auxiliary task contributed -- which is an
 ablation: train the same network with and without each extra loss.
-----------------------------------------------------------------------------

Read `mixnet/models/MixNet.py` for the full three-task reference
implementation (adds a triplet/metric-learning term and adaptive gradient
blending on top of what this file does).
"""

import time

import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, f1_score
from tensorflow.keras import layers
from tensorflow.keras.constraints import max_norm
from tensorflow.keras.models import Model

from mixnet import models
from mixnet.loss import *


class HW_Net(models.base.BaseModel):

    def __init__(self,
                 optimizer,
                 input_shape=(1, 20, 400),
                 num_class=2,
                 loss=[MeanSquaredError(), SparseCategoricalCrossentropy()],
                 loss_names=['mse', 'crossentropy'],
                 loss_weights=[1.0, 1.0],
                 model_name='HW_Net',
                 data_format='channels_first',
                 **kwargs):
        super().__init__(num_class, loss, loss_names, loss_weights,
                         optimizer, data_format, **kwargs)
        self.input_shape = input_shape
        self.num_class = num_class
        self.model_name = model_name
        self._config(**kwargs)

    # =========================================================================
    #  Architecture hyper-parameters
    # =========================================================================
    def _config(self, **kwargs):
        """Declare every hyper-parameter your `build()` reads as `self.<name>`.

        Anything you set here can be overridden from
        `experiments/configs/HW_Net.py`, because the loop at the bottom of
        this method re-applies the keyword arguments coming from the config.
        That is how you tune a model without editing this file.
        """
        self.F1 = 8               # number of temporal filters in the encoder
        self.kernel_length = 64   # ~0.64 s at 100 Hz
        self.latent_dim = 32      # size of the shared latent `z`
        self.dropout_rate = 0.5
        self.norm_rate = 0.25
        if self.data_format == 'channels_first':
            self.Chans, self.Samples = self.input_shape[1], self.input_shape[2]
        else:
            self.Chans, self.Samples = self.input_shape[0], self.input_shape[1]

        # Keep this last: it lets the experiment config override the defaults.
        for k in kwargs.keys():
            self.__setattr__(k, kwargs[k])

    # =========================================================================
    #  The network itself -- shared encoder, two heads
    # =========================================================================
    def build(self, print_summary=True, load_weights=False):
        """Build and return the multi-task architecture.

        Called three times per fold (fit, evaluate, predict), so it must be
        deterministic and free of side effects.

        Args:
            print_summary (bool): print `model.summary()` when True.
            load_weights (bool): restore the best checkpoint of this fold.

        Returns:
            tf.keras.models.Model with THREE outputs, classifier last:
            [decoder_out (batch, *input_shape), latent (batch, latent_dim),
             softmax (batch, num_class)].

        Design: one shared encoder (temporal conv + depthwise spatial conv,
        the same idea as EEGNet) produces a latent vector `z`. Two small
        heads read from `z`: a decoder that reconstructs the input (forces
        `z` to keep information about the whole signal, not just whatever
        is easiest for the classifier to exploit) and a classifier. This is
        the smallest architecture that is genuinely multi-task -- both heads
        share and shape the same representation, rather than being two
        unrelated networks bolted together.
        """
        input1 = layers.Input(shape=self.input_shape)

        # ---- shared encoder: temporal + spatial conv, then a latent vector
        enc = layers.Conv2D(self.F1, (1, self.kernel_length), padding='same',
                            use_bias=False)(input1)
        enc = layers.BatchNormalization()(enc)
        enc = layers.DepthwiseConv2D((self.Chans, 1), use_bias=False,
                                     depth_multiplier=1,
                                     depthwise_constraint=max_norm(1.))(enc)
        enc = layers.BatchNormalization()(enc)
        enc = layers.Activation('elu')(enc)
        enc = layers.AveragePooling2D((1, 8))(enc)
        enc = layers.Dropout(self.dropout_rate)(enc)
        enc = layers.Flatten(name='flatten')(enc)
        z = layers.Dense(self.latent_dim, name='z',
                         kernel_constraint=max_norm(0.5))(enc)

        # ---- decoder head: reconstruct the (filtered, normalized) input
        dec = layers.Dense(self.Chans * self.Samples,
                           name='decoder_dense')(z)
        xr = layers.Reshape(self.input_shape, name='decoder_out')(dec)

        # ---- classifier head: left/right hand from the SAME latent `z`
        clf = layers.Dense(self.num_class, name='dense',
                           kernel_constraint=max_norm(self.norm_rate))(z)
        softmax = layers.Activation('softmax', name='softmax')(clf)

        model = Model(inputs=input1, outputs=[xr, z, softmax],
                     name=self.model_name)
        if print_summary:
            model.summary()
        if load_weights:
            print('loading weights from', self.weights_dir)
            model.load_weights(self.weights_dir)
        return model

    # =========================================================================
    #  Custom training / evaluation steps -- TWO outputs, TWO losses
    # -------------------------------------------------------------------------
    #  self.model(x) returns (xr, z, y_logis); self.loss holds 'mse' and
    #  'crossentropy'. `loss_weights` is used (not decoration): it is how
    #  `Trainer`'s adaptive gradient blending, if enabled, reaches the loop.
    # =========================================================================
    @tf.function
    def train_step(self, x, y, loss_weights):
        with tf.GradientTape() as tape:
            xr, z, y_logis = self.model(x, training=True)
            mse_loss = self.loss.mse(x, xr)
            crossentropy_loss = self.loss.crossentropy(y, y_logis)
            losses = [mse_loss, crossentropy_loss]
            train_loss = tf.reduce_sum(loss_weights * losses)
            self.train_acc_metric.update_state(y, y_logis)
        grads = tape.gradient(train_loss, self.model.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_weights))
        logs = dict({'train_loss': train_loss})
        logs.update(dict(zip(['train_' + loss_name + '_loss'
                              for loss_name in self.loss_names], losses)))
        logs.update(dict({'train_acc': self.train_acc_metric.result()}))
        return logs, (xr, z, y_logis)

    @tf.function
    def val_step(self, x, y, loss_weights):
        xr, z, y_logis = self.model(x, training=False)
        mse_loss = self.loss.mse(x, xr)
        crossentropy_loss = self.loss.crossentropy(y, y_logis)
        losses = [mse_loss, crossentropy_loss]
        val_loss = tf.reduce_sum(loss_weights * losses)
        self.val_acc_metric.update_state(y, y_logis)
        logs = dict({'val_loss': val_loss})
        logs.update(dict(zip(['val_' + loss_name + '_loss'
                              for loss_name in self.loss_names], losses)))
        logs.update(dict({'val_acc': self.val_acc_metric.result()}))
        return logs, (xr, z, y_logis)

    @tf.function
    def test_step(self, x, y, loss_weights):
        xr, z, y_logis = self.model(x, training=False)
        mse_loss = self.loss.mse(x, xr)
        crossentropy_loss = self.loss.crossentropy(y, y_logis)
        losses = [mse_loss, crossentropy_loss]
        test_loss = tf.reduce_sum(loss_weights * losses)
        self.test_acc_metric.update_state(y, y_logis)
        logs = dict({'test_loss': test_loss})
        logs.update(dict(zip(['test_' + loss_name + '_loss'
                              for loss_name in self.loss_names], losses)))
        logs.update(dict({'test_acc': self.test_acc_metric.result()}))
        return logs, (xr, z, y_logis)

    @tf.function
    def pred_step(self, x):
        xr, z, y_logis = self.model(x, training=False)
        return (xr, z, y_logis)

    # =========================================================================
    #  evaluate() override -- required for any multi-output model that is
    #  not named 'MixNet'/'MIN2Net' (see point 4 in the docstring above).
    #  This mirrors the MixNet branch of `mixnet/models/base.py`.
    # =========================================================================
    def evaluate(self, X_test, y_test):
        model = self.build(print_summary=self.print_summary, load_weights=True)
        super().compile(model=model)
        start = time.time()
        evaluation, test_pred = super().testing(x=X_test, y=y_test)
        end = time.time()

        y_pred_decoder, zs, y_pred_clf = test_pred[0], test_pred[1:-1], test_pred[-1]
        zs = zs[0] if len(zs) == 1 else np.array(zs)
        y_pred_argm = np.argmax(y_pred_clf, axis=1)
        Y = {'y_true': y_test, 'y_pred': y_pred_argm, 'y_pred_clf': y_pred_clf,
             'latent': zs, 'y_pred_decoder': y_pred_decoder}

        mem_usage = tf.config.experimental.get_memory_info('GPU:0')['current']
        print('Checking average current GPU memory usage', mem_usage)
        print('F1-score is computed based on {}'.format(self.f1_average))
        f1 = f1_score(y_test, y_pred_argm, average=self.f1_average)
        print(classification_report(y_test, y_pred_argm))
        evaluation.update({'f1-score': f1, 'prediction_time': end - start,
                           'memory_usage': mem_usage})
        evaluation.update(dict(zip(['w_' + name + '_loss'
                                    for name in self.loss_names],
                                   self.best_loss_weights.numpy())))
        return Y, evaluation
