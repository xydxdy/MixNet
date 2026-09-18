"""
=============================================================================
 HOMEWORK PART 2 -- BUILD YOUR OWN MODEL  (HW_Net)
=============================================================================

`HW_Net` plugs into the exact same training machinery as MixNet, EEGNet
and DeepConvNet, so the benchmark in Part 3 is a fair comparison: 
same folds, same subjects, same early stopping, same metrics.

WHAT IS ALREADY DONE FOR YOU (do not change unless you know why):
  * `train_step` / `val_step` / `test_step` / `pred_step` -- the custom
    TensorFlow loops that `mixnet.trainer.Trainer` calls every batch.
    NOTE: they are written for a SINGLE-TASK model -- one output, one loss
    (cross-entropy). If you go multi-task you MUST rewrite them; see
    "IF YOU GO MULTI-TASK" below.
  * inheritance from `BaseModel`, which gives you `.fit()`, `.evaluate()`
    and `.predict()` for free.

WHAT YOU MUST IMPLEMENT:
  1. `_config()` -- declare your architecture hyper-parameters.
  2. `build()`   -- return a `tf.keras.models.Model`.

-----------------------------------------------------------------------------
 CONTRACT YOUR `build()` MUST RESPECT
-----------------------------------------------------------------------------
  * Input : one tensor of shape `self.input_shape` (no batch axis).
  * Output: ONE tensor of shape (batch, num_class) holding probabilities,
            i.e. finish with `layers.Activation('softmax')`.
            Multi-output models take a different code path in
            `mixnet/models/base.py` and need extra work -- see
            "IF YOU GO MULTI-TASK" below before you try one.
  * Name  : pass `name=self.model_name`. Do NOT name it 'MixNet' or
            'MIN2Net' -- `base.py` branches on those names.
  * When `load_weights=True` you must call `model.load_weights(self.weights_dir)`
    (kept in the template below), otherwise `evaluate()` silently scores an
    untrained network.
-----------------------------------------------------------------------------

-----------------------------------------------------------------------------
 NOTE: IF YOU GO MULTI-TASK (optional, advanced -- read this before you start)
-----------------------------------------------------------------------------
 MixNet itself is multi-task: it optimises reconstruction (MSE) + deep metric
 learning (triplet) + classification (cross-entropy) at once, with adaptive
 loss weights. You are allowed to do the same, but the single-task contract
 above then no longer holds and FOUR things must change together:

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

 3. `train_step` / `val_step` / `test_step` / `pred_step` below must compute
    every loss, combine them with the incoming `loss_weights`, and log one
    `*_<loss_name>_loss` entry per task. `mixnet/models/MixNet.py` is the
    reference implementation -- read its steps side by side with the
    single-task ones below. The shape of a multi-task step is:

        xr, z, y_logis = self.model(x, training=True)
        mse_loss   = self.loss.mse(x, xr)
        trp_loss   = self.loss.triplet(y, z)
        ce_loss    = self.loss.crossentropy(y, y_logis)
        losses     = [mse_loss, trp_loss, ce_loss]     # same order as loss_names
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

Minimal smoke-test architecture -- paste this into `build()` to verify the
plumbing end-to-end BEFORE you design anything clever, then replace it:

        input1  = layers.Input(shape=self.input_shape)
        x       = layers.Flatten()(input1)
        x       = layers.Dense(self.num_class)(x)
        softmax = layers.Activation('softmax', name='softmax')(x)
        model   = Model(inputs=input1, outputs=softmax, name=self.model_name)

Read `mixnet/models/EEGNet.py` and `mixnet/models/DeepConvNet.py` for two
complete, working examples of this same contract.
"""

import tensorflow as tf
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
                 loss=[SparseCategoricalCrossentropy()],
                 loss_names=['crossentropy'],
                 loss_weights=[1.0],
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
    #  TODO 1 / 2 -- architecture hyper-parameters
    # =========================================================================
    def _config(self, **kwargs):
        """Declare every hyper-parameter your `build()` reads as `self.<name>`.

        Anything you set here can be overridden from
        `experiments/configs/HW_Net.py`, because the loop at the bottom of
        this method re-applies the keyword arguments coming from the config.
        That is how you tune a model without editing this file.

        `self.input_shape` is already set. Derive the number of channels and
        time samples from it -- the layout depends on `data_format`, exactly
        as in EEGNet:

            channels_first -> input_shape = (depth, n_channels, n_samples)
            channels_last  -> input_shape = (n_channels, n_samples, depth)
        """
        self.F1 = 8               # number of temporal filters
        self.kernel_length = 64   # ~0.64 s at 100 Hz
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
    #  TODO 2 / 2 -- the network itself
    # =========================================================================
    def build(self, print_summary=True, load_weights=False):
        """Build and return your architecture as a `tf.keras.models.Model`.

        Called three times per fold (fit, evaluate, predict), so it must be
        deterministic and free of side effects.

        Args:
            print_summary (bool): print `model.summary()` when True.
            load_weights (bool): restore the best checkpoint of this fold.

        Returns:
            tf.keras.models.Model with a single (batch, num_class) softmax output.

        Design ideas -- you must be able to explain WHY you chose yours:
            * temporal conv + depthwise spatial conv (the EEGNet family)
            * multi-scale / inception-style parallel temporal kernels
            * separable convolutions + squeeze-and-excitation attention
            * a small transformer encoder over time windows
            * a GRU/LSTM head over convolutional features
            * a plain MLP on the band-power features from Part 1

        Useful regularisers on this dataset (it is small -- ~115 training
        trials per fold): `max_norm` kernel constraints, dropout,
        batch normalisation, and keeping the parameter count low.

        This implementation is a minimal, single-branch temporal + spatial
        conv net (an EEGNet-lite): one Conv2D learns `F1` temporal filters
        along the time axis, one DepthwiseConv2D collapses the 20-channel
        axis into a single spatial filter per temporal filter, then a small
        dense head classifies the pooled features. It is intentionally
        shallow so it is easy to read end to end and still gives a
        legitimate (non-toy) baseline to compare against MixNet.
        """
        input1 = layers.Input(shape=self.input_shape)

        # temporal convolution: F1 learned band-pass-like filters over time
        block1 = layers.Conv2D(self.F1, (1, self.kernel_length),
                               padding='same', use_bias=False)(input1)
        block1 = layers.BatchNormalization()(block1)

        # spatial convolution: collapse all 20 channels into one spatial
        # filter per temporal filter (depth_multiplier=1 keeps it minimal)
        block1 = layers.DepthwiseConv2D((self.Chans, 1), use_bias=False,
                                        depth_multiplier=1,
                                        depthwise_constraint=max_norm(1.))(block1)
        block1 = layers.BatchNormalization()(block1)
        block1 = layers.Activation('elu')(block1)
        block1 = layers.AveragePooling2D((1, 8))(block1)
        block1 = layers.Dropout(self.dropout_rate)(block1)

        flatten = layers.Flatten(name='flatten')(block1)
        dense = layers.Dense(self.num_class, name='dense',
                             kernel_constraint=max_norm(self.norm_rate))(flatten)
        softmax = layers.Activation('softmax', name='softmax')(dense)

        # ---- required epilogue, keep it once your graph is defined ----------
        model = Model(inputs=input1, outputs=softmax, name=self.model_name)
        if print_summary:
            model.summary()
        if load_weights:
            print('loading weights from', self.weights_dir)
            model.load_weights(self.weights_dir)
        return model

    # =========================================================================
    #  Provided for you -- custom training / evaluation steps
    # -------------------------------------------------------------------------
    #  SINGLE-TASK ONLY. Every step below assumes `self.model(x)` returns ONE
    #  tensor and that `self.loss` holds exactly one entry, `crossentropy`.
    #  NOTE: that `loss_weights` is accepted but unused here: with a single loss
    #  there is nothing to weight.
    # =========================================================================
    @tf.function
    def train_step(self, x, y, loss_weights):
        with tf.GradientTape() as tape:
            # NOTE: You may rewrite the following line to unpack multiple outputs if you go multi-task 
            # or use other loss functions. See MixNet for reference.
            y_logis = self.model(x, training=True)
            crossentropy_loss = self.loss.crossentropy(y, y_logis)
            losses = [crossentropy_loss]
            train_loss = tf.reduce_sum(crossentropy_loss)
            self.train_acc_metric.update_state(y, y_logis)
        grads = tape.gradient(train_loss, self.model.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_weights))
        logs = dict({'train_loss': train_loss})
        logs.update(dict(zip(['train_' + loss_name + '_loss'
                              for loss_name in self.loss_names], losses)))
        logs.update(dict({'train_acc': self.train_acc_metric.result()}))
        return logs, y_logis

    @tf.function
    def val_step(self, x, y, loss_weights):
        # NOTE: You may rewrite the following line to unpack multiple outputs if you go multi-task 
        # or use other loss functions. See MixNet for reference.
        y_logis = self.model(x, training=False)
        crossentropy_loss = self.loss.crossentropy(y, y_logis)
        losses = [crossentropy_loss]
        val_loss = tf.reduce_sum(crossentropy_loss)
        self.val_acc_metric.update_state(y, y_logis)
        logs = dict({'val_loss': val_loss})
        logs.update(dict(zip(['val_' + loss_name + '_loss'
                              for loss_name in self.loss_names], losses)))
        logs.update(dict({'val_acc': self.val_acc_metric.result()}))
        return logs, y_logis

    @tf.function
    def test_step(self, x, y, loss_weights):
        # NOTE: You may rewrite the following line to unpack multiple outputs if you go multi-task 
        # or use other loss functions. See MixNet for reference.
        y_logis = self.model(x, training=False)
        crossentropy_loss = self.loss.crossentropy(y, y_logis)
        losses = [crossentropy_loss]
        test_loss = tf.reduce_sum(crossentropy_loss)
        self.test_acc_metric.update_state(y, y_logis)
        logs = dict({'test_loss': test_loss})
        logs.update(dict(zip(['test_' + loss_name + '_loss'
                              for loss_name in self.loss_names], losses)))
        logs.update(dict({'test_acc': self.test_acc_metric.result()}))
        return logs, y_logis

    @tf.function
    def pred_step(self, x):
        # NOTE: You may rewrite the following line to unpack multiple outputs if you go multi-task 
        # or use other loss functions. See MixNet for reference.
        y_logis = self.model(x, training=False)
        return y_logis
