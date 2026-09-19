"""
=============================================================================
 EXPERIMENT CONFIG FOR HWNet  (BCIC2a, subject-dependent)
=============================================================================

`experiments/run_HWNet.py` loads this file through
`configs/exp_config.get_params()` and uses the returned dotdict to build the
model, the DataLoader and the log directory.

You will come back to this file constantly -- it is where you tune the
training model without touching model code, because every key of
`model_params` is forwarded to `HWNet.__init__` and re-applied as
`self.<key>` at the end of `HWNet._config()`.

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
matches the input shape you defined in `HWNet.__init__()`.

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

from mixnet.utils import dotdict

# =============================================================================
#  TODO -- keep in sync with your Part 1 pipeline
# =============================================================================
def get_params(**kwargs):
    """Return a dotdict of parameters for `run_HWNet.py`."""
    
    raise NotImplementedError("You must implement `get_params()` in your config file.")
