import warnings

import torch.nn as nn

from neuralhydrology.modelzoo.cudalstm import CudaLSTM
from neuralhydrology.modelzoo.ealstm import EALSTM
from neuralhydrology.utils.config import Config

SINGLE_FREQ_MODELS = [
    "cudalstm",
    "ealstm", 
]

def get_model(cfg: Config) -> nn.Module:
    """Get model object, depending on the run configuration.
    
    Parameters
    ----------
    cfg : Config
        The run configuration.

    Returns
    -------
    nn.Module
        A new model instance of the type specified in the config.
    """

    if cfg.model.lower() == "cudalstm":
        model = CudaLSTM(cfg=cfg)
    elif cfg.model.lower() == "ealstm":
        model = EALSTM(cfg=cfg)
    elif cfg.model.lower() == "lstm":
        warnings.warn(
            "The `LSTM` class has been renamed to `CustomLSTM`. Support for `LSTM` will we dropped in the future.",
            FutureWarning)
        model = CustomLSTM(cfg=cfg)
    else:
        raise NotImplementedError(f"{cfg.model} not implemented or not linked in `get_model()`")

    return model
