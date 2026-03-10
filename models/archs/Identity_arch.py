import torch
import torch.nn as nn
from transformers import PreTrainedModel, PretrainedConfig

class Identity_config(PretrainedConfig):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

class Identity_arch(PreTrainedModel):
    config_class = Identity_config
    def __init__(self, config):
        super().__init__(config)
        self.identity = nn.Identity()

    def forward(self, x):
        return self.identity(x)