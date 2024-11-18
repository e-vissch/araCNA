from functools import partial
from typing import TypeAlias

import torch
import torch.nn as nn
from git import Union
from omegaconf import OmegaConf

from aracna.configs import schemas
from aracna.src.models.backbone import registry as backbone_registry
from aracna.src.models.embeddings import registry as embedding_registry
from aracna.src.models.standalone_hyenadna import _init_weights
from aracna.src.utils.config import get_object_from_registry

# actually OmegaConf but validated against schemas.ModelConfig
# so should have same structure
ModelConfig: TypeAlias = Union[OmegaConf, schemas.ModelConfig]


class Aracna(nn.Module):
    def __init__(self, config: ModelConfig, decoder):
        super().__init__()

        self.embeddings = get_object_from_registry(
            config.embeddings, embedding_registry
        )

        self.backbone = get_object_from_registry(config.backbone, backbone_registry)

        # decoder depends on task
        self.decoder = decoder

        # Initialize weights and apply final processing
        self.apply(
            partial(
                _init_weights,
                n_layer=config.n_layer,
                **(config.get("initializer_cfg", {})),
            )
        )

    def hidden_embeddings(self, inputs, position_features=None):
        input_embeddings = self.embeddings(inputs, position_features=position_features)
        return self.backbone(input_embeddings)

    def forward(self, inputs, position_features=None):
        hidden_states = self.hidden_embeddings(
            inputs, position_features=position_features
        )
        return self.decoder(hidden_states)

    def inference(self, inputs, position_features=None, hidden=False):
        # TODO: is this the best place for this?
        self.eval()
        with torch.inference_mode():
            infer_func = self.hidden_embeddings if hidden else partial(self.forward)
            output = infer_func(inputs, position_features=position_features)
        self.train()
        return output
