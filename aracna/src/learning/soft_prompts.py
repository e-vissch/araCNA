import torch
import torch.nn as nn
from omegaconf import DictConfig

from aracna.src.models.aracna import Aracna


def freeze_model(model):
    for param in model.parameters():
        param.requires_grad = False
    return model


class AracnaSoftPrompt(nn.Module):
    def __init__(self, model: Aracna, config: DictConfig):
        super().__init__()
        self.model = freeze_model(model)
        self.model.eval()

        self.n_soft_tokens = config.task.info.prepend_len
        self.soft_tokens = torch.nn.Parameter(
            torch.zeros(self.n_soft_tokens, config.model.d_model)
        )

    def forward(self, input, position_features=None):
        embeddings = self.model.embeddings(input, position_features=position_features)
        # add soft tokens to the embeddings,
        # should be defined in prepend_len of the model
        embeddings[:, : self.n_soft_tokens] = self.soft_tokens
        hidden_embeds = self.model.backbone(embeddings)
        return self.model.decoder(hidden_embeds)
