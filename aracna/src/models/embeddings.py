import math

import torch
import torch.nn as nn


class SimpleCnaEmbedding(nn.Module):
    def __init__(
        self,
        input_dim,
        embed_dim,
        positional_feature_size=None,
    ):
        super().__init__()

        self.value_embeddings = nn.Linear(input_dim, embed_dim)
        self.feature_embeddings = (
            None
            if positional_feature_size is None
            else nn.Linear(positional_feature_size, embed_dim)
        )

    def forward(self, inputs, position_features=None):
        """
        inputs: (batch, seqlen, input_dim)
        position_features: (batch, seqlen)
        """

        embeddings = self.value_embeddings(inputs)

        if self.feature_embeddings is not None:
            embeddings += self.feature_embeddings(position_features)

        return embeddings


class PositionalEncoder(nn.Module):
    # standard positional encoding adapted for non-sequential positions
    def __init__(self, d_model):
        super().__init__()
        self.register_buffer(
            "div_term",
            torch.exp(-math.log(10000.0) * torch.arange(0.0, d_model, 2.0) / d_model),
        )  # ensures gets moved to device with model
        self.d_model = d_model

    def forward(self, x):
        # x has shape (batch, seqlen, 1), as each element in batch may have different
        # positions for each sequence, need to redefine. Also don't want saved model to
        # be batch_size dependent
        pe = torch.zeros(*x.shape[:-1], self.d_model, device=x.device)
        pe[..., 1::2] = torch.cos(x * self.div_term)
        pe[..., 0::2] = torch.sin(x * self.div_term)
        return pe


def get_embeddings(input_dim, output_dim):
    return nn.Sequential(
        nn.Linear(input_dim, output_dim // 2),
        nn.ReLU(),
        nn.Linear(output_dim // 2, output_dim),
        nn.ReLU(),
        nn.Linear(output_dim, output_dim),
    )


class RealProfileCnaEmbeddings(nn.Module):
    def __init__(
        self,
        input_dim,
        embed_dim,
        chromosome_dim,
        token_dim,
        include_position=True,
        include_chromosome=True,
        lin_embeds=False,
    ):
        super().__init__()

        if lin_embeds:
            self.value_embeddings = nn.Linear(input_dim, embed_dim)
        else:
            self.value_embeddings = get_embeddings(input_dim, embed_dim)

        self.include_position = include_position
        self.include_chromosome = include_chromosome

        self.positional_embeddings = PositionalEncoder(embed_dim)
        self.chromosome_embeddings = nn.Embedding(
            chromosome_dim + 1, embed_dim, padding_idx=0
        )  # +1 for paddin

        self.token_embeddings = nn.Embedding(token_dim, embed_dim)

    def forward(self, inputs, position_features):
        """
        inputs: (batch, seqlen, input_dim)
        position_features: (batch, seqlen)
        """
        positions, chromosomes, tokens = (
            position_features[..., 0:1],
            position_features[..., 1:2],
            position_features[..., 2:],
        )

        embeddings = self.value_embeddings(inputs)

        if self.include_position:
            embeddings += self.positional_embeddings(positions)
        if self.include_chromosome:
            embeddings += self.chromosome_embeddings(chromosomes.squeeze(-1))

        return embeddings + self.token_embeddings(tokens.squeeze(-1))


registry = {
    "simple_cna": SimpleCnaEmbedding,
    "real_cna": RealProfileCnaEmbeddings,
}
