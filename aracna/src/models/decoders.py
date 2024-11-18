from math import ceil

import torch
import torch.nn as nn


def basic_decoder(input_dim, output_dim):
    return nn.Linear(input_dim, output_dim)


def get_decoder(input_dim, output_dim):
    return nn.Sequential(
        nn.Linear(input_dim, input_dim // 2),
        nn.ReLU(),
        nn.Linear(input_dim // 2, input_dim // 4),
        nn.ReLU(),
        nn.Linear(input_dim // 4, output_dim),
    )


def get_cat_decoder(input_dim, output_dim):
    return nn.Sequential(
        nn.Linear(input_dim, 2 * input_dim),
        nn.ReLU(),
        nn.Linear(2 * input_dim, 2 * output_dim),
        nn.ReLU(),
        nn.Linear(2 * output_dim, output_dim),
        nn.ReLU(),
        nn.Linear(output_dim, output_dim),
    )


class SimpleCnaDecoder(nn.Module):
    def __init__(self, decoder_dim, out_dim):
        super().__init__()

        self.decoder = nn.Sequential(get_decoder(decoder_dim, out_dim), nn.Softplus())

    def forward(self, decoder_inputs):
        return self.decoder(decoder_inputs)


class PairedClassificationGlobalDecoder(nn.Module):
    def __init__(self, decoder_dim, max_tot_cn):
        super().__init__()
        tot_incl_0 = max_tot_cn + 1
        # add one for TCN+ category
        self.n_combinations = ceil(tot_incl_0 * (tot_incl_0 + 2) / 4) + 1
        # output for major and minor are logits

        self.decoder = get_cat_decoder(decoder_dim, self.n_combinations)
        self.read_decoder = nn.Sequential(get_decoder(decoder_dim, 1), nn.Softplus())
        self.purity_decoder = nn.Sequential(get_decoder(decoder_dim, 1), nn.Sigmoid())

    def forward(self, decoder_inputs):
        return (
            self.decoder(decoder_inputs),
            torch.cat(
                (
                    self.read_decoder(decoder_inputs),
                    self.purity_decoder(decoder_inputs),
                ),
                dim=-1,
            ),
        )


registry = {
    "simple": SimpleCnaDecoder,
    "paired": PairedClassificationGlobalDecoder,
}
