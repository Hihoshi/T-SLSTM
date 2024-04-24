import torch
import torch.nn as nn
from binarizelayer import BinarizeLayer
import snntorch as snn
from snntorch import surrogate
from snntorch import utils


spike_grad = surrogate.atan()


class TSLSTM(torch.nn.Module):
    def __init__(self, device, num_embeddings, embedding_dim, num_mixers=2):  # num_mixers >= 2 and must even
        super(TSLSTM, self).__init__()
        self.encoder = nn.Sequential(
            nn.Embedding(num_embeddings, embedding_dim, max_norm=1, padding_idx=0),
            nn.Flatten(),
            BinarizeLayer(),
        ).to(device)
        self.mixers = [
            nn.Sequential(
                nn.Linear(embedding_dim, embedding_dim),
                snn.SLSTM(embedding_dim, embedding_dim, spike_grad=spike_grad, init_hidden=True, learn_threshold=True),
            ).to(device)
            for i in range(num_mixers)
        ]
        self.decoder = nn.Sequential(
            nn.Linear(embedding_dim, num_embeddings),
            snn.Leaky(beta=0.5, spike_grad=spike_grad, init_hidden=True, learn_beta=True, output=True)
        ).to(device)
        self.device = device

    def forward(self, data):
        # ret net
        utils.reset(self.encoder)
        utils.reset(self.decoder)
        for i in range(len(self.mixers)):
            utils.reset(self.mixers[i])
        # forward
        encoder_out, decoder_out = [], []

        for step in range(data.size(0)):  # data.size(0) = number of time steps
            spk_out = self.encoder(data[step])
            encoder_out.append(spk_out)

        temp = encoder_out
        for mixer in self.mixers:
            mixer_out = []
            for step in range(len(temp)):
                spk_out = mixer(temp[step])
                mixer_out.append(spk_out)
            temp = mixer_out[::-1]  # invert sequence of output spikes

        for step in range(len(mixer_out)):
            spk_out, _ = self.decoder(mixer_out[step])
            decoder_out.append(spk_out)
        return torch.stack(decoder_out)
