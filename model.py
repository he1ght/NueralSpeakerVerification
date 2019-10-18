__author__ = "Hee-Seung Jung"
__maintainer__ = "Hee-Seung Jung"
__email__ = "heesng.jung@gmail.com"
__status__ = "Production"

import torch
import torch.nn as nn

from configuration import param


class SpeechEmbedder(nn.Module):

    def __init__(self):
        super(SpeechEmbedder, self).__init__()
        self.LSTM_stack = nn.LSTM(param.data.nmels, param.model.hidden, num_layers=param.model.num_layer, batch_first=True)
        for name, p in self.LSTM_stack.named_parameters():
            if 'bias' in name:
                nn.init.constant_(p, 0.0)
            elif 'weight' in name:
                nn.init.xavier_normal_(p)
        self.projection = nn.Linear(param.model.hidden, param.model.proj)

    def forward(self, x):
        x, _ = self.LSTM_stack(x.float())  # (batch, frames, n_mels)
        # only use last frame
        x = x[:, x.size(1) - 1]
        x = self.projection(x.float())
        x = x / torch.norm(x, dim=1).unsqueeze(1)
        return x