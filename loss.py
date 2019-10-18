__author__ = "Hee-Seung Jung"
__maintainer__ = "Hee-Seung Jung"
__email__ = "heesng.jung@gmail.com"
__status__ = "Production"

import torch
import torch.nn as nn

from utils import get_centroids, get_cossim


def calc_loss(sim_mat, eps=1e-6):
    # Calculates loss from (N, M, K) similarity matrix
    per_emb_loss = torch.zeros(sim_mat.size(0), sim_mat.size(1))
    for j in range(len(sim_mat)):
        for i in range(sim_mat.size(1)):
            per_emb_loss[j][i] = -(sim_mat[j][i][j] - ((torch.exp(sim_mat[j][i]).sum() + eps).log_()))
    tot_loss = per_emb_loss.sum()
    return tot_loss, per_emb_loss


class GE2ELoss(nn.Module):

    def __init__(self, device, init_w=10.0, init_b=-5.0, eps=1e-6):
        super(GE2ELoss, self).__init__()
        self.weight = nn.Parameter(torch.tensor(init_w).to(device), requires_grad=True)
        self.bias = nn.Parameter(torch.tensor(init_b).to(device), requires_grad=True)
        self.device = device
        self.eps = eps

    def forward(self, embeddings):
        # 1. prune weight
        torch.clamp(self.weight, self.eps)
        # 2. get centroid from embedding
        centroids = get_centroids(embeddings)
        # 3. get cosin similarity between embedding and centroid.
        cossim = get_cossim(embeddings, centroids)
        # 4. get similarity matrix: Wx + b(x: cosin similarity)
        sim_matrix = self.weight * cossim.to(self.device) + self.bias
        # 5. get loss from similarity matrix
        loss, _ = calc_loss(sim_matrix, eps=self.eps)
        return loss
