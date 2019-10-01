import torch
import torch.nn as nn

from utils import get_centroids, get_cossim


def calc_loss(sim_mat):
    # Calculates loss from (N, M, K) similarity matrix
    per_emb_loss = torch.zeros(sim_mat.size(0), sim_mat.size(1))
    for j in range(len(sim_mat)):
        for i in range(sim_mat.size(1)):
            per_emb_loss[j][i] = -(sim_mat[j][i][j] - ((torch.exp(sim_mat[j][i]).sum() + 1e-6).log_()))
    tot_loss = per_emb_loss.sum()
    return tot_loss, per_emb_loss


class GE2ELoss(nn.Module):

    def __init__(self, device, init_w=10.0, init_b=-5.0):
        super(GE2ELoss, self).__init__()
        self.weight = nn.Parameter(torch.tensor(init_w).to(device), requires_grad=True)
        self.bias = nn.Parameter(torch.tensor(init_b).to(device), requires_grad=True)
        self.device = device

    def forward(self, embeddings):
        # 1. prune data
        torch.clamp(self.weight, 1e-6)
        # 2. get centroid from embedding
        centroids = get_centroids(embeddings)
        # 3. get cosin similarity between embedding and centroid.
        cossim = get_cossim(embeddings, centroids)
        # 4. get similarity matrix: Wx + b(x: cosin similarity)
        sim_matrix = self.weight * cossim.to(self.device) + self.bias
        # 5. get loss from similarity matrix
        loss, _ = calc_loss(sim_matrix)
        return loss
