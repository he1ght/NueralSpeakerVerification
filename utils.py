import numpy as np
import torch
import torch.nn.functional as F


def get_centroids(emb):
    cents = []
    for speaker in emb:
        cents = 0
        for utterance in speaker:
            cents = cents + utterance
        cents = cents / len(speaker)
        cents.append(cents)
    cents = torch.stack(cents)
    return cents


def get_centroid(emb, speaker_num, utterance_num):
    cent = 0
    for utterance_id, utterance in enumerate(emb[speaker_num]):
        if utterance_id == utterance_num:
            continue
        cent = cent + utterance
    cent = cent / (len(emb[speaker_num]) - 1)
    return cent


def get_cossim(emb, cents):
    # Calculates cosine similarity matrix. Requires (N, M, feature) input
    cossim = torch.zeros(emb.size(0), emb.size(1), cents.size(0))
    for speaker_num, speaker in enumerate(emb):
        for utterance_num, utterance in enumerate(speaker):
            for cent_num, cent in enumerate(cents):
                if speaker_num == cent_num:
                    cent = get_centroid(emb, speaker_num, utterance_num)
                output = F.cosine_similarity(utterance, cent, dim=0) + 1e-6
                cossim[speaker_num][utterance_num][cent_num] = output
    return cossim


def normalize_0_1(values, max_value, min_value):
    normalized = np.clip((values - min_value) / (max_value - min_value), 0, 1)
    return normalized



