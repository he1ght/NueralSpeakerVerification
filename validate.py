__author__ = "Hee-Seung Jung"
__maintainer__ = "Hee-Seung Jung"
__email__ = "heesng.jung@gmail.com"
__status__ = "Production"

import random
import random

import torch
from torch.utils.data import DataLoader

from configuration import param
from load_data import ZerothKoreanDataset, ZerothKoreanDatasetPreprocessed
from model import SpeechEmbedder
from utils import get_centroids, get_cossim

if __name__ == "__main__":
    if not param.cuda or not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device("cuda")
    if param.data.data_preprocessed:
        test_dataset = ZerothKoreanDatasetPreprocessed()
    else:
        test_dataset = ZerothKoreanDataset()
    test_loader = DataLoader(test_dataset, batch_size=param.test.batch, shuffle=True,
                             num_workers=param.test.num_workers,
                             drop_last=True)

    embedder_net = SpeechEmbedder().to(device)
    embedder_net.load_state_dict(torch.load(param.test.model_path))
    embedder_net.eval()

    avg_EER = 0
    for e in range(param.test.epochs):
        batch_avg_EER = 0
        batch_id = 0
        print("Epoch {} ...".format(e + 1))
        for batch_id, mel_db_batch in enumerate(test_loader):
            assert param.test.utters % 2 == 0

            enrollment_batch, verification_batch = torch.split(mel_db_batch, int(mel_db_batch.size(1) / 2), dim=1)
            enrollment_batch = enrollment_batch.to(device)
            verification_batch = verification_batch.to(device)

            enrollment_batch = torch.reshape(enrollment_batch, (
                param.test.batch * param.test.utters // 2, enrollment_batch.size(2), enrollment_batch.size(3)))
            verification_batch = torch.reshape(verification_batch, (
                param.test.batch * param.test.utters // 2, verification_batch.size(2), verification_batch.size(3)))

            perm = random.sample(range(0, verification_batch.size(0)), verification_batch.size(0))
            unperm = list(perm)
            for i, j in enumerate(perm):
                unperm[j] = i

            verification_batch = verification_batch[perm]
            enrollment_embeddings = embedder_net(enrollment_batch)
            verification_embeddings = embedder_net(verification_batch)
            verification_embeddings = verification_embeddings[unperm]

            enrollment_embeddings = torch.reshape(enrollment_embeddings,
                                                  (param.test.batch, param.test.utters // 2,
                                                   enrollment_embeddings.size(1)))
            verification_embeddings = torch.reshape(verification_embeddings,
                                                    (param.test.batch, param.test.utters // 2,
                                                     verification_embeddings.size(1)))

            enrollment_centroids = get_centroids(enrollment_embeddings)

            sim_matrix = get_cossim(verification_embeddings, enrollment_centroids)

            # calculating EER
            diff = 1
            EER = 0
            EER_thresh = 0
            EER_FAR = 0  # False acceptance rate
            EER_FRR = 0  # False Rejection Rate

            for thres in [0.01 * tr + 0.5 for tr in range(50)]:
                sim_matrix_thresh = sim_matrix > thres

                FAR = (sum([sim_matrix_thresh[i].float().sum() - sim_matrix_thresh[i, :, i].float().sum() for i in
                            range(int(param.test.batch))])
                       / (param.test.batch - 1.0) / (float(param.test.utters / 2)) / param.test.batch)

                FRR = (sum(
                    [param.test.utters / 2 - sim_matrix_thresh[i, :, i].float().sum() for i in
                     range(int(param.test.batch))])
                       / (float(param.test.utters / 2)) / param.test.batch)

                # Save threshold when FAR = FRR (=EER)
                if diff > abs(FAR - FRR):
                    diff = abs(FAR - FRR)
                    EER = (FAR + FRR) / 2
                    EER_thresh = thres
                    EER_FAR = FAR
                    EER_FRR = FRR
            batch_avg_EER += EER
            print("\nEER : %0.2f (thres:%0.2f, FAR:%0.2f, FRR:%0.2f)" % (EER, EER_thresh, EER_FAR, EER_FRR))
        avg_EER += batch_avg_EER / (batch_id + 1)
    avg_EER = avg_EER / param.test.epochs
    print("\n EER across {0} epochs: {1:.4f}".format(param.test.epochs, avg_EER))
