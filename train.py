import os
import random
import time
import torch
from torch.utils.data import DataLoader

from configuration import param
from load_data import ZerothKoreanDataset, ZerothKoreanDatasetPreprocessed
from model import SpeechEmbedder
from utils import get_centroids, get_cossim
from loss import GE2ELoss


if __name__ == "__main__":
    device = torch.device(param.device)

    if param.data.data_preprocessed:
        train_dataset = ZerothKoreanDatasetPreprocessed(training=True)
    else:
        train_dataset = ZerothKoreanDataset(training=True)
    train_loader = DataLoader(train_dataset, batch_size=param.train.N, shuffle=True,
                              num_workers=param.train.num_workers,
                              drop_last=True)

    embedder_net = SpeechEmbedder().to(device)
    if param.train.restore:
        embedder_net.load_state_dict(torch.load(param.model_path))
    ge2e_loss = GE2ELoss(device)
    # Both net and loss have trainable parameters
    optimizer = torch.optim.SGD([
        {'params': embedder_net.parameters()},
        {'params': ge2e_loss.parameters()}
    ], lr=param.train.lr)

    os.makedirs(param.train.checkpoint_dir, exist_ok=True)

    embedder_net.train()
    iteration = 0
    for e in range(param.train.epochs):
        total_loss = 0
        for batch_id, mel_db_batch in enumerate(train_loader):
            mel_db_batch = mel_db_batch.to(device)

            mel_db_batch = torch.reshape(mel_db_batch,
                                         (param.train.N * param.train.M, mel_db_batch.size(2), mel_db_batch.size(3)))
            perm = random.sample(range(0, param.train.N * param.train.M), param.train.N * param.train.M)
            unperm = list(perm)
            for i, j in enumerate(perm):
                unperm[j] = i
            mel_db_batch = mel_db_batch[perm]
            # gradient accumulates
            optimizer.zero_grad()

            embeddings = embedder_net(mel_db_batch)
            embeddings = embeddings[unperm]
            embeddings = torch.reshape(embeddings, (param.train.N, param.train.M, embeddings.size(1)))

            # get loss, call backward, step optimizer
            loss = ge2e_loss(embeddings)  # wants (Speaker, Utterances, embedding)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(embedder_net.parameters(), 3.0)
            torch.nn.utils.clip_grad_norm_(ge2e_loss.parameters(), 1.0)
            optimizer.step()

            total_loss = total_loss + loss
            iteration += 1
            if (batch_id + 1) % param.train.log_interval == 0:
                mesg = "{0}\tEpoch:{1}[{2}/{3}],Iteration:{4}\tLoss:{5:.4f}\tTLoss:{6:.4f}\t\n".format(time.ctime(),
                                                                                                       e + 1,
                                                                                                       batch_id + 1,
                                                                                                       len(
                                                                                                           train_dataset) // param.train.N,
                                                                                                       iteration, loss,
                                                                                                       total_loss / (
                                                                                                               batch_id + 1))
                print(mesg)
                if param.train.log_file is not None:
                    with open(param.train.log_file, 'a') as f:
                        f.write(mesg)

        if param.train.checkpoint_dir is not None and (e + 1) % param.train.checkpoint_interval == 0:
            embedder_net.eval().cpu()
            ckpt_model_filename = param.train.checkpoint_name + "_e" + str(e + 1) + "_bid_" + str(batch_id + 1) + ".pth"
            ckpt_model_path = os.path.join(param.train.checkpoint_dir, ckpt_model_filename)
            torch.save(embedder_net.state_dict(), ckpt_model_path)
            embedder_net.to(device).train()

    # save model
    embedder_net.eval().cpu()
    save_model_filename = "final_epoch_" + str(e + 1) + "_batch_id_" + str(batch_id + 1) + ".model"
    save_model_path = os.path.join(param.train.checkpoint_dir, save_model_filename)
    torch.save(embedder_net.state_dict(), save_model_path)

    print("\nDone, trained model saved at", save_model_path)
