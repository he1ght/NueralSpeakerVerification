import glob
import os
import random
from random import shuffle

import librosa
import numpy as np
import torch
from torch.utils.data import Dataset

from configuration import param


def mfccs_and_spec(wav_file, wav_process=False, calc_mfccs=False, calc_mag_db=False):
    sound_file, _ = librosa.core.load(wav_file, sr=param.data.sr)
    window_length = int(param.data.window * param.data.sr)
    hop_length = int(param.data.hop * param.data.sr)
    duration = param.data.tisv_frame * param.data.hop + param.data.window

    # Cut silence and fix length
    if wav_process:
        sound_file, index = librosa.effects.trim(sound_file, frame_length=window_length, hop_length=hop_length)
        length = int(param.data.sr * duration)
        sound_file = librosa.util.fix_length(sound_file, length)

    spec = librosa.stft(sound_file, n_fft=param.data.nfft, hop_length=hop_length, win_length=window_length)
    mag_spec = np.abs(spec)

    mel_basis = librosa.filters.mel(param.data.sr, param.data.nfft, n_mels=param.data.nmels)
    mel_spec = np.dot(mel_basis, mag_spec)

    mag_db = librosa.amplitude_to_db(mag_spec)
    # db mel spectrogram
    mel_db = librosa.amplitude_to_db(mel_spec).T

    mfccs = None
    if calc_mfccs:
        mfccs = np.dot(librosa.filters.dct(40, mel_db.shape[0]), mel_db).T

    return mfccs, mel_db, mag_db


class ZerothKoreanDataset(Dataset):
    def __init__(self, training=False):

        if training:
            self.path = param.data.train_path_unprocessed
            self.utter_num = param.train.M
        else:
            self.path = param.data.test_path_unprocessed
            self.utter_num = param.test.M

        self.speakers = glob.glob(os.path.dirname(self.path))
        shuffle(self.speakers)

    def __len__(self):
        return len(self.speakers)

    def __getitem__(self, idx):

        speaker = self.speakers[idx]
        wav_files = glob.glob(speaker + '/*.WAV')
        shuffle(wav_files)
        wav_files = wav_files[0:self.utter_num]

        mel_dbs = []
        for f in wav_files:
            _, mel_db, _ = mfccs_and_spec(f, wav_process=True)
            mel_dbs.append(mel_db)
        return torch.Tensor(mel_dbs)


class ZerothKoreanDatasetPreprocessed(Dataset):

    def __init__(self, training=False, shuffle=True, utter_start=0):

        # data path
        if training:
            self.path = param.data.train_path
            self.utter_num = param.train.M
        else:
            self.path = param.data.test_path
            self.utter_num = param.test.M
        self.file_list = os.listdir(self.path)
        self.shuffle = shuffle
        self.utter_start = utter_start

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):

        np_file_list = os.listdir(self.path)

        if self.shuffle:
            selected_file = random.sample(np_file_list, 1)[0]  # select random speaker
        else:
            selected_file = np_file_list[idx]

        utters = np.load(os.path.join(self.path, selected_file))  # load utterance spectrogram of selected speaker
        if self.shuffle:
            utter_index = np.random.randint(0, utters.shape[0], self.utter_num)  # select M utterances per speaker
            utterance = utters[utter_index]
        else:
            utterance = utters[
                        self.utter_start: self.utter_start + self.utter_num]  # utterances of a speaker [batch(M), n_mels, frames]

        utterance = utterance[:, :, :160]  # TODO implement variable length batch size

        utterance = torch.tensor(np.transpose(utterance, axes=(0, 2, 1)))  # transpose [batch, frames, n_mels]
        return utterance
