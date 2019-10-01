import glob
import os

import librosa
import numpy as np

from configuration import param

# Motivate from "https://github.com/Janghyun1230/Speaker_Verification/blob/master/data_preprocess.py"

# downloaded dataset path
train_path = glob.glob(os.path.dirname(param.data.train_path_unprocessed))
test_path = glob.glob(os.path.dirname(param.data.test_path_unprocessed))


def preprocessing():
    """
        TI-SV (Text Independent - Speaker Verification)
        Each partial utterance is splitted by voice detection using DB
        and the first and the last 180 frames from each partial utterance are saved.
    """
    print("start text independent utterance feature extraction")
    os.makedirs(param.data.train_path, exist_ok=True)  # make folder to save train file
    os.makedirs(param.data.test_path, exist_ok=True)  # make folder to save test file

    utter_min_len = (
                                param.data.tisv_frame * param.data.hop + param.data.window) * param.data.sr  # lower bound of utterance length

    train_speaker_num = len(train_path)
    test_speaker_num = len(test_path)
    print("total speaker number : %d" % (train_speaker_num + test_speaker_num))
    print("train : %d, test : %d" % (train_speaker_num, test_speaker_num))

    def save_tisv(training=True):
        path = train_path if training else test_path
        name_tag = "[Train]" if training else "[Test] "
        for i, folder in enumerate(path):
            print("%s %dth speaker processing..." % (name_tag, i))
            utterances_spec = []
            for utter_name in os.listdir(folder):
                if utter_name[-4:].lower() == '.wav' or utter_name[-5:].lower() == '.flac':
                    utter_path = os.path.join(folder, utter_name)  # path of each utterance
                    utter, sr = librosa.core.load(utter_path, param.data.sr)  # load utterance audio
                    intervals = librosa.effects.split(utter, top_db=30)  # voice activity detection
                    for interval in intervals:
                        if (interval[1] - interval[0]) > utter_min_len:  # If partial utterance is sufficient long,
                            utter_part = utter[
                                         interval[0]:interval[1]]  # save first and last 180 frames of spectrogram.
                            S = librosa.core.stft(y=utter_part, n_fft=param.data.nfft,
                                                  win_length=int(param.data.window * sr),
                                                  hop_length=int(param.data.hop * sr))
                            S = np.abs(S) ** 2
                            mel_basis = librosa.filters.mel(sr=param.data.sr, n_fft=param.data.nfft,
                                                            n_mels=param.data.nmels)
                            S = np.log10(np.dot(mel_basis, S) + 1e-6)  # log mel spectrogram of utterances
                            utterances_spec.append(
                                S[:, :param.data.tisv_frame])  # first 180 frames of partial utterance
                            utterances_spec.append(
                                S[:, -param.data.tisv_frame:])  # last 180 frames of partial utterance

            utterances_spec = np.array(utterances_spec)
            print(utterances_spec.shape)
            np.save(os.path.join(param.data.train_path if training else param.data.test_path, "speaker%d.npy" % i),
                    utterances_spec)

    # preprocess and save train data
    save_tisv(training=True)

    # preprocess and save train data
    save_tisv(training=False)


if __name__ == "__main__":
    preprocessing()
