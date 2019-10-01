import glob
import os
import librosa
import numpy as np
from configuration import param

# Motivate from "https://github.com/Janghyun1230/Speaker_Verification/blob/master/data_preprocess.py"

# downloaded dataset path
audio_path = glob.glob(os.path.dirname(param.unprocessed_data))


def save_spectrogram_tisv():
    """
        TI-SV (Text Independent - Speaker Verification)
        Each partial utterance is splitted by voice detection using DB
        and the first and the last 180 frames from each partial utterance are saved.
    """
    print("start text independent utterance feature extraction")
    os.makedirs(param.data.train_path, exist_ok=True)  # make folder to save train file
    os.makedirs(param.data.test_path, exist_ok=True)  # make folder to save test file

    utter_min_len = (param.data.tisv_frame * param.data.hop + param.data.window) * param.data.sr  # lower bound of utterance length
    total_speaker_num = len(audio_path)
    train_speaker_num = (total_speaker_num // 10) * 9  # split total data 90% train and 10% test
    print("total speaker number : %d" % total_speaker_num)
    print("train : %d, test : %d" % (train_speaker_num, total_speaker_num - train_speaker_num))
    for i, folder in enumerate(audio_path):
        print("%dth speaker processing..." % i)
        utterances_spec = []
        for utter_name in os.listdir(folder):
            if utter_name[-4:].lower() == '.wav' or utter_name[-5:].lower() == '.flac':
                utter_path = os.path.join(folder, utter_name)  # path of each utterance
                utter, sr = librosa.core.load(utter_path, param.data.sr)  # load utterance audio
                intervals = librosa.effects.split(utter, top_db=30)  # voice activity detection
                for interval in intervals:
                    if (interval[1] - interval[0]) > utter_min_len:  # If partial utterance is sufficient long,
                        utter_part = utter[interval[0]:interval[1]]  # save first and last 180 frames of spectrogram.
                        S = librosa.core.stft(y=utter_part, n_fft=param.data.nfft,
                                              win_length=int(param.data.window * sr), hop_length=int(param.data.hop * sr))
                        S = np.abs(S) ** 2
                        mel_basis = librosa.filters.mel(sr=param.data.sr, n_fft=param.data.nfft, n_mels=param.data.nmels)
                        S = np.log10(np.dot(mel_basis, S) + 1e-6)  # log mel spectrogram of utterances
                        utterances_spec.append(S[:, :param.data.tisv_frame])  # first 180 frames of partial utterance
                        utterances_spec.append(S[:, -param.data.tisv_frame:])  # last 180 frames of partial utterance

        utterances_spec = np.array(utterances_spec)
        print(utterances_spec.shape)
        if i < train_speaker_num:  # save spectrogram as numpy file
            np.save(os.path.join(param.data.train_path, "speaker%d.npy" % i), utterances_spec)
        else:
            np.save(os.path.join(param.data.test_path, "speaker%d.npy" % (i - train_speaker_num)), utterances_spec)


if __name__ == "__main__":
    save_spectrogram_tisv()
