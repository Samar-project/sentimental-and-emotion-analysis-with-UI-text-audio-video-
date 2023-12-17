import os, pickle, wave
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

import glob
import librosa
import soundfile
import numpy as np
import pickle


# extract features from video
def runn(filePath):
    def extract_feature(file_name, mfcc, chroma, mel):
        with soundfile.SoundFile(file_name) as sound_file:
            X = sound_file.read(dtype="float32")
            sample_rate = sound_file.samplerate
            if chroma:
                stft = np.abs(librosa.stft(X))
                result = np.array([])
            if mfcc:
                mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
                result = np.hstack((result, mfccs))
            if chroma:
                chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
                result = np.hstack((result, chroma))
            if mel:
                mel = np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T, axis=0)
                result = np.hstack((result, mel))
        return result

    # main function to predict the emotion
    def get_speech_emotion():
        Pkl_Filename = "./data/speech/Emotion_Voice_Detection_Model.pkl"
        with open(Pkl_Filename, 'rb') as file:
            Emotion_Voice_Detection_Model = pickle.load(file)
        # reading audio file
        if filePath:
            file = filePath
        else:
            file = glob.glob('./input/audio/Audio1.mp3')[0]
        ans = []
        new_feature = extract_feature(file, mfcc=True, chroma=True, mel=True)
        ans.append(new_feature)
        ans = np.array(ans)

        emotion = Emotion_Voice_Detection_Model.predict(ans)
        return emotion[0]

    # function to get audio feature
    audio_file = './input/audio/Audio1.mp3'
    y, sr = librosa.load(audio_file, mono=False)  # Load audio as stereo

    # Convert stereo audio to mono
    if y.ndim > 1:
        y_mono = librosa.to_mono(y)
    else:
        y_mono = y
    # Now you can process the mono audio
    mfccs = librosa.feature.mfcc(y=y_mono, sr=sr, n_mfcc=40)
    chroma = librosa.feature.chroma_stft(y=y_mono, sr=sr)
    mel = librosa.feature.melspectrogram(y=y_mono, sr=sr)

    return get_speech_emotion()

