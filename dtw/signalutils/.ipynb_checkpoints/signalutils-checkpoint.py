import librosa
import librosa.display
import IPython.display as ipd
import matplotlib.pyplot as plt
import numpy as np
from os import listdir
from os.path import isfile,join
from os import walk
import pandas as pd
from sklearn import preprocessing
import scipy
import python_speech_features as psf
from scipy.io.wavfile import read

def readFiles(useLibrosa = True, path = "samples"):
    signals= []
    for path, subdirs, files in walk(path):
        for name in files:
            if useLibrosa:
                signal, sampling_rate = librosa.load(path + "/" + name, sr=16000) #uses default SR
            else: 
                sampling_rate,signal = scipy.io.wavfile.read(path + "/" + name)
            signals.append({
                "name":name,
                "signal": signal,
                "sr": sampling_rate,
                "label": name[:2].upper()
            })
    return signals

def padWithZeros(signals):
    longest_sample = max(list(map(lambda element: element["signal"].shape[0], signals)))
    for entry in signals:
        entry["signal"] = np.append(entry["signal"], np.zeros(longest_sample - entry["signal"].shape[0]))
    return signals

def encodedLabels(signals):
    labels = list(map(lambda element: element["label"],signals))
    le = preprocessing.LabelEncoder()
    le.fit(labels)
    for entry in signals:
        entry["encodedLabel"] = le.transform([entry["label"]])[0]
    return signals, le

def get_psf_mfcc(y, sr, window=scipy.signal.windows.hann, n_fft = 2048, win_length = None, n_mfcc=13, hop_length=512):
    if not win_length:
        win_length = 0.025
    if not hop_length:
        hop_length = 0.01
    return psf.mfcc(signal = y,
                   samplerate= sr,
                   winlen = win_length/sr,
                   winfunc = window,
                   numcep = n_mfcc,
                    winstep = hop_length/sr,
                    nfft=n_fft
                   )

def calcMfccs(signals, window=scipy.signal.windows.hann, n_fft = 2048, hop_length = 512, win_length = None, n_mfcc= 13, useLibrosa = True):
    for entry in signals:
        if useLibrosa:
            mfccs = librosa.feature.mfcc(y=entry["signal"], n_mfcc=n_mfcc, sr=entry["sr"],
                                        window = window,
                                        n_fft= n_fft,
                                        hop_length = hop_length,
                                        win_length = win_length)
        else: 
            mfccs = get_psf_mfcc(y=entry["signal"],sr = entry["sr"], n_mfcc = n_mfcc, hop_length = hop_length, n_fft= n_fft, win_length = win_length)
        entry["mfccs"] = mfccs
        entry["delta"] = librosa.feature.delta(mfccs)
        entry["delta2"] = librosa.feature.delta(mfccs, order=2)
    return signals

def export_to_pickle(filename, data):
    import pickle
    with open(filename+'.pickle', 'wb') as f:
        pickle.dump(data, f)

def unpickle(filename):
    import pickle
    with open(filename, 'rb') as fp:
        banana = pickle.load(fp)
    return banana

def test():
    print("test")