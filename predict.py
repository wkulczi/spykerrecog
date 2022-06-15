import click
from pprint import pprint
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
from tensorflow import keras
import numpy as np
SIGNAL_LENGTH = 0

def record_audio(record_seconds  = 4, recording_path = 'recorded.wav'):
    import pyaudio
    import wave

    # the file name output you want to record into
    filename = recording_path
    # set the chunk size of 1024 samples
    chunk = 1024
    # sample format
    FORMAT = pyaudio.paInt16
    # mono, change to 2 if you want stereo
    channels = 1
    # 44100 samples per second
    sample_rate = 16000
    record_seconds = record_seconds
    # initialize PyAudio object
    p = pyaudio.PyAudio()
    # open stream object as input & output
    stream = p.open(format=FORMAT,
                    channels=channels,
                    rate=sample_rate,
                    input=True,
                    output=True,
                    frames_per_buffer=chunk)
    frames = []
    print("Recording...")
    for i in range(int(sample_rate / chunk * record_seconds)):
        data = stream.read(chunk)
        # if you want to hear your voice while recording
        # stream.write(data)
        frames.append(data)
    print("Finished recording.")
    # stop and close stream
    stream.stop_stream()
    stream.close()
    # terminate pyaudio object
    p.terminate()
    # save audio file
    # open the file in 'write bytes' mode
    wf = wave.open(filename, "wb")
    # set the channels
    wf.setnchannels(channels)
    # set the sample format
    wf.setsampwidth(p.get_sample_size(FORMAT))
    # set the sample rate
    wf.setframerate(sample_rate)
    # write the frames as bytes
    wf.writeframes(b"".join(frames))
    # close the file
    wf.close()
    
def trp(l, n): #trim or pad list l to be of n length
    return l[:n] + [0]*(n-len(l))

def librosaReadAudio(filename = "recorded.wav"):
    return librosaGetSampleData(filename).flatten()

def librosaPredictCnn(filename="recorded.wav"):
    data = librosaGetSampleData(filename)
    data = data[np.newaxis,...,np.newaxis]
    return data
    
    
def librosaGetSampleData(filename="recorded.wav"):
    import scipy
    import librosa
    signal, sampling_rate = librosa.load(filename, sr=None)
    signal = trp(signal.tolist(), SIGNAL_LENGTH) 
    mfcc = librosa.feature.mfcc(y=np.array(signal), n_mfcc=13, sr = sampling_rate, window = scipy.signal.windows.hann, n_fft=2048, hop_length=512, win_length = None)
    delta = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)
    data = np.concatenate((mfcc, delta, delta2)).T
    return data

def unpickle(filename):
    import pickle
    with open(filename, 'rb') as fp:
        banana = pickle.load(fp)
    return banana

@click.command()
@click.option('--recording', '-r', default='untitled.wav', help="recording path")
@click.option('--new', '-n', is_flag=True, help="should record new audio")
@click.option('--model', '-m', default='cnn', help="use trained model (neigh\cnn)")
def main(recording = 'untitled.wav', new=False, model = "neigh"):
    global SIGNAL_LENGTH
    SIGNAL_LENGTH = 118272 #always the same length, they were padded earlier
    labelEncoder = unpickle("labelEncoder.pickle")
    if new:
        record_audio(record_seconds = 5, recording_path=recording)
    print(f"Predicting recording {recording}...")

    preds = []
    if model=="neigh":
        neigh=unpickle("knn78.pickle")
        preds=neigh.predict_proba([librosaReadAudio(recording)])
    else: 
        cnnmodel = keras.models.load_model('cnn_librosa_91.h5')
        preds = cnnmodel.predict(librosaPredictCnn(recording))
    
    print(f"Predictions {model}:")
    pprint(list(zip(labelEncoder.classes_, preds[0])))
    print(f"Predicted speaker: {labelEncoder.classes_[preds.argmax()]}")

if __name__ == "__main__":
    main()