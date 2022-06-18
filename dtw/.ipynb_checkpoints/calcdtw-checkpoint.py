from signalutils.signalutils import readFiles, encodedLabels, calcMfccs, export_to_pickle
from dtw import accelerated_dtw
import scipy
from tqdm import tqdm
import sys

signals, labelEncoder = encodedLabels(readFiles(path="../samples"))

signals = calcMfccs(signals, window=scipy.signal.windows.boxcar, n_fft=512, hop_length=256, win_length=512, n_mfcc=13)



for signal in signals:
    signal['name'] = signal['name'].split('.')[0]

apsignals = [x for x in signals if x['label']=='AP']

calculatedData = []
for entry_x in tqdm(apsignals):
    calculatedData=[]   
    for entry_y in tqdm(signals):
        cost, cost_matrix, acc_cost_matrix, path = accelerated_dtw(entry_x['mfccs'].T, entry_y['mfccs'].T, scipy.spatial.distance.cosine)
        data = {
            "x": entry_x['name'],
            "y": entry_y['name'],
            "cost": cost,
            "cost_matrix": cost_matrix,
            "acc_cost_matrix": acc_cost_matrix,
            "path":path
        }
        calculatedData.append(data)
    export_to_pickle(f"data/{entry_x['name']}",calculatedData)    
