import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, multilabel_confusion_matrix

# [('loss', 0.10957802087068558), ('pre', 0.7090774774551392), ('rec', 0.4076753258705139), ('f1', 0.5177034139633179), ('acc', 0.09155555814504623)]

RL = pd.read_pickle("REAL")
PRED = np.load("PRED.npy")
PRED_ROUND = np.round(PRED)

n_usrs = PRED.shape[1]

def expand_real(x):
    ret = np.zeros(n_usrs, dtype=int)
    ret[x]=1
    return ret

REAL = RL.output.apply(expand_real)
REAL = np.row_stack(REAL.to_list())

cfmtx = multilabel_confusion_matrix(REAL, PRED_ROUND)
prc = precision_score(REAL, PRED_ROUND, average="micro")
rcl = recall_score(REAL, PRED_ROUND, average="micro")
f1s = f1_score(REAL, PRED_ROUND, average="micro")
acc = accuracy_score(REAL, PRED_ROUND) # Número de casos que acierta de forma exacta

corr=0
f1a = []
acc = []

for r in range(len(REAL)):
    f1a.append(f1_score(REAL[r], PRED_ROUND[r]))
    if accuracy_score(REAL[r], PRED_ROUND[r])==1: 
        print(RL.iloc[r].rest_name, REAL[r].sum(),PRED_ROUND[r].sum())
        corr+=1

y_true = [[0, 0, 0], [1, 1, 1], [0, 1, 1]]
y_pred = [[0, 0, 0], [1, 1, 1], [1, 1, 0]]
f1_score(y_true, y_pred, average=None)

print(f"Mean accuracy: {np.mean(acc)}")
