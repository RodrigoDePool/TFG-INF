from simulacion_con_incidencias import *
from time import time

print('\n\nTIME TEST FOR THE INCIDENCE CLASSIFIER\n\n')
s = time()
models, scalers = get_trained_model(df, verbose=0)
print('\n\n\nTraining time for full incidence classifier: {} seconds'.format(
    time() - s))

df = df[train_to:]
size = len(df)
print(
    'Prediction time per register (mean of {} register times): '.format(size),
    end='')
s = time()
ys, yhats = scale_and_predict(df, models, scalers)
lstm_umbral_classify(yhats, ys, [1])
print('{} seconds'.format((time() - s) / size))
