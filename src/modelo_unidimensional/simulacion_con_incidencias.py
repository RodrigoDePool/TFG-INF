from pandas import DataFrame
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.models import load_model
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import json
from modules.util import *
from modules.incidencias import generar_incidencias
from modules.evaluacion import no_df_evaluar
"""
EXPERIMENTO:
    Por cada una de las incidencias dada queremos conseguri una
    tablas para nuestras lstms que tengan esta pinta:
        u  |  sensibilidad  | especificidad  | F2-score
        1  |     0.x        |     0.y        |   0.z
        2  |     ...        |     ...        |   ...
"""

# LOAD DF
df = pd.read_csv('../datos/cleancleandf.csv')
df = df.sort_values('tref_start')
df = df[df['targetIP'] == '172.31.190.130']
df.drop_duplicates(subset=['targetIP', 'tref_start'], inplace=True)
df = df.drop('Unnamed: 0', axis=1)
df = df.drop('targetIP', axis=1)
df = df.drop('wday', axis=1)
df = df.drop('hour', axis=1)
# Incidence info
titles = {
    'Up bps received': 'Paquete de incidencias 1',
    'Up rtt': 'Paquete de incidencias 2',
    'Up rtx': 'Paquete de incidencias 3',
    'Up cnx': 'Paquete de incidencias 4',
    'Mixto': 'Paquete de incidencias 5'
}
incidence_files = {
    'Up bps received':
    '../datos/incidencias/incidencias_ejemplo/up_bps_rcv_v.json',
    'Up rtt': '../datos/incidencias/incidencias_ejemplo/up_rtt_v.json',
    'Up rtx': '../datos/incidencias/incidencias_ejemplo/up_rtx_v.json',
    'Up cnx': '../datos/incidencias/incidencias_ejemplo/up_cnx_v.json',
    'Mixto': '../datos/incidencias/incidencias_ejemplo/mixto_v.json'
}
# Train - Validation is 70%-30%
# To train we use a month before validation
train_to = int(len(df) * 0.7)
train_from = train_to - train_size
umbrales_file = 'umbrales.json'


def get_trained_model(df, verbose=1):
    # There is a model and an estimator per
    # column to predict
    models = {}
    scalers = {}
    for col in df.columns:
        if col != 'tref_start':
            model = get_model()
            scaler = MinMaxScaler()
            data = one_feature_series(df[col].values[train_from:train_to])
            data = scaler.fit_transform(data)
            train_window, train_obj = to_window(data)
            model.fit(train_window, train_obj, epochs=epocas, verbose=0)
            models[col] = model
            scalers[col] = scaler
            if verbose == 1:
                print('Entrenada la red para la métrica {}'.format(col))
    return models, scalers


def get_model_predictions(df, models, scalers, inc_file):
    # Generate incidence
    inc = get_working_incidence(inc_file)
    df = df.iloc[train_to:]
    df_inc = generar_incidencias(df, inc)
    # Incidence from fist window
    incidencias = df_inc['incidencia'].values[input_window:-1]
    df_inc = df_inc.drop('incidencia', axis=1)
    # Scale and predict
    ys, yhats = scale_and_predict(df_inc, models, scalers)
    return ys, yhats, incidencias


def scale_and_predict(df_inc, models, scalers):
    column_values = {}
    for col in df_inc.columns:
        if col != 'tref_start':
            data = one_feature_series(df_inc[col].values)
            data = scalers[col].transform(data)
            test_w, ys = to_window(data)
            yhats = models[col].predict(test_w)
            ys = np.array([y[0][0] for y in ys])
            yhats = np.array([y[0] for y in yhats])
            l = len(yhats)
            column_values[col] = (ys, yhats)
    # Transform column_values to shape (sample, cols)
    ys = []
    yhats = []
    cols = [col for col in df_inc.columns if col != 'tref_start']
    for i in range(l):
        aux_y = []
        aux_yhat = []
        for col in cols:
            aux_y.append(column_values[col][0][i])
            aux_yhat.append(column_values[col][1][i])
        ys.append(aux_y)
        yhats.append(aux_yhat)
    return ys, yhats


def lstm_umbral_classify(y_hat, y, u):
    isIncidence = []
    for y0, y1 in zip(y_hat, y):
        # Calculate mse of y0 and y1
        v = np.array(y0) - np.array(y1)
        v = sum(v * v) / len(y0)
        if v > u:
            isIncidence.append(True)
        else:
            isIncidence.append(False)
    return isIncidence


def umbral_testing(y_hat, y, incidencias, umbrales, verbose=False):
    """Testing de varios umbrales. Por cada umbral devuelve un resultado
    de evaluacion
        - y_hat: Lista de outputs predichos
        - y: Lista de valores reales
        - incidencias[i]: si y[i] es o no una incidencia
        - umbrales: Array con cada uno de los umbrales a probar
    NOTA: Utiliza refinación de la clasificación
    """
    resultados = {}
    for u in umbrales:
        clasif = lstm_umbral_classify(y_hat, y, u)
        resultados[u] = no_df_evaluar(incidencias, clasif)
        if verbose: print('Umbral {} computado'.format(u))
    return resultados


def plot_umbrales(title, resultados, file=None):
    title = titles[title]
    plt.clf()
    x = [float(u) for u in resultados]
    y = [resultados[u]['metrica'] for u in resultados]
    plt.plot(x, y)
    plt.title(title)
    plt.ylabel('F2-score')
    plt.xlabel('Umbrales')
    if file:
        plt.savefig(file)
    else:
        plt.show()


# F2-SCORE EXPERIMENT
def f2_score_experiment():
    print('EXPERIMENTO F2-SCORE')
    models, scalers = get_trained_model(df)
    info = {name: {} for name in incidence_files}
    for name in incidence_files:
        print('Predicción con la incidencia: {}'.format(name))
        inc_file = incidence_files[name]
        ys, yhats, isIncidence = get_model_predictions(df, models, scalers,
                                                       inc_file)
        info[name]['ys'] = ys
        info[name]['yhats'] = yhats
        info[name]['isIncidence'] = isIncidence
        resultados = umbral_testing(
            yhats,
            ys,
            isIncidence,
            umbrales=[0.1 / i for i in range(1, 100)],
            verbose=False)
        print('Calculados los resultados por umbral de {}'.format(name))
        info[name]['resultados'] = resultados
        file = 'graphics/umbrales_{}.pdf'.format('_'.join(name.split()))
        plot_umbrales(name, resultados, file)

    print('\n\nMEJORES RESULTADOS POR CADA PAQUETE DE INCIDENCIAS:')
    for name in incidence_files:
        aux = [(info[name]['resultados'][u]['metrica'], u)
               for u in info[name]['resultados']]
        f2, u = max(aux)
        print('- {}:'.format(name))
        print('\tUmbral: {}'.format(u))
        print('\tF2: {}'.format(f2))
        print('\tSensibilidad: {}'.format(
            info[name]['resultados'][u]['sensibilidad']))
        print('\tPrecision: {}'.format(
            info[name]['resultados'][u]['especificidad']))


#f2_score_experiment()