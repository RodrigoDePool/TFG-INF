"""
Algunas cosas que se utilizan en la
generación de las imágenes
"""
from pandas import DataFrame
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.models import load_model
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import json
from time import localtime
from statsmodels.tsa.api import VAR

# Correspondencia número - columna
columna_nombre_lista = [(0, 'bpsPhyRcv'), (1, 'bpsPhySent'), (2, 'dupAck'),
                        (3, 'noRespClient'), (4, 'noRespServer'),
                        (5, 'numberCnx'), (6, 'ppsRcv'), (7, 'ppsSent'),
                        (8, 'resetClient'), (9, 'resetServer'),
                        (10, 'rttPerCnx'), (11, 'rtx'), (12, 'syn'),
                        (13, 'win0'), (14, 'hour'), (15, 'wday')]
col_nom = {}
for num, col in columna_nombre_lista:
    col_nom[num] = col

model_specs = {
    'name': './lstm_models/lstm_basic.model',
    'train_perc': 0.7,
    'epochs': 90,
    'window': 18  # Hora y media
}
# Datos
big_df = pd.read_csv('../../data/cleancleandf.csv')
big_df = big_df.sort_values('tref_start')


def X_to_window(X, window=18):
    """ Convertimos [samples, features] en [samples, window, features]
    """
    tam = len(X)
    window_X = [None for _ in range(window, tam)]
    y = [None for _ in range(window, tam)]
    for i in range(window, tam):
        # Output
        y[i - window] = X[i]
        # Previous timesteps
        w = [ts for ts in X[i - window:i]]
        window_X[i - window] = w
    return np.array(window_X), np.array(y)


def get_trained_model(df, model_specs, load=True):
    """Devuelve el modelo entrenado
    """
    # Limpiamos el df
    df = df.drop('Unnamed: 0', axis=1)
    df = df.drop('tref_start', axis=1)
    train_perc = model_specs['train_perc']
    # Obtenemos estandarizador de valores
    X = df.values[:, :]
    v = int(len(X) * train_perc)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X[:v])
    model = load_model(model_specs['name'])
    return model, scaler


def var_prediction(df, train_perc, incidence_file, window=18, diff=True):
    # Limpiamos el df
    df_aux = df.drop('Unnamed: 0', axis=1)
    df_aux = df_aux.drop('tref_start', axis=1)
    X = df_aux.values[:, :]
    if diff:
        X = np.diff(X, axis=0)
    # Obtenemos estandarizador de valores
    v = int(len(X) * train_perc)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X[:v])
    # Entrenamos el modelo
    model = VAR(X_train)
    results = model.fit(window)
    # df de validación con incidencias
    df = df.iloc[v:]
    incidencias = None
    # Obtenemos valores de la red reales
    df = df.drop('Unnamed: 0', axis=1)
    df = df.drop('tref_start', axis=1)
    X = df.values[:, :]
    if diff:
        X = np.diff(X, axis=0)
    X = scaler.transform(X)
    # Obtengamos predicciones
    ys = X[window:]
    yhats = []
    for i in range(window, len(X)):
        yhats.append(results.forecast(X[i - window:i], 1)[0])
    return ys, np.array(yhats), incidencias


def get_model_predictions(
        df,
        model,
        scaler,
        model_specs,
        incidence_file,
):
    """Devuelve tres arrays ys, yhats e isIncidence:
    - ys[i] corresponde con el valor real de las ys en el momento i
    - yhats[i] corresponde con el valor predicho de las ys en el momento i
    - isIncidence[i] indica si ys[i] corresponde, o no, con una incidencia
    NOTA:
        1. Todos los valores corresponden con tiempos en la zona
           de predicción
        2. Todos los valores están estandarizados con el scaler dado
           (ojo el scaler SOLO debe usar valores del training )
    """
    # Valores
    window = model_specs['window']
    train_perc = model_specs['train_perc']
    v = int(len(df) * train_perc)
    # Obtenemos df con incidencias en la zona de validacion
    df = df.iloc[v:]
    incidencias = None
    df = df.drop('Unnamed: 0', axis=1)
    df = df.drop('tref_start', axis=1)
    X = scaler.transform(df.values[:, :])
    # Obtenemos los ys reales en cada momento de la red
    Xv_inc_windows, ys = X_to_window(X, window=window)
    # Obtenemos las predicciones de nuestra red
    yhats = model.predict(Xv_inc_windows)
    return ys, yhats, incidencias


def dot_to_underline(ip):
    return '_'.join(ip.split('.'))


def tref_to_time(tref_start):
    return localtime(tref_start / 1000)


ips = ['172.31.191.104', '172.31.190.130']
# info[ip][algoritmo] = resultados por umbral
info = {ip: {'VAR': {}, 'LSTM': {}} for ip in ips}
# Para cada IP corremos el test tanto para LSTM como VAR
for ip in ips:
    df = big_df[big_df['targetIP'] == ip].drop('targetIP', axis=1)
    # MODELO LSTM
    model_specs[
        'name'] = '../../models/tfg/lstm_models/lstm_90eps_200neurs_{}.model'.format(
            dot_to_underline(ip))
    model, scaler = get_trained_model(df, model_specs)
    ys, yhats, _ = get_model_predictions(df, model, scaler, model_specs, None)
    info[ip]['LSTM']['raw_data'] = (ys, yhats)
    # MODELO VAR
    ys, yhats, _ = var_prediction(df, 0.7, None)
    info[ip]['VAR']['raw_data'] = (ys, yhats)
