"""
Estudio de alarmas en circunstancias 'reales'
---
Objetivo: 1. Evaluar el número de alarmas que da el sistema en circunstancias 
             reales (con distintos umbrales).
          2. Evaluar si esas alarmas proceden o no.
          3. Comprar los modelos de LSTM y VAR.     
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
from modules.util import trim_dataframe, get_working_incidence
from modules.evaluacion import no_df_evaluar
from modules.incidencias import generar_incidencias
from statsmodels.tsa.api import VAR
import numpy as np

# De índices a nombre de características
columna_nombre_lista = [(0, 'bpsPhyRcv'), (1, 'bpsPhySent'), (2, 'dupAck'),
                        (3, 'noRespClient'), (4, 'noRespServer'),
                        (5, 'numberCnx'), (6, 'ppsRcv'), (7, 'ppsSent'),
                        (8, 'resetClient'), (9, 'resetServer'),
                        (10, 'rttPerCnx'), (11, 'rtx'), (12, 'syn'),
                        (13, 'win0'), (14, 'hour'), (15, 'wday')]
col_nom = {}
for num, col in columna_nombre_lista:
    col_nom[num] = col

# Datos del modelo
model_specs = {
    'name': './lstm_models/lstm_basic.model',
    'train_perc': 0.7,
    'epochs': 90,
    'window': 18  # Hora y media
}
df_limpisimo_con_ips = './cleancleandf.csv'


def get_compiled_model():
    model = Sequential()
    model.add(LSTM(200, activation='relu', input_shape=(18, 16)))
    model.add(Dense(16))
    model.compile(optimizer='adam', loss='mse')
    return model


# FUNCIONES VARIAS PARA EL EXPERIMENTO 2 PARTE DE LSTMs
def get_df(file):
    """Cargamos el dataframe como le hace falta a las funciones de 
    lstms
    """
    # Cargamos el dataframe
    df = pd.read_csv(file)
    df = df.sort_values('tref_start')
    # Estas columnas han pasado a ser inútiles
    #  pero se quedan porque así están entrenados los modelos
    #df = df.drop('hour', axis=1)
    #df = df.drop('wday', axis=1)
    return df


def testeo_umbral(yhats, ys, umbrales):
    """Para cada umbral calcula los valores que serían clasifi
    cados como incidencias. 

    Inputs:
        yhats: Predicción de la red
        ys: Valores reales 
        umbrales: Umbrales a probar en la clasificación
    Output:
        Un JSON: Asignará a cada umbral una pareja de dos arrays
        (isIncidence, mainReason). El primero es un array
        que indica si ese valor fue, o no, marcado como incidencia.
        El segundo indica qué argumento tuvo el mayor peso en la
        decisión.
    """
    outp = {}
    for u in umbrales:
        mainReason = []
        isIncidence = []
        for y0, y1 in zip(yhats, ys):
            # Calculate mse of y0 and y1
            v0 = np.array(y0) - np.array(y1)
            v = sum(v0 * v0) / len(y0)
            if v > u:
                isIncidence.append(True)
                mainReason.append(np.argmax(v0 * v0))
            else:
                isIncidence.append(False)
                mainReason.append(None)
        outp[u] = (isIncidence, mainReason)
    return outp


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
    if load:
        # Intenamos cargar el modelo
        try:
            model = load_model(model_specs['name'])
            return model, scaler
        except:
            pass
    # Entrenamos y guardamos el modelo
    model = get_compiled_model()
    Xtrain_windows, y = X_to_window(X_train, window=model_specs['window'])
    model.fit(Xtrain_windows, y, epochs=model_specs['epochs'], verbose=0)
    model.save(model_specs['name'])
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
    if incidence_file is not None:
        inc = get_working_incidence(incidence_file)
        df = generar_incidencias(df, inc).sort_values(by=['tref_start'])
        # Array de incidencias
        incidencias = df['incidencia'].values[window:]
        df = df.drop('incidencia', axis=1)
    # Obtenemos valores de la red reales
    df = df.drop('Unnamed: 0', axis=1)
    df = df.drop('tref_start', axis=1)
    X = df.values[:, :]
    if diff:
        X = np.diff(X, axis=0)
        if incidence_file is not None:
            incidencias = incidencias[1:]
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
    if incidence_file is not None:
        inc = get_working_incidence(incidence_file)
        df = generar_incidencias(df, inc).sort_values(by=['tref_start'])
        # Nos quedamos con las incidencias a partir de la primera ventana
        incidencias = df['incidencia'].values[window:]
        # Limpiamos y obtenemos los valores
        df = df.drop('incidencia', axis=1)
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


# Entrenamos el modelo


def contar_incidencias_por_rafagas(isInc, mainReasons, previos=12):
    """
    Cuenta por ráfagas las incidencias identificadas.
    Inputs:
        isInc: Array de booleans que indican el momento de 
               incidencia.
        mainReasons: Array con los índices de las columnas que
                     tienen un papel "protagonista" en la 
                     incidencia.
        previos: Si en los pasados 'previos' intervalos hubo 
                 incidencia, no se cuenta como una incidencia
                 nueva.
    Outputs:
        Devuelve un array de incidencias. Cada elemento del
        array contiene un par (inicio, main_set).
            inicio: Es el índice del array donde inicia la
                   incidencia.
            main_set: Es un conjunto con los índices de las 
                      columnas que tienen un papel "protagonista"
                      en la incidencia.
    """
    info_incidencias = []
    for i, v in enumerate(isInc):
        if v and mainReasons[i] < 14:  # Hay incidencia que no sea 14 o 15?
            desde = max(0, i - previos)
            rafaga = any(isInc[desde:i])
            if rafaga:  # Pertence a una rafaga anterior
                _, cjto = info_incidencias[-1]
                cjto.add(mainReasons[i])  # Añadimos columna prota
            else:  # Nueva rafaga
                # (inicio de rafaga, cjto con columna prota)
                info_incidencias.append((i, set([mainReasons[i]])))
    return info_incidencias


from time import localtime


def tref_to_time(tref_start):
    return localtime(tref_start / 1000)


big_df = get_df(file=df_limpisimo_con_ips)
ips = set(big_df['targetIP'].values)
# info[ip][algoritmo] = resultados por umbral
info = {ip: {'VAR': {}, 'LSTM': {}} for ip in ips}
modelos = {
    'VAR': {ip: (None, None)
            for ip in set(big_df['targetIP'].values)},
    'LSTM': {ip: (None, None)
             for ip in set(big_df['targetIP'].values)}
}


# Para cada IP corremos el test tanto para LSTM como VAR
def entrenar_lstm(big_df, ip):
    df = big_df[big_df['targetIP'] == ip].drop('targetIP', axis=1)
    # MODELO LSTM
    model_specs['name'] = './lstm_models/lstm_90eps_200neurs_{}.model'.format(
        dot_to_underline(ip))
    #print('\n---\nIP {}\nModelo LSTM'.format(ip))
    model, scaler = get_trained_model(df, model_specs)
    modelos['LSTM'][ip] = (model, scaler)


def predecir_lstm(big_df, ip, umbral_range=True):
    df = big_df[big_df['targetIP'] == ip].drop('targetIP', axis=1)
    model, scaler = modelos['LSTM'][ip]
    if model is None:
        raise Exception('Modelos no cargados')
    ys, yhats, _ = get_model_predictions(df, model, scaler, model_specs, None)
    info[ip]['LSTM']['raw_data'] = (ys, yhats)
    if umbral_range:
        umbrales = range(1, 51)
    else:
        umbrales = [1]
    info[ip]['LSTM']['clasificacion_umbral'] = testeo_umbral(
        yhats, ys, [u for u in umbrales])


def entrenar_var(big_df, ip):
    df = big_df[big_df['targetIP'] == ip].drop('targetIP', axis=1)
    # MODELO VAR
    print('Modelo VAR')
    ys, yhats, _ = var_prediction(df, 0.7, None)
    info[ip]['VAR']['raw_data'] = (ys, yhats)
    info[ip]['VAR']['clasificacion_umbral'] = testeo_umbral(
        yhats, ys, [u for u in range(1, 51)])


"""
for ip in ips:
    entrenar_lstm(big_df, ip)
    entrenar_Var(big_df, ip)
"""

# GENERA LAS GRÁFICAS DE NÚMERO DE ALARMAS SEGÚN EL UMBRAL
"""
for ip in ips:
    plt.clf()
    lstm_umbral_info = info[ip]['LSTM']['clasificacion_umbral']
    var_umbral_info = info[ip]['VAR']['clasificacion_umbral']
    x = list(lstm_umbral_info.keys())
    x.sort()
    y_lstm = []
    y_var = []
    for u in x:
        a, b = lstm_umbral_info[u]
        lstm_info = contar_incidencias_por_rafagas(a[:4500], b[:4500])
        y_lstm.append(len(lstm_info))
        a, b = var_umbral_info[u]
        var_info = contar_incidencias_por_rafagas(a[:4500], b[:4500])
        y_var.append(len(var_info))
    plt.plot(x, y_lstm, label='LSTM')
    plt.plot(x, y_var, label='VAR')
    plt.title('IP {}'.format(ip))
    plt.legend(loc='upper right')
    plt.ylabel('Número de alarmas')
    plt.xlabel('Umbral')
    plt.show()
"""

# EJEMPLO DE INSPECCIÓN MANUAL DE INCIDENCIAS DETECTADAS
# LSTM
"""
ip = '172.31.191.104'
umbral = 25
ys, yhats_lstm = info[ip]['LSTM']['raw_data']
# Nos quedamos con 4500 intervalos de 5 minutos (15.625 días)
a, b = info[ip]['LSTM']['clasificacion_umbral'][umbral]
incs_lstm = contar_incidencias_por_rafagas(a[:4500], b[:4500])

#VAR
ys_diff, yhats_var = info[ip]['VAR']['raw_data']
# Nos quedamos con 4500 intervalos de 5 minutos (15.625 días)
a, b = info[ip]['VAR']['clasificacion_umbral'][umbral]
incs_var = contar_incidencias_por_rafagas(a[:4500], b[:4500])

#INSPECCIÓN DEL SIGUIENTE DATAFRAME
df = big_df[big_df['targetIP'] == ip].drop('targetIP', axis=1)
df = df.drop('Unnamed: 0', axis=1)
df = df.drop('tref_start', axis=1)
v = int(len(df) * 0.7)
df = df.iloc[v + 18:v + 18 + 4500]
"""


def plot_close(df, col, ind, save=True):
    print(col_nom[col])
    plt.title(col_nom[col])
    ys = df[col_nom[col]]
    xs = [i for i in range(1, len(ys) + 1)]
    plt.plot(xs, ys)
    plt.show()
    plt.clf()
    plt.title(col_nom[col])
    ys = ys[ind - 20:ind + 20]
    xs = [i for i in range(ind - 20, ind + 20)]
    plt.plot(xs, ys)
    plt.show()


"""
TODOS VAR
 (615, {0})
 (1847, {4, 10}), LSTM
 (2629, {0}), LSTM EN EL 13
 (2895, {6, 7}),
 (3094, {7}), LSTM (4,10)
 (3111, {4, 6, 10}),
 (3959, {6, 13}), LSTM LA 13 
 (3982, {13}), LASTM
 (4027, {12}),
"""
"""
#IGNORAR LAS DE 14 Y 15
incs_lstm
incs_var
plot_close(df, 0, 615)  #var
plot_close(df, 4, 1847)  #lstm y var
plot_close(df, 10, 1847)  #lstm y var
plot_close(df, 0, 2629)  # var
plot_close(df, 6, 2895)  # var
plot_close(df, 7, 2895)  # var
plot_close(df, 7, 3094)  # var y lstm
plot_close(df, 4, 3111)  # var
plot_close(df, 6, 3111)  # var
plot_close(df, 10, 3111)  # var
plot_close(df, 6, 3959)  # var
plot_close(df, 13, 3959)  # var y lstm
plot_close(df, 13, 3989)  # var y lstm
plot_close(df, 12, 4027)  # var
"""
# OTRO EXPERIMENTO:
# VEAMOS LA MEDIA DEL NUMERO DE INCIDENCIAS
"""
ip_umbral = {
    '172.31.169.10': 45,
    '172.31.190.128': 25,
    '172.31.190.130': 35,
    '172.31.109.30': 35,
    '172.30.8.203': 25,
    '172.31.190.132': 23,
    '192.168.34.52': 25,
    '172.31.191.104': 25,
    '172.31.190.180': 35,
    '172.31.190.124': 30
}
ip_incidencias = {}
for ip in ip_umbral:
    umbral = ip_umbral[ip]
    a, b = info[ip]['LSTM']['clasificacion_umbral'][umbral]
    incs_var = contar_incidencias_por_rafagas(a[:4500], b[:4500])
    ip_incidencias[ip] = incs_var
numincs = [len(ip_incidencias[ip]) for ip in ip_incidencias]
maxincs = max(numincs)
minincs = min(numincs)
meanincs = sum(numincs) / len(numincs)
print('LSTM:\n- Mean: {}\n- Min: {}\n- Max: {}'.format(meanincs, minincs,
                                                       maxincs))

ip_incidencias = {}
for ip in ip_umbral:
    umbral = ip_umbral[ip]
    a, b = info[ip]['VAR']['clasificacion_umbral'][umbral]
    incs_var = contar_incidencias_por_rafagas(a[:4500], b[:4500])
    ip_incidencias[ip] = incs_var
numincs = [len(ip_incidencias[ip]) for ip in ip_incidencias]
maxincs = max(numincs)
minincs = min(numincs)
meanincs = sum(numincs) / len(numincs)
print('VAR:\n- Mean: {}\n- Min: {}\n- Max: {}'.format(meanincs, minincs,
                                                      maxincs))
"""