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
"""
EXPERIMENTO 2:
    Por cada una de las incidencias dada queremos conseguri una
    tablas para nuestras lstms que tengan esta pinta:
        u  |  sensibilidad  | especificidad  | F2-score
        1  |     0.x        |     0.y        |   0.z
        2  |     ...        |     ...        |   ...

    Y luego comparar eso con los resultados de los algoritmos
    que teniamos antes
"""

df_file = 'clean_172_31_190_130.csv'
# VARIABLES DEL MODELO LSTM
model_specs = {
    'name': './lstm_models/lstm_basic.model',
    'train_perc': 0.7,
    'epochs': 90,
    'window': 18  # Hora y media
}
titles = {
    'Up bps received': 'Paquete de incidencias 1',
    'Up rtt': 'Paquete de incidencias 2',
    'Up rtx': 'Paquete de incidencias 3',
    'Up cnx': 'Paquete de incidencias 4',
    'Mixto': 'Paquete de incidencias 5'
}


def get_compiled_model():
    model = Sequential()
    model.add(LSTM(200, activation='relu', input_shape=(18, 16)))
    model.add(Dense(16))
    model.compile(optimizer='adam', loss='mse')
    return model


# INCIDENCIAS:
#   - LAS DE ENTRENAMIENTO SOLO SE USARÁN PARA LOS ALGORITMOS ANTIGUOS
#   - LAS DE VALIDACION TIENEN QUE TENER INCIDENCIAS EN EL PORCENTAJE
#     QUE CORRESPONDE A LA VALIDACION. POR EJEMPLO: PARA EL 30% TIENEN
#     QUE IR DESDE EL 2 DE SEPT AL 21 DE SEPT
#     (asi engloba al 30% de TODAS las ips seleccionadas en cleandf.csv)
incidence_files = {
    'Up bps received': {
        'train': './datos/incidencias/incidencias_ejemplo/up_bps_rcv_e.json',
        'validate': './datos/incidencias/incidencias_ejemplo/up_bps_rcv_v.json'
    },
    'Up rtt': {
        'train': './datos/incidencias/incidencias_ejemplo/up_rtt_e.json',
        'validate': './datos/incidencias/incidencias_ejemplo/up_rtt_v.json'
    },
    'Up rtx': {
        'train': './datos/incidencias/incidencias_ejemplo/up_rtx_e.json',
        'validate': './datos/incidencias/incidencias_ejemplo/up_rtx_v.json'
    },
    'Up cnx': {
        'train': './datos/incidencias/incidencias_ejemplo/up_cnx_e.json',
        'validate': './datos/incidencias/incidencias_ejemplo/up_cnx_v.json'
    },
    'Mixto': {
        'train': './datos/incidencias/incidencias_ejemplo/mixto_e.json',
        'validate': './datos/incidencias/incidencias_ejemplo/mixto_v.json'
    }
}


# FUNCIONES VARIAS PARA EL EXPERIMENTO 2 PARTE DE LSTMs
def get_df(file=df_file):
    """Cargamos el dataframe como le hace falta a las funciones de 
    lstms
    """
    # Cargamos el dataframe
    df = pd.read_csv(file)
    df = df.sort_values('tref_start')
    return df


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
    inc = get_working_incidence(incidence_file)
    df = df.iloc[v:]
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


def plot_umbrales(title, resultados, file=None):
    title = titles[title]  # Titulo como paquete de incidencias
    plt.clf()
    x = [int(u) for u in resultados]
    y = [resultados[u]['metrica'] for u in resultados]
    plt.plot(x, y)
    plt.title(title)
    plt.ylabel('F2-score')
    plt.xlabel('Umbrales')
    if file:
        plt.savefig(file)
    else:
        plt.show()


def EXPERIMENTO2():
    """Con este código se generó:
    INFORME: Prueba de Long-short term memory neural networks para detección automatizada de incidencias
    CON FECHA: 8 de Marzo del 2020
    """
    # Entrenamos el modelo
    df = get_df()
    model, scaler = get_trained_model(df, model_specs)
    info = {name: {} for name in incidence_files}
    for name in incidence_files:
        print('Predicción con la incidencia {}'.format(name))
        incidence_file = incidence_files[name]['validate']
        ys, yhats, isIncidence = get_model_predictions(
            df, model, scaler, model_specs, incidence_file)
        info[name]['ys'] = ys
        info[name]['yhats'] = yhats
        info[name]['isIncidence'] = isIncidence
        resultados = umbral_testing(
            yhats,
            ys,
            isIncidence,
            umbrales=[i for i in range(1, 100)],
            verbose=False)
        print('Calculados los resultados por umbral de {}'.format(name))
        info[name]['resultados'] = resultados
        file = 'umbrales_{}.svg'.format('_'.join(name.split()))
        plot_umbrales(name, resultados, file)


"""
EXPERIMENTO 3:
Queremos comprobar si los buenos resultados conseguidos en el servicio
172.31.190.130 son replicables en otros servicios. Para ello realizaremos
el mismo análisis en todos los servicios seleccionados anteriormente. 
En vez de hacer una gráfica por cada servicio e incidencia, graficaremos
el rango de umbrales que superan un F2-score del 0.8
"""
# PASO1: Tenemos que dejar el dataframe en las condiciones que
#        que tenemos clean_172_31_190_130.csv, para replicar el
#        experimento
df_limpisimo_con_ips = '../datos/cleancleandf.csv'

# Nuestro df LIMPIO
#big_df = get_df(file=df_limpisimo_con_ips)
# Para tener el df como el del EXPERIMENTO2 hay que:
#df = big_df[big_df['targetIP']=='172.31.190.130'].drop('targetIP', axis=1)


def dot_to_underline(ip):
    return '_'.join(ip.split('.'))


# Entrenamos el modelo


def EXPERIMENTO3():
    big_df = get_df(file=df_limpisimo_con_ips)
    ips = set(big_df['targetIP'].values)
    # info[ip][incidencia] = resultados por umbral
    info = {ip: {name: None for name in incidence_files} for ip in ips}
    for ip in ips:
        df = big_df[big_df['targetIP'] == ip].drop('targetIP', axis=1)
        model_specs[
            'name'] = './lstm_models/lstm_90eps_200neurs_{}.model'.format(
                dot_to_underline(ip))
        #print('\n---\nComenzamos con la IP {}'.format(ip))
        model, scaler = get_trained_model(df, model_specs)
        for name in incidence_files:
            print('Predicción con la incidencia {}'.format(name))
            incidence_file = incidence_files[name]['validate']
            ys, yhats, isIncidence = get_model_predictions(
                df, model, scaler, model_specs, incidence_file)
            info[ip][name] = umbral_testing(
                yhats,
                ys,
                isIncidence,
                umbrales=[i for i in range(1, 50)],
                verbose=False)
            print('Calculados los resultados por umbral de {}'.format(name))
    # Para guardar los resultados umbrales
    import json
    store_f = './store_umbrales.json'
    with open(store_f, 'w') as f:
        f.write(json.dumps(info))


#EXPERIMENTO3()

# Para leer
import json
lim = 0.8


def EXPERIMENTO3_2_plot_umbrales_por_servidor(store_f='./store_umbrales.json'):
    with open(store_f, 'r') as f:
        info = json.loads(f.read())
    for name in incidence_files:
        x = [i for i in range(1, len(info) + 1)]
        y = []
        for ip in info:
            resultados = info[ip][name]
            mayores = []
            for u in resultados:
                if resultados[u]['metrica'] >= lim:
                    mayores.append(int(u))
            if len(mayores) == 0:
                mayores.append(-1)
            mayores.sort()
            y.append(mayores.copy())
        plt.clf()
        title = titles[name]  # Titulo como paquete de incidencias
        plt.title(title)
        plt.ylabel('Umbral con F2-score > {}'.format(lim))
        plt.xlabel('Servidor')
        for i, mayores in zip(x, y):
            for u in mayores:
                if u > 0:
                    plt.scatter([i], [u], color='blue')
                else:
                    plt.scatter([i], [u], color='red')
        plt.show()


#EXPERIMENTO3_2_plot_umbrales_por_servidor()

#EXPERIMENTO 3.3: PROBAMOS LA MISMA TECNICA CON VAR
from statsmodels.tsa.api import VAR
import numpy as np


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
    inc = get_working_incidence(incidence_file)
    df = df.iloc[v:]
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
        incidencias = incidencias[1:]
    X = scaler.transform(X)
    # Obtengamos predicciones
    ys = X[window:]
    yhats = []
    for i in range(window, len(X)):
        yhats.append(results.forecast(X[i - window:i], 1)[0])
    return ys, yhats, incidencias


def EXPERIMENTO3_3():
    """Hace el mismo computo que el experimento 3
    pero con el algoritmo VAR
    """
    big_df = get_df(file=df_limpisimo_con_ips)
    ips = set(big_df['targetIP'].values)
    # info[ip][incidencia] = resultados por umbral
    info = {ip: {name: None for name in incidence_files} for ip in ips}
    for ip in ips:
        df = big_df[big_df['targetIP'] == ip].drop('targetIP', axis=1)
        print('\n---\nComenzamos con la IP {}'.format(ip))
        train_perc = 0.7
        window = 18
        for name in incidence_files:
            print('Predicción con la incidencia {}'.format(name))
            incidence_file = incidence_files[name]['validate']
            ys, yhats, isIncidence = var_prediction(df, train_perc,
                                                    incidence_file)
            info[ip][name] = umbral_testing(
                yhats,
                ys,
                isIncidence,
                umbrales=[i for i in range(1, 100)],
                verbose=False)
            print('Calculados los resultados por umbral de {}'.format(name))
    # Para guardar los resultados umbrales
    import json
    store_f = './store_umbrales_var.json'
    with open(store_f, 'w') as f:
        f.write(json.dumps(info))


#EXPERIMENTO3_3()
#EXPERIMENTO3_2_plot_umbrales_por_servidor(store_f='store_umbrales_var.json')
