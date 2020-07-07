import pandas as pd
import numpy as np
from keras.models import Sequential, load_model
from keras.layers import LSTM, Dense
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from time import struct_time, strptime
import json

#PARÁMETROS FIJADOS PARA ENTRENAMIENTO DE LA RED
train_size = 12 * 24 * 30  # 1 mes
input_window = 24  # 2 Horas
output_window = 1  # Un único intervalo
neuronas = 60
epocas = 15
test_size = 12 * 24 * 7  # 1 semana

# DataFrame
df = pd.read_csv('../cleancleandf.csv')
df = df.drop('Unnamed: 0', axis=1)
# Borramos los duplicados que tengan el
#   mismo IP y tiempos
df.drop_duplicates(subset=['targetIP', 'tref_start'], inplace=True)
ip_example = '172.31.191.104'


def one_feature_series(s):
    s = [[e] for e in s]
    return np.array(s)


def reverse_one_feature(s):
    s = [[x[0] for x in ts] for ts in s]
    return np.array(s)


def get_model(n=neuronas):
    model = Sequential()
    model.add(
        LSTM(
            neuronas,
            activation='relu',
            input_shape=(input_window, output_window)))
    model.add(Dense(output_window))
    model.compile(optimizer='adam', loss='mse')
    return model


"""NOTA INFORMATIVA
    Se obtuvo los mejores resultados
    con el MinMaxScaler
"""


def to_window(X, from_window=input_window, to_window=output_window):
    """ Convertimos [samples, features] en [samples, window, features]
    
        Input:
            - X: Datos con estructura [samples, features]
            - window: Tamaño de la ventana de valores anteriores
        Output:
            - Array de entradas con estructura [samples-window, window, features]
            - Array de salidas esperadas
    """
    tam = len(X)
    window_X = [None for _ in range(from_window, tam - to_window)]
    y = [None for _ in range(from_window, tam - to_window)]
    for i in range(from_window, tam - to_window):
        # Previous timesteps
        window_X[i - from_window] = [ts for ts in X[i - from_window:i]]
        # Próximos timesteps
        y[i - from_window] = [ts for ts in X[i:i + to_window]]
    return np.array(window_X), np.array(y)


def scaled_train_test_division(df,
                               ip,
                               train_size,
                               test_size,
                               column='numberCnx'):
    df_ip = df[df['targetIP'] == ip]
    # Dividimos el dataset en train y test
    data = df_ip[column].values
    train = one_feature_series(data[:train_size])
    test = one_feature_series(data[train_size:train_size + test_size])
    # Escalamos los datos con minmax
    scaler = MinMaxScaler()
    train = scaler.fit_transform(train)
    test = scaler.transform(test)
    # Convertimos la serie en un problema supervisado
    # Ventanas de 4 horas antes y después
    train_window, train_obj = to_window(train)
    test_window, test_obj = to_window(test)
    return (train_window, train_obj, test_window, test_obj, scaler)


def set_time_structs(incidencia):
    """Agrega a cada inicidencia dos campos:
    - desde: Objeto de tipo time con el momento en que empieza
    - hasta: Autoexplicativo
    """
    for ele in incidencia:
        ele['desde'] = strptime(ele['desde'], "%H:%M:%S %d-%m-%Y")
        ele['hasta'] = strptime(ele['hasta'], "%H:%M:%S %d-%m-%Y")
    return incidencia


def get_working_incidence(fichero):
    """Devuelve una incidencia con la estructura apropiada para
    que el generador de incidencias trabaje
    """
    with open(fichero, 'r') as f:
        inc = set_time_structs(json.load(f))
    return inc


def existIncidence(df):
    """Devuelve True si hay alguna incidencia en el dataframe
    """
    return len(df[df['incidencia'] == True]) > 0


# NUEVAS PRUEBAS
"""
# EJEMPLO  1
# Creamos un modelo
model = Sequential()
model.add(LSTM(50, activation='relu', input_shape=(18, 1)))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')
# Entrenamos
model.fit(train_window, train_obj, epochs=epochs, verbose=1)
# Veamos los resultados para este ejemplo
results = model.evaluate(test_window, test_obj)
pred = model.predict(test_window)
print('- MSE {}'.format(results))
yhats = weird_unscale(pred)
ys = weird_unscale(test_obj)
plt.clf()
plt.title('Título')
plt.plot(yhats[:200], label='Pred')
plt.plot(ys[:200], label='Data')
plt.legend(loc='upper right')
plt.show()
"""

# EJEMPLO 4 Veamos qué tal los modelos
"""
for ip in set(df['targetIP'].values):
    # Cargamos modelo y datos
    fname = 'modelos/20n_{}.model'.format(ip.replace('.', '_'))
    model = load_model(fname)
    _, _, testW, testO = scaled_train_test_division(df, ip)
    # Evaluación
    results = model.evaluate(testW, testO)
    print('---\nIP: {} \nLOSS MSE: {}'.format(ip, results))
    # Predicción y graficado
    prediction = model.predict(testW)
    yhats = weird_unscale(prediction)
    ys = weird_unscale(testO)
    plt.clf()
    plt.title('IP {}'.format(ip))
    plt.xlabel('Intervalo')
    plt.ylabel('# Flujos')
    plt.plot(yhats[:200], label='Pred')
    plt.plot(ys[:200], label='Data')
    plt.legend(loc='upper right')
    plt.show()

"""
