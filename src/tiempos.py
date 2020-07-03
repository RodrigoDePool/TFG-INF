from experimentos.simulacion_sin_incidencias import entrenar_lstm, entrenar_var, get_df, df_limpisimo_con_ips, predecir_lstm
from time import time
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# TIEMPO DE ENTRENAMIENTO DE 14 DIMENSIONES
print(
    '\n\n\nTIEMPO DE ENTRENAMIENTO DE LSTM POR SEGMENTO (Media de 10 segmentos)\n---'
)
big_df = get_df(file=df_limpisimo_con_ips)
ips = set(big_df['targetIP'].values)
start = time()
i = 1
for ip in ips:
    entrenar_lstm(big_df, ip)
    print('{} redes entrenadas (Tiempo acumulado hasta ahora  {:.2f})'.format(
        i,
        time() - start))
    i += 1
end = time()
print('\nRESULTADO: {:.2f} segundos de media'.format((end - start) / len(ips)))

# TIEMPO DE PREDICCIÓN
total_de_registros = len(big_df)
print(
    '\n\n\nTIEMPO DE PREDICCIÓN CON  LSTM POR REGISTRO (Media de {} registros)'
    .format(total_de_registros))
start = time()
for ip in ips:
    predecir_lstm(big_df, ip, umbral_range=False)
end = time()
print('\nTiempo medio de predicción por registro {:.8f}'.format(
    (end - start) / total_de_registros))
