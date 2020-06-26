"""
SCRIPT que genera las imágenes
"""
from util import *
import matplotlib.pyplot as plt
IMG_FILE_2_2 = '../pdfs/ejemplo_lstm.pdf'
IMG_FILE_2_3 = '../pdfs/serie_estacionaria.pdf'
IMG_FILE_2_4 = '../pdfs/ejemplo_var.pdf'
IMG_FILE_4_11 = '../pdfs/serv1_1.pdf'
IMG_FILE_4_12 = '../pdfs/serv1_2.pdf'
IMG_FILE_4_21 = '../pdfs/serv2_1.pdf'
IMG_FILE_4_22 = '../pdfs/serv2_2.pdf'

###
# [IMG 2.2] Ejemplo de predicción con LSTM
###
ip = '172.31.191.104'
ys, yhats = info[ip]['LSTM']['raw_data']
# 4500 ~= 15 días de datos
plt.clf()
plt.plot(ys[:4500, 0], label='Datos')
plt.plot(yhats[:4500, 0], label='Predicción')
plt.ylabel('Bits recibidos normalizados')
plt.xlabel('Intervalo de 5 minutos')
plt.legend(loc='upper right')
plt.savefig(IMG_FILE_2_2)

###
# [IMG 2.4] Ejemplo de predicción con VAR
###
ip = '172.31.191.104'
ys, yhats = info[ip]['VAR']['raw_data']
# 4500 ~= 15 días de datos
plt.clf()
plt.plot(ys[:4500, 0], label='Datos')
plt.plot(yhats[:4500, 0], label='Predicción')
plt.ylabel('Variación bits recibidos normalizados')
plt.xlabel('Intervalo de 5 minutos')
plt.legend(loc='upper right')
plt.savefig(IMG_FILE_2_4)

###
# [IMG 2.3] Ejemplo de serie estacionaria
#           cuando se considera la varia
#           ción
###
ip = '172.31.190.124'
df_ip = big_df[big_df['targetIP'] == ip].drop('Unnamed: 0', axis=1)
df_ip = df_ip.drop('tref_start', axis=1)
X = df_ip.values[:1900, 0]
X = np.diff(X, axis=0)
plt.clf()
plt.ylabel('Variación de bits enviados')
plt.xlabel('Intervalo de tiempo')
plt.plot(X)
plt.savefig(IMG_FILE_2_3)


# Funcion auxiliar para las siguientes
#  imagenes. Plottea una caracteristica
#  de red
def plot_metric(ip, col, fname):
    # Nos quedamos con los datos predichos
    df = big_df[big_df['targetIP'] == ip].drop('targetIP', axis=1)
    df = df.drop('Unnamed: 0', axis=1)
    df = df.drop('tref_start', axis=1)
    v = int(len(df) * 0.7)
    df = df.iloc[v + 18:v + 18 + 4500]
    ys = df[col_nom[col]]
    # Gráfica
    plt.clf()
    plt.title(col_nom[col])
    plt.plot(ys)
    plt.savefig(fname)


###
# [IMG 4.1] Gráfica de bits recibidos
#           y RTT por conexión que se
#           utiliza para ilustrar las
#           incidencias detectadas
###
ip = '172.31.191.104'
# Bits recibidos
plot_metric(ip, 0, IMG_FILE_4_11)
# Rtt per cnx
plot_metric(ip, 10, IMG_FILE_4_12)

###
# [IMG 4.2] Gráfica de num de cnx y
#           RTT por conexión que se
#           utiliza para ilustrar las
#           incidencias detectadas
###
ip = '172.31.190.130'
# Bits recibidos
plot_metric(ip, 5, IMG_FILE_4_21)
# Rtt per cnx
plot_metric(ip, 10, IMG_FILE_4_22)
