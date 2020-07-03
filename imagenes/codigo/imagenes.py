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
IMG_FILE_3_5 = '../pdfs/incidencia_sencilla.pdf'
IMG_FILE_3_2 = '../pdfs/Untitled.pdf'
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

###
# [IMG 3.5] Gráfica de ejemplo de inducción
#           de incidencia
###
Ys = df_ip.values[:600, 0]
Ys_alterados = [
    y * 0.95 * 2 + y * 0.05 if i > 150 and i < 300 else y
    for i, y in enumerate(Ys)
]
plt.clf()
plt.title('Incidencia inducida (95% de usuarios con factor de 2)')
plt.xlabel('Intervalo de tiempo')
plt.ylabel('PPS enviados')
plt.plot(Ys_alterados, color='red')
plt.plot(Ys, color='blue')
plt.savefig(IMG_FILE_3_5)

###
# [IMG 3.2] Ejemplo exagerado de incidencia en
#           en bits recibidos
###
ip = '192.168.34.52'
df_ip = big_df[big_df['targetIP'] == ip].drop('Unnamed: 0', axis=1)
Ys = df_ip.values[:600, 1]
Ys_o = []
for i, y in enumerate(Ys):
    if i > 200:
        if i < 220:
            y = y * 1.2
        elif i < 250:
            y = y * 1.5
        elif i < 300:
            y = y * 2
        elif i < 340:
            y = y * 1.5
    Ys_o.append(y)
plt.clf()
plt.title('Ejemplo de incidencia')
plt.xlabel('Intervalo de tiempo')
plt.ylabel('Bps enviados')
plt.plot(Ys_o)
plt.savefig(IMG_FILE_3_2)
