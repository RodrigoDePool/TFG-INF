import pandas as pd
"""Ejemplo de funcionamiento del generador de incidencias
"""
from modules.util import get_working_incidence
from modules.incidencias import generar_incidencias

# Cargamos las métricas de un único servidor
df_file = 'datos/oneServerMetrics.csv'
df = pd.read_csv(df_file)
# Cargamos la incidencia
inc_file = 'datos/incidencias/incidencias_ejemplo/down_bps_rcv_e.json'
inc = get_working_incidence(inc_file)
# Generamos el dataframe con la incidencia
df_mod = generar_incidencias(df, inc)
df_mod.to_csv('df_con_incidencias.csv')
"""
Se requieren dos ficheros:

- df_file: Es el fichero csv donde se
           guarda las métricas de red
           de un servidor concreto.
           Las métricas se deben 
           recoger agregadas por
           servidor y cada fila 
           representará un intervalo
           de tiempo.

- inc_file: Es el fichero json donde
            se define la incidencia.
            La incidencia consta de
            varias "modificaciones"
            que se pueden solapar. 
            Las modificaciones
            afectan a una métrica
            del servidor (una de las
            columnas del csv), con
            una intensidad (factor
            multiplicativo) y proporción
            fijados. Véase el ejemplo.

NOTA IMPORTANTE:

El csv con las métricas de red tiene que tener
una columna que identifique el instante temporal
en que son recogidas. En el caso de ejemplo que
se muestra esta columna es 'tref_start' y guarda
los epochs en milisegundos. 

Si el csv que se utiliza guarda el tiempo en un
formato distinto, será necesario cambiar la función 
*def time_obj(row)* del fichero ./modules/incidencias. 
Esta función recibe una fila del csv y devuelve el
struct_time que responde a ese registro. 
"""