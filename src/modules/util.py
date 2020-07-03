"""
Utilidades para:
1. Cargar datos
2. Cargar algoritmos
3.  LLevar una caché de las incidencias que ya se han generado previamente
4. Limpiar datos
5. Imprimir resultados
"""
import json
import os
from modules.incidencias import generar_incidencias
from modules.evaluacion import evaluar, imprime_metrica
from time import struct_time, strptime
import pandas as pd
import sys


def set_time_structs(incidencia):
    """Agrega a cada inicidencia dos campos:
    - desde: Objeto de tipo time con el momento en que empieza
    - hasta: Autoexplicativo
    """
    for ele in incidencia:
        ele['desde'] = strptime(ele['desde'], "%H:%M:%S %d-%m-%Y")
        ele['hasta'] = strptime(ele['hasta'], "%H:%M:%S %d-%m-%Y")
    return incidencia


def get_df_with_inc(df, incidence, incidence_name, incidenceH):
    """Devuelve el dataframe con la incidencia generada
    Si está en caché se recupera directamente y se devuelve
    Si no lo está entonces se genera, se guarda en caché y se devuelve
    """
    id = incidenceH.getId(incidence)
    if id is not None:  # Si se tiene en caché
        cacheFile = incidenceH.getIncidenceDFFile(id, False)
        if cacheFile is not None:
            df = pd.read_csv(cacheFile)
            return df
    # Si no se tiene en caché
    df = generar_incidencias(df, incidence)
    incidenceH.addDF_toIncidence(incidence, incidence_name, df, False)
    return df


def get_df_difference(df, incidence, incidence_name, incidenceH):
    """Devuelve el dataframe de la diferencia de df
    Se le pasa inc para poder comprobar si el dataframe está en caché
    Si está en caché se recupera y se devuelve.
    Si no lo está entonces se genera la diferencia, se guarda y se retorna
    """
    id = incidenceH.getId(incidence)
    if id is not None:  # Si se tiene en caché
        cacheFile = incidenceH.getIncidenceDFFile(id, True)
        if cacheFile is not None:
            df = pd.read_csv(cacheFile)
            return df
    # Si no se tiene en caché
    df = generar_diferencias_por_ip(df)
    incidenceH.addDF_toIncidence(incidence, incidence_name, df, True)
    return df


def cargarDatos(datafile, incfiles, incidenceH):
    """
    Carga y genera incidencias en  un dataframe
    Input:
        - Datafile: Fichero con el df inicial sobre el que se generarán
                    incidencias
        - Incfiles: Lista de pares ->
                    (fichero_incidencias_entrenamiento, igual_validacion)
    Output:
        - Dataframe sin incidencias
        - Lista de tuplas de 4 dataframes:
            1. DF con incidencias de entrenamiento
            2. DF con incidencias de validación
            3. Igual que 1 pero calculando las diferencias
            4. Igual que 2 pero calculando las diferencias
        - Lista de id de incidencias
        Habrá un elemento en las listas por cada par en incfiles
    """
    df = pd.read_csv(datafile)
    df = trim_dataframe(df)
    # Creamos la lista de id_incidencias y de dataframes
    df_list, id_incidencias = [], []
    for fichero_e, fichero_v in incfiles:
        inc_e = get_working_incidence(fichero_e)
        inc_v = get_working_incidence(fichero_v)
        # Generamos las incidencias
        df_e = get_df_with_inc(df, inc_e, fichero_e, incidenceH)
        df_v = get_df_with_inc(df, inc_v, fichero_e, incidenceH)
        df_e_dif = get_df_difference(df_e, inc_e, fichero_e, incidenceH)
        df_v_dif = get_df_difference(df_v, inc_v, fichero_e, incidenceH)
        # Guardamos los dataframes y los ids de las incidencias generadas
        df_list.append((df_e, df_v, df_e_dif, df_v_dif))
        id_incidencias.append(incidenceH.getId(inc_e))
    return df, df_list, id_incidencias


def get_working_incidence(fichero):
    """Devuelve una incidencia con la estructura apropiada para
    que el generador de incidencias trabaje
    """
    with open(fichero, 'r') as f:
        inc = set_time_structs(json.load(f))
    return inc


def trim_dataframe(df):
    """Elimina columnas poco relevantes o que no son númericas (excepto IP)
    """
    non_numerical_cols = [
        'proto', 'label', 'Unnamed: 0', 'dupAckPerc', 'fallPerc',
        'Unnamed: 0.1'
    ]
    non_useful_cols = [
        'noRespClientPerc', 'noRespServerPerc', 'numberCnxPerc',
        'resetClientPerc', 'resetServerPerc', 'rttPerCnxPerc', 'rtxPerc',
        'synPerc', 'ttl1Perc', 'win0Perc', 'metric'
    ]
    for col in non_numerical_cols + non_useful_cols:
        if col in df.columns:
            df = df.drop(col, axis=1)
    return df


def generar_diferencias_por_ip(
        df,
        no_dif_cols=['targetIP', 'incidencia', 'tref_start', 'hour', 'wday']):
    """Dado un dataframe lo diferencia por ips. Las columnas en
    'no_dif_cols' no son diferenciadas.
    """
    all_ips = set(df['targetIP'].values)
    rows = []
    for ip in all_ips:
        df_by_ip = df[df['targetIP'] == ip]
        df_by_ip.sort_values(by=['tref_start'])
        rows += dataframe_difference_by_rows(df_by_ip, no_dif_cols)
    return pd.DataFrame(rows)


def dataframe_difference_by_rows(
        df,
        no_dif_cols=['targetIP', 'incidencia', 'tref_start', 'hour', 'wday']):
    """Genera una lista de filas donde cada
    fila nueva es la diferencia columna a columna de dos filas 
    consecutivas de 'df'. Las columnas en 'no_dif_cols' se mantienen
    igual, no se les aplica la diferencia.
    """
    rows = []
    prev = None
    no_dif_cols = set([col for col in no_dif_cols if col in set(df.columns)])
    dif_cols = [col for col in set(df.columns) if col not in no_dif_cols]
    for index, row in df.iterrows():
        if prev is not None:
            # Para las columnas que no diferenciamos
            dif_row = {col: row[col] for col in no_dif_cols}
            # Para las columnas que sí
            for col in dif_cols:
                dif_row[col] = row[col] - prev[col]
            # Guardamos la fila
            rows.append(dif_row)
        prev = row
    return rows


def informar_resultados(resultados, indent=''):
    """Imprime el json con una estructura de indentación según la profundidad
    en el json.
    """
    for key in resultados:
        if isinstance(resultados[key], dict):
            print('{}- {}:'.format(indent, key))
            informar_resultados(resultados[key], indent + '\t')
        else:
            imprime_metrica(resultados, indent=indent + '-')
            break


def existIncidence(df):
    """Devuelve True si hay alguna incidencia en el dataframe
    """
    return len(df[df['incidencia'] == True]) > 0
