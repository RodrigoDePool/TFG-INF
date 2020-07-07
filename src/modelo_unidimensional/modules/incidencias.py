"""
Módulo generador de incidencias
"""
from time import struct_time, localtime
from collections import defaultdict
from pandas import DataFrame


def generar_incidencias(df, incidencias):
    """ Devuelve una copia del dataframe df agregando las incidencias indicadas.
    Se copia el dataframe proporcionado, se modifican los valores  según las incidencias 
    y se agregará la columna 'incidencia' que será True si ese vector fue modificado.
   
    Input:
        df: Dataframe que tiene los vectores con los valores de la red cada 5 minutos
        incidencias: Lista de incidencias donde cada incidencia es un diccionario con:
            - desde (struct_time): Fecha en la que inicia la incidencia 
            - hasta (struct_time): Fecha en la que termina la incidencia 
            - proporcion: Proporción de usuarios a los que afecta la incidencia
            - intensidad: Intensidad de la incidencia
            - col: Columna del dataframe al que afecta la incidencia
    
    """
    df = df.copy(deep=True)
    esIncidencia = crea_identificador_de_incidencias(incidencias)

    columna_incidencias = []
    for index, row in df.iterrows():
        incidence_generated = False
        if esIncidencia(row):
            # Agregamos las modificaciones de las incidencias de esta fila
            for incidencia in info_de_incidencias(row, incidencias):
                col = incidencia['columna']
                proporcion = incidencia['proporcion']
                intensidad = incidencia['intensidad']

                initial_value = row[col]
                if initial_value == 0:
                    initial_value = df[col].mean()
                if initial_value != 0:
                    df[col][
                        index] = initial_value * proporcion * intensidad + initial_value * (
                            1 - proporcion)
                    incidence_generated = True
        # Agregamos el flag incidence generated porque cabe la posibilidad
        # de que aunque queramos hacer una incidencia el valor de la col
        # esté a 0 y su media también. De modo que no tenemos forma de
        # generar la incidencia y, en ese caso, no lo hacemos. Las filas
        # en las que esto ocurre no son marcada como filas con incidencias
        columna_incidencias.append(incidence_generated)

    df['incidencia'] = columna_incidencias
    return df


def crea_identificador_de_incidencias(incidencias):
    """Devuelve una lambda que te dice si en una row se debe o no agregar una incidencia
    """
    esIncidencia_lista = []
    for incidencia in incidencias:
        aux = lambda row: time_obj(row) >= incidencia['desde'] and time_obj(
            row) <= incidencia['hasta']
        esIncidencia_lista.append(aux)

    esIncidencia = lambda row: any(aux(row) for aux in esIncidencia_lista)
    return esIncidencia


def info_de_incidencias(row, incidencias):
    """Devuelve una lista con las incidencias que se deben agregar a la fila
    Cada elemento de la lista tiene un diccionario con:
        - columna: Columna que es sujeto de la modificacion
        - proporcion: Proporcion de usuarios afectados por la incidencia
        - intensidad: Intensidad de la incidencia
    """
    info = []
    for incidencia in incidencias:
        if time_obj(row) >= incidencia['desde'] and time_obj(
                row) <= incidencia['hasta']:
            aux = {}
            aux['columna'] = incidencia['columna']
            aux['proporcion'] = incidencia['proporcion']
            aux['intensidad'] = incidencia['intensidad']
            info.append(aux)
    return info


def time_obj(row) -> struct_time:
    """Da la estructura de tiempo de una fila de un dataframe
    """
    return localtime(row['tref_start'] / 1000)
