"""
Dada una clasificación:
1. La "ayuda" a tener en cuenta posibles ráfagas de incidencias
2. Genera una evaluación de la precisión de la clasificación
Nota: Se basa en una métrica F2-score
3. Imprime información acerca de la evaluación
"""
from pandas import DataFrame


def refinar_clasificacion(incidencias_labels,
                          clasif,
                          lag=4,
                          error_classification_label=True):
    """Da una pequeña ayuda a la clasificación. 
    Si una clasificación de incidencia es correcta, arreglamos las clasificaciones alrededor
    
    Básicamente una vez encontrado el inicio de una incidencia, se considera
    que de allí en adelante estará bien clasificada (lo cual es razonable)
    """
    for i in range(lag, len(clasif) - lag):
        if clasif[i] == error_classification_label and incidencias_labels[
                i] == 1:
            for j in range(lag):
                clasif[i + lag] = incidencias_labels[i + lag]
    return clasif


def no_df_evaluar(incidencias, clasificacion, error_classification_label=True):
    # Computamos la precisión antes de la refinación
    fpos = 0.0
    tpos = 0.0
    for i, incidencia in enumerate(incidencias):
        if incidencia:
            if clasificacion[i] == error_classification_label:
                tpos += 1
        else:
            if clasificacion[i] == error_classification_label:
                fpos += 1

    if tpos + fpos == 0:
        especificidad = 0
    else:
        especificidad = tpos / (tpos + fpos)

    clasificacion = refinar_clasificacion(incidencias, clasificacion,
                                          error_classification_label)

    fpos = 0.0
    fneg = 0.0
    tpos = 0.0
    tneg = 0.0
    for i, incidencia in enumerate(incidencias):
        if incidencia:
            if clasificacion[i] == error_classification_label:
                tpos += 1
            else:
                fneg += 1
        else:
            if clasificacion[i] == error_classification_label:
                fpos += 1
            else:
                tneg += 1

    res = {
        'verdaderos positivos': tpos,
        'verdaderos negativos': tneg,
        'falsos positivos': fpos,
        'falsos negativos': fneg
    }
    res['metrica'], res['sensibilidad'], res['especificidad'] = fb_score(
        tpos, tneg, fpos, fneg)
    return res


def evaluar(df, clasificacion, error_classification_label=True):
    incidencias_labels = df['incidencia'].values
    return no_df_evaluar(incidencias_labels, clasificacion,
                         error_classification_label)


def fb_score(tpos, tneg, fpos, fneg, b=2):
    """ Devuelve el F score de la evaluación
    (media armónica de la precisión y la sensibilidad)

    Parámetros:
        tpos: Verdaderos positivos
        tneg: Verdaderos negativos
        fpos: Falsos positivos
        fneg: Falsos negativos
        b: Factor de relevancia de la sensibilidad por encima de
           la precisión
    Retorno:
        Devuelve el F score con el factor b de la evaluación
    Nota:
        - La sensibilidad es el porcentaje de errores que son detectados con éxito
        - La precisión es el porcentaje de acierto que existe cuando se declara una
          incidencia
        La b de nuestro medida indica cuántas veces es más 
        importante la sensibilidad por encima de la precisión.
        En otras palabras, indica cuánto más importante es detectar
        las incidencias a costa de indicar más errores de los que
        existen
    """
    if tpos == 0 or tpos + fneg == 0 or tpos + fpos == 0:
        return 0, 0, 0
    recall = tpos / (tpos + fneg)
    precision = tpos / (tpos + fpos)
    fb = (1 + b * b) * (precision * recall) / (b * b * precision + recall)
    return fb, recall, precision


def imprime_metrica(metr, indent='\t\t'):
    """Imprime el diccionario de resultados en un formato agradable.
    Indica todos los valores que contiene el diccionario.
    Además hace recuento de:
     - Porcentaje de incidencias detectadas que son reales
     - Porcentaje de incidencias detectadas del total de incidencias
    """
    for ele in metr:
        print('{}{}: {}'.format(indent, ele, metr[ele]))
    # Resumen de los valores
    print('{}Resumen:'.format(indent))
    # Estudiamos el porcentaje de incidencias dadas que son correctas
    aux = 0
    if metr['verdaderos positivos'] + metr['falsos positivos'] != 0:
        aux = 100 * metr['verdaderos positivos'] / (
            metr['verdaderos positivos'] + metr['falsos positivos'])
    print('{}\tDe las incidencias detectadas {:.2f}% son ciertas.'.format(
        indent, aux))
    # Estudiamos el porcentaje de incidencias detectadas
    aux = 0
    if metr['verdaderos positivos'] + metr['falsos negativos'] != 0:
        aux = 100 * metr['verdaderos positivos'] / (
            metr['verdaderos positivos'] + metr['falsos negativos'])
    print('{}\tSe detectaron {:.2f}% del total de las incidencias (Había {}).'.
          format(indent, aux,
                 metr['verdaderos positivos'] + metr['falsos negativos']))
    return