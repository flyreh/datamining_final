"""Utilidades compartidas: carga cacheada de datos/modelo y predicción en vivo.

`predecir_corte` replica el feature engineering del entrenamiento (notebook 03):
mismas 51 columnas del bundle y misma imputación con las medianas del train.
"""
from pathlib import Path

import holidays
import joblib
import numpy as np
import pandas as pd
import streamlit as st

RAIZ = Path(__file__).resolve().parents[1]
RUTA_PROCESSED = RAIZ / "data" / "processed"
RUTA_MODELOS = Path(__file__).resolve().parent / "modelos"
RUTA_RECURSOS = Path(__file__).resolve().parent / "recursos"

UMBRAL_DECISION = 0.5   # probabilidad a partir de la cual se predice PROGRAMADA

COLORES_TARGET = {"IMPREVISTA": "#e76f51", "PROGRAMADA": "#2a9d8f"}

# Nombres interpretables de los clusters k=4 (justificados en notebooks/02)
NOMBRES_CLUSTER = {
    0: "Sed crónica (frecuente, breve, reactivo)",
    1: "Crítico (cortes largos e imprevistos)",
    2: "Moderado (problema contenido)",
    3: "Planificado (predominan programados)",
}


# ---------------------------------------------------------------- carga cacheada
@st.cache_data
def cargar_csv(nombre: str) -> pd.DataFrame:
    return pd.read_csv(RUTA_PROCESSED / nombre)


@st.cache_data
def cargar_enriquecido() -> pd.DataFrame:
    df = pd.read_csv(RUTA_PROCESSED / "interrupciones_enriquecido.csv",
                     parse_dates=["fecha_inicio", "fecha_fin"])
    return df


@st.cache_resource
def cargar_clasificador() -> dict:
    """Bundle serializado por el notebook 03: modelo, columnas, medianas del train."""
    return joblib.load(RUTA_MODELOS / "clasificador_final.joblib")


@st.cache_data
def lookup_distritos() -> pd.DataFrame:
    """Mapeo distrito -> (DEPARTAMENTO, población) para el formulario en vivo.

    Los nombres de distrito se repiten entre departamentos: la etiqueta lleva provincia.
    """
    df = cargar_enriquecido()
    look = (
        df[df["DISTRITO"] != "SIN DATO"]
        .groupby(["DEPARTAMENTO", "PROVINCIA", "DISTRITO"], as_index=False)
        .agg(poblacion=("poblacion_distrito", "first"),
             n_registros=("IDINTERRUPCION", "count"))
        .sort_values("n_registros", ascending=False, ignore_index=True)
    )
    look["etiqueta"] = (look["DISTRITO"] + " — " + look["PROVINCIA"]
                        + ", " + look["DEPARTAMENTO"])
    return look


def motivos_disponibles() -> list[str]:
    """Las 11 categorías de motivo_agrupado, leídas de las columnas del bundle."""
    columnas = cargar_clasificador()["columnas"]
    return sorted(c.removeprefix("motivo_agrupado_")
                  for c in columnas if c.startswith("motivo_agrupado_"))


# ------------------------------------------------------------- predicción en vivo
def hora_a_bin(hora: int) -> str:
    """Misma discretización que scripts/01: bins [-1,5,11,17,23]."""
    if hora <= 5:
        return "MADRUGADA"
    if hora <= 11:
        return "MANANA"
    if hora <= 17:
        return "TARDE"
    return "NOCHE"


def predecir_corte(fecha, hora: int, departamento: str, poblacion,
                   motivo: str, conexiones, unidades, camiones: int):
    """Arma la fila de 51 features y devuelve (etiqueta, prob_PROGRAMADA).

    conexiones/unidades = None significa "la EPS no reportó" → NaN y se imputa
    con la mediana del train, igual que en el entrenamiento.
    """
    bundle = cargar_clasificador()
    feriados_pe = holidays.PE(years=[fecha.year])
    reporta = int(conexiones is not None)

    ratio = np.nan
    if reporta and conexiones and conexiones > 0:
        ratio = round((unidades or 0) / conexiones, 3)

    valores = {
        "mes": fecha.month,
        "dia_semana_num": fecha.weekday(),
        "es_fin_de_semana": int(fecha.weekday() >= 5),
        "es_feriado": int(fecha in feriados_pe),
        "hora_inicio_num": hora,
        "NUMCONEXDOM": conexiones if reporta else np.nan,
        "UNIDADESUSO": unidades if reporta else np.nan,
        "NUMCAMIONESPUNTOS": camiones,
        "reporta_conexiones": reporta,
        "ratio_afectados": ratio,
        "poblacion_distrito": poblacion if pd.notna(poblacion) else np.nan,
    }

    # Fila con las MISMAS columnas del entrenamiento: dummies en 0 salvo las del caso
    fila = pd.DataFrame(0.0, index=[0], columns=pd.Index(bundle["columnas"]))
    for col, val in valores.items():
        fila.loc[0, col] = val
    for dummy in (f"motivo_agrupado_{motivo}",
                  f"DEPARTAMENTO_{departamento}",
                  f"hora_inicio_bin_{hora_a_bin(hora)}"):
        if dummy in fila.columns:
            fila.loc[0, dummy] = 1.0

    fila = fila.fillna(pd.Series(bundle["medianas_imputacion"]))

    proba = float(bundle["modelo"].predict_proba(fila)[0, 1])
    etiqueta = "PROGRAMADA" if proba >= UMBRAL_DECISION else "IMPREVISTA"
    return etiqueta, proba
