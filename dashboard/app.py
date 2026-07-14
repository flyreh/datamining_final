"""Portada. Ejecutar desde la raíz del repo: streamlit run dashboard/app.py"""
import streamlit as st

import comunes

st.set_page_config(
    page_title="Cortes de agua SUNASS",
    page_icon=":material/water_drop:",
    layout="wide",
)

st.title("Cortes de agua potable en el Perú (SUNASS 2019-2024)")
st.caption("Trabajo final de Minería de Datos — UNMSM FISI 2026-I")

st.markdown("""
**Problema real**: ¿en qué distritos se corta más el agua, por cuánto tiempo, y se
puede predecir si un corte será **PROGRAMADO** o **IMPREVISTO**?

El dataset fue **construido por el grupo** a partir del registro público de
interrupciones de SUNASS (45,671 filas crudas, 2019-06 → 2024-05,
[datosabiertos.gob.pe](https://www.datosabiertos.gob.pe)): limpieza, features de
calendario/franja horaria, agrupación de motivos y **cruce con la población
distrital del INEI**. La construcción es reproducible con
`scripts/01_construir_dataset.py`.
""")

# ------------------------------------------------------------------ KPIs de portada
enriquecido = comunes.cargar_enriquecido()
matriz = comunes.cargar_csv("matriz_distritos_clusters.csv")
comparativa = comunes.cargar_csv("comparativa_modelos.csv")
metricas_series = comunes.cargar_csv("metricas_series.csv")

pct_imprevistas = (enriquecido["TIPOINTERRUPCION"] == "IMPREVISTA").mean() * 100
ganador = comparativa.iloc[0]  # el CSV viene ordenado por F1 de PROGRAMADA
fila_arima = metricas_series[metricas_series["modelo"].str.startswith("ARIMA")].iloc[0]

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Registros analizados", f"{len(enriquecido):,}")
c2.metric("Cortes imprevistos", f"{pct_imprevistas:.1f}%")
c3.metric("Distritos clusterizados", f"{len(matriz)}")
c4.metric(f"F1 PROGRAMADA ({ganador['modelo']})", f"{ganador['f1_PROG']:.3f}")
c5.metric("MAPE pronóstico (ARIMA)", f"{fila_arima['MAPE_pct']:.1f}%")

st.divider()

# ------------------------------------------------------------------ guía de paneles
st.subheader("Los 4 paneles")
col_a, col_b = st.columns(2)
with col_a:
    st.markdown("""
**Panel 1 — EDA y Clustering**
Estadísticas descriptivas, outliers 1.5·IQR y K-means de distritos con
**método del codo + coeficiente de silueta** (el k se puede cambiar en vivo).

**Panel 2 — Clasificación**
**6 algoritmos comparados**, matriz de confusión interpretada, análisis de
desbalance y leakage, **SHAP/LIME**, y **formulario de predicción en vivo**.
""")
with col_b:
    st.markdown("""
**Panel 3 — Pronóstico semanal**
Serie semanal de cortes, backtest de 3 modelos con **MAPE y RMSE visibles**
y pronóstico ARIMA a **8 semanas** con intervalo de confianza.

**Panel 4 — CRUD de consultas**
Guarda cada predicción con **timestamp automático**; listar, editar y
eliminar desde el navegador (SQLite).
""")

st.divider()
st.markdown("""
**Equipo**: _(completar los 3 integrantes)_ ·
**Repositorio**: [github.com/flyreh/datamining_final](https://github.com/flyreh/datamining_final) ·
**Curso**: Minería de Datos, Dr. José Alfredo Herrera Quispe, UNMSM-FISI 2026-I
""")
