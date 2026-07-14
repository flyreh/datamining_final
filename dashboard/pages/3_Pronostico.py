"""Panel 3 — Pronóstico semanal de cortes (artefactos del notebook 04)."""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import comunes

# Ventana de análisis definida en el notebook 04 (bordes del registro truncados)
FECHA_INICIO_SERIE = "2019-06-24"
FECHA_FIN_SERIE = "2024-04-15"
SEMANAS_TEST = 12       # backtest: últimas 12 semanas de la ventana
HORIZONTE = 8           # semanas pronosticadas (obligatorio: ≥4)

st.set_page_config(page_title="Pronóstico",
                   page_icon=":material/trending_up:", layout="wide")
st.title("Panel 3 — Pronóstico semanal de cortes")

serie = comunes.cargar_csv("serie_semanal.csv")
serie["semana"] = pd.to_datetime(serie["semana"])
metricas = comunes.cargar_csv("metricas_series.csv").set_index("modelo")
pronostico = comunes.cargar_csv("pronostico_semanal.csv")
pronostico["semana"] = pd.to_datetime(pronostico["semana"])

modelo_final = pronostico["modelo"].iloc[0]           # "ARIMA (2, 1, 1)"
fila_final = metricas.loc[modelo_final]

# ══════════════════════════════════════ MAPE y RMSE visibles (obligatorio)
c1, c2, c3, c4 = st.columns(4)
c1.metric("Modelo final", modelo_final)
c2.metric("MAPE (backtest 12 sem)", f"{fila_final['MAPE_pct']:.1f}%")
c3.metric("RMSE (backtest 12 sem)", f"{fila_final['RMSE']:.1f} cortes")
c4.metric("Horizonte", f"{HORIZONTE} semanas")

# ══════════════════════════════════════ 1. Serie + pronóstico
st.header("1. Serie total y pronóstico a 8 semanas")

total = serie[(serie["distrito"] == "TOTAL")
              & (serie["semana"] >= FECHA_INICIO_SERIE)
              & (serie["semana"] <= FECHA_FIN_SERIE)]

semanas_visibles = st.slider("Semanas históricas a mostrar", 26, len(total),
                             104, step=26)
historico = total.tail(semanas_visibles)

fig = go.Figure()
fig.add_trace(go.Scatter(x=historico["semana"], y=historico["n_cortes"],
                         name="histórico", line={"color": "#264653"}))
# banda del intervalo de confianza 95% (se dibuja antes que la línea central)
fig.add_trace(go.Scatter(
    x=pd.concat([pronostico["semana"], pronostico["semana"][::-1]]),
    y=pd.concat([pronostico["ic_superior"], pronostico["ic_inferior"][::-1]]),
    fill="toself", fillcolor="rgba(231,111,81,0.2)",
    line={"color": "rgba(0,0,0,0)"}, name="IC 95%", hoverinfo="skip"))
fig.add_trace(go.Scatter(x=pronostico["semana"], y=pronostico["pronostico"],
                         name=f"pronóstico ({modelo_final})",
                         line={"color": "#e76f51", "dash": "dash"},
                         mode="lines+markers"))
fig.update_layout(title=f"Cortes de agua por semana (nivel país) + pronóstico "
                        f"de {HORIZONTE} semanas",
                  yaxis_title="interrupciones únicas / semana",
                  legend={"orientation": "h", "y": 1.12}, height=480)
st.plotly_chart(fig, width="stretch")

st.dataframe(
    pronostico.assign(semana=pronostico["semana"].dt.date)
    .rename(columns={"pronostico": "pronóstico", "ic_inferior": "IC 95% inf",
                     "ic_superior": "IC 95% sup"}).set_index("semana").round(1),
    width="stretch")

# ══════════════════════════════════════ 2. Backtest de los 3 modelos
st.header("2. ¿Cómo se eligió el modelo? (backtest)")

col_a, col_b = st.columns([2, 3])
with col_a:
    st.dataframe(metricas.round(2), width="stretch")
    st.caption(f"Backtest con corte temporal: entrenamiento hasta la semana "
               f"-{SEMANAS_TEST}, prueba en las últimas {SEMANAS_TEST} semanas "
               "de la ventana (nunca split aleatorio en series).")
with col_b:
    st.markdown("""
- **Empate técnico** entre la media móvil (14.9%) y ARIMA (15.1%): a menos de 1
  punto de MAPE se prefiere el **modelo estadístico**, que da un intervalo de
  confianza fundamentado y reacciona a cambios de nivel (regla de desempate
  explícita en el notebook 04). Holt-Winters queda atrás (23.2%).
- **¿Un MAPE de ~15% es aceptable?** Sí: es el **piso de ruido de la serie** —
  ni el baseline ingenuo baja de ahí. La serie semanal de cortes tiene una
  componente aleatoria irreducible (roturas no se anticipan una a una).
- El orden **(2,1,1)** de ARIMA se eligió por mínimo AIC en una grilla
  p,q ∈ 0..2, d ∈ 0..1.
""")

with st.expander("Complicación → resolución: los bordes truncados del registro"):
    total_completa = serie[serie["distrito"] == "TOTAL"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=total_completa["semana"], y=total_completa["n_cortes"],
                             name="serie completa", line={"color": "#adb5bd"}))
    fig.add_trace(go.Scatter(x=total["semana"], y=total["n_cortes"],
                             name="ventana de análisis", line={"color": "#264653"}))
    fig.update_layout(title="Serie completa vs ventana de análisis",
                      yaxis_title="cortes/semana", height=380)
    st.plotly_chart(fig, width="stretch")
    st.markdown(f"""
Las primeras semanas del registro (3, 1, 38 eventos: arranque del sistema) y las
últimas (67, 8: corte de extracción del dataset) están **truncadas** frente a la
mediana de ~204 cortes/semana. Con esas semanas dentro del test, el MAPE del
backtest superaba **260%**; recortando la serie a
**{FECHA_INICIO_SERIE} → {FECHA_FIN_SERIE}** baja a ~15%. La ventana es una
constante nombrada del notebook 04.
""")

# ══════════════════════════════════════ 3. Series por distrito
st.header("3. Explorar la serie por distrito")

distritos_serie = sorted(serie["distrito"].unique(), key=lambda d: d != "TOTAL")
distrito_sel = st.selectbox("Serie (TOTAL o los 8 distritos con más afectaciones)",
                            distritos_serie)
serie_sel = serie[serie["distrito"] == distrito_sel]

fig = go.Figure()
fig.add_trace(go.Scatter(x=serie_sel["semana"], y=serie_sel["n_cortes"],
                         line={"color": "#457b9d"}, name=distrito_sel))
fig.add_trace(go.Scatter(x=serie_sel["semana"],
                         y=serie_sel["n_cortes"].rolling(8, center=True).mean(),
                         line={"color": "#e76f51", "width": 3},
                         name="media móvil 8 sem"))
fig.update_layout(title=f"Cortes semanales — {distrito_sel}",
                  yaxis_title="cortes/semana", height=420,
                  legend={"orientation": "h", "y": 1.12})
st.plotly_chart(fig, width="stretch")
