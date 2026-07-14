"""Panel 2 — Clasificación PROGRAMADA vs IMPREVISTA (artefactos del notebook 03)."""
import sys
from datetime import date, datetime
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import pandas as pd
import plotly.express as px
import streamlit as st

import bd
import comunes

st.set_page_config(page_title="Clasificación",
                   page_icon=":material/model_training:", layout="wide")
st.title("Panel 2 — ¿El corte será PROGRAMADO o IMPREVISTO?")

comparativa = comunes.cargar_csv("comparativa_modelos.csv").set_index("modelo")
desbalance = comunes.cargar_csv("efecto_desbalance.csv").set_index("configuracion")
ablacion = comunes.cargar_csv("ablacion_motivo.csv").set_index("features")
predicciones = comunes.cargar_csv("predicciones_test.csv")
bundle = comunes.cargar_clasificador()

# ══════════════════════════════════════════ 1. Comparativa de modelos
st.header("1. Comparativa de 6 algoritmos")

ganador = comparativa.iloc[0]  # ordenada por F1 de PROGRAMADA en el notebook
c1, c2, c3, c4 = st.columns(4)
c1.metric("Modelo ganador", bundle["nombre_modelo"])
c2.metric("F1 clase PROGRAMADA", f"{ganador['f1_PROG']:.3f}")
c3.metric("ROC-AUC", f"{ganador['roc_auc']:.3f}")
c4.metric("Recall PROGRAMADA", f"{ganador['rec_PROG']:.3f}")

col_tabla, col_graf = st.columns([3, 2])
with col_tabla:
    st.dataframe(comparativa.round(3), width="stretch")
    st.caption("Métricas sobre el test (30% estratificado). Ordenada por F1 de "
               "PROGRAMADA — la fila 1 es el ganador.")
with col_graf:
    fig = px.bar(comparativa[["f1_PROG", "roc_auc"]].sort_values("f1_PROG"),
                 orientation="h", barmode="group",
                 color_discrete_sequence=["#2a9d8f", "#264653"],
                 title="F1 (PROGRAMADA) y ROC-AUC por modelo")
    fig.update_layout(xaxis_range=[0, 1], yaxis_title="", legend_title="")
    st.plotly_chart(fig, width="stretch")

st.markdown("""
**¿Por qué gana Random Forest?** Se elige por **F1 de la clase minoritaria** (con
76/24 el accuracy premia a quien ignora PROGRAMADA) con ROC-AUC de desempate:
RF **0.912 / 0.986** supera a XGBoost (0.890), MLP (0.885), KNN (0.875) y la
logística (0.842). Además no necesita escalado y captura no-linealidades sin
ingeniería extra. **Naive Bayes colapsa** (F1 0.386): asumir gaussianas
independientes sobre ~40 dummies binarias rompe el modelo — buen ejemplo de por
qué se comparan familias de algoritmos.
""")

# ══════════════════════════════════════════ 2. Matriz de confusión
st.header("2. Matriz de confusión del ganador")

etiquetas = ["IMPREVISTA (0)", "PROGRAMADA (1)"]
cm = pd.crosstab(predicciones["y_real"], predicciones["y_pred"])
cm.index, cm.columns = etiquetas, etiquetas
cm_pct = (cm.div(cm.sum(axis=1), axis=0) * 100).round(1)

col_a, col_b = st.columns(2)
fig = px.imshow(cm, text_auto=True, color_continuous_scale="Blues",
                title=f"Conteos (test = {len(predicciones):,} registros)")
fig.update_layout(xaxis_title="predicho", yaxis_title="real", coloraxis_showscale=False)
col_a.plotly_chart(fig, width="stretch")

fig = px.imshow(cm_pct, text_auto=True, color_continuous_scale="Blues",
                title="Normalizada por fila (%)")
fig.update_layout(xaxis_title="predicho", yaxis_title="real", coloraxis_showscale=False)
col_b.plotly_chart(fig, width="stretch")

st.markdown("""
**¿Qué error es más costoso?** Una **IMPREVISTA clasificada como PROGRAMADA**
(fila superior, columna derecha) implica tratar una emergencia como si estuviera
planificada: no se activan cisternas ni comunicación de contingencia. El error
inverso solo genera una alerta de más. Por eso, además del F1, se vigila la
**precisión de PROGRAMADA** (0.907): cuando el modelo dice "programada", debe
estar seguro.
""")

# ══════════════════════════════════════════ 3. Desbalance y leakage
st.header("3. Desbalance 76/24 y análisis de leakage")

col_a, col_b = st.columns(2)
with col_a:
    st.subheader("Efecto en el recall de la minoritaria")
    st.dataframe(desbalance.round(3), width="stretch")
    st.markdown(
        "`class_weight='balanced'` sube el recall de PROGRAMADA de **0.885 a "
        "0.918** cediendo ~3 pts de precisión, con F1 estable; SMOTE queda a "
        "medio camino (0.887) y genera ~16K filas sintéticas por reentrenamiento. "
        "**Elegido: class_weight**.")
with col_b:
    st.subheader("¿`motivo_agrupado` regala el target?")
    st.dataframe(ablacion.round(3), width="stretch")
    st.markdown(
        "El motivo se conoce **cuando la EPS reporta el corte** (no es leakage "
        "temporal). Sin él, el F1 cae a 0.753 pero el **AUC se mantiene en "
        "0.923**: el modelo no es una tabla de búsqueda — hora, calendario, "
        "geografía y conexiones sostienen un clasificador útil. **Se mantiene.**")

# ══════════════════════════════════════════ 4. Interpretabilidad
st.header("4. ¿Por qué predice lo que predice? (SHAP + LIME)")

col_a, col_b = st.columns(2)
img_summary = comunes.RUTA_RECURSOS / "shap_summary.png"
img_importancia = comunes.RUTA_RECURSOS / "shap_importancia.png"
if img_summary.exists():
    col_a.image(str(img_summary), caption="SHAP summary: cada punto es un registro "
                "del test; rojo = valor alto de la feature. A la derecha empuja "
                "hacia PROGRAMADA.")
    col_b.image(str(img_importancia), caption="Importancia global (|SHAP| medio).")
else:
    st.warning("Ejecutar notebooks/src/03_clasificacion.py para generar los PNG de SHAP.")

with st.expander("Un caso explicado: force plot (SHAP) + LIME"):
    st.markdown(
        "Caso del test: corte de **SEDAPAL en San Juan de Lurigancho por "
        "limpieza y desinfección de reservorio** — el modelo lo clasifica "
        "PROGRAMADA con probabilidad ≈ 1.0. El force plot muestra qué features "
        "empujan desde el valor base (proporción histórica de PROGRAMADA) hacia "
        "la predicción final; LIME confirma localmente las mismas contribuciones.")
    img_force = comunes.RUTA_RECURSOS / "shap_force_caso.png"
    img_lime = comunes.RUTA_RECURSOS / "lime_caso.png"
    if img_force.exists():
        st.image(str(img_force), caption="SHAP force plot del caso")
    if img_lime.exists():
        st.image(str(img_lime), caption="LIME del mismo caso")

# ══════════════════════════════════════════ 5. Predicción en vivo
st.header("5. Predicción en vivo")
st.markdown(
    f"El formulario arma un registro con las **mismas {len(bundle['columnas'])} "
    "features del entrenamiento** (calendario, feriados del Perú, franja horaria, "
    "departamento y población del distrito) y consulta el modelo serializado.")

look = comunes.lookup_distritos()

with st.form("form_prediccion"):
    col1, col2, col3 = st.columns(3)
    with col1:
        etiqueta_distrito = st.selectbox("Distrito", look["etiqueta"])
        motivo = st.selectbox("Motivo reportado", comunes.motivos_disponibles())
    with col2:
        fecha_corte = st.date_input("Fecha del corte", value=date.today())
        hora = st.slider("Hora de inicio", 0, 23, 10)
    with col3:
        reporta = st.checkbox("La EPS reporta conexiones afectadas", value=True)
        conexiones = st.number_input("Conexiones domiciliarias afectadas",
                                     min_value=0, value=500, step=50)
        unidades = st.number_input("Unidades de uso afectadas",
                                   min_value=0, value=550, step=50)
        camiones = st.number_input("Camiones cisterna desplegados",
                                   min_value=0, value=0, step=1)
    enviado = st.form_submit_button("Predecir", type="primary")

if enviado:
    fila_look = look[look["etiqueta"] == etiqueta_distrito].iloc[0]
    etiqueta, proba = comunes.predecir_corte(
        fecha=fecha_corte, hora=hora,
        departamento=fila_look["DEPARTAMENTO"], poblacion=fila_look["poblacion"],
        motivo=motivo,
        conexiones=conexiones if reporta else None,
        unidades=unidades if reporta else None,
        camiones=camiones)

    # se guarda en sesión para poder enviarla al CRUD (Panel 4)
    st.session_state["ultima_consulta"] = {
        "distrito": fila_look["DISTRITO"], "departamento": fila_look["DEPARTAMENTO"],
        "fecha_corte": str(fecha_corte), "hora": hora, "motivo": motivo,
        "conexiones": float(conexiones) if reporta else None,
        "unidades": float(unidades) if reporta else None,
        "prediccion": etiqueta, "prob_programada": round(proba, 4),
        "nota": "",
    }

if "ultima_consulta" in st.session_state:
    consulta = st.session_state["ultima_consulta"]
    c1, c2 = st.columns(2)
    c1.metric("Predicción", consulta["prediccion"])
    c2.metric("P(PROGRAMADA)", f"{consulta['prob_programada']:.1%}")
    st.progress(consulta["prob_programada"],
                text=f"Probabilidad de PROGRAMADA: {consulta['prob_programada']:.1%} "
                     f"(umbral de decisión: {comunes.UMBRAL_DECISION:.0%})")

    if st.button("Guardar esta consulta en el Panel 4 (CRUD)",
                 icon=":material/save:"):
        id_nuevo = bd.crear(consulta)
        st.success(f"Consulta guardada con id **{id_nuevo}** "
                   f"({datetime.now():%Y-%m-%d %H:%M:%S}). Revísala en el Panel 4.")
