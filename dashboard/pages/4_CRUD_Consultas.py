"""Panel 4 — CRUD de consultas de predicción (SQLite, timestamps automáticos)."""
import sys
from datetime import date
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import pandas as pd
import streamlit as st

import bd
import comunes

st.set_page_config(page_title="CRUD de consultas",
                   page_icon=":material/database:", layout="wide")
st.title("Panel 4 — CRUD de consultas")
st.markdown(
    "Cada consulta guarda **los inputs, la predicción devuelta y el timestamp "
    "automático**. Backend: SQLite local (`dashboard/consultas.db`) — se resetea "
    "al redeploy de Streamlit Cloud, suficiente para la demo.")

look = comunes.lookup_distritos()
motivos = comunes.motivos_disponibles()


def formulario_consulta(clave: str, inicial: dict | None = None) -> dict | None:
    """Formulario de consulta reutilizado por Crear y Editar.

    Devuelve el registro con la predicción recalculada al enviarse, o None.
    """
    inicial = inicial or {}
    idx_distrito = 0
    if inicial.get("distrito"):
        coincide = look.index[(look["DISTRITO"] == inicial["distrito"])
                              & (look["DEPARTAMENTO"] == inicial["departamento"])]
        idx_distrito = int(coincide[0]) if len(coincide) else 0
    idx_motivo = motivos.index(inicial["motivo"]) if inicial.get("motivo") in motivos else 0
    fecha_inicial = (pd.to_datetime(inicial["fecha_corte"]).date()
                     if inicial.get("fecha_corte") else date.today())
    reporta_inicial = inicial.get("conexiones") is not None

    with st.form(f"form_{clave}"):
        col1, col2, col3 = st.columns(3)
        with col1:
            etiqueta_distrito = st.selectbox("Distrito", look["etiqueta"],
                                             index=idx_distrito)
            motivo = st.selectbox("Motivo reportado", motivos, index=idx_motivo)
        with col2:
            fecha_corte = st.date_input("Fecha del corte", value=fecha_inicial)
            hora = st.slider("Hora de inicio", 0, 23, int(inicial.get("hora") or 10))
        with col3:
            reporta = st.checkbox("La EPS reporta conexiones afectadas",
                                  value=reporta_inicial or not inicial)
            conexiones = st.number_input("Conexiones afectadas", min_value=0,
                                         value=int(inicial.get("conexiones") or 500))
            unidades = st.number_input("Unidades de uso afectadas", min_value=0,
                                       value=int(inicial.get("unidades") or 550))
        nota = st.text_input("Nota (opcional)", value=inicial.get("nota") or "")
        enviado = st.form_submit_button("Predecir y guardar", type="primary")

    if not enviado:
        return None

    fila_look = look[look["etiqueta"] == etiqueta_distrito].iloc[0]
    etiqueta, proba = comunes.predecir_corte(
        fecha=fecha_corte, hora=hora,
        departamento=fila_look["DEPARTAMENTO"], poblacion=fila_look["poblacion"],
        motivo=motivo,
        conexiones=conexiones if reporta else None,
        unidades=unidades if reporta else None,
        camiones=0)
    return {
        "distrito": fila_look["DISTRITO"], "departamento": fila_look["DEPARTAMENTO"],
        "fecha_corte": str(fecha_corte), "hora": hora, "motivo": motivo,
        "conexiones": float(conexiones) if reporta else None,
        "unidades": float(unidades) if reporta else None,
        "prediccion": etiqueta, "prob_programada": round(proba, 4), "nota": nota,
    }


tab_crear, tab_listar, tab_editar, tab_eliminar = st.tabs(
    ["Crear", "Listar", "Editar", "Eliminar"])

# ─────────────────────────────────────────────────────────────── C — Crear
with tab_crear:
    registro = formulario_consulta("crear")
    if registro:
        id_nuevo = bd.crear(registro)
        st.success(f"Consulta **{id_nuevo}** guardada: {registro['prediccion']} "
                   f"(P(PROGRAMADA) = {registro['prob_programada']:.1%}). "
                   "El timestamp `creado_en` se generó automáticamente.")

# ─────────────────────────────────────────────────────────────── R — Listar
with tab_listar:
    consultas = bd.listar()
    st.metric("Consultas guardadas", len(consultas))
    if consultas.empty:
        st.info("Aún no hay consultas. Créalas aquí o desde el Panel 2.")
    else:
        st.dataframe(consultas.set_index("id"), width="stretch")

# ─────────────────────────────────────────────────────────────── U — Editar
with tab_editar:
    consultas = bd.listar()
    if consultas.empty:
        st.info("No hay consultas para editar.")
    else:
        id_editar = st.selectbox(
            "Consulta a editar", consultas["id"],
            format_func=lambda i: (
                f"#{i} — {consultas.set_index('id').loc[i, 'distrito']} "
                f"({consultas.set_index('id').loc[i, 'prediccion']})"),
            key="sel_editar")
        actual = bd.obtener(int(id_editar))
        st.caption(f"Creada: {actual['creado_en']} · Última edición: "
                   f"{actual['actualizado_en'] or '—'}")
        registro = formulario_consulta("editar", inicial=actual)
        if registro:
            bd.actualizar(int(id_editar), registro)
            st.success(f"Consulta **{id_editar}** actualizada y predicción "
                       f"recalculada: {registro['prediccion']} "
                       f"(P = {registro['prob_programada']:.1%}). "
                       "`actualizado_en` sellado automáticamente.")

# ─────────────────────────────────────────────────────────────── D — Eliminar
with tab_eliminar:
    if st.session_state.pop("id_eliminado", None):
        st.success("Consulta eliminada.")
    consultas = bd.listar()
    if consultas.empty:
        st.info("No hay consultas para eliminar.")
    else:
        id_borrar = st.selectbox(
            "Consulta a eliminar", consultas["id"],
            format_func=lambda i: (
                f"#{i} — {consultas.set_index('id').loc[i, 'distrito']} "
                f"({consultas.set_index('id').loc[i, 'creado_en']})"),
            key="sel_eliminar")
        st.dataframe(consultas[consultas["id"] == id_borrar].set_index("id"),
                     width="stretch")
        if st.button("Eliminar definitivamente", type="primary",
                     icon=":material/delete:"):
            bd.eliminar(int(id_borrar))
            st.session_state["id_eliminado"] = int(id_borrar)
            st.rerun()
