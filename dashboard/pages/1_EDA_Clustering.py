"""Panel 1 — EDA y clustering de distritos (K-means con codo + silueta en vivo)."""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

import comunes

# %% Parámetros (los mismos del notebook 02 — modificables en clase)
RANDOM_STATE = 42
K_DEFECTO = 4           # k elegido con codo + silueta en el notebook 02
K_MIN, K_MAX = 2, 8
FACTOR_IQR = 1.5
UMBRAL_VISUAL_H = 72    # recorte visual del histograma de duración

COLS_CLUSTER = ["cortes_por_10k_hab", "duracion_mediana_h", "pct_imprevistas"]
COLS_LOG = ["cortes_por_10k_hab", "duracion_mediana_h"]  # colas largas → log1p

st.set_page_config(page_title="EDA y Clustering",
                   page_icon=":material/monitoring:", layout="wide")
st.title("Panel 1 — EDA y clustering de distritos")

df = comunes.cargar_enriquecido()
matriz = comunes.cargar_csv("matriz_distritos_clusters.csv")

# ══════════════════════════════════════════════════════════ 1. EDA
st.header("1. Análisis exploratorio")

pct_imprevistas = (df["TIPOINTERRUPCION"] == "IMPREVISTA").mean() * 100
c1, c2, c3, c4 = st.columns(4)
c1.metric("Registros (interrupción × distrito)", f"{len(df):,}")
c2.metric("% IMPREVISTA / PROGRAMADA",
          f"{pct_imprevistas:.1f} / {100 - pct_imprevistas:.1f}")
c3.metric("Duración mediana", f"{df['duracion_horas'].median():.1f} h")
c4.metric("Periodo", "jun-2019 → may-2024")

col_izq, col_der = st.columns(2)

with col_izq:
    # Histograma de duración (recortado para poder verlo: hay outliers de meses)
    pct_sobre = (df["duracion_horas"] > UMBRAL_VISUAL_H).mean() * 100
    fig = px.histogram(
        df[df["duracion_horas"] <= UMBRAL_VISUAL_H], x="duracion_horas", nbins=48,
        title=f"Duración de cortes (≤ {UMBRAL_VISUAL_H}h; el {pct_sobre:.1f}% excede el corte visual)",
        color_discrete_sequence=["#457b9d"])
    fig.update_layout(xaxis_title="duración (horas)", yaxis_title="registros")
    st.plotly_chart(fig, width="stretch")

with col_der:
    # Boxplot en escala log: la asimetría extrema es LA complicación del dataset
    fig = px.box(df[df["duracion_horas"] > 0], x="TIPOINTERRUPCION", y="duracion_horas",
                 color="TIPOINTERRUPCION", color_discrete_map=comunes.COLORES_TARGET,
                 log_y=True, title="Duración por tipo de corte (escala log)")
    fig.update_layout(showlegend=False, yaxis_title="duración (horas, log)")
    st.plotly_chart(fig, width="stretch")

# Outliers 1.5·IQR (obligatorio del panel)
q1, q3 = df["duracion_horas"].quantile([0.25, 0.75])
lim_sup = q3 + FACTOR_IQR * (q3 - q1)
n_outliers = int((df["duracion_horas"] > lim_sup).sum())
st.info(
    f"**Outliers 1.5·IQR en duración**: Q1={q1:.1f}h, Q3={q3:.1f}h → límite superior "
    f"**{lim_sup:.1f}h**; lo exceden **{n_outliers:,} registros "
    f"({n_outliers / len(df) * 100:.1f}%)**, con un máximo de "
    f"{df['duracion_horas'].max():,.0f}h (≈ {df['duracion_horas'].max() / 24:.0f} días). "
    "No se eliminan (son cortes reales): se usan estadísticas robustas (mediana) "
    "en la matriz de distritos.")

col_izq, col_der = st.columns(2)

with col_izq:
    # ¿Cuándo ocurren? — franja horaria vs target (señal predictiva clave)
    orden_bin = ["MADRUGADA", "MANANA", "TARDE", "NOCHE"]
    tabla_hora = (pd.crosstab(df["hora_inicio_bin"], df["TIPOINTERRUPCION"],
                              normalize="index").loc[orden_bin] * 100).round(1)
    fig = px.bar(tabla_hora, barmode="stack",
                 color_discrete_map=comunes.COLORES_TARGET,
                 title="Tipo de corte por franja de inicio (%)")
    fig.update_layout(xaxis_title="", yaxis_title="%")
    st.plotly_chart(fig, width="stretch")

with col_der:
    # Mapa de correlación de las numéricas
    cols_corr = ["duracion_horas", "mes", "dia_semana_num", "es_fin_de_semana",
                 "es_feriado", "hora_inicio_num", "NUMCONEXDOM", "UNIDADESUSO",
                 "NUMCAMIONESPUNTOS", "ratio_afectados", "poblacion_distrito"]
    fig = px.imshow(df[cols_corr].corr().round(2), text_auto=True, zmin=-1, zmax=1,
                    color_continuous_scale="RdBu_r",
                    title="Mapa de correlación (variables numéricas)")
    fig.update_layout(height=500)
    st.plotly_chart(fig, width="stretch")

st.markdown(
    "De **noche el 95%** de los cortes es imprevisto y de madrugada el 87.9% "
    "(nadie programa un corte a esas horas); en fin de semana los imprevistos "
    "suben a 85.9% vs 73.9% de lunes a viernes. Estas variables alimentan el "
    "clasificador del Panel 2.")

with st.expander("Top 15 distritos por registros y evolución mensual"):
    col_a, col_b = st.columns(2)
    top15 = (df[df["DISTRITO"] != "SIN DATO"]["DISTRITO"]
             .value_counts().head(15).sort_values())
    fig = px.bar(top15, orientation="h", title="Top 15 distritos por registros",
                 color_discrete_sequence=["#457b9d"])
    fig.update_layout(showlegend=False, xaxis_title="registros", yaxis_title="")
    col_a.plotly_chart(fig, width="stretch")

    eventos_mes = (df.groupby(df["fecha_inicio"].dt.to_period("M"))["IDINTERRUPCION"]
                   .nunique())
    eventos_mes.index = eventos_mes.index.to_timestamp()
    fig = px.line(eventos_mes, title="Interrupciones únicas por mes",
                  color_discrete_sequence=["#264653"])
    fig.update_layout(showlegend=False, xaxis_title="", yaxis_title="eventos")
    col_b.plotly_chart(fig, width="stretch")

# ══════════════════════════════════════════════════════════ 2. Clustering
st.header("2. Clustering de distritos (K-means)")

st.markdown(f"""
Se agrupan los **{len(matriz)} distritos con ≥30 registros** por su **perfil de
calidad de servicio** — no por tamaño: `n_cortes` y `poblacion` quedan fuera a
propósito. Features: `{"`, `".join(COLS_CLUSTER)}` (las dos primeras con `log1p`
por colas largas), estandarizadas antes del K-means.
""")


@st.cache_data
def matriz_escalada() -> np.ndarray:
    """Mismo pipeline del notebook 02: log1p en las sesgadas + StandardScaler."""
    X = matriz.dropna(subset=COLS_CLUSTER)[COLS_CLUSTER].copy()
    for col in COLS_LOG:
        X[col] = np.log1p(X[col])
    return StandardScaler().fit_transform(X)


@st.cache_data
def curva_codo_silueta() -> pd.DataFrame:
    """Inercia y silueta para k=2..8 (obligatorio: codo Y silueta)."""
    X_esc = matriz_escalada()
    filas = []
    for k in range(K_MIN, K_MAX + 1):
        km = KMeans(n_clusters=k, n_init=10, random_state=RANDOM_STATE)
        etiquetas = km.fit_predict(X_esc)
        filas.append({"k": k, "inercia": km.inertia_,
                      "silueta": silhouette_score(X_esc, etiquetas)})
    return pd.DataFrame(filas)


k_elegido = st.slider("**k (número de clusters)** — cámbialo y todo se recalcula",
                      K_MIN, K_MAX, K_DEFECTO)

# Codo y silueta con el k elegido marcado
curva = curva_codo_silueta()
col_codo, col_sil = st.columns(2)
fig = px.line(curva, x="k", y="inercia", markers=True,
              title="Método del codo (inercia)",
              color_discrete_sequence=["#264653"])
fig.add_vline(x=k_elegido, line_dash="dash", line_color="#e76f51")
col_codo.plotly_chart(fig, width="stretch")

fig = px.line(curva, x="k", y="silueta", markers=True,
              title="Coeficiente de silueta", color_discrete_sequence=["#e76f51"])
fig.add_vline(x=k_elegido, line_dash="dash", line_color="#264653")
col_sil.plotly_chart(fig, width="stretch")

st.markdown(
    "**Elección de k=4** (notebook 02): el codo de la inercia está en 4 y la "
    "silueta es plana en todo el rango (0.27–0.34; máx 0.337 en k=8, inaccionable, "
    "y 0.324 en k=2, demasiado grueso). k=4 combina evidencia numérica con 4 "
    "perfiles interpretables.")

# K-means con el k del slider
X_esc = matriz_escalada()
matriz_k = matriz.dropna(subset=COLS_CLUSTER).copy().reset_index(drop=True)
km = KMeans(n_clusters=k_elegido, n_init=10, random_state=RANDOM_STATE)
matriz_k["cluster"] = km.fit_predict(X_esc)
sil = silhouette_score(X_esc, matriz_k["cluster"])

c1, c2 = st.columns(2)
c1.metric(f"Silueta con k={k_elegido}", f"{sil:.3f}")
c2.metric("Distritos agrupados", f"{len(matriz_k)}")

# PCA 2D para visualizar (los clusters viven en 3 dimensiones)
pca = PCA(n_components=2, random_state=RANDOM_STATE)
coords = pca.fit_transform(X_esc)
var_exp = pca.explained_variance_ratio_ * 100
matriz_k["PC1"], matriz_k["PC2"] = coords[:, 0], coords[:, 1]
matriz_k["cluster_str"] = matriz_k["cluster"].astype(str)
if k_elegido == K_DEFECTO:
    matriz_k["cluster_str"] = matriz_k["cluster"].map(
        lambda c: f"{c} — {comunes.NOMBRES_CLUSTER[c].split(' (')[0]}")

fig = px.scatter(
    matriz_k.sort_values("cluster"), x="PC1", y="PC2", color="cluster_str",
    hover_data={"DISTRITO": True, "DEPARTAMENTO": True, "PC1": False, "PC2": False,
                **{c: ":.1f" for c in COLS_CLUSTER}},
    color_discrete_sequence=px.colors.qualitative.Set2,
    title=f"Clusters de distritos (K-means k={k_elegido}, proyección PCA 2D)")
fig.update_traces(marker={"size": 9, "opacity": 0.85})
fig.update_layout(xaxis_title=f"PC1 ({var_exp[0]:.0f}% var.)",
                  yaxis_title=f"PC2 ({var_exp[1]:.0f}% var.)",
                  legend_title="cluster", height=520)
st.plotly_chart(fig, width="stretch")

# Perfil de cada cluster en unidades originales (interpretación)
st.subheader("Perfil de los clusters")
perfil = matriz_k.groupby("cluster").agg(
    n_distritos=("DISTRITO", "count"),
    cortes_por_10k_hab=("cortes_por_10k_hab", "mean"),
    duracion_mediana_h=("duracion_mediana_h", "mean"),
    pct_imprevistas=("pct_imprevistas", "mean"),
).round(1)
if k_elegido == K_DEFECTO:
    perfil.insert(0, "nombre", perfil.index.map(comunes.NOMBRES_CLUSTER))
st.dataframe(perfil, width="stretch")

if k_elegido == K_DEFECTO:
    st.markdown("""
| Cluster | Nombre | Lectura para SUNASS/EPS |
| --- | --- | --- |
| 0 | **Sed crónica** (57 distritos) | Cortes frecuentes, breves y reactivos (SJL, VMT, VES) → inversión en redes |
| 1 | **Crítico** (24) | Cortes **largos** e imprevistos (Tumbes, Piura, Paita) → capacidad de respuesta |
| 2 | **Moderado** (53) | Problema contenido (Ate, Ventanilla, Comas) → monitoreo |
| 3 | **Planificado** (17) | Predominan mantenimientos programados (Trujillo, Jaén) → gestión referente |
""")

with st.expander("Explorar distritos por cluster"):
    cluster_sel = st.selectbox("Cluster", sorted(matriz_k["cluster"].unique()))
    st.dataframe(
        matriz_k[matriz_k["cluster"] == cluster_sel]
        [["DEPARTAMENTO", "PROVINCIA", "DISTRITO", "n_cortes"] + COLS_CLUSTER]
        .sort_values("cortes_por_10k_hab", ascending=False),
        width="stretch", hide_index=True)

with st.expander("DBSCAN (opcional): distritos atípicos"):
    atipicos = matriz[matriz["cluster_dbscan"] == -1]
    st.markdown(
        f"DBSCAN (eps=0.9, min_samples=5, notebook 02) marca **{len(atipicos)} "
        "distritos como ruido**: perfiles tan extremos que K-means los forzaría "
        "dentro de un cluster (ej. Las Piedras con 167 cortes/10k hab).")
    st.dataframe(
        atipicos[["DEPARTAMENTO", "DISTRITO"] + COLS_CLUSTER]
        .sort_values("cortes_por_10k_hab", ascending=False),
        width="stretch", hide_index=True)
