# Parte del registro crudo de SUNASS (descargas/data/tema_a/Interrupciones_Dataset.csv)
# y produce 3 datasets construidos en data/processed/:
#
#   1. interrupciones_enriquecido.csv  — nivel registro (interrupción × distrito),
#      con features derivadas y cruce de población. Para el Panel 2 (clasificación).
#   2. matriz_distritos.csv            — agregado por distrito. Para el Panel 1 (clustering).
#   3. serie_semanal.csv               — cortes por semana (formato largo, incluye TOTAL
#      y distritos top-N). Para el Panel 3 (pronóstico).
#
# Fuente externa cruzada: población distrital INEI, vía el CSV público
# https://raw.githubusercontent.com/geodir/ubigeo-peru/master/geodir-ubigeo-inei.csv

from pathlib import Path
import unicodedata

import holidays
import pandas as pd

# %% Parámetros (modificables en clase)
RAIZ = Path(__file__).resolve().parents[1]
RUTA_CRUDO = RAIZ / "descargas" / "data" / "tema_a" / "Interrupciones_Dataset.csv"
RUTA_POBLACION = RAIZ / "descargas" / "data" / "externo" / "poblacion_distrital_inei.csv"
DIR_SALIDA = RAIZ / "data" / "processed"

MIN_REGISTROS_DISTRITO = 30  # un distrito entra al clustering solo con ≥ este nº de registros
UMBRAL_MOTIVO_RARO = 500     # motivos con menos registros que esto se agrupan en "OTROS"
TOP_N_DISTRITOS = 8          # distritos con serie semanal propia (además del TOTAL)


def normalizar_nombre(texto):
    """MAYÚSCULAS, sin tildes ni espacios sobrantes — para cruzar nombres de distrito."""
    texto = str(texto).strip().upper()
    texto = unicodedata.normalize("NFKD", texto)
    return "".join(c for c in texto if not unicodedata.combining(c))


# %% 1. Carga del registro crudo
crudo = pd.read_csv(RUTA_CRUDO, encoding="utf-8-sig")  # el archivo trae BOM UTF-8
n_inicial = len(crudo)
print(f"Registros crudos: {n_inicial}")

# %% 2. Limpieza
df = crudo.copy()

# Espacios sobrantes en todas las columnas de texto (ej. "SEDAPAL ")
for col in df.columns[df.dtypes == "str"]:
    df[col] = df[col].str.strip()

# Distrito nulo (188 casos): se conserva el registro pero identificado
df["DISTRITO"] = df["DISTRITO"].fillna("SIN DATO")

# Fechas AAAAMMDD + HH:MM → datetime, y duración del corte
df["fecha_inicio"] = pd.to_datetime(
    df["FECHAINICIO"].astype(str) + " " + df["HORAINICIO"], format="%Y%m%d %H:%M"
)
df["fecha_fin"] = pd.to_datetime(
    df["FECHAFIN"].astype(str) + " " + df["HORAFIN"], format="%Y%m%d %H:%M"
)
df["duracion_horas"] = ((df["fecha_fin"] - df["fecha_inicio"]).dt.total_seconds() / 3600).round(2)

# Duraciones negativas: dato imposible (fin antes del inicio), se elimina
negativas = (df["duracion_horas"] < 0).sum()
df = df[df["duracion_horas"] >= 0]
print(f"Eliminadas {negativas} filas con duración negativa")

# Duplicados exactos (misma fila completa): error de registro, se eliminan
exactos = df.duplicated().sum()
df = df.drop_duplicates()
print(f"Eliminados {exactos} duplicados exactos de fila completa")

# OJO: un mismo IDINTERRUPCION puede repetirse legítimamente — un corte que abarca
# varios distritos genera una fila por distrito. La unidad de análisis es
# interrupción × distrito. Solo se consolidan los duplicados de (ID, DISTRITO):
# se conserva la fila con mayor NUMCONEXDOM (el reporte más completo).
antes = len(df)
df = (
    df.sort_values("NUMCONEXDOM", ascending=False, na_position="last")
    .drop_duplicates(subset=["IDINTERRUPCION", "DISTRITO"], keep="first")
    .sort_index()
)
print(f"Consolidados {antes - len(df)} duplicados de (IDINTERRUPCION, DISTRITO)")

# El problema del proyecto es cortes de AGUA; ALCANTARILLADO es marginal (~1.5%)
alcantarillado = (df["TIPOSERVICIO"] != "AGUA POTABLE").sum()
df = df[df["TIPOSERVICIO"] == "AGUA POTABLE"]
print(f"Excluidas {alcantarillado} filas de ALCANTARILLADO (el estudio es agua potable)")

# NUMCAMIONESPUNTOS: ~95% nulo; nulo = no se desplegaron camiones cisterna → 0
df["NUMCAMIONESPUNTOS"] = df["NUMCAMIONESPUNTOS"].fillna(0).astype(int)

# %% 3. Features derivadas (esto convierte el registro crudo en dataset propio)
NOMBRES_DIA = ["LUNES", "MARTES", "MIERCOLES", "JUEVES", "VIERNES", "SABADO", "DOMINGO"]

df["anio"] = df["fecha_inicio"].dt.year
df["mes"] = df["fecha_inicio"].dt.month
df["dia_semana_num"] = df["fecha_inicio"].dt.dayofweek  # 0 = lunes
df["dia_semana"] = df["dia_semana_num"].map(dict(enumerate(NOMBRES_DIA)))
df["es_fin_de_semana"] = (df["dia_semana_num"] >= 5).astype(int)

# Feriados nacionales de Perú (librería holidays)
feriados_pe = holidays.PE(years=range(df["anio"].min(), df["anio"].max() + 1))
df["es_feriado"] = df["fecha_inicio"].dt.date.isin(feriados_pe).astype(int)

# Franja horaria del inicio del corte
df["hora_inicio_num"] = df["fecha_inicio"].dt.hour
df["hora_inicio_bin"] = pd.cut(
    df["hora_inicio_num"],
    bins=[-1, 5, 11, 17, 23],
    labels=["MADRUGADA", "MANANA", "TARDE", "NOCHE"],
).astype(str)

# Proporción de unidades de uso por conexión domiciliaria afectada.
# La mitad de los registros no reporta conexiones → queda NaN + columna indicadora
# (la decisión de imputación se toma en la fase de modelado, no aquí).
df["reporta_conexiones"] = df["NUMCONEXDOM"].notna().astype(int)
con_conexiones = df["NUMCONEXDOM"] > 0
df["ratio_afectados"] = pd.NA
df.loc[con_conexiones, "ratio_afectados"] = (
    df.loc[con_conexiones, "UNIDADESUSO"] / df.loc[con_conexiones, "NUMCONEXDOM"]
).round(3)

# Motivos raros agrupados en "OTROS" (reduce cardinalidad para el encoding del Panel 2)
frecuencia_motivo = df["MOTIVOINTERRUPCION"].value_counts()
motivos_raros = frecuencia_motivo[frecuencia_motivo < UMBRAL_MOTIVO_RARO].index
df["motivo_agrupado"] = df["MOTIVOINTERRUPCION"].where(
    ~df["MOTIVOINTERRUPCION"].isin(motivos_raros), "OTROS"
)
print(f"Motivos: {frecuencia_motivo.size} originales -> {df['motivo_agrupado'].nunique()} agrupados")

# %% 4. Cruce con fuente externa: población distrital INEI
pob = pd.read_csv(RUTA_POBLACION, encoding="utf-8-sig", thousands=",")

# Clave de cruce: DEPARTAMENTO|PROVINCIA|DISTRITO normalizados
pob["clave"] = (
    pob["Departamento"].map(normalizar_nombre)
    + "|" + pob["Provincia"].map(normalizar_nombre)
    + "|" + pob["Distrito"].map(normalizar_nombre)
)
df["clave"] = (
    df["DEPARTAMENTO"].map(normalizar_nombre)
    + "|" + df["PROVINCIA"].map(normalizar_nombre)
    + "|" + df["DISTRITO"].map(normalizar_nombre)
)

# Variantes de nombre conocidas entre SUNASS y el archivo INEI
ALIAS = {
    "CALLAO|CALLAO|CARMEN DE LA LEGUA REYNOSO": "CALLAO|CALLAO|CARMEN DE LA LEGUA",
    "TUMBES|CONTRALMIRANTE VILLAR|ZORRITOS": "TUMBES|CONTRALMIRANTE VILLA|ZORRITOS",
    "TUMBES|CONTRALMIRANTE VILLAR|CASITAS": "TUMBES|CONTRALMIRANTE VILLA|CASITAS",
    "TUMBES|CONTRALMIRANTE VILLAR|CANOAS DE PUNTA SAL": "TUMBES|CONTRALMIRANTE VILLA|CANOAS DE PUNTA SAL",
}
df["clave"] = df["clave"].replace(ALIAS)

df = df.merge(
    pob[["clave", "Poblacion"]].rename(columns={"Poblacion": "poblacion_distrito"}),
    on="clave", how="left",
)
pct_match = 100 * df["poblacion_distrito"].notna().mean()
print(f"Cruce de población: {pct_match:.1f}% de registros con población del distrito")

# %% 5. Salida 1 — nivel registro, para el Panel 2 (clasificación)
enriquecido = df[[
    "IDINTERRUPCION", "EPS", "TIPOINTERRUPCION", "MOTIVOINTERRUPCION", "motivo_agrupado",
    "DEPARTAMENTO", "PROVINCIA", "DISTRITO",
    "fecha_inicio", "fecha_fin", "duracion_horas",
    "anio", "mes", "dia_semana", "dia_semana_num", "es_fin_de_semana", "es_feriado",
    "hora_inicio_num", "hora_inicio_bin",
    "NUMCONEXDOM", "UNIDADESUSO", "NUMCAMIONESPUNTOS",
    "reporta_conexiones", "ratio_afectados", "poblacion_distrito",
]]
DIR_SALIDA.mkdir(parents=True, exist_ok=True)
enriquecido.to_csv(DIR_SALIDA / "interrupciones_enriquecido.csv", index=False)

# %% 6. Salida 2 — matriz por distrito, para el Panel 1 (clustering)
con_distrito = df[df["DISTRITO"] != "SIN DATO"]
matriz = con_distrito.groupby(["DEPARTAMENTO", "PROVINCIA", "DISTRITO"]).agg(
    n_cortes=("IDINTERRUPCION", "count"),
    n_imprevistas=("TIPOINTERRUPCION", lambda s: (s == "IMPREVISTA").sum()),
    duracion_media_h=("duracion_horas", "mean"),
    duracion_mediana_h=("duracion_horas", "median"),
    conexiones_afectadas_total=("NUMCONEXDOM", "sum"),
    unidades_uso_total=("UNIDADESUSO", "sum"),
    poblacion=("poblacion_distrito", "first"),
).reset_index()

matriz["pct_imprevistas"] = (100 * matriz["n_imprevistas"] / matriz["n_cortes"]).round(1)
matriz["cortes_por_10k_hab"] = (10_000 * matriz["n_cortes"] / matriz["poblacion"]).round(2)
matriz[["duracion_media_h", "duracion_mediana_h"]] = matriz[
    ["duracion_media_h", "duracion_mediana_h"]
].round(2)

# Distritos con pocos registros dan estadísticas inestables → fuera del clustering
total_distritos = len(matriz)
matriz = matriz[matriz["n_cortes"] >= MIN_REGISTROS_DISTRITO]
print(f"Matriz de distritos: {len(matriz)} de {total_distritos} distritos "
      f"(>= {MIN_REGISTROS_DISTRITO} registros)")
matriz.to_csv(DIR_SALIDA / "matriz_distritos.csv", index=False)

# %% 7. Salida 3 — serie semanal (formato largo), para el Panel 3 (pronóstico)
df["semana"] = df["fecha_inicio"].dt.to_period("W").dt.start_time  # semana que inicia lunes
con_distrito = df[df["DISTRITO"] != "SIN DATO"]  # re-tomar el slice, ya con la columna semana

# TOTAL: interrupciones únicas por semana (un corte multi-distrito cuenta una vez)
serie_total = (
    df.groupby("semana")["IDINTERRUPCION"].nunique().reset_index(name="n_cortes")
)
serie_total["distrito"] = "TOTAL"

# Top-N distritos con más registros: una serie por distrito (filas = afectaciones)
top_distritos = con_distrito["DISTRITO"].value_counts().head(TOP_N_DISTRITOS).index
serie_top = (
    con_distrito[con_distrito["DISTRITO"].isin(top_distritos)]
    .groupby(["semana", "DISTRITO"]).size().reset_index(name="n_cortes")
    .rename(columns={"DISTRITO": "distrito"})
)

serie = pd.concat([serie_total, serie_top], ignore_index=True)

# Rellenar semanas sin cortes con 0 para que la serie sea continua (clave para ARIMA)
todas_semanas = pd.date_range(df["semana"].min(), df["semana"].max(), freq="W-MON")
completas = []
for nombre, grupo in serie.groupby("distrito"):
    conteos = grupo.set_index("semana")["n_cortes"].reindex(todas_semanas, fill_value=0)
    completas.append(
        pd.DataFrame({"semana": todas_semanas, "distrito": nombre, "n_cortes": conteos.values})
    )
serie = pd.concat(completas, ignore_index=True)
serie.to_csv(DIR_SALIDA / "serie_semanal.csv", index=False)

# %% 8. Resumen final
print("\n=== RESUMEN ===")
print(f"interrupciones_enriquecido.csv: {len(enriquecido)} filas, {enriquecido.shape[1]} columnas")
print(f"matriz_distritos.csv:           {len(matriz)} distritos")
print(f"serie_semanal.csv:              {len(serie)} filas "
      f"({serie['semana'].nunique()} semanas × {serie['distrito'].nunique()} series)")
print(f"Target en enriquecido: \n{enriquecido['TIPOINTERRUPCION'].value_counts(normalize=True).round(3).to_string()}")
