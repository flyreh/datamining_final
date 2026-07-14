"""CRUD del Panel 4 sobre SQLite local (dashboard/consultas.db)."""
import sqlite3
from contextlib import closing
from datetime import datetime
from pathlib import Path

import pandas as pd

RUTA_BD = Path(__file__).resolve().parent / "consultas.db"

COLUMNAS = ["id", "creado_en", "actualizado_en", "distrito", "departamento",
            "fecha_corte", "hora", "motivo", "conexiones", "unidades",
            "prediccion", "prob_programada", "nota"]


def _conexion() -> sqlite3.Connection:
    con = sqlite3.connect(RUTA_BD)
    con.execute("""
        CREATE TABLE IF NOT EXISTS consultas (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            creado_en       TEXT NOT NULL,
            actualizado_en  TEXT,
            distrito        TEXT,
            departamento    TEXT,
            fecha_corte     TEXT,
            hora            INTEGER,
            motivo          TEXT,
            conexiones      REAL,
            unidades        REAL,
            prediccion      TEXT,
            prob_programada REAL,
            nota            TEXT
        )
    """)
    return con


def _ahora() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


# OJO: "with sqlite3.connect(...)" solo maneja la transacción (commit/rollback),
# NO cierra la conexión — por eso cada operación usa closing() además del with.

def crear(registro: dict) -> int:
    """Inserta una consulta; el timestamp creado_en se genera automáticamente."""
    campos = ["distrito", "departamento", "fecha_corte", "hora", "motivo",
              "conexiones", "unidades", "prediccion", "prob_programada", "nota"]
    with closing(_conexion()) as con, con:
        cursor = con.execute(
            f"INSERT INTO consultas (creado_en, {', '.join(campos)}) "
            f"VALUES (?, {', '.join('?' * len(campos))})",
            [_ahora()] + [registro.get(c) for c in campos],
        )
        return cursor.lastrowid


def listar() -> pd.DataFrame:
    with closing(_conexion()) as con:
        return pd.read_sql_query(
            "SELECT * FROM consultas ORDER BY id DESC", con)


def obtener(id_consulta: int) -> dict | None:
    with closing(_conexion()) as con:
        fila = con.execute(
            "SELECT * FROM consultas WHERE id = ?", (id_consulta,)).fetchone()
    return dict(zip(COLUMNAS, fila)) if fila else None


def actualizar(id_consulta: int, registro: dict) -> None:
    """Actualiza los campos editables y sella actualizado_en automáticamente."""
    campos = ["distrito", "departamento", "fecha_corte", "hora", "motivo",
              "conexiones", "unidades", "prediccion", "prob_programada", "nota"]
    asignaciones = ", ".join(f"{c} = ?" for c in campos)
    with closing(_conexion()) as con, con:
        con.execute(
            f"UPDATE consultas SET {asignaciones}, actualizado_en = ? WHERE id = ?",
            [registro.get(c) for c in campos] + [_ahora(), id_consulta],
        )


def eliminar(id_consulta: int) -> None:
    with closing(_conexion()) as con, con:
        con.execute("DELETE FROM consultas WHERE id = ?", (id_consulta,))
