"""
flask/app.py
============
Interfaz web del proyecto.
Ejecutar desde la raíz del proyecto:
    python flask/app.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))  # agrega raíz al path

from flask import Flask, render_template, request, jsonify
import pandas as pd
from modules.segmentacion.kmeans import (
    preparar_features,
    calcular_elbow,
    entrenar,
    perfiles,
)

app = Flask(__name__)

BASE_DIR  = Path(__file__).parent.parent
DATA_PATH = BASE_DIR / "data" / "processed" / "datos_limpios.csv"

MALLS = [
    "Todos", "Mall of Istanbul", "Kanyon", "Metrocity",
    "Metropol AVM", "Istinye Park", "Zorlu Center",
    "Cevahir AVM", "Forum Istanbul", "Viaport Outlet", "Emaar Square Mall",
]


def cargar(mall: str) -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    # Compatibilidad con columnas en español de versiones anteriores
    df = df.rename(columns={
        "año": "year", "mes": "month",
        "dia_semana": "day_of_week", "total_compra": "total_spend"
    })
    df["invoice_date"] = pd.to_datetime(df["invoice_date"])
    if mall and mall != "Todos":
        df = df[df["shopping_mall"] == mall].reset_index(drop=True)
    return df


# ── Rutas ─────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html", malls=MALLS)


@app.route("/api/segmentar", methods=["POST"])
def segmentar():
    body = request.get_json()
    mall = body.get("mall", "Todos")
    k    = int(body.get("k", 4))

    try:
        df = cargar(mall)

        # Muestra máx 6000 puntos para el scatter (rendimiento del navegador)
        df_plot = df.sample(min(6000, len(df)), random_state=42).reset_index(drop=True)

        feat_raw, feat_scaled = preparar_features(df_plot)
        resultado, sil, inertia = entrenar(feat_raw, feat_scaled, k=k)
        tabla = perfiles(df_plot, resultado)
        elbow = calcular_elbow(feat_scaled, k_max=10)

        # Scatter PCA
        scatter = [
            {
                "x":       round(float(row.pca_x), 4),
                "y":       round(float(row.pca_y), 4),
                "cluster": int(row.cluster),
                "age":     int(df_plot.loc[i, "age"]),
                "spend":   round(float(df_plot.loc[i, "total_spend"]), 2),
                "cat":     str(df_plot.loc[i, "category"]),
                "gender":  str(df_plot.loc[i, "gender"]),
            }
            for i, row in resultado.iterrows()
        ]

        # Perfiles
        perfiles_list = [
            {
                "cluster":     int(idx),
                "n":           int(row["n"]),
                "pct":         float(row["pct_total"]),
                "edad_prom":   float(row["edad_prom"]),
                "gasto_prom":  float(row["gasto_prom"]),
                "cat_top":     str(row["categoria_top"]),
                "pct_fem":     float(row["pct_femenino"]),
                "pago_top":    str(row["pago_top"]),
            }
            for idx, row in tabla.iterrows()
        ]

        return jsonify({
            "ok":        True,
            "n_total":   len(df),
            "silhouette": round(sil, 4),
            "inertia":   round(inertia, 2),
            "k":         k,
            "scatter":   scatter,
            "perfiles":  perfiles_list,
            "elbow":     elbow.to_dict(orient="records"),
        })

    except Exception as e:
        import traceback
        return jsonify({"ok": False, "error": str(e), "trace": traceback.format_exc()}), 500


if __name__ == "__main__":
    app.run(debug=True, port=5000)
