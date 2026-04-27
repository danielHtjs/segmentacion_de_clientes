"""
modules/segmentacion/kmeans.py
==============================
Módulo reutilizable de segmentación K-Means.
Importado por el notebook 02_kmeans.ipynb y por la app Flask.

Uso:
    from modules.segmentacion.kmeans import preparar_features, entrenar, perfiles
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA


# ── 1. PREPARAR FEATURES ─────────────────────────────────────────────────────
def preparar_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Construye la matriz de features para clustering a partir del dataset limpio.

    Nota: cada customer_id aparece una sola vez en el dataset — no hay historial
    multi-visita real. Se usan variables transaccionales como proxy del perfil
    de cliente (decisión documentada en el informe).

    Features usadas:
        age             → perfil demográfico
        total_spend     → poder adquisitivo por transacción
        quantity        → volumen de compra
        category_code   → categoría de producto (codificada numéricamente)
        gender_code     → género (0 = Female, 1 = Male)

    Retorna:
        feat_raw   : DataFrame con customer_id + features sin escalar (para análisis)
        feat_scaled: DataFrame con features escaladas (para el modelo)
    """
    feat = df[["customer_id", "age", "total_spend", "quantity",
               "category", "gender"]].copy()

    feat["category_code"] = feat["category"].astype("category").cat.codes
    feat["gender_code"]   = (feat["gender"] == "Male").astype(int)
    feat = feat.drop(columns=["category", "gender"])

    feature_cols = ["age", "total_spend", "quantity", "category_code", "gender_code"]
    feat_raw = feat.copy()

    scaler     = StandardScaler()
    X_scaled   = scaler.fit_transform(feat[feature_cols].values)
    feat_scaled = pd.DataFrame(X_scaled, columns=feature_cols)

    return feat_raw, feat_scaled


# ── 2. MÉTODO DEL CODO ───────────────────────────────────────────────────────
def calcular_elbow(feat_scaled: pd.DataFrame, k_max: int = 10) -> pd.DataFrame:
    """
    Calcula inercia y Silhouette Score para k = 2 ... k_max.
    Úsalo para elegir el k óptimo antes de entrenar el modelo final.

    Retorna un DataFrame con columnas: k, inertia, silhouette
    """
    X = feat_scaled.values
    resultados = []
    for k in range(2, k_max + 1):
        km  = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(X)
        sil = silhouette_score(X, km.labels_)
        resultados.append({
            "k":          k,
            "inertia":    round(km.inertia_, 2),
            "silhouette": round(float(sil), 4),
        })
        print(f"  k={k}  inercia={km.inertia_:,.0f}  silhouette={sil:.4f}")

    return pd.DataFrame(resultados)


# ── 3. ENTRENAR MODELO FINAL ─────────────────────────────────────────────────
def entrenar(feat_raw: pd.DataFrame,
             feat_scaled: pd.DataFrame,
             k: int) -> tuple[pd.DataFrame, float, float]:
    """
    Entrena K-Means con el k elegido y agrega PCA 2D para visualización.

    Retorna:
        resultado   : feat_raw + columna 'cluster' + 'pca_x' + 'pca_y'
        silhouette  : Silhouette Score del modelo (objetivo ≥ 0.40)
        inertia     : Inercia final (WCSS)
    """
    X = feat_scaled.values

    modelo = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = modelo.fit_predict(X)

    sil     = float(silhouette_score(X, labels))
    inertia = float(modelo.inertia_)

    # PCA 2D para scatter plot
    pca    = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(X)

    resultado = feat_raw.copy().reset_index(drop=True)
    resultado["cluster"] = labels
    resultado["pca_x"]   = coords[:, 0]
    resultado["pca_y"]   = coords[:, 1]

    return resultado, sil, inertia


# ── 4. PERFILES POR SEGMENTO ─────────────────────────────────────────────────
def perfiles(df_original: pd.DataFrame, resultado: pd.DataFrame) -> pd.DataFrame:
    """
    Genera una tabla resumen con el perfil cualitativo de cada cluster.

    Parámetros:
        df_original : dataset limpio completo (con category, gender, payment_method)
        resultado   : salida de entrenar() — tiene columna 'cluster'

    Retorna DataFrame con una fila por cluster y columnas descriptivas.
    """
    # Unir cluster con datos originales
    df_c = df_original.reset_index(drop=True).copy()
    df_c["cluster"] = resultado["cluster"].values

    resumen = []
    for cl in sorted(df_c["cluster"].unique()):
        sub = df_c[df_c["cluster"] == cl]
        resumen.append({
            "cluster":        int(cl),
            "segmento":       f"Segmento {cl + 1}",
            "n":              len(sub),
            "pct_total":      round(len(sub) / len(df_c) * 100, 1),
            "edad_prom":      round(sub["age"].mean(), 1),
            "gasto_prom":     round(sub["total_spend"].mean(), 0),
            "cantidad_prom":  round(sub["quantity"].mean(), 2),
            "categoria_top":  sub["category"].value_counts().index[0],
            "pct_femenino":   round((sub["gender"] == "Female").mean() * 100, 1),
            "pago_top":       sub["payment_method"].value_counts().index[0],
        })

    return pd.DataFrame(resumen).set_index("cluster")


# ── 5. GUARDAR RESULTADOS ────────────────────────────────────────────────────
def guardar_resultados(df_original: pd.DataFrame,
                       resultado: pd.DataFrame,
                       output_path: str | Path) -> None:
    """
    Exporta el dataset original enriquecido con la etiqueta de cluster.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cols_export = ["invoice_no", "customer_id", "age", "gender", "category",
                   "quantity", "price", "total_spend", "payment_method",
                   "invoice_date", "shopping_mall", "cluster"]

    df_out = df_original.reset_index(drop=True).copy()
    df_out["cluster"] = resultado["cluster"].values
    df_out["cluster"] = df_out["cluster"].apply(lambda x: f"Segmento {x + 1}")

    # Solo exportar columnas que existan
    cols_export = [c for c in cols_export if c in df_out.columns]
    df_out[cols_export].to_csv(output_path, index=False)
    print(f"✅ Resultados guardados en: {output_path}")
