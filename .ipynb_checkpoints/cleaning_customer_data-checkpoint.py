"""
Limpieza de datos — customer_shopping_data.csv
==============================================
Ejecutar desde la raíz del proyecto:
    python cleaning_customer_data.py

Diagnóstico previo encontró:
  ✓ Sin valores nulos
  ✓ Sin filas duplicadas
  ✗ invoice_date con formato mixto (D/M/YYYY y DD/MM/YYYY)
  ✗ invoice_date es string → debe ser datetime
  ✗ Variables categóricas como string → deben ser category
  ✗ Outliers en price (~5.1% de filas — se documentan, no se eliminan)
  ✗ Columna total_spend ausente (price × quantity)
  ✗ Columnas derivadas en español mezcladas con inglés → unificadas a inglés
"""

import pandas as pd
import numpy as np
from pathlib import Path

# ── Rutas relativas a la raíz del proyecto ───────────────────────────────────
BASE_DIR    = Path(__file__).parent
INPUT_PATH  = BASE_DIR / "data" / "raw" / "customer_shopping_data.csv"
OUTPUT_PATH = BASE_DIR / "data" / "processed" / "datos_limpios.csv"


# ── 1. CARGA ─────────────────────────────────────────────────────────────────
df = pd.read_csv(INPUT_PATH)
print(f"Filas cargadas: {len(df):,}")


# ── 2. NORMALIZAR FECHAS ─────────────────────────────────────────────────────
# El campo mezcla D/M/YYYY y DD/MM/YYYY.
# dayfirst=True resuelve la ambigüedad: el primer componente siempre es el día.
df["invoice_date"] = pd.to_datetime(df["invoice_date"], dayfirst=True)

# Extraer componentes útiles para los módulos de temporada y eventos
df["year"]        = df["invoice_date"].dt.year
df["month"]       = df["invoice_date"].dt.month
df["day_of_week"] = df["invoice_date"].dt.day_name()

print(f"Rango de fechas: {df['invoice_date'].min().date()} → {df['invoice_date'].max().date()}")


# ── 3. CORREGIR TIPOS DE DATOS ───────────────────────────────────────────────
# Variables categóricas: reducen memoria y aceleran groupby
categorical_cols = ["gender", "category", "payment_method", "shopping_mall"]
for col in categorical_cols:
    df[col] = df[col].astype("category")

print("Tipos asignados:")
print(df.dtypes.to_string())


# ── 4. COLUMNA DERIVADA: GASTO TOTAL ─────────────────────────────────────────
# price representa precio unitario → total_spend = price × quantity
df["total_spend"] = (df["price"] * df["quantity"]).round(2)


# ── 5. DETECCIÓN DE OUTLIERS EN PRICE ────────────────────────────────────────
# Método IQR. NO se eliminan: pueden ser compras legítimas de tecnología o lujo.
# Se marcan con bandera para que cada módulo decida si filtrarlos.
Q1  = df["price"].quantile(0.25)
Q3  = df["price"].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

df["is_price_outlier"] = (
    (df["price"] < lower_bound) | (df["price"] > upper_bound)
)

n_outliers = df["is_price_outlier"].sum()
print(f"\nOutliers en price: {n_outliers:,} ({100 * n_outliers / len(df):.1f}%)")
print(f"  Rango normal: ${lower_bound:.2f} – ${upper_bound:.2f}")
print(f"  Rango real:   ${df['price'].min():.2f} – ${df['price'].max():.2f}")


# ── 6. REORDENAR COLUMNAS ────────────────────────────────────────────────────
column_order = [
    "invoice_no", "customer_id",
    "gender", "age",
    "category", "quantity", "price", "total_spend", "is_price_outlier",
    "payment_method",
    "invoice_date", "year", "month", "day_of_week",
    "shopping_mall",
]
df = df[column_order]


# ── 7. REPORTE FINAL ─────────────────────────────────────────────────────────
print("\n" + "="*55)
print("REPORTE DE LIMPIEZA")
print("="*55)
print(f"  Filas finales:        {len(df):,}")
print(f"  Columnas finales:     {len(df.columns)}")
print(f"  Nulos restantes:      {df.isnull().sum().sum()}")
print(f"  Duplicados restantes: {df.duplicated().sum()}")
print(f"  Memoria usada:        {df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
print("="*55)
print("\nEstadísticas de columnas numéricas:")
print(df[["age", "quantity", "price", "total_spend"]].describe().round(2))


# ── 8. GUARDAR ───────────────────────────────────────────────────────────────
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(OUTPUT_PATH, index=False)
print(f"\nArchivo guardado: {OUTPUT_PATH}")