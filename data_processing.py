from pathlib import Path
import pandas as pd

DIR = Path(__file__).resolve().parent

df_csv = pd.read_csv(DIR / "dataframe" / "ventas_bodega_csv_200.csv", delimiter=";")
df_json = pd.read_json(DIR / "dataframe" / "ventas_bodega_json_200.json")

df = pd.concat([df_csv, df_json])
df = df.reset_index(drop=True)
df["ID"] = df.index
df = df.drop(columns=["id_venta", "origen_archivo", "ciudad"])

def limpiar_porcentaje(valor):
    if pd.isna(valor):
        return valor
    if isinstance(valor, str) and "%" in valor:
        return float(valor.replace("%", "").strip()) / 100
    return float(valor)

df["descuento"] = df["descuento"].apply(limpiar_porcentaje)
df["total_venta"] = df["total_venta"].astype(str).str.replace("$", "", regex=False).astype(float)
df["fecha_venta"] = pd.to_datetime(df["fecha_venta"], dayfirst=True, errors="coerce")
df["fecha_venta"] = df["fecha_venta"].fillna(df["fecha_venta"].mode()[0])
df["costo_envio"] = df["costo_envio"].fillna(df["costo_envio"].median())
df["provincia"] = df["provincia"].replace({"Bs As": "Buenos Aires", "Mza": "Mendoza"})
df["producto"] = df["producto"].str.strip().str.title()

df["mes_num"] = df["fecha_venta"].dt.month
MESES = {
    1: "Enero", 2: "Febrero", 3: "Marzo", 4: "Abril", 5: "Mayo", 6: "Junio",
    7: "Julio", 8: "Agosto", 9: "Septiembre", 10: "Octubre", 11: "Noviembre", 12: "Diciembre",
}
df["mes"] = df["mes_num"].map(MESES)
