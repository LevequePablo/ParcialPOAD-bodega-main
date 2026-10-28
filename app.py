import dash
from data_processing import df
from backend_analitico import (
    analizar_estadistica_descriptiva,
    calcular_correlaciones,
    entrenar_modelo_deep_learning
)
from view.dashboard import crear_dashboard_layout

print("1/3 Analizando estadísticas de la muestra...")
stats_ventas = analizar_estadistica_descriptiva(df, "total_venta")
corr_ventas = calcular_correlaciones(df)

print("2/3 Entrenando red neuronal (MLP Deep Learning)...")
modelo_dl = entrenar_modelo_deep_learning(df)

print("3/3 Montando servidor web Dash...")
app = dash.Dash(__name__, title="Dashboard Bodega Aura de los Andes")
server = app.server

app.layout = crear_dashboard_layout(df, stats_ventas, corr_ventas, modelo_dl)

if __name__ == "__main__":
    app.run(debug=True, port=8050)