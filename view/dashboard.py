from dash import html, dcc
from view.plotly_utils import (
    grafico_evolucion_mensual,
    grafico_ventas_canal,
    grafico_top_productos,
    grafico_ventas_provincia,
    grafico_heatmap_provincia_mes,
    grafico_ranking_vendedores,
    grafico_curva_densidad,
    grafico_curva_perdida_dl
)

def crear_tarjeta_kpi(valor, titulo, bg_color):
    return html.Div([
        html.H3(valor, style={"margin": "0", "fontSize": "26px", "fontWeight": "bold", "color": "white"}),
        html.P(titulo, style={"margin": "5px 0 0 0", "fontSize": "14px", "color": "#f0f0f0"})
    ], style={
        "backgroundColor": bg_color,
        "borderRadius": "8px",
        "padding": "20px",
        "textAlign": "center",
        "flex": "1",
        "margin": "0 10px",
        "boxShadow": "0 2px 6px rgba(0,0,0,0.15)"
    })


def crear_dashboard_layout(df, stats_ventas, corr_ventas, modelo_dl):
    total_ventas = df["total_venta"].sum()
    botellas_totales = df["cantidad_botellas"].sum()
    ticket_promedio = df["total_venta"].mean()

    # Gráficos Parcial 1
    fig_lineas = grafico_evolucion_mensual(df)
    fig_torta_canal = grafico_ventas_canal(df)
    fig_barras_prod = grafico_top_productos(df)
    fig_torta_prov = grafico_ventas_provincia(df)
    fig_heatmap = grafico_heatmap_provincia_mes(df)
    fig_vendedores = grafico_ranking_vendedores(df)

    # Gráficos Parcial 2
    fig_densidad = grafico_curva_densidad(df)
    fig_loss = grafico_curva_perdida_dl(modelo_dl["curva_perdida"])

    normalidad_texto = "Sigue distribución normal" if stats_ventas["es_normal"] else "No sigue distribución normal (p < 0.05)"

    return html.Div([
        html.H1("Dashboard de Ventas - Bodega Aura de los Andes", 
                style={"textAlign": "center", "color": "#6D071A", "marginBottom": "25px"}),

        # KPIs Principales
        html.Div([
            crear_tarjeta_kpi(f"${total_ventas:,.0f}", "Ventas totales", "#6D071A"),
            crear_tarjeta_kpi(f"{botellas_totales:,.0f}", "Botellas vendidas", "#8B1E3F"),
            crear_tarjeta_kpi(f"${ticket_promedio:,.0f}", "Ticket promedio", "#C9A227")
        ], style={"display": "flex", "justifyContent": "space-between", "marginBottom": "30px"}),

        # Fila 1
        html.Div([
            html.Div([dcc.Graph(figure=fig_lineas)], style={"width": "49%"}),
            html.Div([dcc.Graph(figure=fig_torta_canal)], style={"width": "49%"})
        ], style={"display": "flex", "justifyContent": "space-between", "marginBottom": "20px"}),

        # Fila 2
        html.Div([
            html.Div([dcc.Graph(figure=fig_barras_prod)], style={"width": "49%"}),
            html.Div([dcc.Graph(figure=fig_torta_prov)], style={"width": "49%"})
        ], style={"display": "flex", "justifyContent": "space-between", "marginBottom": "20px"}),

        # Heatmap y Ranking
        html.Div([dcc.Graph(figure=fig_heatmap)], style={"marginBottom": "20px"}),
        html.Div([dcc.Graph(figure=fig_vendedores)], style={"marginBottom": "40px"}),

        # SECCIÓN ANALÍTICA
        html.Div([
            html.H2("Módulo Backend: Análisis Estadístico y Aprendizaje Profundo", 
                    style={"textAlign": "center", "color": "#6D071A", "marginTop": "20px", "marginBottom": "20px"}),

            # Tarjetas Estadísticas y Deep Learning
            html.Div([
                html.Div([
                    html.H4("Forma de Distribución", style={"color": "#6D071A", "marginBottom": "8px"}),
                    html.P(f"Asimetría (Fisher g1): {stats_ventas['asimetria_fisher']:.2f}"),
                    html.P(f"Curtosis en exceso: {stats_ventas['curtosis']:.2f}"),
                    html.P(f"IQR: ${stats_ventas['iqr']:,.0f}")
                ], style={"flex": "1", "backgroundColor": "white", "padding": "15px", "borderRadius": "8px", "boxShadow": "0 2px 4px rgba(0,0,0,0.1)", "margin": "0 10px"}),

                html.Div([
                    html.H4("Prueba Shapiro-Wilk", style={"color": "#6D071A", "marginBottom": "8px"}),
                    html.P(f"Estadístico W: {stats_ventas['shapiro_w']:.4f}"),
                    html.P(f"Valor p: {stats_ventas['shapiro_p']:.4e}"),
                    html.P(f"Resultado: {normalidad_texto}", style={"fontWeight": "bold", "color": "#8B1E3F"})
                ], style={"flex": "1", "backgroundColor": "white", "padding": "15px", "borderRadius": "8px", "boxShadow": "0 2px 4px rgba(0,0,0,0.1)", "margin": "0 10px"}),

                html.Div([
                    html.H4("Deep Learning (MLP)", style={"color": "#6D071A", "marginBottom": "8px"}),
                    html.P("Partición: 80% Train / 20% Val"),
                    html.P(f"RMSE Validación: ${modelo_dl['rmse']:,.2f}"),
                    html.P(f"R² Score: {modelo_dl['r2']:.3f}")
                ], style={"flex": "1", "backgroundColor": "white", "padding": "15px", "borderRadius": "8px", "boxShadow": "0 2px 4px rgba(0,0,0,0.1)", "margin": "0 10px"})
            ], style={"display": "flex", "justifyContent": "space-between", "marginBottom": "25px"}),

            # Fila de gráficos analíticos
            html.Div([
                html.Div([dcc.Graph(figure=fig_densidad)], style={"width": "49%"}),
                html.Div([dcc.Graph(figure=fig_loss)], style={"width": "49%"})
            ], style={"display": "flex", "justifyContent": "space-between", "marginBottom": "30px"})
        ], style={"backgroundColor": "#f0ede8", "padding": "20px", "borderRadius": "10px"})

    ], style={"padding": "25px", "backgroundColor": "#f9f9f9", "fontFamily": "Arial, sans-serif"})