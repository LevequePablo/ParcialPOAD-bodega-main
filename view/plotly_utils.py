import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff

# Paleta oficial de Aura de los Andes
COLOR_VINO_OSCURO = "#6D071A"
COLOR_VINO_MEDIO = "#8B1E3F"
COLOR_VINO_CLARO = "#A23B5A"
COLOR_DORADO = "#C9A227"
COLOR_DORADO_CLARO = "#DCCB8F"

PALETA_DISCRETA = [COLOR_VINO_OSCURO, COLOR_VINO_MEDIO, COLOR_VINO_CLARO, COLOR_DORADO, COLOR_DORADO_CLARO]


def grafico_evolucion_mensual(df):
    ventas_mes = df.groupby(["mes_num", "mes"])["total_venta"].sum().reset_index()
    ventas_mes = ventas_mes.sort_values("mes_num")
    fig = px.line(
        ventas_mes,
        x="mes",
        y="total_venta",
        markers=True,
        title="Evolución mensual de ventas"
    )
    fig.update_traces(line_color=COLOR_VINO_OSCURO)
    fig.update_layout(paper_bgcolor="#f9f9f9", plot_bgcolor="#f9f9f9")
    return fig


def grafico_ventas_canal(df):
    ventas_canal = df.groupby("canal_venta")["total_venta"].sum().reset_index()
    fig = px.pie(
        ventas_canal,
        names="canal_venta",
        values="total_venta",
        title="Distribución de ventas por canal",
        color_discrete_sequence=PALETA_DISCRETA
    )
    fig.update_layout(paper_bgcolor="#f9f9f9")
    return fig


def grafico_top_productos(df):
    prod = df.groupby("producto")["cantidad_botellas"].sum().reset_index()
    prod = prod.sort_values("cantidad_botellas", ascending=False).head(10)
    fig = px.bar(
        prod,
        x="producto",
        y="cantidad_botellas",
        title="Top 10 productos más vendidos"
    )
    fig.update_traces(marker_color=COLOR_VINO_MEDIO)
    fig.update_layout(paper_bgcolor="#f9f9f9", plot_bgcolor="#f9f9f9")
    return fig


def grafico_ventas_provincia(df):
    ventas_prov = df.groupby("provincia")["total_venta"].sum().reset_index()
    fig = px.pie(
        ventas_prov,
        names="provincia",
        values="total_venta",
        title="Distribución de ventas por provincia",
        color_discrete_sequence=PALETA_DISCRETA
    )
    fig.update_layout(paper_bgcolor="#f9f9f9")
    return fig


def grafico_heatmap_provincia_mes(df):
    pivot = df.pivot_table(index="provincia", columns="mes", values="total_venta", aggfunc="sum", fill_value=0)
    fig = px.imshow(
        pivot,
        title="Ventas totales por provincia y mes",
        color_continuous_scale="Reds"
    )
    fig.update_layout(paper_bgcolor="#f9f9f9")
    return fig


def grafico_ranking_vendedores(df):
    vend = df.groupby("vendedor")["ID"].count().reset_index()
    vend.columns = ["vendedor", "cantidad_ventas"]
    vend = vend.sort_values("cantidad_ventas", ascending=False)
    fig = px.bar(
        vend,
        x="vendedor",
        y="cantidad_ventas",
        title="Ranking de vendedores por cantidad de ventas"
    )
    fig.update_traces(marker_color=COLOR_VINO_MEDIO)
    fig.update_layout(paper_bgcolor="#f9f9f9", plot_bgcolor="#f9f9f9")
    return fig


# Gráficos del parcial 2 (Estadística + Deep Learning)
def grafico_curva_densidad(df):
    valores = df["total_venta"].dropna().tolist()
    fig = ff.create_distplot(
        [valores],
        group_labels=["Total Venta"],
        bin_size=15000,
        colors=[COLOR_VINO_OSCURO],
        show_hist=True,
        show_rug=False
    )
    fig.update_layout(
        title="Distribución y Curva de Densidad (KDE) de Ventas",
        xaxis_title="Total Venta ($ ARS)",
        yaxis_title="Densidad",
        template="plotly_white",
        paper_bgcolor="#f9f9f9",
        plot_bgcolor="#f9f9f9"
    )
    return fig


def grafico_curva_perdida_dl(curva_perdida):
    epocas = list(range(1, len(curva_perdida) + 1))
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=epocas,
        y=curva_perdida,
        mode="lines+markers",
        line=dict(color=COLOR_VINO_MEDIO, width=2),
        marker=dict(size=4, color=COLOR_DORADO),
        name="Loss (MSE)"
    ))
    fig.update_layout(
        title="Curva de Aprendizaje - Red Neuronal (Deep Learning)",
        xaxis_title="Época / Iteración",
        yaxis_title="Pérdida (Loss)",
        template="plotly_white",
        paper_bgcolor="#f9f9f9",
        plot_bgcolor="#f9f9f9"
    )
    return fig