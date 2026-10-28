"""
backend_analitico.py
Módulo backend de análisis estadístico e inferencial y Aprendizaje Profundo
implementado con Scikit-Learn, Keras y TensorFlow.
"""

import os
# Silenciar advertencias informativas de inicialización de TensorFlow
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import numpy as np
import pandas as pd
import scipy.stats as stats
import statistics

# Scikit-Learn: Partición, escalado y codificación
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# TensorFlow y Keras: Red Neuronal Artificial (Deep Learning)
import tensorflow as tf
from tensorflow import keras
from keras import layers, callbacks


# ==========================================================
# 1. DESCRIPCIÓN E INFERENCIA ESTADÍSTICA
# ==========================================================

def analizar_estadistica_descriptiva(df: pd.DataFrame, columna: str = "total_venta") -> dict:
    """
    Calcula medidas de tendencia central, dispersión, forma (Fisher, curtosis),
    rango intercuartílico (IQR) y prueba formal de normalidad de Shapiro-Wilk.
    """
    valores = df[columna].dropna().to_numpy()

    # Medidas de tendencia central
    media = float(np.mean(valores))
    mediana = float(np.median(valores))
    try:
        modas = statistics.multimode(np.round(valores, 0))
    except Exception:
        modas = [mediana]

    # Medidas de dispersión muestral (ddof=1: corrección de Bessel)
    std_muestral = float(np.std(valores, ddof=1))
    var_muestral = float(np.var(valores, ddof=1))

    # Medidas de forma
    asimetria_fisher = float(stats.skew(valores, bias=False))
    curtosis = float(stats.kurtosis(valores, bias=False))

    # Rango Intercuartílico (IQR)
    q25, q75 = np.percentile(valores, [25, 75])
    iqr = float(q75 - q25)

    # Test de Shapiro-Wilk (estadístico W y p-valor)
    # Si p < 0.05 se rechaza la hipótesis nula de normalidad
    shapiro_w, shapiro_p = stats.shapiro(valores)

    return {
        "media": media,
        "mediana": mediana,
        "moda": modas[0] if len(modas) > 0 else media,
        "std": std_muestral,
        "varianza": var_muestral,
        "asimetria_fisher": asimetria_fisher,
        "curtosis": curtosis,
        "q25": float(q25),
        "q75": float(q75),
        "iqr": iqr,
        "shapiro_w": float(shapiro_w),
        "shapiro_p": float(shapiro_p),
        "es_normal": bool(shapiro_p >= 0.05)
    }


def calcular_correlaciones(df: pd.DataFrame) -> dict:
    """
    Calcula la correlación lineal paramétrica de Karl Pearson y
    la correlación monótona de rangos de Charles Spearman con sus p-valores.
    """
    sub_df = df[["cantidad_botellas", "total_venta"]].dropna()
    r_pearson, p_pearson = stats.pearsonr(sub_df["cantidad_botellas"], sub_df["total_venta"])
    rho_spearman, p_spearman = stats.spearmanr(sub_df["cantidad_botellas"], sub_df["total_venta"])

    return {
        "pearson_r": float(r_pearson),
        "pearson_p": float(p_pearson),
        "spearman_rho": float(rho_spearman),
        "spearman_p": float(p_spearman)
    }


# ==========================================================
# 2. APRENDIZAJE PROFUNDO (Red Neuronal Densa / Keras & TensorFlow)
# ==========================================================

def entrenar_modelo_deep_learning(df: pd.DataFrame):
    """
    Entrena una Red Neuronal Artificial profunda (MLP) utilizando TensorFlow y Keras
    para pronosticar total_venta a partir de características operativas y comerciales.
    
    Aplica:
    - Scikit-Learn: ColumnTransformer, StandardScaler y OneHotEncoder para inputs.
    - Escalado de la variable objetivo (Y) para estabilizar gradientes en Adam.
    - TensorFlow/Keras: Sequential con capas Dense, Dropout y EarlyStopping.
    - Partición supervisada: 80% entrenamiento, 20% validación representativa.
    """
    features_num = ["cantidad_botellas", "precio_unitario", "descuento"]
    features_cat = ["canal_venta", "categoria", "provincia"]
    target = "total_venta"

    datos = df.dropna(subset=features_num + features_cat + [target]).copy()
    X = datos[features_num + features_cat]
    y = datos[target].to_numpy().astype(float)

    # 1. Partición 80% entrenamiento / 20% validación
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.20, random_state=42
    )

    # 2. Pipeline de Preprocesamiento de entradas con Scikit-Learn
    preprocesador_X = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), features_num),
            ("cat", OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore"), features_cat)
        ]
    )

    X_train_proc = preprocesador_X.fit_transform(X_train)
    X_val_proc = preprocesador_X.transform(X_val)

    # 3. Escalado de la variable objetivo Y (indispensable para convergencia en redes neuronales)
    scaler_y = StandardScaler()
    y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
    y_val_scaled = scaler_y.transform(y_val.reshape(-1, 1)).flatten()

    # 4. Arquitectura de Red Neuronal Profunda con Keras (Sequential API)
    tf.random.set_seed(42)
    np.random.seed(42)

    model = keras.Sequential([
        layers.Input(shape=(X_train_proc.shape[1],), name="capa_entrada"),
        layers.Dense(64, activation="relu", name="capa_oculta_1"),
        layers.Dropout(0.05, name="regularizacion_dropout"),
        layers.Dense(32, activation="relu", name="capa_oculta_2"),
        layers.Dense(1, activation="linear", name="capa_salida_regresion")
    ], name="Red_Neuronal_Ventas_Bodega")

    # 5. Compilación con optimizador Adam y función de costo MSE
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.01),
        loss="mean_squared_error",
        metrics=["mae"]
    )

    # 6. Callback de parada temprana (EarlyStopping) para evitar sobreajuste
    early_stop = callbacks.EarlyStopping(
        monitor="val_loss",
        patience=25,
        restore_best_weights=True
    )

    # 7. Entrenamiento
    history = model.fit(
        X_train_proc,
        y_train_scaled,
        validation_data=(X_val_proc, y_val_scaled),
        epochs=120,
        batch_size=16,
        callbacks=[early_stop],
        verbose=0
    )

    # 8. Predicciones desescaladas a valores reales monetarios ($ ARS)
    y_pred_scaled = model.predict(X_val_proc, verbose=0)
    y_pred = scaler_y.inverse_transform(y_pred_scaled).flatten()

    # 9. Cálculo de métricas formales de validación
    rmse = float(np.sqrt(mean_squared_error(y_val, y_pred)))
    mae = float(mean_absolute_error(y_val, y_pred))
    r2 = float(r2_score(y_val, y_pred))
    curva_perdida = [float(loss_val) for loss_val in history.history["loss"]]

    return {
        "model": model,
        "preprocesador_X": preprocesador_X,
        "scaler_y": scaler_y,
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "curva_perdida": curva_perdida,
        "y_val": y_val,
        "y_pred": y_pred
    }


def predecir_nuevas_ventas(modelo_dl_dict: dict, nuevos_datos: pd.DataFrame) -> np.ndarray:
    """
    Función utilitaria para realizar inferencias sobre nuevas órdenes o cotizaciones.
    """
    model = modelo_dl_dict["model"]
    preprocesador_X = modelo_dl_dict["preprocesador_X"]
    scaler_y = modelo_dl_dict["scaler_y"]

    X_proc = preprocesador_X.transform(nuevos_datos)
    y_pred_scaled = model.predict(X_proc, verbose=0)
    return scaler_y.inverse_transform(y_pred_scaled).flatten()
