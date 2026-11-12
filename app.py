import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Configuración de la página
st.set_page_config(page_title="Predictor de Calificaciones UAP", layout="wide")

st.title("🎓 Sistema de Predicción de Calificaciones UAP")
st.markdown("Aplicación interactiva basada en Machine Learning (**Regresión Ridge** y **Random Forest**) usando datos reales de estudiantes.")

# Cargar los datos reales
@st.cache_data
def cargar_datos():
    try:
        df = pd.read_csv("datos_estudiantes.csv", encoding="latin1")
    except Exception as e:
        df = pd.read_csv("datos_estudiantes.csv", encoding="utf-8")
    return df

df_raw = cargar_datos()

# Limpieza y selección de variables relevantes para el modelo
# Identificamos las columnas clave basadas en la estructura del archivo
col_target = [c for c in df_raw.columns if "last period" in c][0]
col_estudio = [c for c in df_raw.columns if "study outside class" in c][0]
col_sueno = [c for c in df_raw.columns if "sleep per night" in c][0]
col_asistencia = [c for c in df_raw.columns if "attend classes" in c][0]
col_trabajo = [c for c in df_raw.columns if "work per day" in c][0]

# Preprocesamiento de un DataFrame limpio para el modelo
df_model = pd.DataFrame({
    'Calificacion_Anterior': pd.to_numeric(df_raw[col_target], errors='coerce'),
    'Horas_Estudio': pd.to_numeric(df_raw[col_estudio], errors='coerce'),
    'Horas_Sueno': pd.to_numeric(df_raw[col_sueno], errors='coerce'),
    'Horas_Trabajo': pd.to_numeric(df_raw[col_trabajo], errors='coerce')
})

# Rellenar valores nulos si los hay y limpiar
df_model = df_model.dropna()

if len(df_model) < 10:
    st.error("El conjunto de datos limpio tiene muy pocas filas para entrenar. Revisa los datos.")
else:
    # Variables predictoras (X) y objetivo (y) -> Calificación del periodo anterior
    X = df_model[['Horas_Estudio', 'Horas_Sueno', 'Horas_Trabajo']]
    y = df_model['Calificacion_Anterior']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 1. Entrenamiento: Regresión Ridge
    ridge = Ridge(alpha=1.0)
    ridge.fit(X_train, y_train)
    ridge_pred = ridge.predict(X_test)
    ridge_r2 = r2_score(y_test, ridge_pred)
    ridge_rmse = np.sqrt(mean_squared_error(y_test, ridge_pred))

    # 2. Entrenamiento: Random Forest Regressor
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)
    rf_r2 = r2_score(y_test, rf_pred)
    rf_rmse = np.sqrt(mean_squared_error(y_test, rf_pred))

    # Panel lateral para la predicción interactiva
    st.sidebar.header("🔍 Ingrese las métricas del estudiante")
    input_estudio = st.sidebar.slider("Horas de estudio fuera de clase", 0.0, 15.0, 4.0)
    input_sueno = st.sidebar.slider("Horas de sueño por noche", 0.0, 12.0, 7.0)
    input_trabajo = st.sidebar.slider("Horas de trabajo por día", 0.0, 12.0, 0.0)

    datos_usuario = pd.DataFrame({
        'Horas_Estudio': [input_estudio],
        'Horas_Sueno': [input_sueno],
        'Horas_Trabajo': [input_trabajo]
    })

    # Layout principal
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📊 Métricas de Rendimiento de los Modelos")
        tabla_metricas = pd.DataFrame({
            "Modelo": ["Regresión Ridge", "Random Forest Regressor"],
            "R² Score": [f"{ridge_r2:.4f}", f"{rf_r2:.4f}"],
            "RMSE (Error)": [f"{ridge_rmse:.4f}", f"{rf_rmse:.4f}"]
        })
        st.table(tabla_metricas)
        st.info("El $R^2$ evalúa qué tan bien se ajusta el modelo a los datos reales de la UAP.")

    with col2:
        st.subheader("🎯 Resultado de la Predicción")
        modelo_elegido = st.selectbox("Seleccione el modelo para predecir", ["Random Forest", "Regresión Ridge"])
        
        if modelo_elegido == "Random Forest":
            prediccion = rf.predict(datos_usuario)[0]
        else:
            prediccion = ridge.predict(datos_usuario)[0]
            
        st.metric(label="Calificación Estimada (Periodo Anterior)", value=f"{prediccion:.2f}")

    st.markdown("---")
    st.subheader("📁 Vista previa de los datos procesados de la UAP")
    st.dataframe(df_model.head(10))