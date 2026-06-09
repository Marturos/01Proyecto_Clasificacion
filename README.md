# Proyecto de Clasificación ENAHO 2024 - NBI

Este proyecto predice si un hogar presenta al menos una Necesidad Básica Insatisfecha usando datos ENAHO 2024.

## Archivos principales

- `app_streamlit.py`: aplicación web en Streamlit.
- `modelo_enaho_nbi.joblib`: modelo entrenado con pipeline completo.
- `importancia_variables_enaho.csv`: importancia de variables.
- `casos_prueba_streamlit_enaho.csv`: casos para validar predicciones.
- `requirements.txt`: librerías necesarias.
- `ejecutar_app_streamlit.bat`: archivo para ejecutar la app en Windows.

## Ruta del proyecto

```text
D:\INEI_DESPLIEGUE_WEB\PROYECTO_FINAL\01Proyecto_Clasificacion
```

## Ejecutar aplicación

```bash
cd /d "D:\INEI_DESPLIEGUE_WEB\PROYECTO_FINAL\01Proyecto_Clasificacion"
streamlit run app_streamlit.py
```

## Target

- `0`: hogar sin NBI.
- `1`: hogar con al menos una NBI.
