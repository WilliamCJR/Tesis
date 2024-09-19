import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import StandardScaler
from info import retorna_info as info
from graficos import *
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import scipy.stats as stats

st.set_page_config(
    page_title="Dashboard",
    page_icon="📊",
    layout="centered",
    initial_sidebar_state="expanded"
)
st.markdown("### Indicadores de disponibilidad de los servicios de telecomunicaciones en las principales ciudades de Colombia")

# Cargar el modelo y los datos
modelo = joblib.load('../V1_Modelo.pkl')
data = pd.read_csv('../DatosFiltradosDisponibilidad.csv', delimiter=';')
data_eval = pd.read_csv('../EvaluacionModelo.csv', delimiter=',')

# Configuración del escalador
scaler_y = StandardScaler()
scaler_y.mean_ = 98.35182591026732 
scaler_y.scale_ = 1.1051598391283655

# Diccionarios 
empresas = {'Claro': 800153993, 'Tigo': 830114921, 'Movistar': 830122566, 'Wom': 901354361, 'DirecTV': 805006014}
municipios = {'Bogota': 11001, 'Medellin': 5001,'Cali': 76001, 'Barranquilla': 8001}
meses = {'Enero': 1, 'Febrero': 2, 'Marzo': 3, 'Abril': 4, 'Mayo': 5, 'Junio': 6, 'Julio': 7, 'Agosto': 8, 'Septiembre': 9, 'Octubre': 10, 'Noviembre': 11, 'Diciembre': 12}
tecnologias = {'Movil 2G': 1, 'Movil 3G': 2,'Movil 4G': 3, 'HFC': 4, 'Fibra optica (PON)': 4}
meses_dict = {1: 'Enero', 2: 'Febrero', 3: 'Marzo', 4: 'Abril',5: 'Mayo', 6: 'Junio', 7: 'Julio', 8: 'Agosto',9: 'Septiembre', 10: 'Octubre', 11: 'Noviembre', 12: 'Diciembre'}
data['MES'] = data['MES'].map(meses_dict)
categoria_meses = pd.Categorical(
    data['MES'],
    categories=['Enero', 'Febrero', 'Marzo', 'Abril', 'Mayo', 'Junio','Julio', 'Agosto', 'Septiembre', 'Octubre', 'Noviembre', 'Diciembre'],
    ordered=True
)
data['MES'] = categoria_meses
empresas_dict = {800153993: 'Claro', 830114921: 'Tigo', 830122566: 'Movistar', 901354361: 'Wom',805006014: 'DirectTv'}
data['ID_EMPRESA'] = data['ID_EMPRESA'].map(empresas_dict)
tecnologia_dict = {1: 'Movil 2G', 2: 'Movil 3G', 3: 'Movil 4G', 4: 'HFC',5: 'PON'}
data['ID_TECNOLOGIA'] = data['ID_TECNOLOGIA'].map(tecnologia_dict)
municipio_dict = {11001: 'Bogota', 76001: 'Cali', 5001: 'Medellin', 8001: 'Barranquilla'}
data['ID_MUNICIPIO'] = data['ID_MUNICIPIO'].map(municipio_dict)

# Inicialización de las pestañas
tabs = st.tabs(["Estimacion de Disponibilidad","Prediccion conjunto de datos",
                "Evaluacion del Modelo", "Analisis Descriptivo",
                "Analisis porcentaje disponibilidad","Informacion Conjunto Datos"])
if 'predicciones' not in st.session_state:
    st.session_state.predicciones = pd.DataFrame(columns=['Anno', 'Mes', 'Empresa', 'Tecnologia', 'Ciudad', 'Disponibilidad Estimada (%)'])

data_info = info()
with tabs[0]:
    st.header('Estimación de Disponibilidad')
    # Selección de parámetros
    empresa = st.selectbox('Empresa', list(empresas.keys()))
    municipio = st.selectbox('Ciudad', list(municipios.keys()))
    mes = st.selectbox('Mes', list(meses.keys()))
    tecnologia = st.selectbox('Tecnología', list(tecnologias.keys()))
    anno = st.number_input('Año', min_value=2017, max_value=2027, step=1)
    id_empresa = empresas[empresa]
    id_municipio = municipios[municipio]
    id_mes = meses[mes]
    id_tecnologia = tecnologias[tecnologia]

    if st.button("Calcular"):
        entrada_modelo = pd.DataFrame({
            'ANNO': [anno],
            'MES': [id_mes],
            'ID_EMPRESA': [id_empresa],
            'ID_TECNOLOGIA': [id_tecnologia],
            'ID_MUNICIPIO': [id_municipio]
        })
        
        prediction = modelo.predict(entrada_modelo)
        prediction = scaler_y.inverse_transform(prediction.reshape(-1, 1)).flatten()[0]
        nueva_fila = pd.DataFrame({
            'Anno': [anno], 'Mes': [mes], 'Empresa': [empresa],
            'Tecnologia': [tecnologia], 'Ciudad': [municipio],
            'Disponibilidad Estimada (%)': [f"{prediction:.2f}"]
        })
        st.session_state.predicciones = pd.concat([st.session_state.predicciones, nueva_fila], ignore_index=True)
     
    st.write("Estimaciones acumuladas:")
    st.table(st.session_state.predicciones)
    csv = st.session_state.predicciones.to_csv(index=False)
    st.download_button(
        label="Descargar CSV",
        data=csv,
        file_name='predicciones.csv',
        mime='text/csv',
    )   
with tabs[1]:
    st.header('Estimacion conjunto de datos')
    # Subir un archivo CSV para predecir
    uploaded_file = st.file_uploader("Cargar archivo CSV para realziar estimaciones", type="csv")

    if uploaded_file is not None:
        datos = pd.read_csv(uploaded_file)
        st.write("Datos cargados:")
        datos['ANNO'] =  datos['ANNO'].apply(lambda x: f"{int(x)}") 
        datos['ID_EMPRESA'] = datos['ID_EMPRESA'].apply(lambda x: f"{int(x)}") 
        datos['ID_MUNICIPIO'] = datos['ID_MUNICIPIO'].apply(lambda x: f"{int(x)}") 
        st.write(datos.head())

        if st.button("Calcular Resultados"):

            predictions = modelo.predict(datos)
            predictions = scaler_y.inverse_transform(predictions.reshape(-1, 1)).flatten()

            # Añadir las predicciones a los datos
            datos['Disponibilidad Estimada (%)'] = predictions

            st.write("Resultado:")
            st.dataframe(datos)

            # Opción para descargar los resultados
            csv = datos.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Descargar Resultados",
                data=csv,
                file_name='Resultados.csv',
                mime='text/csv',
            )
    else:
        st.write("Cargar un archivo CSV para realizar estimaciones.")   
with tabs[2]:
    st.header('Evaluación del Modelo')

    st.write("Datos de evaluación cargados:")
    st.write(data_eval.head())

    # Cálculo de métricas de evaluación
    y_true = data_eval['PORCENTAJE_DISPONIBILIDAD']
    y_pred = data_eval['PREDICCION']
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    # Estadísticas del modelo
    st.subheader('Estadísticas del Modelo')
    st.write(f"Error Absoluto Medio (MAE): {mae:.2f}")
    st.write(f"Error Cuadrático Medio (MSE): {mse:.2f}")
    st.write(f"Raíz del Error Cuadrático Medio (RMSE): {np.sqrt(mse):.2f}")
    st.write(f"Coeficiente de Determinación (R²): {r2:.2f}")

    # Gráfico de dispersión de PORCENTAJE_DISPONIBILIDAD vs PREDICCION
    st.subheader('Dispersión de Disponibilidad Real vs Predicción')
    plt.figure(figsize=(12, 6))
    sns.scatterplot(data=data_eval, x='PORCENTAJE_DISPONIBILIDAD', y='PREDICCION', alpha=0.6)
    plt.plot([data_eval['PORCENTAJE_DISPONIBILIDAD'].min(), data_eval['PORCENTAJE_DISPONIBILIDAD'].max()],
             [data_eval['PORCENTAJE_DISPONIBILIDAD'].min(), data_eval['PORCENTAJE_DISPONIBILIDAD'].max()],
             color='red', linestyle='--')
    plt.title('Disponibilidad Real vs Predicción')
    plt.xlabel('Disponibilidad Real')
    plt.ylabel('Predicción')
    st.pyplot(plt.gcf())
    plt.clf()


    # Gráfico de residuos (errores)
    st.subheader('Gráfico de Residuos')
    residuals = y_true - y_pred
    plt.figure(figsize=(12, 6))
    sns.scatterplot(x=y_pred, y=residuals, alpha=0.5)
    plt.axhline(0, color='red', linestyle='--')
    plt.xlabel('Predicciones')
    plt.ylabel('Residuos')
    plt.title('Gráfico de Residuos')
    st.pyplot(plt.gcf())
    plt.clf()

    # Gráfico de distribución de errores
    st.subheader('Distribución de Errores')
    plt.figure(figsize=(12, 6))
    plt.hist(residuals, bins=30, alpha=0.7, color='blue')
    plt.xlabel('Residuos')
    plt.ylabel('Frecuencia')
    plt.title('Distribución de Errores')
    st.pyplot(plt.gcf())
    plt.clf()

    # Gráfico Q-Q para residuos
    st.subheader('Gráfico Q-Q de Residuos')
    plt.figure(figsize=(12, 6))
    stats.probplot(residuals, dist="norm", plot=plt)
    plt.title('Gráfico Q-Q de Residuos')
    st.pyplot(plt.gcf())
    plt.clf()


with tabs[3]:
    st.header('Análisis Descriptivo de Datos')
    st.subheader('Distribución de ANNO')
    crear_grafico_barras(data, x='ANNO', titulo='Distribución de ANNO', xlabel='Año', ylabel='Frecuencia')
    st.subheader('Distribución de MES')
    crear_grafico_barras(data, x='MES', titulo='Distribución de MES', xlabel='Mes', ylabel='Frecuencia')
    st.subheader('Distribución de ID_EMPRESA')
    crear_grafico_barras(data, x='ID_EMPRESA', titulo='Distribución de ID_EMPRESA', xlabel='Empresa', ylabel='Frecuencia',
                        order=data['ID_EMPRESA'].value_counts().index)
    st.subheader('Distribución de ID_TECNOLOGIA')
    crear_grafico_barras(data, x='ID_TECNOLOGIA', titulo='Distribución de ID_TECNOLOGIA', xlabel='Tecnología', ylabel='Frecuencia')
    st.subheader('Distribución de ID_MUNICIPIO')
    crear_grafico_barras(data, x='ID_MUNICIPIO', titulo='Distribución de ID_MUNICIPIO', xlabel='Municipio', ylabel='Frecuencia',
                        order=data['ID_MUNICIPIO'].value_counts().index)
    st.subheader('Distribución de PORCENTAJE_DISPONIBILIDAD')
    crear_grafico_histograma(data, x='PORCENTAJE_DISPONIBILIDAD', bins=30, titulo='Distribución de PORCENTAJE_DISPONIBILIDAD',
                            xlabel='Porcentaje de Disponibilidad', ylabel='Frecuencia', color='blue')   
with tabs[4]:
    st.header('Análisis disponibilidad')

    st.subheader('ANNO vs PORCENTAJE_DISPONIBILIDAD')
    crear_grafico_lineal(data, x='ANNO', y='PORCENTAJE_DISPONIBILIDAD',
                        titulo='ANNO vs PORCENTAJE_DISPONIBILIDAD', xlabel='Año',
                        ylabel='Porcentaje de Disponibilidad', ylim=(98.3, 98.4))
    st.subheader('MES vs PORCENTAJE_DISPONIBILIDAD')
    crear_grafico_lineal(data, x='MES', y='PORCENTAJE_DISPONIBILIDAD',
                        titulo='MES vs PORCENTAJE_DISPONIBILIDAD', xlabel='Mes',
                        ylabel='Porcentaje de Disponibilidad', ylim=(98.3, 98.4), rotar_etiquetas=True)
    st.subheader('EMPRESA vs PORCENTAJE_DISPONIBILIDAD')
    crear_grafico_lineal(data, x='ID_EMPRESA', y='PORCENTAJE_DISPONIBILIDAD',
                        titulo='EMPRESA vs PORCENTAJE_DISPONIBILIDAD', xlabel='Empresa',
                        ylabel='Porcentaje de Disponibilidad', ylim=(98.3, 98.4), rotar_etiquetas=True)
    st.subheader('TECNOLOGIA vs PORCENTAJE_DISPONIBILIDAD')
    crear_grafico_lineal(data, x='ID_TECNOLOGIA', y='PORCENTAJE_DISPONIBILIDAD',
                        titulo='TECNOLOGIA vs PORCENTAJE_DISPONIBILIDAD', xlabel='Tecnología',
                        ylabel='Porcentaje de Disponibilidad', ylim=(97, 99), rotar_etiquetas=True)
    st.subheader('MUNICIPIO vs PORCENTAJE_DISPONIBILIDAD')
    crear_grafico_lineal(data, x='ID_MUNICIPIO', y='PORCENTAJE_DISPONIBILIDAD',
                        titulo='MUNICIPIO vs PORCENTAJE_DISPONIBILIDAD', xlabel='Municipio',
                        ylabel='Porcentaje de Disponibilidad', ylim=(98.3, 98.4), rotar_etiquetas=True)
with tabs[5]:
    st.header('Conjunto de datos')
    st.markdown("[Disponibilidad de Elementos de Red de Acceso](https://www.postdata.gov.co/dataset/indicadores-de-disponibilidad-para-los-servicios-de-telecomunicaciones/resource/e94521a2#)")
    st.markdown("Diccionario de datos: ") 
    info= pd.DataFrame(data_info)
    st.table(info)
    
    
