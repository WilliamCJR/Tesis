import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#Cargamos datos
data = pd.read_csv('IndicadoresDeDisponibilidadRed.csv', delimiter=';')

# Eliminar filas con valores nulos
data_no_null = data.dropna(subset=['DESC_DIVISION_ADMIN']).copy()  
data_no_null['PORCENTAJE_DISPONIBILIDAD'] = pd.to_numeric(data_no_null['PORCENTAJE_DISPONIBILIDAD'], errors='coerce')
data_clean = data_no_null.dropna(subset=['PORCENTAJE_DISPONIBILIDAD'])

# Calculamos límites para detectar valores atípicos
Q1 = data_clean['PORCENTAJE_DISPONIBILIDAD'].quantile(0.25)
Q3 = data_clean['PORCENTAJE_DISPONIBILIDAD'].quantile(0.75)
IQR = Q3 - Q1
limite_inferior = Q1 - 1.5 * IQR
limite_superior = Q3 + 1.5 * IQR

# Filtrar los datos para eliminar valores atípicos
data_filtrada = data_clean[(data_clean['PORCENTAJE_DISPONIBILIDAD'] >= limite_inferior) & 
                           (data_clean['PORCENTAJE_DISPONIBILIDAD'] <= limite_superior)].copy() 

# Filtrar datos relevantes
data_final = data_filtrada[['ANNO', 'MES', 'ID_EMPRESA', 'ID_TECNOLOGIA', 'ID_MUNICIPIO', 'PORCENTAJE_DISPONIBILIDAD']]
data_final['PORCENTAJE_DISPONIBILIDAD'] = pd.to_numeric(data_final['PORCENTAJE_DISPONIBILIDAD'], errors='coerce')
data_final = data_final.dropna(subset=['PORCENTAJE_DISPONIBILIDAD'])

# Guardamos los datos filtrados en un archivo CSV
data_final.to_csv('DatosFiltradosDisponibilidad.csv', sep=';', index=False)
