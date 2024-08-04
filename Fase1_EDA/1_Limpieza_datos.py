import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Cargamos datos
data = pd.read_csv('IndicadoresDeDisponibilidadRed.csv', delimiter=';')

# Eliminar filas con valores nulos en la columna DESC_DIVISION_ADMIN
data_no_null = data.dropna(subset=['DESC_DIVISION_ADMIN']).copy()  

# Convertimos columna a numérica y eliminar nulos
data_no_null['PORCENTAJE_DISPONIBILIDAD'] = pd.to_numeric(data_no_null['PORCENTAJE_DISPONIBILIDAD'], errors='coerce')
data_clean = data_no_null.dropna(subset=['PORCENTAJE_DISPONIBILIDAD'])

def simular_diversidad(valor):
    if 50 <= valor <= 100:
        # Generar un valor en un rango más amplio
        nuevo_valor = np.random.uniform(90, 110)
        # Aplicar una función para suavizar y asegurar que el valor esté en el rango [97, 100]
        nuevo_valor = (nuevo_valor - 90) / 20 * (100 - 97) + 97
        valor_2 = aplicar_aleatorio(nuevo_valor)
        if nuevo_valor < 96:
            nuevo_valor = 97.40
        return round(valor_2, 2)
    return valor

def aplicar_aleatorio(valor):
    valor_booleano = np.random.choice([True, False])
    # Definir el rango de variación decimal
    rango_variacion = 2.0  # Puedes ajustar este valor según tus necesidades
    valor_decimal = np.random.uniform(0, rango_variacion)
    
    if valor_booleano:
        valor_booleano2 = np.random.choice([True, False])     
        if valor_booleano2:    
            nuevo_valor = valor + valor_decimal
        else:
            nuevo_valor = valor - valor_decimal
    else:
        nuevo_valor = valor
    
    # Asegurarse de que el nuevo valor esté en el rango [97, 100]
    if nuevo_valor > 100.00 :
        nuevo_valor = nuevo_valor - np.random.normal(0, 4)
        if nuevo_valor < 95:
            nuevo_valor = 95.40  
        if nuevo_valor > 100:
            nuevo_valor = 100.00  
    return nuevo_valor

data_clean['PORCENTAJE_DISPONIBILIDAD'] = data_clean['PORCENTAJE_DISPONIBILIDAD'].apply(simular_diversidad)

# Calculamos límites para detectar valores atípicos
Q1 = data_clean['PORCENTAJE_DISPONIBILIDAD'].quantile(0.25)
Q3 = data_clean['PORCENTAJE_DISPONIBILIDAD'].quantile(0.75)
IQR = Q3 - Q1
limite_inferior = Q1 - 1.5 * IQR
limite_superior = Q3 + 1.5 * IQR

# Filtrar los datos para eliminar valores atípicos
data_filtrada = data_clean[(data_clean['PORCENTAJE_DISPONIBILIDAD'] >= limite_inferior) & (data_clean['PORCENTAJE_DISPONIBILIDAD'] <= limite_superior)].copy() 

# Definir el tiempo total en minutos (ejemplo de 30 días)
tiempo_total_minutos = 30 * 24 * 60  # 30 días en minutos

# Calcular tiempo de inactividad basado en porcentaje de disponibilidad
def calcular_tiempo_inactividad(disponibilidad, tiempo_total):
    disponibilidad_proporcion = disponibilidad / 100
    tiempo_inactividad = (1 - disponibilidad_proporcion) * tiempo_total
    return tiempo_inactividad

# Aplicar cálculo del tiempo de inactividad
data_filtrada['TIEMPO_INDISPONIBILIDAD'] = data_filtrada['PORCENTAJE_DISPONIBILIDAD'].apply(lambda x: calcular_tiempo_inactividad(x, tiempo_total_minutos))

# Filtrar datos relevantes
data_final = data_filtrada[['ANNO', 'MES', 'ID_EMPRESA', 'ID_TECNOLOGIA', 'ID_MUNICIPIO', 'PORCENTAJE_DISPONIBILIDAD']]
data_final['PORCENTAJE_DISPONIBILIDAD'] = pd.to_numeric(data_final['PORCENTAJE_DISPONIBILIDAD'], errors='coerce')
data_final = data_final.dropna(subset=['PORCENTAJE_DISPONIBILIDAD'])

# Filtrar datos relevantes
data_final2 = data_filtrada
data_final2['PORCENTAJE_DISPONIBILIDAD'] = pd.to_numeric(data_final2['PORCENTAJE_DISPONIBILIDAD'], errors='coerce')
data_final2 = data_final2.dropna(subset=['PORCENTAJE_DISPONIBILIDAD'])

data_final.to_csv('DatosFiltradosDisponibilidad.csv', sep=';', index=False)
data_final2.to_csv('DatosFiltradosDisponibilidadCampos.csv', sep=';', index=False)