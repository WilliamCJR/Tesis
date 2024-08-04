import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Cargamos datos
data = pd.read_csv('DatosFiltradosDisponibilidad.csv', delimiter=';')

# Configurar estilo de los gráficos
sns.set(style="whitegrid")

# Gráfico lineal para ANNO vs PORCENTAJE_DISPONIBILIDAD sin banda de confianza
plt.figure(figsize=(12, 6))
sns.lineplot(data=data, x='ANNO', y='PORCENTAJE_DISPONIBILIDAD', marker='o', palette='viridis', ci=None)
plt.title('ANNO vs PORCENTAJE_DISPONIBILIDAD')
plt.xlabel('Año')
plt.ylabel('Porcentaje de Disponibilidad')
plt.ylim(98.3, 98.4)
plt.show()

meses_dict = {
    1: 'Enero', 2: 'Febrero', 3: 'Marzo', 4: 'Abril',
    5: 'Mayo', 6: 'Junio', 7: 'Julio', 8: 'Agosto',
    9: 'Septiembre', 10: 'Octubre', 11: 'Noviembre', 12: 'Diciembre'
}

# Mapear los números de meses a nombres de meses
data['MES'] = data['MES'].map(meses_dict)

# Crear una categoría ordenada para los meses
categoria_meses = pd.Categorical(
    data['MES'],
    categories=[
        'Enero', 'Febrero', 'Marzo', 'Abril', 'Mayo', 'Junio',
        'Julio', 'Agosto', 'Septiembre', 'Octubre', 'Noviembre', 'Diciembre'
    ],
    ordered=True
)

# Asignar la categoría ordenada a la columna 'MES'
data['MES'] = categoria_meses

# Configurar estilo de los gráficos
sns.set(style="whitegrid")

# Gráfico lineal para MES vs PORCENTAJE_DISPONIBILIDAD sin banda de confianza
plt.figure(figsize=(12, 6))
sns.lineplot(data=data, x='MES', y='PORCENTAJE_DISPONIBILIDAD', marker='o', palette='viridis', ci=None)
plt.title('MES vs PORCENTAJE_DISPONIBILIDAD')
plt.xlabel('Mes')
plt.ylabel('Porcentaje de Disponibilidad')
plt.ylim(98.3, 98.4)
plt.xticks(rotation=45)  # Rotar etiquetas de los meses para mejor legibilidad
plt.show()




empresas_dict = {
    800153993: 'Claro', 830114921: 'Tigo', 
    830122566: 'Movistar', 901354361: 'Wom',
    805006014: 'DirectTv'
}

# Mapear los números de meses a nombres de meses
data['ID_EMPRESA'] = data['ID_EMPRESA'].map(empresas_dict)


# Configurar estilo de los gráficos
sns.set(style="whitegrid")

# Gráfico lineal para MES vs PORCENTAJE_DISPONIBILIDAD sin banda de confianza
plt.figure(figsize=(12, 6))
sns.lineplot(data=data, x='ID_EMPRESA', y='PORCENTAJE_DISPONIBILIDAD', marker='o', palette='viridis', ci=None)
plt.title('EMPRESA vs PORCENTAJE_DISPONIBILIDAD')
plt.xlabel('Empresa')
plt.ylabel('Porcentaje de Disponibilidad')
plt.xticks(rotation=45)  # Rotar etiquetas de los meses para mejor legibilidad
plt.ylim(98.3, 98.4)
plt.show()







tecnologia_dict = {
    1: 'Móvil 2G', 2: 'Móvil 3G', 
    3: 'Móvil 4G', 4: 'HFC',
    5: 'PON'
}

# Mapear los números de meses a nombres de meses
data['ID_TECNOLOGIA'] = data['ID_TECNOLOGIA'].map(tecnologia_dict)


# Configurar estilo de los gráficos
sns.set(style="whitegrid")

# Gráfico lineal para MES vs PORCENTAJE_DISPONIBILIDAD sin banda de confianza
plt.figure(figsize=(12, 6))
sns.lineplot(data=data, x='ID_TECNOLOGIA', y='PORCENTAJE_DISPONIBILIDAD', marker='o', palette='viridis', ci=None)
plt.title('TECNOLOGIA vs PORCENTAJE_DISPONIBILIDAD')
plt.xlabel('Tecnologia')
plt.ylabel('Porcentaje de Disponibilidad')
plt.xticks(rotation=45)  # Rotar etiquetas de los meses para mejor legibilidad
plt.ylim(97, 99)
plt.show()



municipio_dict = {
    11001: 'Bogota', 76001: 'Cali', 
    5001: 'Medellín', 8001: 'Barranquilla'
}

# Mapear los números de meses a nombres de meses
data['ID_MUNICIPIO'] = data['ID_MUNICIPIO'].map(municipio_dict)


# Configurar estilo de los gráficos
sns.set(style="whitegrid")

# Gráfico lineal para MES vs PORCENTAJE_DISPONIBILIDAD sin banda de confianza
plt.figure(figsize=(12, 6))
sns.lineplot(data=data, x='ID_MUNICIPIO', y='PORCENTAJE_DISPONIBILIDAD', marker='o', palette='viridis', ci=None)
plt.title('MUNICIPIO vs PORCENTAJE_DISPONIBILIDAD')
plt.xlabel('Municipio')
plt.ylabel('Porcentaje de Disponibilidad')
plt.xticks(rotation=45)  # Rotar etiquetas de los meses para mejor legibilidad
plt.ylim(98.3, 98.4)
plt.show()