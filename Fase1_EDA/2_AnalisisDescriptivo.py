import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Cargamos datos
data = pd.read_csv('DatosFiltradosDisponibilidad.csv', delimiter=';')

# Configurar estilo de los gráficos
sns.set(style="whitegrid")

# Gráfico de barras para ANNO
plt.figure(figsize=(12, 6))
sns.countplot(data=data, x='ANNO', palette='viridis')
plt.title('Distribución de ANNO')
plt.xlabel('Año')
plt.ylabel('Frecuencia')
plt.show()

# Gráfico de barras para MES
plt.figure(figsize=(12, 6))
sns.countplot(data=data, x='MES', palette='viridis')
plt.title('Distribución de MES')
plt.xlabel('Mes')
plt.ylabel('Frecuencia')
plt.show()

# Gráfico de barras para ID_EMPRESA
plt.figure(figsize=(12, 6))
sns.countplot(data=data, x='ID_EMPRESA', palette='viridis', order=data['ID_EMPRESA'].value_counts().index)
plt.title('Distribución de ID_EMPRESA')
plt.xlabel('Empresa')
plt.ylabel('Frecuencia')
plt.show()

# Gráfico de barras para ID_TECNOLOGIA
plt.figure(figsize=(12, 6))
sns.countplot(data=data, x='ID_TECNOLOGIA', palette='viridis')
plt.title('Distribución de ID_TECNOLOGIA')
plt.xlabel('Tecnología')
plt.ylabel('Frecuencia')
plt.show()

# Gráfico de barras para ID_MUNICIPIO
plt.figure(figsize=(12, 6))
sns.countplot(data=data, x='ID_MUNICIPIO', palette='viridis', order=data['ID_MUNICIPIO'].value_counts().index)
plt.title('Distribución de ID_MUNICIPIO')
plt.xlabel('Municipio')
plt.ylabel('Frecuencia')
plt.show()

# Gráfico de barras para PORCENTAJE_DISPONIBILIDAD
plt.figure(figsize=(12, 6))
sns.histplot(data=data, x='PORCENTAJE_DISPONIBILIDAD', bins=30, kde=True, color='blue')
plt.title('Distribución de PORCENTAJE_DISPONIBILIDAD')
plt.xlabel('Porcentaje de Disponibilidad')
plt.ylabel('Frecuencia')
plt.show()
