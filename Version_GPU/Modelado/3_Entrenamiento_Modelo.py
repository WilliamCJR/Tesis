import time
import numpy as np
import pandas as pd
import pycuda.autoinit
import pycuda.driver as cuda
from pycuda import gpuarray
import pycuda.compiler as compiler
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import joblib

# Codigo del kernel de normalizacion
normalize_kernel_code = """
__global__ void normalize(float *data, float *mean, float *std_dev, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n) {
        data[idx] = (data[idx] - mean[0]) / std_dev[0];
    }
}
"""

# Cargar los datos
data = pd.read_csv('DatosFiltradosDisponibilidad.csv', delimiter=';')

# Separar características y variable objetivo
X = data[['ANNO', 'MES', 'ID_EMPRESA', 'ID_TECNOLOGIA', 'ID_MUNICIPIO']]
y = data['PORCENTAJE_DISPONIBILIDAD'].values

# Obtener la media y desviación estándar de la columna 'ANNO'
mean = X['ANNO'].mean()
std_dev = X['ANNO'].std()

# Convertir la columna 'ANNO' a un array de tipo float32
X_anno = X['ANNO'].values.astype(np.float32)

# Copiar datos a la GPU
X_anno_gpu = gpuarray.to_gpu(X_anno)
mean_gpu = gpuarray.to_gpu(np.array([mean], dtype=np.float32))
std_dev_gpu = gpuarray.to_gpu(np.array([std_dev], dtype=np.float32))

# Compilar el kernel
normalize_kernel = compiler.SourceModule(normalize_kernel_code)
normalize_function = normalize_kernel.get_function("normalize")

# Ejecutar el kernel de normalización
n = X_anno_gpu.size
block_size = 256
grid_size = (n + block_size - 1) // block_size
normalize_function(X_anno_gpu, mean_gpu, std_dev_gpu, np.int32(n), block=(block_size, 1, 1), grid=(grid_size, 1))

# Obtener los datos normalizados
X['ANNO'] = X_anno_gpu.get()

# Definir el preprocesamiento para variables categóricas y numéricas
preprocessor = ColumnTransformer(
    transformers=[
        ('anno', 'passthrough', ['ANNO']),
        ('mes', OneHotEncoder(handle_unknown='ignore'), ['MES']),
        ('id_empresa', OneHotEncoder(handle_unknown='ignore'), ['ID_EMPRESA']),
        ('id_tecnologia', OneHotEncoder(handle_unknown='ignore'), ['ID_TECNOLOGIA']),
        ('id_municipio', OneHotEncoder(handle_unknown='ignore'), ['ID_MUNICIPIO'])
    ])

# Crear el pipeline con un modelo de kernel polinomial
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', SVR(kernel='poly', degree=3, C=10, epsilon=0.01, coef0=1, gamma='scale'))
])

# Dividir los datos en conjuntos de entrenamiento 80% y prueba 20%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

# Medir el tiempo de entrenamiento
print("Inicia entrenamiento")
start_train_time = time.time()
pipeline.fit(X_train, y_train)
train_time = time.time() - start_train_time
print("Finaliza entrenamiento")

# Medir el tiempo de predicción
start_pred_time = time.time()
y_pred_scaled = pipeline.predict(X_test)
prediction_time = time.time() - start_pred_time

# Desnormalizar las predicciones para evaluar el modelo
# Nota: Aquí asumimos que y no fue normalizado. Si lo haces, aplica scaler_y.inverse_transform.
y_pred = y_pred_scaled

# Evaluar el modelo en el conjunto de prueba
mse = mean_squared_error(y_test, y_pred)

print(f"Error cuadrático medio: {mse:.2f}")
print(f"Tiempo de entrenamiento: {train_time:.2f} segundos")
print(f"Tiempo de predicción: {prediction_time:.2f} segundos")

# Guardar el modelo
joblib.dump(pipeline, 'V1_Modelo.pkl')

# Crear un DataFrame con las características de prueba, los valores reales y las predicciones
results_df = X_test.copy()
results_df['PORCENTAJE_DISPONIBILIDAD_REAL'] = y_test
results_df['PORCENTAJE_DISPONIBILIDAD_PREDICHO'] = y_pred

# Guardar el DataFrame en un archivo CSV
results_df.to_csv('EvaluacionModelo.csv', index=False, float_format='%.2f')
