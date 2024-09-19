import time
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import joblib

# Cargar los datos
data = pd.read_csv('DatosFiltradosDisponibilidad.csv', delimiter=';')

# Separar caracteristicas y variable objetivo
X = data[['ANNO', 'MES', 'ID_EMPRESA', 'ID_TECNOLOGIA', 'ID_MUNICIPIO']]
y = data['PORCENTAJE_DISPONIBILIDAD']

# Usar StandardScaler para normalizar el objetivo
scaler_y = StandardScaler()
y_scaled = scaler_y.fit_transform(y.values.reshape(-1, 1)).flatten()

# Obtener la media y desviación estándar
mean_y = scaler_y.mean_[0]
std_y = scaler_y.scale_[0]

print(f"Media: {mean_y}, Desviación estándar: {std_y}")
print("Datos escalados: ", y_scaled[:10])

# Definir el preprocesamiento para variables categóricas y numéricas
preprocessor = ColumnTransformer(
    transformers=[
        ('anno', StandardScaler(), ['ANNO']),
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
X_train, X_test, y_train, y_test = train_test_split(X, y_scaled, test_size=0.20, random_state=42)

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

# Redondear el tiempo de predicción a dos decimales
prediction_time = round(prediction_time, 2)

# Desnormalizar las predicciones para evaluar el modelo
y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()

# Evaluar el modelo en el conjunto de prueba
y_test = scaler_y.inverse_transform(y_test.reshape(-1, 1)).flatten()

# Redondear valores
y_pred = y_pred.round(2)
y_test = y_test.round(2)

# Calcular el error cuadrático medio
mse = mean_squared_error(y_test, y_pred)

print(f"Error cuadrático medio: {mse:.2f}")
print(f"Tiempo de entrenamiento: {train_time:.2f} segundos")
print(f"Tiempo de predicción: {prediction_time} segundos")

# Guardar el modelo
joblib.dump(pipeline, 'V1_Modelo.pkl')

# Crear un DataFrame con las características de prueba, los valores reales y las predicciones
results_df = X_test.copy()
results_df['PORCENTAJE_DISPONIBILIDAD_REAL'] = y_test
results_df['PORCENTAJE_DISPONIBILIDAD_PREDICHO'] = y_pred

# Guardar el DataFrame en un archivo CSV
results_df.to_csv('EvaluacionModelo.csv', index=False, float_format='%.2f')

# Gráfico de tiempos de predicción
plt.bar(['Tiempo entrenamiento', 'Tiempo prediccion'], [train_time, prediction_time])
plt.xlabel('Valor')
plt.ylabel('T (segundos)')
plt.title('Tiempo de entrenamiento y predicción del modelo')
plt.show()
