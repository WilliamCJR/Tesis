import time
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error
import joblib
from cuml.svm import SVR as cuSVR  

# Cargar los datos
data = pd.read_csv('DatosFiltradosDisponibilidad.csv', delimiter=';')

# Separar características y objetivo
X = data[['ANNO', 'MES', 'ID_EMPRESA', 'ID_TECNOLOGIA', 'ID_MUNICIPIO']]
y = data['PORCENTAJE_DISPONIBILIDAD']

# Usar StandardScaler para estandarizar el objetivo
scaler_y = StandardScaler()
y_scaled = scaler_y.fit_transform(y.values.reshape(-1, 1)).flatten()

# Definir el preprocesamiento para variables categóricas y numéricas
preprocessor = ColumnTransformer(
    transformers=[
        ('anno', StandardScaler(), ['ANNO']),
        ('mes', OneHotEncoder(handle_unknown='ignore'), ['MES']),
        ('id_empresa', OneHotEncoder(handle_unknown='ignore'), ['ID_EMPRESA']),
        ('id_tecnologia', OneHotEncoder(handle_unknown='ignore'), ['ID_TECNOLOGIA']),
        ('id_municipio', OneHotEncoder(handle_unknown='ignore'), ['ID_MUNICIPIO'])
    ])

# Crear el pipeline de preprocesamiento
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
])

# Transformar los datos de entrenamiento y prueba
X_transformed = pipeline.fit_transform(X)

# Dividir los datos en conjunto de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X_transformed, y_scaled, test_size=0.10, random_state=42)

# Parámetros para Grid Search
param_grid = {
    'kernel': ['linear', 'poly', 'rbf'],
    'C': [0.1, 1, 10, 100],
    'epsilon': [0.01, 0.1, 1],
    'gamma': ['scale', 'auto'],
}

# Función para realizar la búsqueda con cuML y SVR
def grid_search_cuml(X_train, y_train, param_grid):
    best_params = None
    best_mse = float('inf')
    best_model = None

    for kernel in param_grid['kernel']:
        for C in param_grid['C']:
            for epsilon in param_grid['epsilon']:
                for gamma in param_grid['gamma']:
                    print(f"Entrenando con kernel={kernel}, C={C}, epsilon={epsilon}, gamma={gamma}")
                    model = cuSVR(kernel=kernel, C=C, epsilon=epsilon, gamma=gamma)
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_train)

                    # Calcula el MSE en el conjunto de entrenamiento
                    mse = mean_squared_error(y_train, y_pred)
                    print(f"Train MSE: {mse}")

                    # Guardar el mejor modelo basado en el MSE
                    if mse < best_mse:
                        best_mse = mse
                        best_params = {'kernel': kernel, 'C': C, 'epsilon': epsilon, 'gamma': gamma}
                        best_model = model

    return best_model, best_params, best_mse

# Ejecutar la búsqueda
start_time = time.time()
best_model, best_params, best_mse = grid_search_cuml(X_train, y_train, param_grid)
end_time = time.time()

print(f"Mejores hiperparámetros: {best_params}")
print(f"Mejor MSE en conjunto de entrenamiento: {best_mse}")
print(f"Tiempo total de entrenamiento: {end_time - start_time:.2f} segundos")

# Evaluar el modelo en los datos de prueba
y_pred_test = best_model.predict(X_test)
mse_test = mean_squared_error(y_test, y_pred_test)
print(f"MSE en el conjunto de prueba: {mse_test}")
