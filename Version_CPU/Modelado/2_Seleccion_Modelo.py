import time
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error
import joblib

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

# Crear el pipeline de modelo
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('svr', SVR())
])

# Definir los parametros para grid search
param_grid = {
    'svr__kernel': ['linear', 'poly', 'rbf'],
    'svr__C': [0.1, 1, 10, 100],
    'svr__epsilon': [0.01, 0.1, 1],
    'svr__gamma': ['scale', 'auto'],
}

# Dividir los datos en conjunto de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y_scaled, test_size=0.90, random_state=42)

# Implementar GridSearchCV
grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)

# Entrenar el modelo
grid_search.fit(X_train, y_train)

# Imprimir los mejores parámetros encontrados por GridSearchCV
print("Mejores hiperparámetros:", grid_search.best_params_)

# Evaluar el modelo en los datos de prueba
y_pred = grid_search.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print("MSE en el conjunto de prueba:", mse)

# Crear un DataFrame con los resultados del GridSearchCV
results = pd.DataFrame(grid_search.cv_results_)

# Imprimir los resultados de la búsqueda
print(results[['param_svr__kernel', 'param_svr__C', 'param_svr__epsilon', 'param_svr__gamma', 'mean_test_score']])
