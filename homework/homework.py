import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.svm import SVC
from sklearn.metrics import precision_score, balanced_accuracy_score, recall_score, f1_score, confusion_matrix
import json
import gzip
import pickle
import os

# flake8: noqa: E501
#
# En este dataset se desea pronosticar el default (pago) del cliente el próximo
# mes a partir de 23 variables explicativas.
#
#   LIMIT_BAL: Monto del credito otorgado. Incluye el credito individual y el
#              credito familiar (suplementario).
#         SEX: Genero (1=male; 2=female).
#   EDUCATION: Educacion (0=N/A; 1=graduate school; 2=university; 3=high school; 4=others).
#    MARRIAGE: Estado civil (0=N/A; 1=married; 2=single; 3=others).
#         AGE: Edad (years).
#       PAY_0: Historia de pagos pasados. Estado del pago en septiembre, 2005.
#       PAY_2: Historia de pagos pasados. Estado del pago en agosto, 2005.
#       PAY_3: Historia de pagos pasados. Estado del pago en julio, 2005.
#       PAY_4: Historia de pagos pasados. Estado del pago en junio, 2005.
#       PAY_5: Historia de pagos pasados. Estado del pago en mayo, 2005.
#       PAY_6: Historia de pagos pasados. Estado del pago en abril, 2005.
#   BILL_AMT1: Historia de pagos pasados. Monto a pagar en septiembre, 2005.
#   BILL_AMT2: Historia de pagos pasados. Monto a pagar en agosto, 2005.
#   BILL_AMT3: Historia de pagos pasados. Monto a pagar en julio, 2005.
#   BILL_AMT4: Historia de pagos pasados. Monto a pagar en junio, 2005.
#   BILL_AMT5: Historia de pagos pasados. Monto a pagar en mayo, 2005.
#   BILL_AMT6: Historia de pagos pasados. Monto a pagar en abril, 2005.
#    PAY_AMT1: Historia de pagos pasados. Monto pagado en septiembre, 2005.
#    PAY_AMT2: Historia de pagos pasados. Monto pagado en agosto, 2005.
#    PAY_AMT3: Historia de pagos pasados. Monto pagado en julio, 2005.
#    PAY_AMT4: Historia de pagos pasados. Monto pagado en junio, 2005.
#    PAY_AMT5: Historia de pagos pasados. Monto pagado en mayo, 2005.
#    PAY_AMT6: Historia de pagos pasados. Monto pagado en abril, 2005.
#
# La variable "default payment next month" corresponde a la variable objetivo.
#
# El dataset ya se encuentra dividido en conjuntos de entrenamiento y prueba
# en la carpeta "files/input/".
#
# Los pasos que debe seguir para la construcción de un modelo de
# clasificación están descritos a continuación.
#
#
# Paso 1.
# Realice la limpieza de los datasets:
# - Renombre la columna "default payment next month" a "default".
# - Remueva la columna "ID".
# - Elimine los registros con informacion no disponible.
# - Para la columna EDUCATION, valores > 4 indican niveles superiores
#   de educación, agrupe estos valores en la categoría "others".
# - Renombre la columna "default payment next month" a "default"
# - Remueva la columna "ID".
#
#
# Paso 2.
# Divida los datasets en x_train, y_train, x_test, y_test.
#
#
# Paso 3.
# Cree un pipeline para el modelo de clasificación. Este pipeline debe
# contener las siguientes capas:
# - Transforma las variables categoricas usando el método
#   one-hot-encoding.
# - Descompone la matriz de entrada usando PCA. El PCA usa todas las componentes.
# - Estandariza la matriz de entrada.
# - Selecciona las K columnas mas relevantes de la matrix de entrada.
# - Ajusta una maquina de vectores de soporte (svm).
#
#
# Paso 4.
# Optimice los hiperparametros del pipeline usando validación cruzada.
# Use 10 splits para la validación cruzada. Use la función de precision
# balanceada para medir la precisión del modelo.
#
#
# Paso 5.
# Guarde el modelo (comprimido con gzip) como "files/models/model.pkl.gz".
# Recuerde que es posible guardar el modelo comprimido usanzo la libreria gzip.
#
#
# Paso 6.
# Calcule las metricas de precision, precision balanceada, recall,
# y f1-score para los conjuntos de entrenamiento y prueba.
# Guardelas en el archivo files/output/metrics.json. Cada fila
# del archivo es un diccionario con las metricas de un modelo.
# Este diccionario tiene un campo para indicar si es el conjunto
# de entrenamiento o prueba. Por ejemplo:
#
# {'dataset': 'train', 'precision': 0.8, 'balanced_accuracy': 0.7, 'recall': 0.9, 'f1_score': 0.85}
# {'dataset': 'test', 'precision': 0.7, 'balanced_accuracy': 0.6, 'recall': 0.8, 'f1_score': 0.75}
#
#
# Paso 7.
# Calcule las matrices de confusion para los conjuntos de entrenamiento y
# prueba. Guardelas en el archivo files/output/metrics.json. Cada fila
# del archivo es un diccionario con las metricas de un modelo.
# de entrenamiento o prueba. Por ejemplo:
#
# {'type': 'cm_matrix', 'dataset': 'train', 'true_0': {"predicted_0": 15562, "predicte_1": 666}, 'true_1': {"predicted_0": 3333, "predicted_1": 1444}}
# {'type': 'cm_matrix', 'dataset': 'test', 'true_0': {"predicted_0": 15562, "predicte_1": 650}, 'true_1': {"predicted_0": 2490, "predicted_1": 1420}}
#


# Paso 1: Cargar y limpiar los datos
train_data = pd.read_csv('../files/input/train_data.csv.zip')
test_data = pd.read_csv('../files/input/test_data.csv.zip')

def clean_data(df):
    df = df.rename(columns={'default payment next month': 'default'})
    df = df.drop(columns=['ID'])
    df = df.dropna()
    df = df.iloc[df[(df['EDUCATION'] != 0) & (df['MARRIAGE'] != 0)].index]  
    df['EDUCATION'] = df['EDUCATION'].apply(lambda x: 4 if x > 4 else x)
    return df

train_data = clean_data(train_data)
test_data = clean_data(test_data)

# Paso 2: Dividir los datasets en x_train, y_train, x_test, y_test
x_train = train_data.drop(columns=['default'])
y_train = train_data['default']
x_test = test_data.drop(columns=['default'])
y_test = test_data['default']

# Paso 3: Crear el pipeline

categorical_features = ['SEX', 'EDUCATION', 'MARRIAGE']
numeric_features = [col for col in x_train.columns if col not in categorical_features]

#preprocesador
preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ("num", StandardScaler(), numeric_features)
    ]
)
pipeline = Pipeline([
    ("preprocessor", preprocessor),   
    ('pca', PCA()),
    ('selectkbest', SelectKBest(score_func=f_classif)),
    ('svc', SVC(random_state=42))
])

# Paso 4: Optimizar hiperparámetros
param_grid = {
        'pca__n_components': [20, x_train.shape[1]-2],
        'selectkbest__k': [12],
        'svc__kernel': ['rbf'],
        'svc__gamma': [0.1]
        
    }
grid_search = GridSearchCV(pipeline, param_grid, cv=10, scoring='balanced_accuracy',n_jobs=-1,
    refit=True)

    
grid_search.fit(x_train, y_train)

# Guardar el modelo
output_dir = '../files/models'
os.makedirs(output_dir, exist_ok=True)
with gzip.open('../files/models/model.pkl.gz', 'wb') as f:
    pickle.dump(grid_search, f)

# Paso 6: Calcular métricas
def calculate_metrics(model, x, y, dataset_type):
    y_pred = model.predict(x)
    metrics = {
        'type': 'metrics',
        'dataset': dataset_type,
        'precision': precision_score(y, y_pred, zero_division=0),
        'balanced_accuracy': balanced_accuracy_score(y, y_pred),
        'recall': recall_score(y, y_pred, zero_division=0),
        'f1_score': f1_score(y, y_pred, zero_division=0)
    }
    return metrics

train_metrics = calculate_metrics(grid_search.best_estimator_, x_train, y_train, 'train')
test_metrics = calculate_metrics(grid_search.best_estimator_, x_test, y_test, 'test')

# Guardar métricas
output_dir = '../files/output'
os.makedirs(output_dir, exist_ok=True)
metrics = [train_metrics, test_metrics]
with open('../files/output/metrics.json', 'w') as f:
    for metric in metrics:
        f.write(json.dumps(metric) + '\n')

# Paso 7: Calcular matrices de confusión
def calculate_confusion_matrix(model, x, y, dataset_type):
    y_pred = model.predict(x)
    cm = confusion_matrix(y, y_pred)
    cm_dict = {
        'type': 'cm_matrix',
        'dataset': dataset_type,
        'true_0': {'predicted_0': int(cm[0, 0]), 'predicted_1': int(cm[0, 1])},
        'true_1': {'predicted_0': int(cm[1, 0]), 'predicted_1': int(cm[1, 1])}
    }
    return cm_dict

train_cm = calculate_confusion_matrix(grid_search.best_estimator_, x_train, y_train, 'train')
test_cm = calculate_confusion_matrix(grid_search.best_estimator_, x_test, y_test, 'test')

# Guardar matrices de confusión
metrics_extend = [train_cm, test_cm]
with open('../files/output/metrics.json', 'a') as f:
    for metric in metrics_extend:
        f.write(json.dumps(metric) + '\n')