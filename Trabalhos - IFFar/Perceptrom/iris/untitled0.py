import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt

# Carregando o conjunto de dados Iris
iris = load_iris()
x = iris.data
y = iris.target

# Dividindo o conjunto de dados em treino e teste
x_treino, x_teste, y_treino, y_teste = train_test_split(x, y, test_size=0.2, random_state=42)

# Normalizando os dados
scaler = StandardScaler()
x_treino = scaler.fit_transform(x_treino)
x_teste = scaler.transform(x_teste)

# Criando e treinando o modelo
model = MLPClassifier(hidden_layer_sizes=(5, 5), max_iter=1000, random_state=42, verbose=False)
model.fit(x_treino, y_treino)

# Realizando previsões no conjunto de teste
y_pred = model.predict(x_teste)

# Calculando e exibindo a acurácia
accuracy = accuracy_score(y_teste, y_pred)
print(f"Acurácia no conjunto de teste: {accuracy * 100:.0f}%")