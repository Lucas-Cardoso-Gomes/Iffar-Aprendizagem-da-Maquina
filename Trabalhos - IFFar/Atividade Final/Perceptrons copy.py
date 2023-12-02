import numpy as np
import pandas as pd
import tensorflow as tf
import os
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt

os.system("cls")

cancer = load_breast_cancer()
x, y = cancer.data, cancer.target

x_treino, x_teste, y_treino, y_teste = train_test_split(x, y, test_size=0.2, random_state=75)

scaler = StandardScaler()
x_treino = scaler.fit_transform(x_treino)
x_teste = scaler.transform(x_teste)

model = MLPClassifier(hidden_layer_sizes=(30, 250), max_iter=5000, random_state=75, verbose=True)

train_loss = []
test_accuracy = []

for i in range(1, 100):
    model.partial_fit(x_treino, y_treino, classes=np.unique(y))
    
    if i % 10 == 0:
        train_loss.append(model.loss_)
        y_pred = model.predict(x_teste)
        accuracia = accuracy_score(y_teste, y_pred)
        test_accuracy.append(accuracia)
        print(f"Iteração {i}, Acurácia no conjunto de teste: {accuracia * 100:.2f}%")

plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(train_loss, label='Erro de Treinamento', color='blue')
plt.title('Treinamento - Perda ao longo das iterações')
plt.xlabel('Iteração')
plt.ylabel('Erro de Treinamento')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(np.arange(10, 100, 10), test_accuracy, label='Acurácia no Conjunto de Teste', color='green')
plt.title('Avaliação - Acurácia no Conjunto de Teste ao longo das iterações')
plt.xlabel('Iteração')
plt.ylabel('Acurácia no Conjunto de Teste')
plt.legend()

plt.tight_layout()
plt.show()