import tensorflow as tf
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
from tensorflow.keras.utils import plot_model
import os

os.system("cls")

cancer = load_breast_cancer()
x, y = cancer.data, cancer.target

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=75)

scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(x_train.shape[1],)),
    tf.keras.layers.Dense(200, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.fit(x_train_scaled, y_train, epochs=50, batch_size=64, validation_data=(x_test_scaled, y_test), verbose=False)

accuracy = model.evaluate(x_test_scaled, y_test)[1]
print(f'A precisão do modelo no conjunto de teste é de {accuracy * 100:.2f}%')

plot_model(model, to_file='ordem-rede-neural.png', show_shapes=True, show_layer_names=True)