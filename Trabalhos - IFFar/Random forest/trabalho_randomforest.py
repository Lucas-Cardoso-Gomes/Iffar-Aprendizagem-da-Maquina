import numpy as np # biblioteca para funções matematicas complexas
import pandas as pd # Biblioteca para realizar analise e consulta em arquivos
import matplotlib.pyplot as plt # Biblioteca para criação de graficos
import graphviz # Biblioteca para gricação de graficos usando arquivos DOT
import os # Bibliotecas para interagir com o OS

#Bibliotecas para ramdomflorest / treiro / acuracia / aprendizagem de maquina
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.tree import export_graphviz
from sklearn.utils import resample

base_credit = pd.read_csv('//shadowserver/dados/2. Trabalhos/Code/Python/0. Trabalhos/Random forest/credit_data.csv')

base_credit

# Exibe linhas onde age é invalido
base_credit.loc[pd.isnull(base_credit['age'])]

# Preenche campos age onde dados são nulos
base_credit['age'].fillna(base_credit['age'].mean(), inplace = True)
base_credit.loc[pd.isnull(base_credit['age'])]

# Exibe age negativos
base_credit[base_credit['age'] < 0]

# Torna os valores negativos em positivos
base_credit['age'] = base_credit['age'].abs()
base_credit[base_credit['age'] < 0]

"""##################################################


Termino do pré-processamento


##################################################"""

# Realiza a separação das colunas
X = base_credit[['income', 'age', 'loan']]
y = base_credit['default']

# Define os dados - dados X tem resultado y - test_size= quantidade de dados que serão usados para treinar = 90% - ramdom_state= é o Seed/Semente para geração das arvores
X_treino, X_teste, y_treino, y_teste = train_test_split(X, y, test_size=0.1, random_state=25)

#Cria as asvores em RandomForest, total de arvores= 500 - seed/semente: 25
rf_classifier = RandomForestClassifier(n_estimators=500, random_state=25)

#Realiza o treino
rf_classifier.fit(X_treino, y_treino)

#Realiza previsões dos resultados
y_previsao = rf_classifier.predict(X_teste)

#Realiza o teste da acuracia do algorito com base dos dados anteriormente importados e treinados
accuracy = accuracy_score(y_teste, y_previsao)*100
print ("Acurácia foi de: {:.2f}%".format(accuracy))

print("Relatório de Classificação:")
print(classification_report(y_teste, y_previsao))

# Criação do modelo bootstrap para verificar variabilidade do sistema
n_bootstraps = 1

bootstrap_accuracies = []

for _ in range(n_bootstraps):
    X_bootstrap, y_bootstrap = resample(X_treino, y_treino, random_state=25)

    rf_classifier = RandomForestClassifier(n_estimators=500, random_state=25)
    rf_classifier.fit(X_bootstrap, y_bootstrap)

    y_previsao = rf_classifier.predict(X_teste)

    accuracy = accuracy_score(y_teste, y_previsao)
    bootstrap_accuracies.append(accuracy)

# Calcule a média da acuracias do bootstrap
mean_accuracy = np.mean(bootstrap_accuracies)
std_accuracy = np.std(bootstrap_accuracies)

print("Média da Acurácia das Amostras Bootstrap: {:.2f}%".format(mean_accuracy * 100))
print("Desvio Padrão da Acurácia das Amostras Bootstrap: {:.2f}".format(std_accuracy))

# Realiza apresentação da acuracia e telatório da classificação pós bootstrap
accuracy = accuracy_score(y_teste, y_previsao)*100
print ("Acurácia foi de: {:.2f}%".format(accuracy))

print("Relatório de Classificação:")
print(classification_report(y_teste, y_previsao))

output_dir = '//shadowserver/dados/2. Trabalhos/Code/Python/0. Trabalhos/Random forest/trees'
os.makedirs(output_dir, exist_ok=True)

#salva primeira e ultima arvores no drive
dot_data = export_graphviz(rf_classifier.estimators_[0], out_file=None,
                           feature_names=X.columns,
                           filled=True, rounded=True,
                           special_characters=True, class_names=["0", "1"],
                           proportion=True)

graph = graphviz.Source(dot_data)
filename = os.path.join(output_dir, 'tree_first')
graph.render(filename, format='png', cleanup=True)

dot_data = export_graphviz(rf_classifier.estimators_[-1], out_file=None,
                           feature_names=X.columns,
                           filled=True, rounded=True,
                           special_characters=True, class_names=["0", "1"],
                           proportion=True)

graph = graphviz.Source(dot_data)
filename = os.path.join(output_dir, 'tree_last')
graph.render(filename, format='png', cleanup=True)

"""##################################################

Termino da criação/treino da arvore de decisões

##################################################"""


# Prova real, tem que ser: 0
Prova1 = rf_classifier.predict([[66155.925095,59.017015,8106.532131]])
print(f"Previsões: {Prova1}")

# Prova real, tem que ser: 1
Prova2 = rf_classifier.predict([[66952.688845,18.584336,8770.099235]])
print(f"Previsões: {Prova2}")

# Prova real, tem que ser: 0
Prova3 = rf_classifier.predict([[69516.127573,23.162104,3503.176156]])
print(f"Previsões: {Prova3}")

# Prova real, tem que ser: 1
Prova4 = rf_classifier.predict([[44311.449262,28.017167,5522.786693]])
print(f"Previsões: {Prova4}")

# Prova real, tem que ser: 0
Prova5 = rf_classifier.predict([[69436.579552,56.152617,7378.833599]])
print(f"Previsões: {Prova5}")

# Teste - provavel: 0
Teste1 = rf_classifier.predict([[70000, 35, 8000]])
print(f"Previsões: {Teste1}")

# Teste - provavel: 1
Teste2 = rf_classifier.predict([[70000, 34, 8000]])
print(f"Previsões: {Teste2}")

# Resultados dos testes
nomes_testes = ['Prova1', 'Prova2', 'Prova3', 'Prova4', 'Prova5', 'Teste1', 'Teste2']
resultados = [Prova1[0], Prova2[0], Prova3[0], Prova4[0], Prova5[0], Teste1[0], Teste2[0]]

# Crie um gráfico de barras
plt.bar(nomes_testes, resultados)
plt.xlabel('0 = Aprovado / 1 = Recusado')
plt.ylabel('Resultado')
plt.title('Resultados dos Testes')

plt.show()