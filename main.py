import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.linear_model import Perceptron
from google.colab import drive

# Montando o Google Drive na mesma conta do Google Colab
drive.mount('/content/drive')

# Carregando o Dataset
df = pd.read_csv("/content/drive/MyDrive/RNA_Datasets/BreastCancerWisconsinDataSet.csv")

# Explorando o Dataset
print(df.head())
print(df.tail())
print(df.info())
print(df.describe())
print(df.shape)
print(df.isnull().sum())

# Pré-Processando os Dados
df.drop(columns=['id', 'Unnamed: 32'], inplace=True)
print(df.shape)

encoder = LabelEncoder()
df['diagnosis'] = encoder.fit_transform(df['diagnosis'])

X = df.drop(columns=['diagnosis'])
y = df['diagnosis']

scaler = MinMaxScaler()
X = scaler.fit_transform(X)

# Plotando o gráfico para verificação da distribuição amostral das variáveis do dataset
X_plot = df.iloc[:,:].values  # transformando o df em ndarray para usar no scatter
plt.scatter(X_plot[:,2], X_plot[:,3], c=y)
plt.title("Distribuição do Diagnóstico por Variáveis de Entrada")
plt.xlabel('Var 1')
plt.ylabel('Var 2')
plt.show()

# Separando os Dados para Treinamento e Teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Construindo e Testando o Modelo
p = Perceptron()
p.fit(X_train, y_train)
y_pred = p.predict(X_test)

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True)
plt.show()

# Apresentação das Métricas dos Testes Validados
test_score = accuracy_score(y_pred, y_test)
print("Acurácia dos testes: ", test_score)
print(classification_report(y_pred, y_test))
print("Número de épocas no treinamento: ", p.n_iter_)
print("Lista de parâmetros configurados na Perceptron: ", p.get_params())
