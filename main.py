#Importando as Bibliotecas Necessárias
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.linear_model import Perceptron
from google.colab import drive
drive.mount('/content/drive') # Montando o Google Drive na mesma conta do Google Colab
#Carregando o Dataset
# Descrição do dataset: https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data
# Caminho do dataset no Google Drive que será carregado em df
df = pd.read_csv("/content/drive/MyDrive/RNA_Datasets/BreastCancerWisconsinDataSet.csv")
#Explorando o Dataset
# Diagnosis (M = malignant, B = benign)
df.head()
df.tail()
df.info()
df.describe()
df.shape
df.isnull().sum()
#Pré-Processando os Dados
# Excluindo as variáveis id e Unnamed, pois são irrelevantes no contexto de informações
df.drop(columns=['id', 'Unnamed: 32'], inplace=True)
df.shape

# Codificando a coluna de destino usando o codificador de rótulo - Transforma em 0 e 1 as Classes - Diagnosis (M = malignant = 1, B = benign = 0)
encoder = LabelEncoder()
df['diagnosis'] = encoder.fit_transform(df['diagnosis'])

X = df.drop(columns=['diagnosis'])
y = df['diagnosis']

# Normalizando os Dados
scaler = MinMaxScaler()
X = scaler.fit_transform(X)

# plotando o gráfico para verificação da distribuição amostral das variáveis do dataset
X = df.iloc[:,:].values # transformando o df em ndarray para usar no scatter
pl = plt
pl.scatter(X[:,2],X[:,3],c=y)
pl.title("Distribuição do Diagnóstico por Variáveis de Entrada" )
pl.xlabel('Var 1')
pl.ylabel('Var 2')
pl.show

#Separando os Dados para Treinamento e Teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

#Construindo e Testando o Modelo
p = Perceptron()
p.fit(X_train, y_train)
y_pred = p.predict(X_test)

cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True)
plt.show()

#Apresentação das Métricas dos Testes Validados
test_score = accuracy_score(y_pred, y_test)
print("Acurácia dos testes: ", test_score)

print(classification_report(y_pred, y_test))

print("Número de épocas no treinamento: ", p.n_iter_)
print("Lista de parâmetros configurados na Perceptron: ", p.get_params())


