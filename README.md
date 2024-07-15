# Classificação de Câncer de Mama

Este projeto é focado em classificar tumores de câncer de mama como malignos ou benignos usando técnicas de aprendizado de máquina. O dataset utilizado é o Breast Cancer Wisconsin disponível no Kaggle.

## Índice

- [Visão Geral](#visão-geral)

- [Dataset](#dataset)

- [Instalação](#instalação)

- [Uso](#uso)

- [Resultados](#resultados)

- [Contribuindo](#contribuindo)

- [Licença](#licença)

- [Agradecimentos](#agradecimentos)

## Visão Geral

O câncer de mama é uma doença comum e potencialmente mortal. O diagnóstico precoce é crucial para um tratamento eficaz. Este projeto utiliza aprendizado de máquina para classificar tumores de câncer de mama com base em várias características extraídas de imagens.

## Dataset

O dataset utilizado neste projeto é o [Breast Cancer Wisconsin dataset](https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data) disponível no Kaggle. Ele contém características computadas a partir de imagens de câncer de mama, incluindo média, erro padrão e piores (maiores) valores de várias medições (por exemplo, raio, textura, perímetro, área, suavidade).

### Informações do Dataset:

- **Características:** 30 características numéricas

- **Alvo:** Binário (maligno ou benigno)

- **Número de Instâncias:** 569

## Instalação

Para executar este projeto, você precisa ter Python e as seguintes bibliotecas instaladas:

- numpy

- pandas

- matplotlib

- seaborn

- scikit-learn

- google.colab (se usar o Google Colab)

Você pode instalar essas bibliotecas usando pip:

`pip install numpy pandas matplotlib seaborn scikit-learn`

Uso

---

### Clone o repositório:

`git clone https://github.com/seu-usuario/breast-cancer-classification.git

cd breast-cancer-classification`

### Execute o script Python:

Certifique-se de que você tem o arquivo do dataset no caminho correto (`/content/drive/MyDrive/RNA_Datasets/BreastCancerWisconsinDataSet.csv`). Você pode ajustar o caminho no script, se necessário.

`python main.py`

### Google Colab:

Se você estiver usando o Google Colab, certifique-se de que seu Google Drive está montado corretamente e o dataset está no diretório certo.

Resultados

----------

Após executar o script, o modelo irá apresentar os seguintes resultados:

-   Matriz de Confusão

-   Pontuação de Acurácia

-   Relatório de Classificação (Precisão, Recall, F1-Score)

-   Número de épocas treinadas

-   Parâmetros configurados no Perceptron

### Exemplo de Saída:

Acurácia dos testes: 0.956

`precision    recall  f1-score   support
 0       0.96      0.98      0.97       100
       1       0.95      0.93      0.94        71
accuracy                           0.96       171`

macro avg 0.96 0.95 0.96 171 weighted avg 0.96 0.96 0.96 171

Número de épocas no treinamento: 10

Lista de parâmetros configurados na Perceptron: {'penalty': None, 'alpha': 0.0001, 'fit_intercept': True, 'max_iter': 1000, 'tol': 0.001, 'shuffle': True, 'eta0': 1.0, 'n_jobs': None, 'random_state': 42, 'early_stopping': False, 'validation_fraction': 0.1, 'n_iter_no_change': 5, 'class_weight': None, 'warm_start': False}

Contribuindo

------------

Contribuições são bem-vindas! Se você tiver qualquer ideia ou melhoria, sinta-se à vontade para abrir uma issue ou enviar um pull request.

Agradecimentos

--------------

-   [Kaggle](https://www.kaggle.com/) por fornecer o dataset.

-   A biblioteca scikit-learn por fornecer ferramentas de aprendizado de máquina fáceis de usar.

-   A plataforma Google Colab por fornecer um excelente ambiente para executar e compartilhar projetos de aprendizado de máquina.
