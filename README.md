# 📊 Churn Prediction com LGBM e RegLog

Este projeto utiliza um modelo de aprendizado de máquina para prever a evasão de clientes (churn) com o algoritmo **LightGBM** e **LogisticRegression**. Todo o fluxo inclui pipeline de pré-processamento, tuning do threshold de decisão e análise interpretativa com SHAP (Shapley Additive Explanations).

## 🔧 Tecnologias Utilizadas

- Python 3.10
- Scikit-learn
- LightGBM
- Pandas / NumPy
- SHAP
- Matplotlib / Seaborn

##🧠 Objetivos do Desafio

✅Preparar os dados para a modelagem (tratamento, encoding, normalização).

✅Realizar análise de correlação e seleção de variáveis.

✅Treinar dois ou mais modelos de classificação.

✅Avaliar o desempenho dos modelos com métricas.

✅Interpretar os resultados, incluindo a importância das variáveis.

✅Criar uma conclusão estratégica apontando os principais fatores que influenciam a evasão.



## 🧪 Etapas do Projeto

### 1. Pré-processamento

Foi utilizado o `ColumnTransformer` e `Pipeline` do Scikit-learn para tratamento dos dados:
- Codificação de variáveis categóricas (`OneHotEncoder`)
- Escalonamento de variáveis numéricas (`StandardScaler`)
- Separação entre treino, teste e validação

### 2. Modelagem

Utilizamos o **LightGBMClassifier** como modelo final, com validação cruzada para hiperparâmetros e tuning do threshold de decisão:
- Ajuste de `threshold` com base na maximização do **KS (Kolmogorov-Smirnov)**
- Avaliação por métricas como AUC, Precision, Recall e F1-Score
Além do LGBM, utilizamos o **LogisticRegression** que obteve performance bem parecida ao LGBM.
### 3. Interpretação com SHAP

Análise dos impactos de cada variável na predição de churn usando SHAP:
- Variáveis com maior impacto positivo (aumentam chance de churn);
- Variáveis com maior impacto negativo (reduzem chance de churn);
- Visualização das variáveis importantes: (`shap.plots.bar(shap_values, max_display = 20)`);
- Visualização impacto das variáveis: (`shap.plots.beeswarm(shap_values, max_display=20)`).

### 📈 Principais Variáveis Identificadas pelo SHAP

| Variável                                 | Impacto                                                   |
|------------------------------------------|------------------------------------------------------------|
| `cat__contrato_Month-to-month`          | Aumenta significativamente o churn                        |
| `num__tempo_de_contrato`                | Reduz churn quanto maior o tempo de contrato              |
| `num__total_servicos_mes`               | Serviços intermediários aumentam churn                   |
| `cat__servico_internet_Fiber optic`     | Aumenta churn                                              |
| `cat__contrato_Two year`                | Reduz churn de forma expressiva                           |
| `cat__metodo_pagamento_Electronic check`| Aumenta churn                                              |
| `cat__servico_internet_No`              | Reduz churn (usuários sem internet permanecem mais)       |
| `cat__metodo_pagamento_Credit card...`  | Reduz churn (pagamento automático é bom sinal)            |
| `cat__metodo_pagamento_Bank transfer...`| Reduz churn                                                |

## 📌 Conclusões

- Contratos mensais e serviços de internet via fibra estão fortemente associados à evasão de clientes.
- Métodos de pagamento automáticos e contratos de longa duração (2 anos) reduzem significativamente o risco de churn.
- O modelo apresentou desempenho estável entre treino, teste e validação com KS acima de 0.48.

## 🙋‍♀️ Autora

Leticia Maia 

---
