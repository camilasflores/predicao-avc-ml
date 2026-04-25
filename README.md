# 🧠 Predição de AVC com Machine Learning

Este projeto tem como objetivo prever a ocorrência de AVC (Acidente Vascular Cerebral) com base em dados demográficos e clínicos, utilizando técnicas de Machine Learning.

A abordagem prioriza a **detecção de casos positivos (AVC)**, considerando o forte desbalanceamento do dataset.

---

## 🎯 Objetivo

Desenvolver um modelo capaz de identificar indivíduos com maior risco de AVC, com foco em:

- Maximizar o **recall (sensibilidade)**  
- Reduzir falsos negativos (casos de AVC não detectados)  
- Controlar o volume de falsos positivos  
- Ajustar o modelo de acordo com o objetivo do problema  

---

## 📊 Dataset

Foi utilizado o dataset **Healthcare Stroke Dataset**, contendo variáveis como:

- Idade (`age`)
- Gênero (`gender`)
- Tipo de trabalho (`work_type`)
- Tipo de residência (`Residence_type`)
- Estado civil (`ever_married`)
- Nível médio de glicose (`avg_glucose_level`)
- IMC (`bmi`)
- Hipertensão (`hypertension`)
- Doença cardíaca (`heart_disease`)
- Status de fumante (`smoking_status`)

### Variável alvo:
- `stroke`  
  - 0 → Não teve AVC  
  - 1 → Teve AVC  

---

## ⚙️ Etapas do Projeto

### 1. Análise Exploratória (EDA)
- Análise de distribuição das variáveis
- Identificação de padrões e hipóteses
- Relação entre fatores de risco e ocorrência de AVC

### 2. Pré-processamento
- Tratamento de valores ausentes (mediana e moda)
- Padronização de variáveis numéricas
- Codificação de variáveis categóricas (One-Hot Encoding)

### 3. Balanceamento de dados
- Uso de **SMOTE** para tratar o desbalanceamento da variável alvo

### 4. Modelagem
Foram testados diferentes modelos:

- Regressão Logística ✅ (principal)
- Random Forest (comparação)
- Gradient Boosting (comparação)

### 5. Feature Engineering
- Criação da variável `risk_count` (combinação de fatores de risco)
- Avaliação do impacto da feature no modelo

### 6. Ajuste de Threshold
- Definição de um **recall mínimo desejado de 0.7**
- Seleção do threshold que atinge esse recall com a **melhor precisão possível**
- Avaliação do trade-off entre recall e precision

### 7. Avaliação
- Precision
- Recall
- F1-score
- ROC AUC
- Matriz de confusão

---

## 📈 Resultados

### 🔹 Regressão Logística (Threshold padrão 0.5)
- Alto **recall (~0.80)** → boa detecção de AVC  
- Baixa precisão → maior número de falsos positivos  

### 🔹 Regressão Logística (Threshold ajustado)
Foi aplicado um ajuste de threshold com o objetivo de garantir um **recall mínimo de 0.7**.

Entre os thresholds possíveis, foi escolhido aquele que apresentou a melhor precisão mantendo esse nível de recall.

👉 Resultado:
- Redução de falsos positivos  
- Manutenção da capacidade de detectar casos de AVC  
- Melhor equilíbrio entre precision e recall  

---

## 🧠 Principais Insights

- A **acurácia não é adequada** para avaliar modelos com dados desbalanceados  
- O **recall é a métrica mais importante** em problemas de saúde  
- O uso de **SMOTE foi essencial** para melhorar a detecção de AVC  
- A feature `risk_count` **não trouxe ganho significativo**, indicando que o modelo já captura essas relações  
- O ajuste de threshold permite adaptar o modelo ao objetivo do problema  
- Modelos mais complexos (como Random Forest) não necessariamente performam melhor  

---

## 📁 Estrutura do Projeto
predicao-avc-ml/
├── dataset/
├── notebooks/
├── src/
│ ├── preprocessing.py
│ ├── features.py
│ ├── model.py
│ └── models/
└── README.md


---

## ▶️ Como Executar

1. Clone o repositório:
git clone https://github.com/camilasflores/predicao-avc-ml.git


2. Acesse a pasta:
cd predicao-avc-ml


3. Instale as dependências:
pip install pandas numpy scikit-learn imbalanced-learn matplotlib seaborn

4. Execute o modelo:
python -m src.model


---

## 🛠️ Tecnologias Utilizadas

- Python
- Pandas
- NumPy
- Scikit-learn
- Imbalanced-learn (SMOTE)
- Matplotlib / Seaborn

---

## 🏆 Conclusão

A **Regressão Logística com SMOTE e ajuste de threshold baseado em recall mínimo (0.7)** se mostrou a abordagem mais adequada para o problema.

Essa estratégia permitiu:

- manter foco na detecção de casos de AVC  
- reduzir falsos negativos  
- controlar melhor o número de falsos positivos  

O projeto reforça a importância de alinhar o modelo ao objetivo do problema, especialmente em cenários críticos como saúde.

---

## 👩‍💻 Autora

Camila Flores  
🔗 https://github.com/camilasflores