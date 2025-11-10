# 📊 Análise dos Datasets - Projeto DDoS Data Analytics

## 📋 Resumo Executivo

Você possui dois datasets de ataques DDoS para análise:

| Aspecto | Dataset Balanceado | Dataset Desbalanceado |
|--------|-------------------|----------------------|
| **Arquivo** | `ddos_balanced/final_dataset.csv` | `ddos_imbalanced/unbalaced_20_80_dataset.csv` |
| **Tamanho** | 6.48 GB | 3.93 GB |
| **Linhas** | ~14.7 milhões (estimado) | ~7.6 milhões |
| **Colunas** | 85 | 85 (mesmas) |
| **Labels** | Apenas 'ddos' (necessário validar) | 'ddos' (~20%) + 'Benign' (~80%) |
| **Objetivo** | Análise com distribuição equilibrada | Análise com proporção realista |

---

## 🏗️ Estrutura dos Dados

### 1. **Colunas de Identificação do Fluxo** (7 colunas)
```
Flow ID, Src IP, Src Port, Dst IP, Dst Port, Protocol, Timestamp
```
- **Uso**: Identificar e agrupar pacotes relacionados
- **Relevância para análise**: Não usar diretamente em ML (exceto agregar)

### 2. **Características de Duração e Taxa** (4 colunas)
```
Flow Duration, Flow Byts/s, Flow Pkts/s, Flow IAT Mean/Std/Max/Min
```
- **Insight**: DDoS geralmente têm padrões de taxa muito elevados
- **Esperar em análise**: Picos anormais de velocidade

### 3. **Características de Pacotes Forward (Fwd)** (~15 colunas)
```
Tot Fwd Pkts, TotLen Fwd Pkts, Fwd Pkt Len Max/Min/Mean/Std
Fwd IAT Tot/Mean/Std/Max/Min, Fwd PSH Flags, Fwd URG Flags
Fwd Header Len, Fwd Pkts/s
```
- **Insight**: Padrão de envio pode diferir em ataques

### 4. **Características de Pacotes Backward (Bwd)** (~15 colunas)
```
Tot Bwd Pkts, TotLen Bwd Pkts, Bwd Pkt Len Max/Min/Mean/Std
Bwd IAT Tot/Mean/Std/Max/Min, Bwd PSH Flags, Bwd URG Flags
Bwd Header Len, Bwd Pkts/s
```
- **Insight**: Respostas podem ser bloqueadas/reduzidas em DDoS

### 5. **Flags TCP** (8 colunas)
```
FIN Flag Cnt, SYN Flag Cnt, RST Flag Cnt, PSH Flag Cnt
ACK Flag Cnt, URG Flag Cnt, CWE Flag Count, ECE Flag Cnt
```
- **Insight**: Ataques DDoS frequentemente apresentam padrões anormais de flags

### 6. **Estatísticas Agregadas** (~30 colunas)
```
Pkt Len Min/Max/Mean/Std/Var, Down/Up Ratio, Pkt Size Avg
Subflow Fwd/Bwd Pkts/Byts, Active/Idle Mean/Std/Max/Min
Fwd Act Data Pkts, Init Fwd/Bwd Win Byts, Fwd Seg Size Min/Avg, etc.
```
- **Insight**: Comportamento geral do fluxo de tráfego

### 7. **Variável Alvo**
```
Label: 'ddos' ou 'Benign'
```

---

## 📊 Estatísticas Descritivas (Amostra de 5000 linhas)

| Métrica | Flow Duration | Tot Fwd Pkts | Tot Bwd Pkts | Flow Byts/s | Flow Pkts/s |
|---------|---------------|--------------|--------------|-------------|-------------|
| **Média** | 7.28M μs | 6.46 | 13.15 | 5,482.58 | 24,944.99 |
| **Mediana** | 213.4K μs | 2.00 | 5.00 | 242.12 | 33.12 |
| **Std Dev** | 18.27M μs | 78.21 | 150.24 | 14,129.85 | 71,510.40 |
| **Min** | 6 μs | 0 | 1 | 0 | 0.017 |
| **Max** | 119.98M μs | 5,251 | 10,156 | 268,409.75 | 333,333.33 |

**Observações importantes:**
- Sem valores faltantes detectados
- Sem valores infinitos na amostra
- Grande variabilidade nos dados (alto desvio padrão)
- Distribuição muito assimétrica (mediana ≪ média)

---

## 🎯 Dados Relevantes para Cada Etapa do Projeto

### **1. Análise Univariada**
**Objetivo**: Entender a distribuição individual de cada variável

**Colunas mais relevantes:**
- Taxa de fluxo: `Flow Byts/s`, `Flow Pkts/s`, `Flow Duration`
- Características de pacotes: `Tot Fwd Pkts`, `Tot Bwd Pkts`, `Fwd Pkt Len Mean`, `Bwd Pkt Len Mean`
- Flags TCP: `SYN Flag Cnt`, `ACK Flag Cnt`, `RST Flag Cnt`
- Razão: `Down/Up Ratio`, `Pkt Size Avg`

**Técnicas recomendadas:**
- Histogramas e boxplots
- Estatísticas descritivas (média, mediana, desvio padrão)
- Análise de outliers
- Distribuição de probabilidade

**Insight esperado:** Ataques DDoS mostram picos anormais em velocidade/taxa

---

### **2. Análise Multivariada**
**Objetivo**: Entender relações entre variáveis

**Correlações esperadas:**
- `Tot Fwd Pkts` ↔ `Flow Byts/s` (pacotes maiores = mais bytes)
- `Flow Duration` ↔ `Tot Fwd Pkts + Tot Bwd Pkts`
- `SYN Flag Cnt` ↔ `Tot Fwd Pkts` (em conexões normais)
- DDoS específicas: Razões anormais entre variáveis

**Técnicas recomendadas:**
- Matriz de correlação (Pearson/Spearman)
- Análise de componentes principais (PCA)
- Testes estatísticos (t-test, ANOVA, qui-quadrado)
- Análise de agrupamento (clustering)

**Insight esperado:** Variáveis comportam-se diferentemente em DDoS vs Benign

---

### **3. Visualização**
**Objetivo**: Comunicar padrões através de gráficos

**Visualizações recomendadas:**
- **Boxplots comparativos**: Variáveis por classe (ddos vs Benign)
- **Scatter plots**: Relações entre pares de variáveis
- **Heatmaps**: Correlação entre variáveis
- **Distribuições**: Histogramas por classe
- **Série temporal**: Padrões ao longo do tempo
- **Análise de redes**: Relações IP origem-destino (se relevante)

---

### **4. Classificação/Regressão**
**Objetivo**: Construir modelo preditivo para detectar DDoS

**Abordagem recomendada:**

#### 4.1 Pré-processamento
```
1. Remover colunas não-numéricas (Flow ID, IPs, Ports, Timestamp)
   ou codificá-las (Protocol como numérico)
2. Normalizar/Padronizar variáveis (StandardScaler ou MinMaxScaler)
3. Lidar com outliers (valores muito extremos)
4. Feature selection: Remover variáveis com baixa variância
5. Tratar desbalanceamento (SMOTE, class weights, threshold adjustment)
```

#### 4.2 Modelos Candidatos
```
- Classificação binária (DDoS vs Benign):
  * Logistic Regression (baseline)
  * Random Forest (interpretabilidade)

- Avaliação:
  * Accuracy (mas cuidado com desbalanceamento)
  * Precision, Recall, F1-Score
  * ROC-AUC
  * Confusion Matrix
```

#### 4.3 Considerações Especiais
```
- Dataset desbalanceado (20-80):
  * Use stratified k-fold cross-validation
  * Ajuste class weights nos modelos
  * Considere threshold tuning

- Tamanho dos dados:
  * Use sampling se necessário para treinar
  * Considere algoritmos escaláveis (XGBoost, SGD)
```

---

