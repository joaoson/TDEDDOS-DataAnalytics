# 📊 Projeto DDoS Data Analytics - Guia de Análise

## 🎯 Objetivo do Projeto

Aplicar técnicas de **Data Analytics** em um cenário prático de **detecção de ataques DDoS**, através de análises univariada, multivariada, visualização de dados e construção de modelos preditivos.

---

## 📁 Estrutura dos Datasets

### Dataset Balanceado
- **Arquivo**: `ddos_balanced/final_dataset.csv`
- **Tamanho**: 6.48 GB
- **Características**: Distribuição equilibrada de exemplos
- **Melhor para**: Aprender padrões com viés reduzido

### Dataset Desbalanceado
- **Arquivo**: `ddos_imbalanced/unbalaced_20_80_dataset.csv`
- **Tamanho**: 3.93 GB
- **Características**: 20% DDoS, 80% Benign (realista)
- **Melhor para**: Treinar modelos que funcionem em produção

---

## 📚 Guias Disponíveis

| Documento | Descrição |
|-----------|-----------|
| **ANALISE_DATASETS.md** | 📊 Análise completa dos datasets com estatísticas, estrutura e recomendações |
| **GUIA_PRATICO.md** | 🛠️ Snippets de código prontos para executar em cada fase |
| Este arquivo | 📋 Visão geral e roteiro de execução |

---

## 🚀 Começando Rápido

### Pré-requisitos
```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost
```

### Teste Rápido dos Dados
```python
import pandas as pd

# Carregar amostra
df = pd.read_csv('ddos_balanced/final_dataset.csv', nrows=10000)
print(df.shape)
print(df['Label'].value_counts())
print(df.describe())
```

---

## 📊 Fases da Análise

### **1. Análise Univariada**
Entender cada variável individualmente

**O que fazer:**
- [ ] Histogramas de frequência
- [ ] Boxplots por classe (DDoS vs Benign)
- [ ] Estatísticas descritivas
- [ ] Identificar outliers

**Variáveis-chave:**
- Taxa: `Flow Byts/s`, `Flow Pkts/s`
- Duração: `Flow Duration`
- Pacotes: `Tot Fwd Pkts`, `Tot Bwd Pkts`
- Flags TCP: `SYN Flag Cnt`, `ACK Flag Cnt`, `RST Flag Cnt`

**Ferramenta recomendada:** Pandas + Matplotlib/Seaborn

---

### **2. Análise Multivariada**
Entender relacionamentos entre variáveis

**O que fazer:**
- [ ] Matriz de correlação
- [ ] Teste ANOVA para diferenças entre classes
- [ ] Análise de Componentes Principais (PCA)
- [ ] Agrupamento exploratório (clustering)

**Esperado encontrar:**
- Padrões de comportamento DDoS distintos
- Variáveis mais discriminativas
- Redução de dimensionalidade via PCA

**Ferramenta recomendada:** Pandas + SciPy + Scikit-learn

---

### **3. Visualização**
Comunicar padrões através de gráficos

**O que fazer:**
- [ ] Boxplots comparativos (DDoS vs Benign)
- [ ] Scatter plots de variáveis principais
- [ ] Heatmaps de correlação
- [ ] Pair plots
- [ ] Distribuições por classe

**Ferramenta recomendada:** Matplotlib + Seaborn

---

### **4. Classificação**
Construir modelo preditivo para detectar DDoS

**O que fazer:**
- [ ] Preparar dados (normalizar, feature selection)
- [ ] Treinar modelo baseline (Logistic Regression)
- [ ] Treinar modelos avançados (Random Forest, XGBoost)
- [ ] Comparar desempenho
- [ ] Fine-tuning do melhor modelo
- [ ] Análise de features importantes

**Modelos recomendados:**
1. **Logistic Regression** (baseline rápido)
2. **Random Forest** (interpretabilidade)
3. **XGBoost** (melhor desempenho)
4. **Neural Networks** (complexo, melhor AUC)

**Métricas:**
- Accuracy (cuidado: desbalanceamento)
- Precision/Recall/F1-Score
- ROC-AUC (recomendado)
- Confusion Matrix

**Ferramenta recomendada:** Scikit-learn + XGBoost

---

## 📈 Matriz de Colunas vs Análise

```
┌─────────────────────────────────────┬──────────┬───────────────┬────────────┐
│ Tipo de Coluna                      │ Univariada│ Multivariada │ Classificação
├─────────────────────────────────────┼──────────┼───────────────┼────────────┤
│ Taxa/Velocidade (Flow Byts/s)       │    ✅    │      ✅      │     ✅     │
│ Duração (Flow Duration)             │    ✅    │      ✅      │     ✅     │
│ Pacotes (Tot Fwd/Bwd Pkts)          │    ✅    │      ✅      │     ✅     │
│ Comprimento pacotes (Pkt Len)       │    ✅    │      ✅      │     ✅     │
│ Flags TCP (SYN, ACK, RST)           │    ✅    │      ✅      │     ✅     │
│ Estatísticas IAT (Inter-Arrival)    │    ✅    │      ✅      │     ✅     │
│ IDs e IPs (Flow ID, Src IP)         │    ❌    │      ⚠️      │     ❌     │
│ Timestamp                           │    ⚠️    │      ⚠️      │     ❌     │
└─────────────────────────────────────┴──────────┴───────────────┴────────────┘

✅ = Usar diretamente
⚠️  = Usar com cuidado (agrupar/agregar)
❌ = Não usar ou descartar
```

---

## 💡 Insights Esperados

### Características de Ataque DDoS
- **Taxa elevada** de pacotes por segundo
- **Padrão repetitivo** de tipos de pacotes
- **Razão anormal** entre Fwd e Bwd
- **Flags TCP anormais** (muitos SYN, poucos ACK)
- **Comprimento consistente** de pacotes
- **Duração longa** do fluxo

### Características de Tráfego Benign
- **Taxa variada** de pacotes
- **Padrão misto** de tipos
- **Razão Fwd/Bwd balanceada**
- **Flags TCP normais**
- **Variação** no comprimento de pacotes
- **Duração curta** a média

---

## 🔍 Checklist de Execução

### Fase 1: Setup
- [ ] Verificar dataset está acessível
- [ ] Instalar bibliotecas necessárias
- [ ] Carregar amostra dos dados
- [ ] Entender estrutura básica

### Fase 2: Exploração
- [ ] Executar análise univariada
- [ ] Criar histogramas e boxplots
- [ ] Documentar distribuições
- [ ] Identificar colunas mais importantes

### Fase 3: Análise Multivariada
- [ ] Calcular correlações
- [ ] Executar PCA
- [ ] Realizar testes estatísticos
- [ ] Documentar padrões encontrados

### Fase 4: Visualizações
- [ ] Criar pair plots
- [ ] Fazer heatmaps
- [ ] Gráficos comparativos por classe
- [ ] Preparar para apresentação

### Fase 5: Modelagem
- [ ] Preparar dados (normalizar, split)
- [ ] Treinar baseline
- [ ] Treinar modelos avançados
- [ ] Comparar desempenho
- [ ] Fine-tuning

### Fase 6: Documentação
- [ ] Summarizar descobertas
- [ ] Escrever conclusões
- [ ] Documentar limitações
- [ ] Propor trabalhos futuros

---

## 📊 Métricas de Sucesso

| Métrica | Bom | Excelente |
|---------|-----|-----------|
| **ROC-AUC** | > 0.85 | > 0.95 |
| **Precision** | > 0.80 | > 0.95 |
| **Recall** | > 0.80 | > 0.95 |
| **F1-Score** | > 0.80 | > 0.90 |

---

## ⚙️ Dicas para Lidar com Dados Grandes

### Para ler arquivo grande em partes:
```python
# Opção 1: nrows
df = pd.read_csv('arquivo.csv', nrows=100000)

# Opção 2: skiprows
df = pd.read_csv('arquivo.csv', skiprows=range(1, 1000000), nrows=100000)

# Opção 3: chunks
for chunk in pd.read_csv('arquivo.csv', chunksize=100000):
    process(chunk)
```

### Para salvar resultados:
```python
# Modelos
import joblib
joblib.dump(model, 'model.pkl')

# Dados
df.to_csv('resultados.csv', index=False)

# Metadados
import json
with open('metadata.json', 'w') as f:
    json.dump(metadata, f)
```

---

## 📚 Referências Úteis

- **Pandas**: https://pandas.pydata.org/
- **Matplotlib**: https://matplotlib.org/
- **Seaborn**: https://seaborn.pydata.org/
- **Scikit-learn**: https://scikit-learn.org/
- **XGBoost**: https://xgboost.readthedocs.io/

---

## 🤝 Próximos Passos

1. **Leia ANALISE_DATASETS.md** para entender os dados em detalhe
2. **Consulte GUIA_PRATICO.md** para snippets de código
3. **Comece com exploração inicial** (rode o código de teste rápido)
4. **Siga as fases** na ordem sugerida
5. **Documente suas descobertas** ao longo do caminho
6. **Crie um notebook Jupyter** compilando tudo

---

## ❓ Dúvidas Comuns

**P: Por que dois datasets diferentes?**
R: Balanceado para aprender padrões, desbalanceado para simular produção.

**P: Por onde começo?**
R: Comece com análise univariada do dataset balanceado (menor), depois compare com desbalanceado.

**P: Devo usar todas as 85 colunas?**
R: Não. Feature selection reduz dimensionalidade e overfitting.

**P: Qual modelo é melhor?**
R: Depende do trade-off: Logistic Regression é rápida, Random Forest é balanceada, XGBoost geralmente melhor AUC.

**P: Como lidar com desbalanceamento?**
R: Class weights, SMOTE, threshold tuning, stratified cross-validation.

---

**Última atualização**: 2025-11-08
**Autor**: Claude Code
**Status**: 🟢 Pronto para começar
