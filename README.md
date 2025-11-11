# TDEDDOS-DataAnalytics

Projeto de análise de dados para detecção de ataques DDoS utilizando técnicas de Data Analytics, incluindo análises univariada, multivariada, visualização de dados e modelagem de Machine Learning.

## 📋 Apresentação

- **Slides**: [Apresentação Google Slides](https://docs.google.com/presentation/d/12N9lQqO7e-k8rsf8mTp5UEl8uZPTjkepeNjyu4d7RbE/edit?usp=sharing)

---

## 📁 Estrutura do Projeto

### 📂 Scripts Principais

#### `analise_univariada.py`
Script para análise univariada dos dados. Realiza:
- Cálculo de métricas estatísticas (média, mediana, moda, desvio padrão, percentis)
- Geração de histogramas para variáveis selecionadas
- Análise de distribuições das features
- Gera relatórios em CSV e visualizações em PNG

**Saída**: `analise_univariada_output/`

#### `analise_correlacao.py`
Script para análise de correlação entre variáveis. Realiza:
- Cálculo da matriz de correlação
- Geração de heatmaps de correlação
- Identificação de variáveis altamente correlacionadas
- Análise de correlações top 10

**Saída**: `analise_correlacao_output/`

#### `modelagem_ml.py`
Script para modelagem de Machine Learning. Implementa:
- Engenharia de features
- Treinamento de múltiplos modelos (Random Forest, Logistic Regression, SVM)
- Avaliação de métricas (accuracy, precision, recall, F1-score, ROC-AUC)
- Geração de matrizes de confusão e curvas ROC
- Comparação de modelos

**Saída**: `modelagem_ml_output/`

---

### 📂 Pastas de Saída

#### `analise_univariada_output/`
Contém os resultados da análise univariada:
- `resumo_metricas.csv`: Tabela com métricas estatísticas (média, mediana, moda, desvio padrão, percentis)
- `relatorio_metricas.txt`: Relatório textual com resumo das análises
- `histograma_*.png`: Histogramas individuais para cada variável analisada

#### `analise_correlacao_output/`
Contém os resultados da análise de correlação:
- `matriz_correlacao.csv`: Matriz completa de correlação entre variáveis
- `heatmap_correlacao_completo.png`: Heatmap com todas as correlações
- `heatmap_correlacao_top10.png`: Heatmap das 10 maiores correlações
- `relatorio_correlacao.txt`: Relatório textual da análise
- `conclusao_correlacao.txt`: Conclusões extraídas da análise

#### `modelagem_ml_output/`
Contém os resultados da modelagem de ML:
- `comparacao_modelos.csv`: Tabela comparativa de métricas entre modelos
- `comparacao_metricas.png`: Gráfico comparativo de métricas
- `curvas_roc.png`: Curvas ROC para cada modelo
- `matrizes_confusao.png`: Matrizes de confusão para cada modelo
- `relatorio_modelos.txt`: Relatório detalhado dos modelos
- `conclusao_modelos.txt`: Conclusões sobre o desempenho dos modelos

---

### 📂 Pastas de Apoio

#### `apoio/`
Scripts auxiliares para processamento de dados:
- `converter_parquet.py`: Converte arquivos CSV para formato Parquet (mais eficiente)
- `reduzir_csv.py`: Script para reduzir o tamanho de arquivos CSV (amostragem ou filtragem)

#### `infos/`
Documentação e guias do projeto:
- `README_ANALISE.md`: Guia completo de análise com fases, objetivos e recomendações
- `ANALISE_DATASETS.md`: Análise detalhada dos datasets utilizados
- `GUIA_PRATICO.md`: Snippets de código prontos para uso
- `RESUMO_ANALISE.txt`: Resumo executivo das análises realizadas

#### `Analises iniciais/previa/`
Análises exploratórias iniciais do projeto:
- `analise_ddos.py`: Script de análise exploratória inicial
- `grafico_*.png`: Gráficos gerados nas análises iniciais
  - Distribuição de classes
  - Features por classe
  - Top features
  - Análises univariadas de variáveis específicas
  - Matriz de correlação
- `resumo_univariada.csv`: Resumo das análises univariadas iniciais

---

### 📄 Arquivos de Documentação

#### `EXPLICACAO_COLUNAS.md`
Documentação detalhada explicando o significado de cada coluna/feature do dataset DDoS. Contém 479 linhas com descrições completas das variáveis.

#### `colunas_eliminadas.txt`
Lista das colunas que foram eliminadas durante o processo de limpeza e preparação dos dados (53 colunas).

#### `README_univariate.txt`
Guia rápido para gerar métricas e histogramas para 10 variáveis, incluindo instruções de configuração e execução.

#### `dicionario_dados_ddos.pdf`
Dicionário de dados em formato PDF com definições e explicações das features do dataset.

#### `Apresentação TDE Data.pdf`
Apresentação em PDF sobre o projeto TDE Data Analytics.

#### `gráficosTDE.pbit`
Arquivo Power BI Template (.pbit) com visualizações e gráficos do projeto.

---

## 🚀 Como Usar

### Pré-requisitos

```bash
pip install polars pandas numpy matplotlib seaborn scikit-learn
```

### Executar Análises

1. **Análise Univariada**:
   ```bash
   python analise_univariada.py
   ```

2. **Análise de Correlação**:
   ```bash
   python analise_correlacao.py
   ```

3. **Modelagem de ML**:
   ```bash
   python modelagem_ml.py
   ```

### Converter Dados

Para converter CSV para Parquet (mais eficiente):
```bash
python apoio/converter_parquet.py
```

---

## 📊 Objetivo do Projeto

Aplicar técnicas de **Data Analytics** em um cenário prático de **detecção de ataques DDoS**, através de:

1. **Análise Univariada**: Entender cada variável individualmente
2. **Análise Multivariada**: Identificar relacionamentos entre variáveis
3. **Visualização**: Comunicar padrões através de gráficos
4. **Modelagem de ML**: Construir modelos preditivos para classificação

---

## 📝 Notas

- Os scripts utilizam **Polars** para processamento eficiente de grandes volumes de dados
- As visualizações são geradas com **Matplotlib** e **Seaborn**
- Os modelos de ML utilizam **Scikit-learn**
- Todos os outputs são salvos em pastas específicas para fácil organização

---

## 🔗 Links Úteis

- [Apresentação do Projeto](https://docs.google.com/presentation/d/12N9lQqO7e-k8rsf8mTp5UEl8uZPTjkepeNjyu4d7RbE/edit?usp=sharing)
- Consulte `infos/README_ANALISE.md` para um guia completo de análise
- Consulte `EXPLICACAO_COLUNAS.md` para entender as features do dataset
