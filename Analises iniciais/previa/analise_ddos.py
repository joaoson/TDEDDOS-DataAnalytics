# ================================================================================================
# IMPORTACAO DE BIBLIOTECAS
# ================================================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Machine Learning
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, precision_score, recall_score, f1_score
)
from imblearn.over_sampling import SMOTE

# Configuracoes de visualizacao
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (14, 6)
plt.rcParams['font.size'] = 11

print("\n" + "="*100)
print("BIBLIOTECAS IMPORTADAS COM SUCESSO!")
print("="*100)


# ================================================================================================
# CONFIGURACAO DO DATASET
# ================================================================================================


CAMINHO_DATASET = 'final_dataset.csv'

# Numero de linhas para analise (None = todas)
AMOSTRA_LINHAS = 150000  # Usar 150k para rodar mais rapido. Trocar para None para tudo

print(f"\nCarregando dataset de: {CAMINHO_DATASET}")
print(f"Amostra: {AMOSTRA_LINHAS if AMOSTRA_LINHAS else 'Dataset completo'}")


# ================================================================================================
# 1. DESCRICAO DO DATASET
# ================================================================================================

print("\n" + "="*100)
print("1. DESCRICAO DO DATASET")
print("="*100)

print("""
O DDoS Dataset contem dados de trafego de rede para analise de seguranca cibernetica.

Caracteristicas:
- Fonte: Kaggle
- Tamanho: ~6.6 GB
- Colunas: 84 features + 1 label
- Tipos: Metricas de fluxo de rede, flags TCP/IP, estatisticas temporais
""")


# ================================================================================================
# 2. APRESENTACAO DO PROBLEMA
# ================================================================================================

print("\n" + "="*100)
print("2. APRESENTACAO DO PROBLEMA")
print("="*100)

print("""
PROBLEMA: Como identificar automaticamente ataques DDoS em tempo real?

OBJETIVOS:
1. Analise exploratoria de padroes estatisticos
2. Identificar features discriminativas
3. Desenvolver classificadores de alta acuracia
4. Comparar performance de diferentes algoritmos

META: Acuracia > 95%, Recall > 90%, F1-Score balanceado
""")


# ================================================================================================
# 3. ETL/ELT - PREPARACAO DOS DADOS
# ================================================================================================

print("\n" + "="*100)
print("3. ETL/ELT - EXTRACAO, TRANSFORMACAO E CARGA")
print("="*100)

print("\nCarregando dados...")
df = pd.read_csv(CAMINHO_DATASET, nrows=AMOSTRA_LINHAS)

print(f"\nDataset carregado!")
print(f"Dimensoes: {df.shape[0]:,} linhas x {df.shape[1]} colunas")
print(f"Memoria: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

print("\nPrimeiras 5 linhas:")
print(df.head())

print("\nInformacoes do dataset:")
print(df.info())

# Remover colunas desnecessarias
print("\nLimpeza de dados...")
colunas_remover = ['Flow ID', 'Src IP', 'Dst IP', 'Timestamp']
colunas_remover_existentes = [col for col in colunas_remover if col in df.columns]

if colunas_remover_existentes:
    df_clean = df.drop(columns=colunas_remover_existentes)
    print(f"Removidas: {colunas_remover_existentes}")
else:
    df_clean = df.copy()

# Verificar nulos
print("\nVerificando valores nulos...")
nulos = df_clean.isnull().sum()
if nulos.sum() > 0:
    print(f"Total de nulos: {nulos.sum():,}")
    for col in nulos[nulos > 0].index:
        if df_clean[col].dtype in ['float64', 'int64']:
            df_clean[col].fillna(df_clean[col].median(), inplace=True)
        else:
            df_clean[col].fillna(df_clean[col].mode()[0], inplace=True)
    print("Valores nulos tratados!")
else:
    print("Sem valores nulos!")

# Verificar duplicatas
duplicatas = df_clean.duplicated().sum()
print(f"\nDuplicatas: {duplicatas:,}")
if duplicatas > 0:
    df_clean = df_clean.drop_duplicates()
    print("Duplicatas removidas!")

# Verificar Label
if 'Label' in df_clean.columns:
    print("\nDistribuicao da variavel alvo (Label):")
    print(df_clean['Label'].value_counts())
    print("\nPercentual:")
    print(df_clean['Label'].value_counts(normalize=True).mul(100).round(2))

print(f"\nETL Concluido! Dataset final: {df_clean.shape}")


# ================================================================================================
# 4. ANALISE UNIVARIADA (10 VARIAVEIS)
# ================================================================================================

print("\n" + "="*100)
print("4. ANALISE UNIVARIADA - 10 VARIAVEIS")
print("="*100)

# Variaveis selecionadas
variaveis_analise = [
    'Flow Duration',
    'Tot Fwd Pkts',
    'Tot Bwd Pkts',
    'Flow Byts/s',
    'Flow Pkts/s',
    'Pkt Len Mean',
    'Fwd Pkt Len Mean',
    'Bwd Pkt Len Mean',
    'Flow IAT Mean',
    'Pkt Len Std'
]

print("\nVariaveis selecionadas:")
for i, var in enumerate(variaveis_analise, 1):
    print(f"{i:2d}. {var}")

# Funcao de analise
def analise_univariada(df, variavel):
    """Analise univariada completa de uma variavel"""
    
    print(f"\n{'='*100}")
    print(f"ANALISE: {variavel.upper()}")
    print(f"{'='*100}")
    
    dados = df[variavel].replace([np.inf, -np.inf], np.nan).dropna()
    
    # Calcular estatisticas
    media = dados.mean()
    mediana = dados.median()
    moda_valores = dados.mode()
    moda = moda_valores[0] if len(moda_valores) > 0 else np.nan
    desvio_padrao = dados.std()
    
    # Percentis
    p25 = dados.quantile(0.25)
    p50 = dados.quantile(0.50)
    p75 = dados.quantile(0.75)
    minimo = dados.min()
    maximo = dados.max()
    
    # Exibir resultados
    print(f"\nTENDENCIA CENTRAL:")
    print(f"   Media:    {media:>20,.4f}")
    print(f"   Mediana:  {mediana:>20,.4f}")
    print(f"   Moda:     {moda:>20,.4f}")
    
    print(f"\nDISPERSAO:")
    print(f"   Desvio Padrao:  {desvio_padrao:>15,.4f}")
    print(f"   Variancia:      {desvio_padrao**2:>15,.4f}")
    
    print(f"\nPERCENTIS:")
    print(f"   Minimo:   {minimo:>20,.4f}")
    print(f"   P25:      {p25:>20,.4f}")
    print(f"   P50:      {p50:>20,.4f}")
    print(f"   P75:      {p75:>20,.4f}")
    print(f"   Maximo:   {maximo:>20,.4f}")
    
    # Criar visualizacoes
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Histograma
    axes[0].hist(dados, bins=50, edgecolor='black', alpha=0.75, color='steelblue')
    axes[0].axvline(media, color='red', linestyle='--', linewidth=2.5, label=f'Media: {media:.2f}')
    axes[0].axvline(mediana, color='green', linestyle='--', linewidth=2.5, label=f'Mediana: {mediana:.2f}')
    axes[0].set_xlabel(variavel, fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Frequencia', fontsize=12, fontweight='bold')
    axes[0].set_title(f'Histograma: {variavel}', fontsize=13, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Boxplot
    axes[1].boxplot(dados, vert=True, patch_artist=True,
                   boxprops=dict(facecolor='lightblue', alpha=0.7),
                   medianprops=dict(color='red', linewidth=2.5))
    axes[1].set_ylabel(variavel, fontsize=12, fontweight='bold')
    axes[1].set_title(f'Boxplot: {variavel}', fontsize=13, fontweight='bold')
    axes[1].grid(True, alpha=0.3, axis='y')
    
    # Densidade
    dados.plot(kind='density', ax=axes[2], color='darkgreen', linewidth=2.5)
    axes[2].axvline(media, color='red', linestyle='--', linewidth=2, label='Media')
    axes[2].axvline(mediana, color='blue', linestyle='--', linewidth=2, label='Mediana')
    axes[2].set_xlabel(variavel, fontsize=12, fontweight='bold')
    axes[2].set_ylabel('Densidade', fontsize=12, fontweight='bold')
    axes[2].set_title(f'Densidade: {variavel}', fontsize=13, fontweight='bold')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'grafico_univariada_{variavel.replace(" ", "_").replace("/", "_")}.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    return {
        'Variavel': variavel,
        'Media': media,
        'Mediana': mediana,
        'Moda': moda,
        'Desvio Padrao': desvio_padrao,
        'P25': p25,
        'P50': p50,
        'P75': p75
    }

# Executar analise para todas as variaveis
resultados_univariada = []
for variavel in variaveis_analise:
    if variavel in df_clean.columns:
        resultado = analise_univariada(df_clean, variavel)
        resultados_univariada.append(resultado)

# Criar tabela resumo
df_resumo = pd.DataFrame(resultados_univariada)
print("\n" + "="*100)
print("TABELA RESUMO - ANALISE UNIVARIADA")
print("="*100)
print(df_resumo.to_string(index=False))

# Salvar tabela
df_resumo.to_csv('resumo_univariada.csv', index=False)
print("\nTabela salva em: resumo_univariada.csv")


# ================================================================================================
# 5. ANALISE MULTIVARIADA - CORRELACAO DE PEARSON
# ================================================================================================

print("\n" + "="*100)
print("5. ANALISE MULTIVARIADA - CORRELACAO DE PEARSON")
print("="*100)

# Preparar dados
df_multi = df_clean.copy()

print("\nConvertendo variaveis categoricas...")
label_encoders = {}

for col in df_multi.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    df_multi[f'{col}_encoded'] = le.fit_transform(df_multi[col].astype(str))
    label_encoders[col] = le
    print(f"{col}: {len(le.classes_)} categorias")

# Selecionar variaveis para correlacao
variaveis_correlacao = variaveis_analise.copy()
if 'Label_encoded' in df_multi.columns:
    variaveis_correlacao.append('Label_encoded')

# Adicionar mais algumas variaveis numericas
colunas_numericas = df_multi.select_dtypes(include=[np.number]).columns.tolist()
outras = [c for c in colunas_numericas if c not in variaveis_correlacao][:10]
variaveis_correlacao.extend(outras)
variaveis_correlacao = variaveis_correlacao[:20]

print(f"\nCalculando correlacao para {len(variaveis_correlacao)} variaveis...")
matriz_correlacao = df_multi[variaveis_correlacao].corr(method='pearson')

# Criar heatmap
plt.figure(figsize=(18, 16))
sns.heatmap(matriz_correlacao,
            annot=True,
            fmt='.2f',
            cmap='RdBu_r',
            center=0,
            square=True,
            linewidths=1,
            cbar_kws={"shrink": 0.8},
            vmin=-1, vmax=1,
            annot_kws={'size': 8})

plt.title('Matriz de Correlacao de Pearson - Dataset DDoS', fontsize=18, fontweight='bold', pad=20)
plt.xticks(rotation=45, ha='right', fontsize=10)
plt.yticks(fontsize=10)
plt.tight_layout()
plt.savefig('matriz_correlacao.png', dpi=150, bbox_inches='tight')
plt.close()
print("Heatmap salvo em: matriz_correlacao.png")

# Encontrar correlacoes fortes
print("\nCORRELACOES FORTES (|r| >= 0.7):")
correlacoes_fortes = []
for i in range(len(matriz_correlacao.columns)):
    for j in range(i+1, len(matriz_correlacao.columns)):
        var1 = matriz_correlacao.columns[i]
        var2 = matriz_correlacao.columns[j]
        corr = matriz_correlacao.iloc[i, j]
        if abs(corr) >= 0.7:
            correlacoes_fortes.append({
                'Variavel 1': var1,
                'Variavel 2': var2,
                'Correlacao': corr,
                'Tipo': 'Positiva' if corr > 0 else 'Negativa'
            })

if correlacoes_fortes:
    df_corr = pd.DataFrame(correlacoes_fortes)
    df_corr = df_corr.sort_values('Correlacao', key=abs, ascending=False)
    print(df_corr.to_string(index=False))
else:
    print("Nenhuma correlacao >= 0.7 encontrada")

print("""
\nIMPACTO DAS RELACOES:
1. Multicolinearidade: Variaveis com r > 0.9 podem ser redundantes
2. Features discriminativas: Correlacao com Label indica poder preditivo
3. Padroes de trafego: Ataques alteram correlacoes normais
4. Feature engineering: Oportunidade de criar features derivadas
""")


# ================================================================================================
# 6. VISUALIZACOES
# ================================================================================================

print("\n" + "="*100)
print("6. VISUALIZACOES")
print("="*100)

# GRAFICO 1: Distribuicao de Classes
print("\nGrafico 1: Distribuicao de Classes...")
if 'Label' in df_clean.columns:
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    contagem = df_clean['Label'].value_counts()
    colors = sns.color_palette('Set2', len(contagem))
    
    # Barras
    axes[0].bar(range(len(contagem)), contagem.values, color=colors, edgecolor='black', linewidth=2)
    axes[0].set_xticks(range(len(contagem)))
    axes[0].set_xticklabels(contagem.index, rotation=45, ha='right')
    axes[0].set_ylabel('Quantidade', fontsize=13, fontweight='bold')
    axes[0].set_title('Distribuicao de Classes', fontsize=15, fontweight='bold')
    axes[0].grid(True, alpha=0.3, axis='y')
    
    for i, v in enumerate(contagem.values):
        axes[0].text(i, v, f'{v:,}\n({v/len(df_clean)*100:.1f}%)',
                    ha='center', va='bottom', fontweight='bold')
    
    # Pizza
    axes[1].pie(contagem.values, labels=contagem.index, autopct='%1.1f%%',
               colors=colors, startangle=90, textprops={'fontsize': 11, 'fontweight': 'bold'})
    axes[1].set_title('Proporcao de Classes', fontsize=15, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('grafico_distribuicao_classes.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Salvo: grafico_distribuicao_classes.png")

# GRAFICO 2: Features por Classe
print("\nGrafico 2: Features por Classe...")
if 'Label' in df_clean.columns:
    features_comp = variaveis_analise[:4]
    
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    axes = axes.ravel()
    
    for idx, feature in enumerate(features_comp):
        if feature in df_clean.columns:
            df_clean.boxplot(column=feature, by='Label', ax=axes[idx], patch_artist=True, grid=True)
            axes[idx].set_title(f'{feature} por Classe', fontsize=13, fontweight='bold')
            axes[idx].set_xlabel('Classe', fontsize=11, fontweight='bold')
            axes[idx].set_ylabel(feature, fontsize=11, fontweight='bold')
            plt.sca(axes[idx])
            plt.xticks(rotation=45, ha='right')
    
    plt.suptitle('Comparacao de Features por Classe', fontsize=16, fontweight='bold', y=1.00)
    plt.tight_layout()
    plt.savefig('grafico_features_por_classe.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Salvo: grafico_features_por_classe.png")

# GRAFICO 3: Top Features
print("\nGrafico 3: Top Features Correlacionadas...")
if 'Label_encoded' in df_multi.columns and 'Label_encoded' in matriz_correlacao.columns:
    corr_label = matriz_correlacao['Label_encoded'].sort_values(ascending=False)
    corr_label = corr_label[corr_label.index != 'Label_encoded']
    
    top_10 = pd.concat([corr_label.head(5), corr_label.tail(5)])
    
    plt.figure(figsize=(14, 8))
    colors_bar = ['#d62728' if x < 0 else '#2ca02c' for x in top_10.values]
    
    plt.barh(range(len(top_10)), top_10.values, color=colors_bar, edgecolor='black', linewidth=2)
    plt.yticks(range(len(top_10)), top_10.index, fontsize=11)
    plt.xlabel('Correlacao com Label', fontsize=13, fontweight='bold')
    plt.title('Top 10 Features Correlacionadas com Ataques DDoS', fontsize=15, fontweight='bold', pad=20)
    plt.axvline(x=0, color='black', linewidth=2)
    plt.grid(True, alpha=0.3, axis='x')
    
    for i, v in enumerate(top_10.values):
        plt.text(v + (0.01 if v > 0 else -0.01), i, f'{v:.3f}',
                ha='left' if v > 0 else 'right', va='center', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('grafico_top_features.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Salvo: grafico_top_features.png")


# ================================================================================================
# 7. MACHINE LEARNING
# ================================================================================================

print("\n" + "="*100)
print("7. MACHINE LEARNING - 3 MODELOS")
print("="*100)

# Preparar X e y
if 'Label' in df_clean.columns:
    X = df_multi.select_dtypes(include=[np.number]).drop(columns=['Label_encoded'], errors='ignore')
    
    if df_clean['Label'].dtype == 'object':
        le_target = LabelEncoder()
        y = le_target.fit_transform(df_clean['Label'])
        print(f"\nClasses: {list(le_target.classes_)}")
    else:
        y = df_clean['Label'].values
    
    print(f"\nDimensoes: X={X.shape}, y={y.shape}")
    
    # Train/Test Split
    print("\nDividindo em treino e teste...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    print(f"Treino: {len(X_train):,} | Teste: {len(X_test):,}")
    
    # Normalizacao
    print("\nNormalizando...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Balanceamento
    print("\nVerificando balanceamento...")
    unique, counts = np.unique(y_train, return_counts=True)
    for c, cnt in zip(unique, counts):
        print(f"Classe {c}: {cnt:,} ({cnt/len(y_train)*100:.1f}%)")
    
    ratio = max(counts) / min(counts) if len(counts) > 1 else 1
    if ratio > 1.5:
        print(f"Desbalanceado (ratio {ratio:.1f}:1) - Aplicando SMOTE...")
        smote = SMOTE(random_state=42)
        X_train_final, y_train_final = smote.fit_resample(X_train_scaled, y_train)
        print(f"Balanceado: {len(y_train_final):,}")
    else:
        X_train_final = X_train_scaled
        y_train_final = y_train
    
    # MODELO 1: Random Forest
    print("\n" + "="*100)
    print("MODELO 1: RANDOM FOREST")
    print("="*100)
    rf = RandomForestClassifier(n_estimators=100, max_depth=20, random_state=42, n_jobs=-1)
    print("Treinando...")
    rf.fit(X_train_final, y_train_final)
    y_pred_rf = rf.predict(X_test_scaled)
    print("\nRESULTADOS:")
    print(classification_report(y_test, y_pred_rf))
    
    # MODELO 2: XGBoost
    print("\n" + "="*100)
    print("MODELO 2: XGBOOST")
    print("="*100)
    xgb = XGBClassifier(n_estimators=100, max_depth=10, learning_rate=0.1, random_state=42, n_jobs=-1)
    print("Treinando...")
    xgb.fit(X_train_final, y_train_final)
    y_pred_xgb = xgb.predict(X_test_scaled)
    print("\nRESULTADOS:")
    print(classification_report(y_test, y_pred_xgb))
    
    # MODELO 3: Decision Tree
    print("\n" + "="*100)
    print("MODELO 3: DECISION TREE")
    print("="*100)
    dt = DecisionTreeClassifier(max_depth=15, min_samples_split=20, min_samples_leaf=10, random_state=42)
    print("Treinando...")
    dt.fit(X_train_final, y_train_final)
    y_pred_dt = dt.predict(X_test_scaled)
    print("\nRESULTADOS:")
    print(classification_report(y_test, y_pred_dt))
    
    # Comparacao
    print("\n" + "="*100)
    print("COMPARACAO DE PERFORMANCE")
    print("="*100)
    
    resultados = []
    for nome, y_pred in [('Random Forest', y_pred_rf), ('XGBoost', y_pred_xgb), ('Decision Tree', y_pred_dt)]:
        resultados.append({
            'Modelo': nome,
            'Acuracia': accuracy_score(y_test, y_pred),
            'Precisao': precision_score(y_test, y_pred, average='weighted'),
            'Recall': recall_score(y_test, y_pred, average='weighted'),
            'F1-Score': f1_score(y_test, y_pred, average='weighted')
        })
    
    df_comp = pd.DataFrame(resultados)
    print("\n" + df_comp.to_string(index=False))
    
    melhor = df_comp.loc[df_comp['F1-Score'].idxmax(), 'Modelo']
    melhor_f1 = df_comp['F1-Score'].max()
    print(f"\nMELHOR MODELO: {melhor} (F1-Score: {melhor_f1:.4f})")
    
    # Salvar resultados
    df_comp.to_csv('comparacao_modelos.csv', index=False)
    print("\nResultados salvos em: comparacao_modelos.csv")
    
    # Grafico comparativo
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    metricas = ['Acuracia', 'Precisao', 'Recall', 'F1-Score']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    
    for idx, metrica in enumerate(metricas):
        ax = axes[idx // 2, idx % 2]
        valores = df_comp[metrica].values
        ax.bar(df_comp['Modelo'], valores, color=colors, edgecolor='black', linewidth=2)
        ax.set_ylabel(metrica, fontsize=12, fontweight='bold')
        ax.set_title(f'{metrica} por Modelo', fontsize=13, fontweight='bold')
        ax.set_ylim([0, 1.1])
        ax.grid(True, alpha=0.3, axis='y')
        for i, v in enumerate(valores):
            ax.text(i, v + 0.02, f'{v:.4f}', ha='center', fontweight='bold')
    
    plt.suptitle('Comparacao de Performance - 3 Modelos', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('comparacao_modelos.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Grafico salvo em: comparacao_modelos.png")


# ================================================================================================
# 8. CONCLUSOES
# ================================================================================================

print("\n" + "="*100)
print("ANALISE COMPLETA CONCLUIDA!")
print("="*100)
print("\nArquivos gerados:")
print("   - resumo_univariada.csv")
print("   - matriz_correlacao.png")
print("   - grafico_distribuicao_classes.png")
print("   - grafico_features_por_classe.png")
print("   - grafico_top_features.png")
print("   - comparacao_modelos.csv")
print("   - comparacao_modelos.png")
print("   - + graficos individuais de cada variavel")
print("\nProjeto finalizado com sucesso!")
