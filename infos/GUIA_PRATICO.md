# 🛠️ Guia Prático - Começando a Análise

Este guia fornece snippets de código para cada fase da análise.

---

## 1️⃣ Exploração Inicial

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ============ CARREGAR DADOS ============
# Para o dataset balanceado (maior)
df_balanced = pd.read_csv('ddos_balanced/final_dataset.csv', nrows=100000)

# Para o dataset desbalanceado
df_imbalanced = pd.read_csv('ddos_imbalanced/unbalaced_20_80_dataset.csv', nrows=100000)

# ============ INFORMAÇÕES BÁSICAS ============
print("Shape:", df_balanced.shape)
print("\nDados faltantes:")
print(df_balanced.isnull().sum())

print("\nTipos de dados:")
print(df_balanced.dtypes)

print("\nDistribuição de labels:")
print(df_balanced['Label'].value_counts())
print(df_balanced['Label'].value_counts(normalize=True) * 100)

print("\nDescrição estatística:")
print(df_balanced.describe())
```

---

## 2️⃣ Análise Univariada

```python
# ============ COLUNAS RELEVANTES ============
numeric_cols = df_balanced.select_dtypes(include=[np.number]).columns.tolist()
# Remover coluna de índice se existir
numeric_cols = [col for col in numeric_cols if col != 'Unnamed: 0']

# ============ HISTOGRAMAS ============
fig, axes = plt.subplots(5, 5, figsize=(20, 20))
axes = axes.flatten()

for idx, col in enumerate(numeric_cols[:25]):
    axes[idx].hist(df_balanced[col], bins=50, edgecolor='black')
    axes[idx].set_title(col, fontsize=10)
    axes[idx].set_yscale('log')  # Escala logarítmica para melhor visualização

plt.tight_layout()
plt.savefig('univariada_histogramas.png', dpi=100, bbox_inches='tight')
plt.show()

# ============ BOXPLOTS POR CLASSE ============
# Selecionar top 10 variáveis mais interessantes
top_vars = ['Flow Duration', 'Tot Fwd Pkts', 'Tot Bwd Pkts', 'Flow Byts/s',
            'Flow Pkts/s', 'Fwd Pkt Len Mean', 'Bwd Pkt Len Mean',
            'SYN Flag Cnt', 'ACK Flag Cnt', 'RST Flag Cnt']

fig, axes = plt.subplots(2, 5, figsize=(20, 10))
axes = axes.flatten()

for idx, col in enumerate(top_vars):
    df_balanced.boxplot(column=col, by='Label', ax=axes[idx])
    axes[idx].set_title(col)
    axes[idx].set_ylabel(col)
    axes[idx].set_xlabel('Classe')

plt.suptitle('Distribuição por Classe', fontsize=16, y=1.00)
plt.tight_layout()
plt.savefig('univariada_boxplots.png', dpi=100, bbox_inches='tight')
plt.show()

# ============ ESTATÍSTICAS POR CLASSE ============
print("Estatísticas por classe:")
for col in top_vars:
    print(f"\n{col}:")
    print(df_balanced.groupby('Label')[col].describe())
```

---

## 3️⃣ Análise Multivariada

```python
# ============ MATRIZ DE CORRELAÇÃO ============
# Selecionar apenas colunas numéricas
numeric_df = df_balanced.select_dtypes(include=[np.number])
numeric_df = numeric_df.drop('Unnamed: 0', axis=1, errors='ignore')

correlation_matrix = numeric_df.corr()

# Visualizar matriz de correlação
plt.figure(figsize=(20, 18))
sns.heatmap(correlation_matrix, cmap='coolwarm', center=0,
            vmin=-1, vmax=1, square=True, cbar_kws={"shrink": 0.8})
plt.title('Matriz de Correlação - Todas as Variáveis', fontsize=16)
plt.tight_layout()
plt.savefig('multivariada_correlacao.png', dpi=100, bbox_inches='tight')
plt.show()

# ============ CORRELAÇÃO COM LABEL ============
# Converter label para numérico
label_numeric = (df_balanced['Label'] == 'ddos').astype(int)
numeric_df_with_label = numeric_df.copy()
numeric_df_with_label['Label_numeric'] = label_numeric

correlations_with_label = numeric_df_with_label.corr()['Label_numeric'].sort_values(ascending=False)
print("Top 15 variáveis correlacionadas com DDoS:")
print(correlations_with_label.head(15))
print("\nTop 15 variáveis anti-correlacionadas com DDoS:")
print(correlations_with_label.tail(15))

# Visualizar
plt.figure(figsize=(12, 8))
top_corr = pd.concat([correlations_with_label.head(10), correlations_with_label.tail(10)])
top_corr = top_corr.drop('Label_numeric')
top_corr.plot(kind='barh', color=['green' if x > 0 else 'red' for x in top_corr])
plt.xlabel('Correlação com DDoS')
plt.title('Top 10 Features Correlacionadas e Anti-correlacionadas com DDoS')
plt.tight_layout()
plt.savefig('multivariada_correlacao_label.png', dpi=100, bbox_inches='tight')
plt.show()

# ============ ANÁLISE DE COMPONENTES PRINCIPAIS (PCA) ============
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Normalizar dados
scaler = StandardScaler()
X_scaled = scaler.fit_transform(numeric_df)

# Aplicar PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Visualizar
plt.figure(figsize=(12, 8))
colors = ['red' if label == 'ddos' else 'blue' for label in df_balanced['Label']]
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=colors, alpha=0.5, s=20)
plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%})')
plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%})')
plt.title('PCA - Primeiros 2 Componentes Principais')
plt.legend(['DDoS', 'Benign'])
plt.tight_layout()
plt.savefig('multivariada_pca.png', dpi=100, bbox_inches='tight')
plt.show()

print(f"Variância explicada: {pca.explained_variance_ratio_.sum():.2%}")

# ============ TESTES ESTATÍSTICOS ============
from scipy import stats

# ANOVA para cada variável
print("\nTeste ANOVA - Verificar se há diferença significativa entre classes:")
for col in top_vars:
    ddos_data = df_balanced[df_balanced['Label'] == 'ddos'][col]
    benign_data = df_balanced[df_balanced['Label'] == 'Benign'][col]

    f_stat, p_value = stats.f_oneway(ddos_data, benign_data)
    print(f"{col}: F={f_stat:.4f}, p-value={p_value:.4e}")
```

---

## 4️⃣ Visualizações Avançadas

```python
# ============ PAIR PLOT DE VARIÁVEIS PRINCIPAIS ============
import warnings
warnings.filterwarnings('ignore')

# Selecionar top 5 variáveis + label
top_5_vars = ['Flow Byts/s', 'Tot Fwd Pkts', 'Tot Bwd Pkts', 'Flow Pkts/s', 'Flow Duration']
plot_df = df_balanced[top_5_vars + ['Label']].sample(5000)  # Amostra para velocidade

sns.pairplot(plot_df, hue='Label', diag_kind='kde', plot_kws={'alpha': 0.6}, height=2)
plt.suptitle('Pair Plot - Top 5 Variáveis', y=1.00, fontsize=14)
plt.tight_layout()
plt.savefig('visualizacao_pairplot.png', dpi=100, bbox_inches='tight')
plt.show()

# ============ DENSIDADE POR CLASSE ============
fig, axes = plt.subplots(2, 5, figsize=(20, 10))
axes = axes.flatten()

numeric_cols_sample = numeric_df.columns[:10]

for idx, col in enumerate(numeric_cols_sample):
    # Plotar densidade para cada classe
    df_balanced[df_balanced['Label'] == 'ddos'][col].plot(kind='density', ax=axes[idx],
                                                           label='DDoS', color='red', linewidth=2)
    df_balanced[df_balanced['Label'] == 'Benign'][col].plot(kind='density', ax=axes[idx],
                                                             label='Benign', color='blue', linewidth=2)
    axes[idx].set_title(col, fontsize=10)
    axes[idx].legend()
    axes[idx].set_xlabel('')

plt.suptitle('Densidade de Distribuição por Classe', fontsize=16)
plt.tight_layout()
plt.savefig('visualizacao_densidade.png', dpi=100, bbox_inches='tight')
plt.show()

# ============ VIOLIN PLOT ============
fig, axes = plt.subplots(2, 5, figsize=(20, 10))
axes = axes.flatten()

for idx, col in enumerate(top_vars):
    sns.violinplot(data=df_balanced, x='Label', y=col, ax=axes[idx])
    axes[idx].set_title(col)

plt.suptitle('Violin Plot - Distribuição por Classe', fontsize=16)
plt.tight_layout()
plt.savefig('visualizacao_violinplot.png', dpi=100, bbox_inches='tight')
plt.show()
```

---

## 5️⃣ Preparação de Dados para ML

```python
# ============ PRÉ-PROCESSAMENTO ============
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder

# Copiar dataframe
df_ml = df_balanced.copy()

# Remover colunas não-numéricas que não são o label
cols_to_drop = ['Unnamed: 0', 'Flow ID', 'Src IP', 'Dst IP', 'Timestamp']
df_ml = df_ml.drop(cols_to_drop, axis=1, errors='ignore')

# Codificar Protocol se for texto
if df_ml['Protocol'].dtype == 'object':
    le_protocol = LabelEncoder()
    df_ml['Protocol'] = le_protocol.fit_transform(df_ml['Protocol'])

# Codificar Label
le_label = LabelEncoder()
df_ml['Label'] = le_label.fit_transform(df_ml['Label'])

# Separar features e target
X = df_ml.drop('Label', axis=1)
y = df_ml['Label']

print(f"Features shape: {X.shape}")
print(f"Target shape: {y.shape}")
print(f"Distribuição de classes:\n{y.value_counts()}")

# ============ NORMALIZAÇÃO ============
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Converter de volta para DataFrame mantendo colunas
X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

# ============ TRAIN-TEST SPLIT ============
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTrain set: {X_train.shape}")
print(f"Test set: {X_test.shape}")
print(f"\nDistribuição no train:\n{y_train.value_counts()}")
print(f"\nDistribuição no test:\n{y_test.value_counts()}")
```

---

## 6️⃣ Modelagem

```python
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (classification_report, confusion_matrix,
                            roc_auc_score, roc_curve, auc)

# ============ MODELO 1: LOGISTIC REGRESSION (BASELINE) ============
print("="*50)
print("LOGISTIC REGRESSION")
print("="*50)

lr_model = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
lr_model.fit(X_train, y_train)

y_pred_lr = lr_model.predict(X_test)
y_pred_proba_lr = lr_model.predict_proba(X_test)[:, 1]

print("\nClassification Report:")
print(classification_report(y_test, y_pred_lr, target_names=['Benign', 'DDoS']))
print(f"ROC-AUC: {roc_auc_score(y_test, y_pred_proba_lr):.4f}")

# ============ MODELO 2: RANDOM FOREST ============
print("\n" + "="*50)
print("RANDOM FOREST")
print("="*50)

rf_model = RandomForestClassifier(n_estimators=100, random_state=42,
                                  class_weight='balanced', n_jobs=-1)
rf_model.fit(X_train, y_train)

y_pred_rf = rf_model.predict(X_test)
y_pred_proba_rf = rf_model.predict_proba(X_test)[:, 1]

print("\nClassification Report:")
print(classification_report(y_test, y_pred_rf, target_names=['Benign', 'DDoS']))
print(f"ROC-AUC: {roc_auc_score(y_test, y_pred_proba_rf):.4f}")

# Feature Importance
feature_importance_rf = pd.DataFrame({
    'feature': X_train.columns,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False)

plt.figure(figsize=(12, 8))
top_features = feature_importance_rf.head(15)
plt.barh(range(len(top_features)), top_features['importance'].values)
plt.yticks(range(len(top_features)), top_features['feature'].values)
plt.xlabel('Importância')
plt.title('Top 15 Features - Random Forest')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig('ml_feature_importance_rf.png', dpi=100, bbox_inches='tight')
plt.show()

# ============ MODELO 3: XGBOOST ============
print("\n" + "="*50)
print("XGBOOST")
print("="*50)

xgb_model = XGBClassifier(n_estimators=100, random_state=42,
                         scale_pos_weight=(y_train == 0).sum() / (y_train == 1).sum())
xgb_model.fit(X_train, y_train)

y_pred_xgb = xgb_model.predict(X_test)
y_pred_proba_xgb = xgb_model.predict_proba(X_test)[:, 1]

print("\nClassification Report:")
print(classification_report(y_test, y_pred_xgb, target_names=['Benign', 'DDoS']))
print(f"ROC-AUC: {roc_auc_score(y_test, y_pred_proba_xgb):.4f}")

# ============ COMPARAÇÃO DE MODELOS ============
print("\n" + "="*50)
print("COMPARAÇÃO DE MODELOS")
print("="*50)

models_comparison = pd.DataFrame({
    'Modelo': ['Logistic Regression', 'Random Forest', 'XGBoost'],
    'ROC-AUC': [
        roc_auc_score(y_test, y_pred_proba_lr),
        roc_auc_score(y_test, y_pred_proba_rf),
        roc_auc_score(y_test, y_pred_proba_xgb)
    ]
})

print(models_comparison.sort_values('ROC-AUC', ascending=False))

# ============ CURVA ROC ============
plt.figure(figsize=(10, 8))

fpr_lr, tpr_lr, _ = roc_curve(y_test, y_pred_proba_lr)
fpr_rf, tpr_rf, _ = roc_curve(y_test, y_pred_proba_rf)
fpr_xgb, tpr_xgb, _ = roc_curve(y_test, y_pred_proba_xgb)

plt.plot(fpr_lr, tpr_lr, label=f'LR (AUC={roc_auc_score(y_test, y_pred_proba_lr):.4f})', linewidth=2)
plt.plot(fpr_rf, tpr_rf, label=f'RF (AUC={roc_auc_score(y_test, y_pred_proba_rf):.4f})', linewidth=2)
plt.plot(fpr_xgb, tpr_xgb, label=f'XGB (AUC={roc_auc_score(y_test, y_pred_proba_xgb):.4f})', linewidth=2)
plt.plot([0, 1], [0, 1], 'k--', label='Random', linewidth=2)

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - Comparação de Modelos')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('ml_roc_curves.png', dpi=100, bbox_inches='tight')
plt.show()

# ============ CONFUSION MATRIX ============
from sklearn.metrics import ConfusionMatrixDisplay

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

for idx, (y_pred, model_name) in enumerate([
    (y_pred_lr, 'Logistic Regression'),
    (y_pred_rf, 'Random Forest'),
    (y_pred_xgb, 'XGBoost')
]):
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Benign', 'DDoS'])
    disp.plot(ax=axes[idx], cmap='Blues')
    axes[idx].set_title(model_name)

plt.tight_layout()
plt.savefig('ml_confusion_matrices.png', dpi=100, bbox_inches='tight')
plt.show()
```

---

## 💾 Salvando Resultados

```python
# Salvar modelos
import joblib

joblib.dump(scaler, 'scaler.pkl')
joblib.dump(rf_model, 'best_model.pkl')

# Salvar resultados em CSV
models_comparison.to_csv('model_comparison.csv', index=False)
feature_importance_rf.to_csv('feature_importance.csv', index=False)

# Salvar metadados
metadata = {
    'train_size': X_train.shape,
    'test_size': X_test.shape,
    'features': list(X_train.columns),
    'best_model': 'Random Forest',
    'best_auc': roc_auc_score(y_test, y_pred_proba_rf)
}

import json
with open('metadata.json', 'w') as f:
    json.dump(metadata, f, indent=4)
```

---

## 🎯 Execução Passo a Passo

1. **Comece com exploração inicial** para entender os dados
2. **Execute análise univariada** e identifique padrões
3. **Faça análise multivariada** para correlações
4. **Crie visualizações** claras e comunicativas
5. **Prepare dados** e treine modelos
6. **Documente descobertas** e conclusões

Boa análise! 🚀
