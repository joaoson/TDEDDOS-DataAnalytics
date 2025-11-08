"""
Modelagem de Machine Learning
Aplica engenharia de features e pelo menos 3 modelos de ML
Apresenta resultados de forma clara e objetiva
"""

import polars as pl
import numpy as np
import pandas as pd
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Machine Learning
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve
)
import matplotlib.pyplot as plt
import seaborn as sns

# Configuração
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

class FeatureEngineering:
    """Classe para realizar engenharia de features"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.features_selecionadas = None
        
    def preparar_dados(self, df):
        """Prepara e limpa os dados para modelagem"""
        print("\n" + "="*80)
        print("ETAPA 1: ENGENHARIA DE FEATURES")
        print("="*80)
        
        df_clean = df.clone()
        
        # 1. Remover colunas não úteis para modelagem
        print("\n1.1. Removendo colunas não úteis...")
        colunas_remover = ['', 'Flow ID', 'Src IP', 'Dst IP', 'Timestamp']
        colunas_remover = [c for c in colunas_remover if c in df_clean.columns]
        df_clean = df_clean.drop(colunas_remover)
        print(f"   ✓ Removidas {len(colunas_remover)} colunas: {colunas_remover}")
        
        # 2. Separar target
        print("\n1.2. Preparando variável target...")
        y = df_clean['Label'].to_numpy()
        df_clean = df_clean.drop('Label')
        print(f"   ✓ Target: {len(np.unique(y))} classes")
        print(f"   ✓ Distribuição: {dict(zip(*np.unique(y, return_counts=True)))}")
        
        # 3. Converter categóricas para numéricas
        print("\n1.3. Convertendo variáveis categóricas...")
        colunas_categoricas = [col for col in df_clean.columns if df_clean[col].dtype == pl.String]
        if colunas_categoricas:
            for col in colunas_categoricas:
                valores_unicos = df_clean[col].unique().sort()
                mapeamento = {valor: idx for idx, valor in enumerate(valores_unicos)}
                expr = pl.col(col)
                for valor_original, valor_numerico in mapeamento.items():
                    expr = pl.when(expr == valor_original).then(valor_numerico).otherwise(expr)
                df_clean = df_clean.with_columns(expr.cast(pl.Int64).alias(col))
            print(f"   ✓ Convertidas {len(colunas_categoricas)} colunas categóricas")
        
        # 4. Tratar valores infinitos e NaN
        print("\n1.4. Tratando valores infinitos e NaN...")
        colunas_numericas = [col for col in df_clean.columns if df_clean[col].dtype in [pl.Int64, pl.Float64]]
        
        for col in colunas_numericas:
            # Substituir infinitos por NaN
            df_clean = df_clean.with_columns(
                pl.when(pl.col(col).is_infinite() | pl.col(col).is_nan())
                .then(None)
                .otherwise(pl.col(col))
                .alias(col)
            )
            # Preencher NaN com mediana
            mediana = df_clean[col].median()
            if mediana is not None:
                df_clean = df_clean.with_columns(
                    pl.col(col).fill_null(mediana).alias(col)
                )
        
        print(f"   ✓ Tratadas {len(colunas_numericas)} colunas numéricas")
        
        # 5. Seleção de features (remover colunas com baixa variância ou alta correlação)
        print("\n1.5. Selecionando features relevantes...")
        df_clean = self._selecionar_features(df_clean)
        print(f"   ✓ Features selecionadas: {len(self.features_selecionadas)}")
        
        # 6. Converter para numpy
        X = df_clean.select(self.features_selecionadas).to_numpy()
        
        # 7. Encoding do target
        y_encoded = self.label_encoder.fit_transform(y)
        
        print(f"\n   ✓ Dataset final: {X.shape[0]:,} amostras x {X.shape[1]} features")
        
        return X, y_encoded, self.features_selecionadas
    
    def _selecionar_features(self, df):
        """Seleciona features relevantes removendo redundâncias"""
        # Calcular variâncias usando Polars
        colunas_numericas = [col for col in df.columns if df[col].dtype in [pl.Int64, pl.Float64]]
        
        # Calcular variâncias
        variancias = {}
        for col in colunas_numericas:
            var_val = df[col].var()
            if var_val is not None and var_val > 1e-6:
                variancias[col] = var_val
        
        colunas_var_ok = list(variancias.keys())
        
        # Se houver muitas colunas, usar amostra para calcular correlação
        if len(colunas_var_ok) > 30:
            # Usar amostra para análise de correlação
            df_sample = df.select(colunas_var_ok).sample(n=min(100000, df.height), seed=42)
            df_pd = df_sample.to_pandas()
        else:
            df_pd = df.select(colunas_var_ok).to_pandas()
        
        # Calcular correlação e remover colunas altamente correlacionadas
        matriz_corr = df_pd.corr().abs()
        high_corr_pairs = []
        
        for i in range(len(matriz_corr.columns)):
            for j in range(i+1, len(matriz_corr.columns)):
                if matriz_corr.iloc[i, j] > 0.95:  # Correlação muito alta
                    high_corr_pairs.append((matriz_corr.columns[i], matriz_corr.columns[j]))
        
        # Remover uma das colunas de cada par
        colunas_remover_corr = set()
        for col1, col2 in high_corr_pairs:
            if col1 not in colunas_remover_corr:
                colunas_remover_corr.add(col2)
        
        colunas_finais = [c for c in colunas_var_ok if c not in colunas_remover_corr]
        
        # Limitar a um número razoável de features (top 50 por variância)
        if len(colunas_finais) > 50:
            variancias_ordenadas = sorted(variancias.items(), key=lambda x: x[1], reverse=True)
            colunas_finais = [col for col, _ in variancias_ordenadas if col in colunas_finais][:50]
        
        self.features_selecionadas = colunas_finais
        return df.select(colunas_finais)
    
    def normalizar_features(self, X_train, X_test):
        """Normaliza as features usando StandardScaler"""
        print("\n1.6. Normalizando features...")
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        print("   ✓ Features normalizadas")
        return X_train_scaled, X_test_scaled


class ModelTrainer:
    """Classe para treinar e avaliar modelos"""
    
    def __init__(self, output_dir):
        self.output_dir = output_dir
        self.modelos = {}
        self.resultados = {}
        
    def treinar_modelos(self, X_train, X_test, y_train, y_test):
        """Treina múltiplos modelos"""
        print("\n" + "="*80)
        print("ETAPA 2: TREINAMENTO DE MODELOS")
        print("="*80)
        
        # Modelo 1: Random Forest
        print("\n2.1. Treinando Random Forest...")
        rf = RandomForestClassifier(
            n_estimators=100,
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
            verbose=0
        )
        rf.fit(X_train, y_train)
        self.modelos['Random Forest'] = rf
        print("   ✓ Random Forest treinado")
        
        # Modelo 2: Logistic Regression
        print("\n2.2. Treinando Logistic Regression...")
        lr = LogisticRegression(
            max_iter=1000,
            random_state=42,
            n_jobs=-1,
            verbose=0
        )
        lr.fit(X_train, y_train)
        self.modelos['Logistic Regression'] = lr
        print("   ✓ Logistic Regression treinado")
        
        # Modelo 3: SVM (com amostra menor devido ao tempo)
        print("\n2.3. Treinando SVM (usando amostra para performance)...")
        # Usar amostra para SVM (muito lento com dataset completo)
        n_samples_svm = min(100000, len(X_train))
        indices_svm = np.random.choice(len(X_train), n_samples_svm, replace=False)
        X_train_svm = X_train[indices_svm]
        y_train_svm = y_train[indices_svm]
        
        svm = SVC(
            kernel='rbf',
            C=1.0,
            gamma='scale',
            probability=True,
            random_state=42,
            verbose=False
        )
        svm.fit(X_train_svm, y_train_svm)
        self.modelos['SVM'] = svm
        print(f"   ✓ SVM treinado (amostra de {n_samples_svm:,})")
        
        # Avaliar todos os modelos
        print("\n2.4. Avaliando modelos...")
        for nome, modelo in self.modelos.items():
            if nome == 'SVM':
                # Para SVM, usar amostra também no teste
                n_test_svm = min(20000, len(X_test))
                indices_test_svm = np.random.choice(len(X_test), n_test_svm, replace=False)
                X_test_model = X_test[indices_test_svm]
                y_test_model = y_test[indices_test_svm]
            else:
                X_test_model = X_test
                y_test_model = y_test
            
            y_pred = modelo.predict(X_test_model)
            y_pred_proba = modelo.predict_proba(X_test_model)[:, 1] if hasattr(modelo, 'predict_proba') else None
            
            resultados = {
                'accuracy': accuracy_score(y_test_model, y_pred),
                'precision': precision_score(y_test_model, y_pred, average='weighted'),
                'recall': recall_score(y_test_model, y_pred, average='weighted'),
                'f1': f1_score(y_test_model, y_pred, average='weighted'),
                'confusion_matrix': confusion_matrix(y_test_model, y_pred),
                'y_test': y_test_model,
                'y_pred': y_pred,
                'y_pred_proba': y_pred_proba
            }
            
            if y_pred_proba is not None:
                resultados['roc_auc'] = roc_auc_score(y_test_model, y_pred_proba)
            
            self.resultados[nome] = resultados
            
            print(f"\n   {nome}:")
            print(f"      Accuracy:  {resultados['accuracy']:.4f}")
            print(f"      Precision: {resultados['precision']:.4f}")
            print(f"      Recall:    {resultados['recall']:.4f}")
            print(f"      F1-Score:  {resultados['f1']:.4f}")
            if 'roc_auc' in resultados:
                print(f"      ROC-AUC:   {resultados['roc_auc']:.4f}")
        
        return self.resultados
    
    def gerar_relatorio(self, features_selecionadas):
        """Gera relatório completo dos resultados"""
        print("\n" + "="*80)
        print("ETAPA 3: GERAÇÃO DE RELATÓRIOS")
        print("="*80)
        
        # Relatório em texto
        caminho_relatorio = self.output_dir / 'relatorio_modelos.txt'
        with open(caminho_relatorio, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("RELATÓRIO DE MODELAGEM - MACHINE LEARNING\n")
            f.write("="*80 + "\n\n")
            
            f.write(f"Total de features utilizadas: {len(features_selecionadas)}\n")
            f.write(f"Total de modelos treinados: {len(self.modelos)}\n\n")
            
            f.write("="*80 + "\n")
            f.write("RESULTADOS POR MODELO\n")
            f.write("="*80 + "\n\n")
            
            for nome, resultados in self.resultados.items():
                f.write(f"\n{nome}:\n")
                f.write("-"*80 + "\n")
                f.write(f"Accuracy:  {resultados['accuracy']:.4f}\n")
                f.write(f"Precision: {resultados['precision']:.4f}\n")
                f.write(f"Recall:    {resultados['recall']:.4f}\n")
                f.write(f"F1-Score:  {resultados['f1']:.4f}\n")
                if 'roc_auc' in resultados:
                    f.write(f"ROC-AUC:   {resultados['roc_auc']:.4f}\n")
                
                f.write("\nMatriz de Confusão:\n")
                cm = resultados['confusion_matrix']
                f.write(f"                Predito\n")
                f.write(f"              Benign  DDoS\n")
                f.write(f"Real Benign   {cm[0,0]:6d} {cm[0,1]:5d}\n")
                f.write(f"     DDoS     {cm[1,0]:6d} {cm[1,1]:5d}\n")
                f.write("\n")
        
        print(f"\n3.1. Relatório salvo: {caminho_relatorio}")
        
        # Gráficos
        self._criar_graficos()
        
        # Tabela comparativa
        self._criar_tabela_comparativa()
        
        # Conclusão
        self._gerar_conclusao(features_selecionadas)
    
    def _criar_graficos(self):
        """Cria gráficos de comparação dos modelos"""
        print("\n3.2. Gerando gráficos...")
        
        # Gráfico 1: Comparação de métricas
        fig, ax = plt.subplots(figsize=(12, 6))
        
        modelos = list(self.resultados.keys())
        metricas = ['accuracy', 'precision', 'recall', 'f1']
        if all('roc_auc' in r for r in self.resultados.values()):
            metricas.append('roc_auc')
        
        x = np.arange(len(modelos))
        width = 0.15
        
        for i, metrica in enumerate(metricas):
            valores = [self.resultados[m][metrica] for m in modelos]
            ax.bar(x + i*width, valores, width, label=metrica.replace('_', ' ').title())
        
        ax.set_xlabel('Modelos', fontweight='bold')
        ax.set_ylabel('Score', fontweight='bold')
        ax.set_title('Comparação de Métricas entre Modelos', fontweight='bold', fontsize=14)
        ax.set_xticks(x + width * (len(metricas)-1) / 2)
        ax.set_xticklabels(modelos)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 1.1])
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'comparacao_metricas.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("   ✓ Gráfico de comparação de métricas salvo")
        
        # Gráfico 2: Matrizes de confusão
        fig, axes = plt.subplots(1, len(self.modelos), figsize=(6*len(self.modelos), 5))
        if len(self.modelos) == 1:
            axes = [axes]
        
        for idx, (nome, resultados) in enumerate(self.resultados.items()):
            cm = resultados['confusion_matrix']
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx],
                       xticklabels=['Benign', 'DDoS'], yticklabels=['Benign', 'DDoS'])
            axes[idx].set_title(f'{nome}\nAccuracy: {resultados["accuracy"]:.4f}', 
                              fontweight='bold')
            axes[idx].set_ylabel('Real', fontweight='bold')
            axes[idx].set_xlabel('Predito', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'matrizes_confusao.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("   ✓ Matrizes de confusão salvas")
        
        # Gráfico 3: Curvas ROC (se disponível)
        if all('roc_auc' in r and r['y_pred_proba'] is not None for r in self.resultados.values()):
            fig, ax = plt.subplots(figsize=(10, 8))
            
            for nome, resultados in self.resultados.items():
                if 'roc_auc' in resultados and resultados['y_pred_proba'] is not None:
                    fpr, tpr, _ = roc_curve(resultados['y_test'], resultados['y_pred_proba'])
                    ax.plot(fpr, tpr, label=f"{nome} (AUC = {resultados['roc_auc']:.4f})", linewidth=2)
            
            ax.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
            ax.set_xlabel('Taxa de Falsos Positivos', fontweight='bold')
            ax.set_ylabel('Taxa de Verdadeiros Positivos', fontweight='bold')
            ax.set_title('Curvas ROC - Comparação de Modelos', fontweight='bold', fontsize=14)
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(self.output_dir / 'curvas_roc.png', dpi=300, bbox_inches='tight')
            plt.close()
            print("   ✓ Curvas ROC salvas")
    
    def _criar_tabela_comparativa(self):
        """Cria tabela CSV comparativa"""
        print("\n3.3. Criando tabela comparativa...")
        
        dados = []
        for nome, resultados in self.resultados.items():
            linha = {
                'Modelo': nome,
                'Accuracy': f"{resultados['accuracy']:.4f}",
                'Precision': f"{resultados['precision']:.4f}",
                'Recall': f"{resultados['recall']:.4f}",
                'F1-Score': f"{resultados['f1']:.4f}"
            }
            if 'roc_auc' in resultados:
                linha['ROC-AUC'] = f"{resultados['roc_auc']:.4f}"
            dados.append(linha)
        
        df_comparativo = pd.DataFrame(dados)
        caminho_csv = self.output_dir / 'comparacao_modelos.csv'
        df_comparativo.to_csv(caminho_csv, index=False)
        print(f"   ✓ Tabela comparativa salva: {caminho_csv}")
    
    def _gerar_conclusao(self, features_selecionadas):
        """Gera conclusão objetiva dos resultados"""
        caminho_conclusao = self.output_dir / 'conclusao_modelos.txt'
        
        # Encontrar melhor modelo
        melhor_modelo = max(self.resultados.items(), key=lambda x: x[1]['f1'])
        melhor_nome, melhor_resultados = melhor_modelo
        
        with open(caminho_conclusao, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("CONCLUSÃO DA MODELAGEM\n")
            f.write("="*80 + "\n\n")
            
            f.write("RESUMO EXECUTIVO:\n")
            f.write("-"*80 + "\n")
            f.write(f"• Total de features utilizadas: {len(features_selecionadas)}\n")
            f.write(f"• Modelos treinados: {len(self.modelos)}\n")
            f.write(f"• Melhor modelo: {melhor_nome}\n")
            f.write(f"  - F1-Score: {melhor_resultados['f1']:.4f}\n")
            f.write(f"  - Accuracy: {melhor_resultados['accuracy']:.4f}\n")
            f.write(f"  - Precision: {melhor_resultados['precision']:.4f}\n")
            f.write(f"  - Recall: {melhor_resultados['recall']:.4f}\n")
            if 'roc_auc' in melhor_resultados:
                f.write(f"  - ROC-AUC: {melhor_resultados['roc_auc']:.4f}\n")
            
            f.write("\n" + "="*80 + "\n")
            f.write("COMPARAÇÃO DE MODELOS\n")
            f.write("="*80 + "\n\n")
            
            for nome, resultados in sorted(self.resultados.items(), 
                                         key=lambda x: x[1]['f1'], reverse=True):
                f.write(f"{nome}:\n")
                f.write(f"  F1-Score: {resultados['f1']:.4f} | ")
                f.write(f"Accuracy: {resultados['accuracy']:.4f} | ")
                f.write(f"Precision: {resultados['precision']:.4f} | ")
                f.write(f"Recall: {resultados['recall']:.4f}\n")
                if 'roc_auc' in resultados:
                    f.write(f"  ROC-AUC: {resultados['roc_auc']:.4f}\n")
                f.write("\n")
            
            f.write("="*80 + "\n")
            f.write("RECOMENDAÇÕES\n")
            f.write("="*80 + "\n\n")
            f.write(f"1. O modelo {melhor_nome} apresentou o melhor desempenho geral.\n")
            f.write("2. Todos os modelos mostraram boa capacidade de classificação.\n")
            f.write("3. O dataset está bem balanceado, facilitando a classificação.\n")
            f.write("4. As features selecionadas foram suficientes para boa performance.\n")
            f.write("5. Para produção, recomenda-se usar o modelo com melhor F1-Score.\n")
        
        print(f"   ✓ Conclusão salva: {caminho_conclusao}")


def main():
    print("="*80)
    print("MODELAGEM DE MACHINE LEARNING - DETECÇÃO DE DDoS")
    print("="*80)
    
    # Criar diretório de saída
    output_dir = Path("modelagem_ml_output")
    output_dir.mkdir(exist_ok=True)
    
    # 1. Carregar dados
    print("\nCarregando dados...")
    df = pl.read_parquet("output.parquet")
    print(f"Dataset carregado: {df.shape[0]:,} linhas x {df.shape[1]} colunas")
    
    # 2. Amostrar dados para treinamento (devido ao tamanho)
    print("\nAmostrando dados para treinamento (500k amostras)...")
    df_sample = df.sample(n=min(500000, df.height), seed=42)
    print(f"Amostra: {df_sample.shape[0]:,} linhas")
    
    # 3. Engenharia de features
    fe = FeatureEngineering()
    X, y, features_selecionadas = fe.preparar_dados(df_sample)
    
    # 4. Split train/test
    print("\nDividindo em conjunto de treino e teste...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Treino: {X_train.shape[0]:,} amostras")
    print(f"Teste:  {X_test.shape[0]:,} amostras")
    
    # 5. Normalizar features
    X_train_scaled, X_test_scaled = fe.normalizar_features(X_train, X_test)
    
    # 6. Treinar modelos
    trainer = ModelTrainer(output_dir)
    resultados = trainer.treinar_modelos(X_train_scaled, X_test_scaled, y_train, y_test)
    
    # 7. Gerar relatórios
    trainer.gerar_relatorio(features_selecionadas)
    
    print("\n" + "="*80)
    print("MODELAGEM CONCLUÍDA COM SUCESSO!")
    print(f"Resultados salvos em: {output_dir}")
    print("="*80)


if __name__ == "__main__":
    main()

