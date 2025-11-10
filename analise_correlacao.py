
import polars as pl
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Configuração de estilo
try:
    plt.style.use('seaborn-v0_8-darkgrid')
except:
    try:
        plt.style.use('seaborn-darkgrid')
    except:
        plt.style.use('default')
plt.rcParams['figure.figsize'] = (16, 12)
plt.rcParams['font.size'] = 9

def converter_categoricas_para_numericas(df):
    """Converte colunas categóricas (String) para numéricas usando label encoding"""
    df_encoded = df.clone()
    mapeamentos = {}
    
    colunas_categoricas = [col for col in df.columns if df[col].dtype == pl.String]
    
    print(f"\n   Convertendo {len(colunas_categoricas)} colunas categóricas para numéricas:")
    
    for col in colunas_categoricas:
        # Obter valores únicos
        valores_unicos = df[col].unique().sort()
        
        # Criar mapeamento
        mapeamento = {valor: idx for idx, valor in enumerate(valores_unicos)}
        mapeamentos[col] = mapeamento
        
        # Aplicar label encoding usando uma abordagem mais eficiente
        # Para colunas com muitos valores únicos, usar rank
        if len(valores_unicos) > 1000:
            # Para muitas categorias, usar rank (mais eficiente)
            df_encoded = df_encoded.with_columns(
                (pl.col(col).rank("dense") - 1).cast(pl.Int64).alias(f"{col}_encoded")
            )
        else:
            # Para poucas categorias, usar when/then
            expr = pl.col(col)
            for valor_original, valor_numerico in mapeamento.items():
                expr = pl.when(expr == valor_original).then(valor_numerico).otherwise(expr)
            
            df_encoded = df_encoded.with_columns(
                expr.cast(pl.Int64).alias(f"{col}_encoded")
            )
        
        # Remover coluna original e renomear
        df_encoded = df_encoded.drop(col).rename({f"{col}_encoded": col})
        
        print(f"      ✓ {col}: {len(valores_unicos)} valores únicos")
    
    return df_encoded, mapeamentos

def limpar_dados_para_correlacao(df):
    """Remove colunas não numéricas, categóricas codificadas e a variável alvo 'Label'"""
    df_clean = df.clone()

    # Remover coluna de índice vazia se existir
    if '' in df_clean.columns:
        df_clean = df_clean.drop('')

    # Excluir variável alvo (Label) se existir
    if 'Label' in df_clean.columns:
        print("   Removendo coluna 'Label' (variável alvo) da análise de correlação...")
        df_clean = df_clean.drop('Label')

    # Selecionar apenas colunas numéricas
    colunas_numericas = [
        col for col in df_clean.columns
        if df_clean[col].dtype in [pl.Int64, pl.Float64]
    ]

    print(f"   Mantendo {len(colunas_numericas)} colunas numéricas para correlação")

    # Substituir infinitos e NaN por None
    for col in colunas_numericas:
        df_clean = df_clean.with_columns(
            pl.when(pl.col(col).is_infinite() | pl.col(col).is_nan())
            .then(None)
            .otherwise(pl.col(col))
            .alias(col)
        )

    # Selecionar apenas as colunas válidas
    df_clean = df_clean.select(colunas_numericas)

    return df_clean


def calcular_matriz_correlacao(df):
    """Calcula matriz de correlação de Pearson usando método eficiente"""
    print("      Calculando correlações...")
    
    colunas = df.columns
    
    # Converter para numpy para cálculo eficiente de correlação
    # Usar amostra se o dataset for muito grande para melhor performance
    n_rows = df.height
    if n_rows > 1_000_000:
        print(f"      Dataset grande ({n_rows:,} linhas). Usando amostra de 1M linhas para cálculo de correlação...")
        df_sample = df.sample(n=1_000_000, seed=42)
        dados = df_sample.to_numpy()
    else:
        dados = df.to_numpy()
    
    # Remover colunas com variância zero (não podem ter correlação)
    variancias = np.var(dados, axis=0)
    indices_validos = ~np.isnan(variancias) & (variancias > 1e-10)
    
    if not np.all(indices_validos):
        print(f"      Removendo {np.sum(~indices_validos)} colunas com variância zero...")
        dados = dados[:, indices_validos]
        colunas = [colunas[i] for i in range(len(colunas)) if indices_validos[i]]
    
    # Calcular correlação de Pearson
    matriz_corr = np.corrcoef(dados, rowvar=False)
    
    # Substituir NaN por 0 (ocorre quando há variância zero)
    matriz_corr = np.nan_to_num(matriz_corr, nan=0.0)
    
    # Criar DataFrame com nomes das colunas
    df_corr = pl.DataFrame(matriz_corr, schema=colunas)
    
    return df_corr, matriz_corr, colunas

def criar_heatmap_correlacao(matriz_corr, colunas, output_dir):
    """Cria heatmap QUADRADO da matriz de correlação (sem mascarar a metade superior)."""
    fig, ax = plt.subplots(figsize=(20, 16))

    sns.heatmap(
        matriz_corr,
        annot=False,          # sem valores em cada célula (muita info)
        fmt='.2f',
        cmap='coolwarm',
        center=0,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8},
        xticklabels=colunas,
        yticklabels=colunas,
        ax=ax
    )

    ax.set_title(
        'Matriz de Correlação de Pearson - Todas as Variáveis',
        fontsize=16,
        fontweight='bold',
        pad=20
    )
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)

    plt.tight_layout()
    caminho_arquivo = output_dir / 'heatmap_correlacao_completo.png'
    plt.savefig(caminho_arquivo, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"   Heatmap completo salvo: {caminho_arquivo}")

    # Mantém o heatmap focado no top 30 como está
    criar_heatmap_focado(matriz_corr, colunas, output_dir)


def criar_heatmap_focado(matriz_corr, colunas, output_dir):
    """Cria heatmap QUADRADO das 10 variáveis com maior variância das correlações."""
    # Calcular variância das correlações para cada variável
    var_corr = np.var(matriz_corr, axis=0)
    
    # Selecionar top 10 variáveis com maior variância
    top_indices = np.argsort(var_corr)[-10:][::-1]
    top_colunas = [colunas[i] for i in top_indices]
    
    # Extrair submatriz
    submatriz = matriz_corr[np.ix_(top_indices, top_indices)]
    
    fig, ax = plt.subplots(figsize=(14, 12))
    
    sns.heatmap(
        submatriz,
        annot=True,
        fmt='.2f',
        cmap='coolwarm',
        center=0,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8},
        xticklabels=top_colunas,
        yticklabels=top_colunas,
        ax=ax
    )
    
    ax.set_title(
        'Matriz de Correlação de Pearson - Top 10 Variáveis (Maior Variância)',
        fontsize=14,
        fontweight='bold',
        pad=20
    )
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    plt.tight_layout()
    caminho_arquivo = output_dir / 'heatmap_correlacao_top10.png'
    plt.savefig(caminho_arquivo, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"   Heatmap top 10 salvo: {caminho_arquivo}")


def analisar_correlacoes_fortes(matriz_corr, colunas, threshold=0.7):
    """Identifica e analisa correlações fortes (acima do threshold)"""
    correlacoes_fortes = []
    
    n = len(colunas)
    for i in range(n):
        for j in range(i + 1, n):
            corr_val = matriz_corr[i, j]
            if abs(corr_val) >= threshold:
                correlacoes_fortes.append({
                    'var1': colunas[i],
                    'var2': colunas[j],
                    'correlacao': corr_val,
                    'abs_correlacao': abs(corr_val)
                })
    
    # Ordenar por valor absoluto de correlação
    correlacoes_fortes.sort(key=lambda x: x['abs_correlacao'], reverse=True)
    
    return correlacoes_fortes

def gerar_relatorio_correlacao(df_corr, matriz_corr, colunas, correlacoes_fortes, output_dir):
    """Gera relatório detalhado de correlação"""
    caminho_relatorio = output_dir / 'relatorio_correlacao.txt'
    
    with open(caminho_relatorio, 'w', encoding='utf-8') as f:
        f.write("=" * 100 + "\n")
        f.write("RELATÓRIO DE ANÁLISE DE CORRELAÇÃO DE PEARSON\n")
        f.write("=" * 100 + "\n\n")
        
        f.write(f"Total de variáveis analisadas: {len(colunas)}\n")
        f.write(f"Total de pares de variáveis: {len(colunas) * (len(colunas) - 1) // 2}\n\n")
        
        # Estatísticas gerais
        valores_corr = matriz_corr
        valores_corr_triang = valores_corr[np.triu_indices_from(valores_corr, k=1)]
        
        f.write("=" * 100 + "\n")
        f.write("ESTATÍSTICAS GERAIS DAS CORRELAÇÕES\n")
        f.write("=" * 100 + "\n\n")
        f.write(f"Média das correlações: {np.mean(valores_corr_triang):.4f}\n")
        f.write(f"Mediana das correlações: {np.median(valores_corr_triang):.4f}\n")
        f.write(f"Desvio padrão das correlações: {np.std(valores_corr_triang):.4f}\n")
        f.write(f"Correlação mínima: {np.min(valores_corr_triang):.4f}\n")
        f.write(f"Correlação máxima: {np.max(valores_corr_triang):.4f}\n\n")
        
        # Distribuição de correlações
        f.write("=" * 100 + "\n")
        f.write("DISTRIBUIÇÃO DE CORRELAÇÕES\n")
        f.write("=" * 100 + "\n\n")
        f.write(f"Correlações muito fortes (|r| >= 0.9): {np.sum(np.abs(valores_corr_triang) >= 0.9)}\n")
        f.write(f"Correlações fortes (0.7 <= |r| < 0.9): {np.sum((np.abs(valores_corr_triang) >= 0.7) & (np.abs(valores_corr_triang) < 0.9))}\n")
        f.write(f"Correlações moderadas (0.5 <= |r| < 0.7): {np.sum((np.abs(valores_corr_triang) >= 0.5) & (np.abs(valores_corr_triang) < 0.7))}\n")
        f.write(f"Correlações fracas (0.3 <= |r| < 0.5): {np.sum((np.abs(valores_corr_triang) >= 0.3) & (np.abs(valores_corr_triang) < 0.5))}\n")
        f.write(f"Correlações muito fracas (|r| < 0.3): {np.sum(np.abs(valores_corr_triang) < 0.3)}\n\n")
        
        # Correlações fortes
        f.write("=" * 100 + "\n")
        f.write("CORRELAÇÕES FORTES (|r| >= 0.7)\n")
        f.write("=" * 100 + "\n\n")
        
        if len(correlacoes_fortes) == 0:
            f.write("Nenhuma correlação forte encontrada (threshold: 0.7)\n\n")
        else:
            f.write(f"Total de correlações fortes encontradas: {len(correlacoes_fortes)}\n\n")
            for idx, corr in enumerate(correlacoes_fortes[:50], 1):  # Top 50
                sinal = "positiva" if corr['correlacao'] > 0 else "negativa"
                f.write(f"{idx:3d}. {corr['var1']:30s} <-> {corr['var2']:30s} | "
                       f"r = {corr['correlacao']:7.4f} ({sinal})\n")
        
        # Top 20 correlações mais fortes
        f.write("\n" + "=" * 100 + "\n")
        f.write("TOP 20 CORRELAÇÕES MAIS FORTES\n")
        f.write("=" * 100 + "\n\n")
        
        todas_correlacoes = []
        n = len(colunas)
        for i in range(n):
            for j in range(i + 1, n):
                todas_correlacoes.append({
                    'var1': colunas[i],
                    'var2': colunas[j],
                    'correlacao': matriz_corr[i, j],
                    'abs_correlacao': abs(matriz_corr[i, j])
                })
        
        todas_correlacoes.sort(key=lambda x: x['abs_correlacao'], reverse=True)
        
        for idx, corr in enumerate(todas_correlacoes[:20], 1):
            sinal = "+" if corr['correlacao'] > 0 else "-"
            f.write(f"{idx:2d}. {corr['var1']:35s} <-> {corr['var2']:35s} | "
                   f"r = {sinal}{corr['abs_correlacao']:.4f}\n")
    
    print(f"   Relatório salvo: {caminho_relatorio}")

def gerar_conclusao(correlacoes_fortes, valores_corr_triang, output_dir):
    """Gera conclusão objetiva e clara da análise"""
    caminho_conclusao = output_dir / 'conclusao_correlacao.txt'
    
    with open(caminho_conclusao, 'w', encoding='utf-8') as f:
        f.write("=" * 100 + "\n")
        f.write("CONCLUSÃO DA ANÁLISE DE CORRELAÇÃO DE PEARSON\n")
        f.write("=" * 100 + "\n\n")
        
        # Resumo estatístico
        f.write("RESUMO ESTATÍSTICO:\n")
        f.write("-" * 100 + "\n")
        f.write(f"• Total de pares de variáveis analisados: {len(valores_corr_triang):,}\n")
        f.write(f"• Correlação média: {np.mean(valores_corr_triang):.4f}\n")
        f.write(f"• Correlação mediana: {np.median(valores_corr_triang):.4f}\n")
        f.write(f"• Correlações muito fortes (|r| ≥ 0.9): {np.sum(np.abs(valores_corr_triang) >= 0.9)}\n")
        f.write(f"• Correlações fortes (0.7 ≤ |r| < 0.9): {np.sum((np.abs(valores_corr_triang) >= 0.7) & (np.abs(valores_corr_triang) < 0.9))}\n")
        f.write(f"• Correlações moderadas (0.5 ≤ |r| < 0.7): {np.sum((np.abs(valores_corr_triang) >= 0.5) & (np.abs(valores_corr_triang) < 0.7))}\n\n")
        
        # Principais descobertas
        f.write("PRINCIPAIS DESCOBERTAS:\n")
        f.write("-" * 100 + "\n")
        
        if len(correlacoes_fortes) > 0:
            f.write(f"1. Foram identificadas {len(correlacoes_fortes)} correlações fortes (|r| ≥ 0.7), indicando:\n")
            f.write("   • Relações lineares significativas entre essas variáveis\n")
            f.write("   • Possível redundância de informação (multicolinearidade)\n")
            f.write("   • Oportunidade de redução de dimensionalidade\n\n")
            
            # Agrupar por tipo de correlação
            positivas = [c for c in correlacoes_fortes if c['correlacao'] > 0]
            negativas = [c for c in correlacoes_fortes if c['correlacao'] < 0]
            
            f.write(f"2. Distribuição das correlações fortes:\n")
            f.write(f"   • Correlações positivas: {len(positivas)} ({len(positivas)/len(correlacoes_fortes)*100:.1f}%)\n")
            f.write(f"   • Correlações negativas: {len(negativas)} ({len(negativas)/len(correlacoes_fortes)*100:.1f}%)\n\n")
            
            if len(correlacoes_fortes) >= 3:
                f.write("3. Top 3 correlações mais fortes:\n")
                for idx, corr in enumerate(correlacoes_fortes[:3], 1):
                    direcao = "aumentam juntas" if corr['correlacao'] > 0 else "variam inversamente"
                    f.write(f"   {idx}. {corr['var1']} e {corr['var2']} (r = {corr['correlacao']:.4f}) - {direcao}\n")
                f.write("\n")
        else:
            f.write("1. Não foram encontradas correlações muito fortes (|r| ≥ 0.7)\n")
            f.write("   • As variáveis são relativamente independentes\n")
            f.write("   • Baixa multicolinearidade no dataset\n\n")
        
        # Impacto e recomendações
        f.write("IMPACTO E RECOMENDAÇÕES:\n")
        f.write("-" * 100 + "\n")
        
        if len(correlacoes_fortes) > 0:
            f.write("1. Multicolinearidade: Variáveis altamente correlacionadas podem:\n")
            f.write("   • Inflacionar a variância dos coeficientes em modelos de regressão\n")
            f.write("   • Tornar a interpretação dos modelos mais difícil\n")
            f.write("   • Sugerir que algumas variáveis podem ser removidas sem perda significativa de informação\n\n")
            
            f.write("2. Redução de Dimensionalidade:\n")
            f.write("   • Considere aplicar técnicas como PCA ou seleção de features\n")
            f.write("   • Remova variáveis redundantes para simplificar modelos\n\n")
            
            f.write("3. Análise de Features:\n")
            f.write("   • Variáveis com correlação muito alta (|r| > 0.95) podem ser consideradas redundantes\n")
            f.write("   • Mantenha apenas uma delas ou crie uma variável combinada\n\n")
        else:
            f.write("1. Independência das Variáveis:\n")
            f.write("   • As variáveis são relativamente independentes\n")
            f.write("   • Todas as features podem ser úteis para modelagem\n")
            f.write("   • Baixo risco de multicolinearidade\n\n")
        
        f.write("4. Visualização:\n")
        f.write("   • Os heatmaps gerados mostram padrões de correlação visualmente\n")
        f.write("   • Use o heatmap completo para visão geral\n")
        f.write("   • Use o heatmap top 30 para análise detalhada das variáveis mais relevantes\n\n")
        
        # Conclusão final
        f.write("=" * 100 + "\n")
        f.write("CONCLUSÃO FINAL\n")
        f.write("=" * 100 + "\n\n")
        
        if len(correlacoes_fortes) > 0:
            f.write(f"A análise revelou {len(correlacoes_fortes)} correlações fortes entre variáveis, ")
            f.write("indicando que há redundância de informação no dataset. ")
            f.write("Recomenda-se aplicar técnicas de redução de dimensionalidade ou ")
            f.write("seleção de features antes de construir modelos preditivos. ")
            f.write("As correlações identificadas podem ser exploradas para melhor compreensão ")
            f.write("das relações entre as características do tráfego de rede.\n")
        else:
            f.write("A análise revelou que as variáveis são relativamente independentes, ")
            f.write("com baixa multicolinearidade. Isso sugere que todas as features podem ")
            f.write("ser úteis para modelagem, sem necessidade imediata de redução de ")
            f.write("dimensionalidade. As correlações moderadas e fracas indicam que cada ")
            f.write("variável contribui com informação única para a análise.\n")
    
    print(f"   Conclusão salvo: {caminho_conclusao}")

def salvar_matriz_correlacao(df_corr, colunas, output_dir):
    """Salva matriz de correlação em CSV"""
    # Adicionar nomes das colunas
    df_corr_completo = df_corr.with_columns(
        pl.Series("Variável", colunas)
    ).select(["Variável"] + colunas)
    
    caminho_csv = output_dir / 'matriz_correlacao.csv'
    df_corr_completo.write_csv(caminho_csv)
    print(f"   Matriz de correlação salva: {caminho_csv}")

def main():
    print("Iniciando análise de correlação de Pearson...")
    print("=" * 100)
    
    # Ler arquivo parquet
    print("\n1. Carregando dados do arquivo parquet...")
    df = pl.read_parquet("./ddos_balanced/balanced_reduzido.parquet")
    print(f"   Dataset carregado: {df.shape[0]:,} linhas x {df.shape[1]} colunas")
    
    # Converter colunas categóricas para numéricas
    print("\n2. Convertendo colunas categóricas para numéricas...")
    df_encoded, mapeamentos = converter_categoricas_para_numericas(df)
    print(f"   ✓ Conversão concluída")
    
    # Limpar dados
    print("\n3. Limpando dados (removendo infinitos e NaN)...")
    df_clean = limpar_dados_para_correlacao(df_encoded)
    print(f"   Variáveis numéricas finais: {len(df_clean.columns)}")
    
    # Calcular matriz de correlação
    print("\n4. Calculando matriz de correlação de Pearson...")
    df_corr, matriz_corr, colunas = calcular_matriz_correlacao(df_clean)
    print(f"   ✓ Matriz de correlação calculada: {len(colunas)}x{len(colunas)}")
    
    # Criar diretório de saída
    output_dir = Path("analise_correlacao_output")
    output_dir.mkdir(exist_ok=True)
    print(f"\n5. Diretório de saída: {output_dir}")
    
    # Criar heatmaps
    print("\n6. Gerando heatmaps de correlação...")
    criar_heatmap_correlacao(matriz_corr, colunas, output_dir)
    
    # Analisar correlações fortes
    print("\n7. Analisando correlações fortes...")
    correlacoes_fortes = analisar_correlacoes_fortes(matriz_corr, colunas, threshold=0.7)
    print(f"   Correlações fortes encontradas (|r| >= 0.7): {len(correlacoes_fortes)}")
    
    # Gerar relatório
    print("\n8. Gerando relatório detalhado...")
    valores_corr_triang = matriz_corr[np.triu_indices_from(matriz_corr, k=1)]
    gerar_relatorio_correlacao(df_corr, matriz_corr, colunas, correlacoes_fortes, output_dir)
    
    # Gerar conclusão
    print("\n9. Gerando conclusão da análise...")
    gerar_conclusao(correlacoes_fortes, valores_corr_triang, output_dir)
    
    # Salvar matriz de correlação
    print("\n10. Salvando matriz de correlação...")
    salvar_matriz_correlacao(df_corr, colunas, output_dir)
    
    print("\n" + "=" * 100)
    print("Análise de correlação concluída com sucesso!")
    print(f"Resultados salvos em: {output_dir}")
    print("=" * 100)

if __name__ == "__main__":
    main()

