"""
Análise Univariada - Estatística Descritiva e Visualização
Calcula métricas e gera histogramas para 10 variáveis numéricas
"""

import polars as pl
import matplotlib.pyplot as plt
import numpy as np
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
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

def calcular_metricas(serie):
    """Calcula todas as métricas estatísticas para uma série"""
    # Remove valores nulos
    serie_clean = serie.drop_nulls()
    
    if len(serie_clean) == 0:
        return None
    
    # Converter para numpy para cálculos
    valores = serie_clean.to_numpy()
    
    # Remover valores infinitos e NaN
    valores = valores[np.isfinite(valores)]
    
    if len(valores) == 0:
        return None
    
    metricas = {
        'media': float(np.mean(valores)),
        'mediana': float(np.median(valores)),
        'moda': float(pl.Series(valores).mode().to_list()[0]) if len(pl.Series(valores).mode()) > 0 else None,
        'desvio_padrao': float(np.std(valores)),
        'percentis': {
            'p5': float(np.percentile(valores, 5)),
            'p10': float(np.percentile(valores, 10)),
            'p25': float(np.percentile(valores, 25)),
            'p50': float(np.percentile(valores, 50)),
            'p75': float(np.percentile(valores, 75)),
            'p90': float(np.percentile(valores, 90)),
            'p95': float(np.percentile(valores, 95)),
            'p99': float(np.percentile(valores, 99))
        },
        'min': float(np.min(valores)),
        'max': float(np.max(valores)),
        'count': len(valores),
        'nulos': len(serie) - len(serie_clean)
    }
    
    return metricas

def criar_histograma(serie, nome_variavel, output_dir):
    """Cria histograma para uma variável"""
    serie_clean = serie.drop_nulls()
    
    if len(serie_clean) == 0:
        print(f"Variável {nome_variavel} não tem dados válidos para histograma")
        return
    
    valores = serie_clean.to_numpy()
    
    # Remover valores infinitos e NaN
    valores = valores[np.isfinite(valores)]
    
    if len(valores) == 0:
        print(f"Variável {nome_variavel} não tem dados finitos para histograma")
        return
    
    # Criar figura
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Criar histograma
    n_bins = min(50, int(np.sqrt(len(valores))))
    ax.hist(valores, bins=n_bins, edgecolor='black', alpha=0.7, color='steelblue')
    
    # Adicionar linha para média e mediana
    media = np.mean(valores)
    mediana = np.median(valores)
    
    ax.axvline(media, color='red', linestyle='--', linewidth=2, label=f'Média: {media:.2f}')
    ax.axvline(mediana, color='green', linestyle='--', linewidth=2, label=f'Mediana: {mediana:.2f}')
    
    ax.set_xlabel(nome_variavel, fontsize=12, fontweight='bold')
    ax.set_ylabel('Frequência', fontsize=12, fontweight='bold')
    ax.set_title(f'Histograma - {nome_variavel}', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Salvar figura
    nome_arquivo = nome_variavel.replace('/', '_').replace(' ', '_').replace('\\', '_')
    caminho_arquivo = output_dir / f'histograma_{nome_arquivo}.png'
    plt.tight_layout()
    plt.savefig(caminho_arquivo, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Histograma salvo: {caminho_arquivo}")

def gerar_relatorio(metricas_dict, output_dir):
    """Gera relatório em texto com todas as métricas"""
    caminho_relatorio = output_dir / 'relatorio_metricas.txt'
    
    with open(caminho_relatorio, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("RELATÓRIO DE ANÁLISE UNIVARIADA - ESTATÍSTICA DESCRITIVA\n")
        f.write("=" * 80 + "\n\n")
        
        for nome_var, metricas in metricas_dict.items():
            if metricas is None:
                continue
                
            f.write(f"\n{'=' * 80}\n")
            f.write(f"VARIÁVEL: {nome_var}\n")
            f.write(f"{'=' * 80}\n\n")
            
            f.write(f"Estatísticas Básicas:\n")
            f.write(f"  - Média: {metricas['media']:.6f}\n")
            f.write(f"  - Mediana: {metricas['mediana']:.6f}\n")
            if metricas['moda'] is not None:
                f.write(f"  - Moda: {metricas['moda']:.6f}\n")
            else:
                f.write(f"  - Moda: N/A\n")
            f.write(f"  - Desvio Padrão: {metricas['desvio_padrao']:.6f}\n")
            f.write(f"  - Mínimo: {metricas['min']:.6f}\n")
            f.write(f"  - Máximo: {metricas['max']:.6f}\n")
            f.write(f"  - Contagem: {metricas['count']:,}\n")
            f.write(f"  - Valores Nulos: {metricas['nulos']:,}\n")
            
            f.write(f"\nPercentis:\n")
            p = metricas['percentis']
            f.write(f"  - P5:   {p['p5']:.6f}\n")
            f.write(f"  - P10:  {p['p10']:.6f}\n")
            f.write(f"  - P25:  {p['p25']:.6f}\n")
            f.write(f"  - P50:  {p['p50']:.6f} (Mediana)\n")
            f.write(f"  - P75:  {p['p75']:.6f}\n")
            f.write(f"  - P90:  {p['p90']:.6f}\n")
            f.write(f"  - P95:  {p['p95']:.6f}\n")
            f.write(f"  - P99:  {p['p99']:.6f}\n")
            
            f.write("\n")
    
    print(f"\nRelatório salvo: {caminho_relatorio}")

def main():
    print("Iniciando análise univariada...")
    print("=" * 80)
    
    # Ler arquivo parquet
    print("\n1. Carregando dados do arquivo parquet...")
    df = pl.read_parquet("output.parquet")
    print(f"   Dataset carregado: {df.shape[0]:,} linhas x {df.shape[1]} colunas")
    
    # Selecionar 10 variáveis numéricas interessantes
    variaveis_selecionadas = [
        'Flow Duration',
        'Tot Fwd Pkts',
        'Tot Bwd Pkts',
        'Flow Byts/s',
        'Flow Pkts/s',
        'Fwd Pkt Len Mean',
        'Bwd Pkt Len Mean',
        'Flow IAT Mean',
        'Pkt Len Mean',
        'Active Mean'
    ]
    
    # Verificar quais variáveis existem no dataset
    variaveis_disponiveis = [v for v in variaveis_selecionadas if v in df.columns]
    
    if len(variaveis_disponiveis) < 10:
        print(f"\nAviso: Apenas {len(variaveis_disponiveis)} das 10 variáveis selecionadas foram encontradas.")
        print("Variáveis encontradas:", variaveis_disponiveis)
        
        # Selecionar outras variáveis numéricas se necessário
        colunas_numericas = [col for col in df.columns 
                           if df[col].dtype in [pl.Int64, pl.Float64] 
                           and col not in variaveis_disponiveis
                           and col != '']
        
        variaveis_adicionais = colunas_numericas[:10 - len(variaveis_disponiveis)]
        variaveis_disponiveis.extend(variaveis_adicionais)
        print(f"Variáveis adicionais selecionadas: {variaveis_adicionais}")
    
    print(f"\n2. Analisando {len(variaveis_disponiveis)} variáveis:")
    for i, var in enumerate(variaveis_disponiveis, 1):
        print(f"   {i}. {var}")
    
    # Criar diretório de saída
    output_dir = Path("analise_univariada_output")
    output_dir.mkdir(exist_ok=True)
    print(f"\n3. Diretório de saída: {output_dir}")
    
    # Calcular métricas para cada variável
    print("\n4. Calculando métricas estatísticas...")
    metricas_dict = {}
    
    for var in variaveis_disponiveis:
        print(f"   Processando: {var}...")
        metricas = calcular_metricas(df[var])
        metricas_dict[var] = metricas
        
        if metricas:
            print(f"      ✓ Média: {metricas['media']:.2f}, "
                  f"Mediana: {metricas['mediana']:.2f}, "
                  f"DP: {metricas['desvio_padrao']:.2f}")
    
    # Gerar histogramas
    print("\n5. Gerando histogramas...")
    for var in variaveis_disponiveis:
        print(f"   Criando histograma: {var}...")
        criar_histograma(df[var], var, output_dir)
    
    # Gerar relatório
    print("\n6. Gerando relatório de métricas...")
    gerar_relatorio(metricas_dict, output_dir)
    
    # Criar resumo em formato tabular
    print("\n7. Criando resumo tabular...")
    criar_resumo_tabular(metricas_dict, output_dir)
    
    print("\n" + "=" * 80)
    print("Análise univariada concluída com sucesso!")
    print(f"Resultados salvos em: {output_dir}")
    print("=" * 80)

def criar_resumo_tabular(metricas_dict, output_dir):
    """Cria um resumo tabular das métricas"""
    caminho_resumo = output_dir / 'resumo_metricas.csv'
    
    linhas = []
    linhas.append("Variável,Média,Mediana,Moda,Desvio Padrão,Min,Max,Count,Nulos,P5,P10,P25,P50,P75,P90,P95,P99")
    
    for nome_var, metricas in metricas_dict.items():
        if metricas is None:
            continue
        
        moda_str = f"{metricas['moda']:.6f}" if metricas['moda'] is not None else "N/A"
        linha = (
            f"{nome_var},"
            f"{metricas['media']:.6f},"
            f"{metricas['mediana']:.6f},"
            f"{moda_str},"
            f"{metricas['desvio_padrao']:.6f},"
            f"{metricas['min']:.6f},"
            f"{metricas['max']:.6f},"
            f"{metricas['count']},"
            f"{metricas['nulos']},"
            f"{metricas['percentis']['p5']:.6f},"
            f"{metricas['percentis']['p10']:.6f},"
            f"{metricas['percentis']['p25']:.6f},"
            f"{metricas['percentis']['p50']:.6f},"
            f"{metricas['percentis']['p75']:.6f},"
            f"{metricas['percentis']['p90']:.6f},"
            f"{metricas['percentis']['p95']:.6f},"
            f"{metricas['percentis']['p99']:.6f}"
        )
        linhas.append(linha)
    
    with open(caminho_resumo, 'w', encoding='utf-8') as f:
        f.write('\n'.join(linhas))
    
    print(f"Resumo tabular salvo: {caminho_resumo}")

if __name__ == "__main__":
    main()

