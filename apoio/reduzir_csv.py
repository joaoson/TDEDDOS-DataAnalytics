import polars as pl
import time

print("=" * 80)
print("REDUÇÃO DE DIMENSIONALIDADE - CSV Original")
print("=" * 80)

# ============================================================================
# CONFIGURAÇÃO: Ajuste o nome do arquivo CSV de entrada
# ============================================================================
ARQUIVO_CSV_ENTRADA = "unbalaced_20_80_dataset.csv"  # ⚠️ AJUSTE AQUI O NOME DO SEU CSV!
ARQUIVO_CSV_SAIDA = "dataset_reduzido.csv"

# ============================================================================
# Lista de colunas para eliminar (mesma lista do código anterior)
# ============================================================================
COLUNAS_ELIMINAR = [
    # 1. Identificadores
    '', 'Flow ID', 'Src IP', 'Dst IP', 'Src Port', 'Timestamp',
    
    # 2. Redundantes - Totais
    'TotLen Fwd Pkts', 'TotLen Bwd Pkts',
    
    # 3. Tamanho - Max/Min redundantes
    'Fwd Pkt Len Max', 'Fwd Pkt Len Min', 'Bwd Pkt Len Max', 'Bwd Pkt Len Min',
    'Pkt Len Min', 'Pkt Len Max', 'Pkt Len Var',
    
    # 4. IAT redundantes
    'Flow IAT Max', 'Flow IAT Min', 'Fwd IAT Tot', 'Fwd IAT Std', 'Fwd IAT Max',
    'Bwd IAT Tot', 'Bwd IAT Mean', 'Bwd IAT Std', 'Bwd IAT Max', 'Bwd IAT Min',
    
    # 5. Flags raros
    'Fwd URG Flags', 'Bwd URG Flags', 'URG Flag Cnt', 'CWE Flag Count',
    
    # 6. PSH Flags
    'Fwd PSH Flags', 'Bwd PSH Flags',
    
    # 7. Header Length
    'Fwd Header Len', 'Bwd Header Len',
    
    # 8. Segment Size
    'Fwd Seg Size Avg', 'Bwd Seg Size Avg', 'Fwd Seg Size Min',
    
    # 9. Bulk (geralmente zeros)
    'Fwd Byts/b Avg', 'Fwd Pkts/b Avg', 'Fwd Blk Rate Avg',
    'Bwd Byts/b Avg', 'Bwd Pkts/b Avg', 'Bwd Blk Rate Avg',
    
    # 10. Subflows (geralmente redundantes)
    'Subflow Fwd Pkts', 'Subflow Fwd Byts', 'Subflow Bwd Pkts', 'Subflow Bwd Byts',
    
    # 11. Active/Idle redundantes
    'Active Std', 'Active Max', 'Active Min', 'Idle Std', 'Idle Max', 'Idle Min',
]

print(f"\n📂 Arquivo de entrada: {ARQUIVO_CSV_ENTRADA}")
print(f"📂 Arquivo de saída: {ARQUIVO_CSV_SAIDA}")
print(f"🗑️  Colunas a eliminar: {len(COLUNAS_ELIMINAR)}")

# ============================================================================
# 1. Carregar CSV
# ============================================================================
print("\n" + "=" * 80)
print("ETAPA 1: Carregando CSV...")
print("=" * 80)

inicio = time.time()

try:
    # Ler CSV (Polars detecta automaticamente delimitador e tipos)
    df = pl.read_csv(ARQUIVO_CSV_ENTRADA)
    
    tempo_leitura = time.time() - inicio
    
    print(f"✓ CSV carregado com sucesso!")
    print(f"  - Linhas: {df.shape[0]:,}")
    print(f"  - Colunas originais: {df.shape[1]}")
    print(f"  - Tempo de leitura: {tempo_leitura:.2f} segundos")
    print(f"  - Tamanho em memória: {df.estimated_size() / 1024 / 1024:.2f} MB")

except FileNotFoundError:
    print(f"\n❌ ERRO: Arquivo '{ARQUIVO_CSV_ENTRADA}' não encontrado!")
    print(f"   Por favor, ajuste o nome na linha 10 do código.")
    exit(1)
except Exception as e:
    print(f"\n❌ ERRO ao carregar CSV: {e}")
    exit(1)

# ============================================================================
# 2. Verificar quais colunas existem
# ============================================================================
print("\n" + "=" * 80)
print("ETAPA 2: Verificando colunas existentes...")
print("=" * 80)

# Mostrar primeiras colunas do dataset
print(f"\nPrimeiras 10 colunas do CSV:")
for i, col in enumerate(df.columns[:10], 1):
    print(f"  {i}. {col}")

if len(df.columns) > 10:
    print(f"  ... (e mais {len(df.columns) - 10} colunas)")

# Filtrar apenas colunas que realmente existem no CSV
colunas_existentes = [col for col in COLUNAS_ELIMINAR if col in df.columns]
colunas_nao_encontradas = [col for col in COLUNAS_ELIMINAR if col not in df.columns]

print(f"\n✓ Colunas marcadas para eliminação encontradas: {len(colunas_existentes)}")
print(f"⚠️  Colunas não encontradas no CSV: {len(colunas_nao_encontradas)}")

if colunas_nao_encontradas and len(colunas_nao_encontradas) <= 10:
    print(f"\nColunas não encontradas:")
    for col in colunas_nao_encontradas:
        print(f"  - {col}")

# ============================================================================
# 3. Eliminar colunas
# ============================================================================
print("\n" + "=" * 80)
print("ETAPA 3: Eliminando colunas...")
print("=" * 80)

if len(colunas_existentes) == 0:
    print("\n⚠️  AVISO: Nenhuma coluna será eliminada!")
    print("   Verifique se os nomes das colunas no CSV correspondem à lista.")
else:
    print(f"\nEliminando {len(colunas_existentes)} colunas...")
    
    # Eliminar colunas
    df_reduzido = df.drop(colunas_existentes)
    
    print(f"✓ Colunas eliminadas com sucesso!")

# ============================================================================
# 4. Mostrar resumo
# ============================================================================
print("\n" + "=" * 80)
print("ETAPA 4: Resumo da Redução")
print("=" * 80)

reducao_percentual = (len(colunas_existentes) / len(df.columns)) * 100
reducao_tamanho = (1 - (df_reduzido.estimated_size() / df.estimated_size())) * 100

print(f"\n📊 ESTATÍSTICAS:")
print(f"  - Colunas originais:    {len(df.columns)}")
print(f"  - Colunas eliminadas:   {len(colunas_existentes)}")
print(f"  - Colunas mantidas:     {len(df_reduzido.columns)}")
print(f"  - Redução de features:  {reducao_percentual:.1f}%")
print(f"  - Redução de memória:   {reducao_tamanho:.1f}%")

print(f"\n📋 COLUNAS MANTIDAS ({len(df_reduzido.columns)}):")
for i, col in enumerate(df_reduzido.columns, 1):
    print(f"  {i:2d}. {col}")

# ============================================================================
# 5. Salvar CSV reduzido
# ============================================================================
print("\n" + "=" * 80)
print("ETAPA 5: Salvando CSV reduzido...")
print("=" * 80)

inicio_escrita = time.time()

try:
    # Salvar CSV reduzido
    df_reduzido.write_csv(ARQUIVO_CSV_SAIDA)
    
    tempo_escrita = time.time() - inicio_escrita
    
    print(f"✓ CSV reduzido salvo com sucesso!")
    print(f"  - Arquivo: {ARQUIVO_CSV_SAIDA}")
    print(f"  - Tempo de escrita: {tempo_escrita:.2f} segundos")
    
    # Verificar tamanho do arquivo
    import os
    if os.path.exists(ARQUIVO_CSV_SAIDA):
        tamanho_original = os.path.getsize(ARQUIVO_CSV_ENTRADA) / 1024 / 1024
        tamanho_reduzido = os.path.getsize(ARQUIVO_CSV_SAIDA) / 1024 / 1024
        reducao_arquivo = (1 - (tamanho_reduzido / tamanho_original)) * 100
        
        print(f"\n💾 TAMANHO DOS ARQUIVOS:")
        print(f"  - Original:  {tamanho_original:.2f} MB")
        print(f"  - Reduzido:  {tamanho_reduzido:.2f} MB")
        print(f"  - Economia:  {reducao_arquivo:.1f}%")

except Exception as e:
    print(f"\n❌ ERRO ao salvar CSV: {e}")
    exit(1)

# ============================================================================
# 6. Verificação de integridade
# ============================================================================
print("\n" + "=" * 80)
print("ETAPA 6: Verificação de Integridade")
print("=" * 80)

print(f"\n🔍 Verificando consistência dos dados...")
print(f"  - Linhas no original:  {df.shape[0]:,}")
print(f"  - Linhas no reduzido:  {df_reduzido.shape[0]:,}")

if df.shape[0] == df_reduzido.shape[0]:
    print(f"  ✓ Número de linhas preservado!")
else:
    print(f"  ⚠️  AVISO: Número de linhas diferente!")

# Verificar se Label foi mantida
if 'Label' in df_reduzido.columns:
    print(f"  ✓ Coluna 'Label' (target) foi mantida!")
    
    # Mostrar distribuição de labels
    distribuicao = df_reduzido['Label'].value_counts()
    print(f"\n📊 Distribuição de Labels:")
    for row in distribuicao.iter_rows():
        label, count = row
        percentual = (count / df_reduzido.shape[0]) * 100
        print(f"  - {label}: {count:,} ({percentual:.2f}%)")
else:
    print(f"  ⚠️  AVISO: Coluna 'Label' não encontrada!")

# ============================================================================
# FINALIZAÇÃO
# ============================================================================
print("\n" + "=" * 80)
print("✅ REDUÇÃO CONCLUÍDA COM SUCESSO!")
print("=" * 80)

print(f"\n🎯 PRÓXIMOS PASSOS:")
print(f"  1. Use '{ARQUIVO_CSV_SAIDA}' para suas análises")
print(f"  2. O arquivo tem {len(df_reduzido.columns)} features essenciais")
print(f"  3. Economizou ~{reducao_arquivo:.0f}% de espaço em disco")
print(f"  4. Modelos de ML treinarão mais rápido!")

print("\n" + "=" * 80)