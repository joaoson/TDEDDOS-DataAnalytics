import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import os

# Pergunta o caminho do arquivo CSV
csv_path = input("Informe o caminho completo do arquivo CSV: ").strip()

if not os.path.isfile(csv_path):
    raise FileNotFoundError(f"Arquivo não encontrado: {csv_path}")

# Caminho de saída (mesmo nome, extensão .parquet)
parquet_path = os.path.splitext(csv_path)[0] + ".parquet"

# Tamanho do chunk (ajuste se ainda faltar memória)
chunksize = 500_000  # 500k linhas por vez é um bom começo

print(f"Lendo em chunks de {chunksize} linhas...")
print(f"Gerando: {parquet_path}")

parquet_writer = None

for i, chunk in enumerate(pd.read_csv(csv_path, chunksize=chunksize)):
    # Converte o chunk (DataFrame) para Arrow Table
    table = pa.Table.from_pandas(chunk, preserve_index=False)

    # Cria o writer na 1ª iteração com o schema detectado
    if parquet_writer is None:
        parquet_writer = pq.ParquetWriter(parquet_path, table.schema)

    # Escreve o chunk no arquivo Parquet
    parquet_writer.write_table(table)

    print(f"Chunk {i+1} concluído.")

# Fecha o writer
if parquet_writer is not None:
    parquet_writer.close()

print("Conversão concluída com sucesso!")
print(f"Arquivo Parquet salvo em: {parquet_path}")
