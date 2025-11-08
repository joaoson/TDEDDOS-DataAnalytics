
Como gerar as métricas para 10 variáveis (média, mediana, moda, desvio padrão, percentis) e histogramas

1) Suba seu CSV para /mnt/data e anote o caminho, por ex.: /mnt/data/ddos.csv
2) Edite o arquivo de configuração: /mnt/data/config_univariate.json
   - Defina "csv_path" com o caminho do seu CSV.
   - (Opcional) Em "variables", liste até 10 colunas numéricas. Se deixar vazio, o script escolherá automaticamente as 10 de maior variância.
3) Execute o script:
   - Em um terminal Python:  python /mnt/data/univariate_ddos_report.py
4) Saída será gerada em: /mnt/data/univariate_out/
   - univariate_metrics.csv  -> tabela com as métricas
   - histograms/*.png        -> histogramas (e _hist_log.png quando aplicável)
   - univariate_report.pdf   -> relatório consolidado com tabela + figuras

Observações:
- Histogramas são gerados com matplotlib (sem seaborn). Um gráfico por figura e sem cores definidas manualmente.
- Para variáveis de tempo como *Duration* e *IAT*, o script usa ms (÷1000) somente para visualização; as métricas numéricas são calculadas nos valores originais do CSV.
- A moda retorna até 3 valores empatados (se houver).
