# Case Técnico iFood - Otimização de Cupons

## Estrutura do Projeto

- `data/raw/` : Dados originais JSON.
- `data/processed/` : Dados processados e prontos para modelagem (formato Parquet).
- `notebooks/1_data_processing.ipynb` : Pipeline PySpark para limpeza, imputação e feature engineering.
- `notebooks/2_modeling.ipynb` : Treino, avaliação e comparação de modelos para recomendação.
- `presentation/src/` : Slides para apresentação aos stakeholders.
- `requirements.txt` : Lista de pacotes Python necessários.

## Como executar

1. **Preparar o ambiente no Databricks Community Edition:**

- Criar cluster com runtime que suporte PySpark e Python 3.x. O utilizado foi com a seguinte configuração:
```yaml
Databricks Runtime Version: 14.3 LTS (includes Apache Spark 3.5.0, Scala 2.12)
Driver Type: Community Optimized 15.3 GB Memory, 2 Cores
```

- Fazer upload dos arquivos no Workspace seguindo a seguinte estrutura:
```csharp
ifood-case/
├── data/ # Datasets
│ ├── raw/ # Dados originais
│ └── processed/ # Dados processados
├── notebooks/ # Jupyter notebooks
│ ├── 1_data_processing.ipynb
│ └── 2_modeling.ipynb
├── presentation/
├── src/
├── README.md
└── requirements.txt 
``` 

- Fazer upload dos dados `.json` disponíveis no diretório `data/raw/` para `FileStore/tables/` de forma manual:
> 1) Abra um notebook no Databrick
> 2) Clique em **File > Upload data to DBFS...**
> 4) Selecione os arquivos e aguarde o upload

- No Databricks Community não foi necessário instalar pacotes Python, mas caso utilize fora desse ambiente importe os listados no `requirements.txt` via notebook (célula mágica):

```python
%pip install -r requirements.txt
```

> Se for executar os notebooks fora do ambiente Databricks, é recomendado ler as células que possuem indicações de ajustes de caminhos

2. **Executar notebook de processamento:**

- Rodar `1_data_processing.ipynb` para gerar dataset processado em `dbfs:/FileStore/tables/final_dataset.parquet`.


3. **Executar notebook de modelagem:**

- Rodar `2_modeling.ipynb` para treinar modelos, avaliar e gerar métricas e recomendações.

4. **Apresentação:**

- Slides prontos em `presentation` para apresentar a solução ao time de negócio.

## Pacotes necessários
Veja o arquivo `requirements.txt` para referência.