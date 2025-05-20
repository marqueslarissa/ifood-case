# Databricks notebook source
# MAGIC %md
# MAGIC # Case Ifood
# MAGIC
# MAGIC ## Processamentos de Dados
# MAGIC
# MAGIC Utiliza PySpark para ler os da­dos his­tó­ri­cos de tran­sa­ções, ofer­tas e cli­en­tes para ma­ni­pu­la­ção e lim­pe­za dos da­dos afim de uni­fi­ca­r os datasets.

# COMMAND ----------

# Importar bibliotecas e configurar Spark

from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import *
from pyspark.ml.feature import VectorAssembler, MinMaxScaler, StringIndexer, OneHotEncoder
from pyspark.sql.window import Window
from pyspark.ml import Pipeline

user = spark.sql("SELECT current_user()").collect()[0][0]

spark = SparkSession.builder.appName("iFoodCase").getOrCreate()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Extração e Exploração dos Dados

# COMMAND ----------

# Carrega dados raw json
offers_raw = spark.read.json("/FileStore/tables/offers.json")
customers_raw = spark.read.json("/FileStore/tables/profile.json")
transactions_raw = spark.read.json("/FileStore/tables/transactions.json")

# COMMAND ----------

## Explorar e entender os dados
#print('--- Ofertas ---')
#offers_raw.printSchema()
#offers_raw.show(5)
#
#print('--- Clientes ---')
#customers_raw.printSchema()
#customers_raw.show(5)
#
#print('--- Eventos ---')
#transactions_raw.printSchema()
#transactions_raw.show(5)
#

# COMMAND ----------

# MAGIC %md
# MAGIC ### Tratamento dos Dados
# MAGIC
# MAGIC 1) Remover duplicados
# MAGIC 2) Corrigir tipos
# MAGIC 3) Separar `value.offer_id`, `value.amount`, `value.reward` e unificar `offer id` e `offer_id` em uma unica coluna para não perder informação
# MAGIC
# MAGIC
# MAGIC
# MAGIC

# COMMAND ----------

## Ofertas: Explodir canais e corrigir tipos
offers = offers_raw \
    .withColumn("channel", explode(col("channels"))).drop("channels") \
    .withColumn("min_value", col("min_value").cast(IntegerType())) \
    .withColumn("duration", col("duration").cast(IntegerType())) \
    .withColumn("discount_value", col("discount_value").cast(IntegerType())) \
    .dropDuplicates()

## Clientes: Corrigir tipos
customers = customers_raw \
    .withColumn("age", col("age").cast(IntegerType())) \
    .withColumn("registered_on", col("registered_on").cast(IntegerType())) \
    .withColumn("credit_card_limit", col("credit_card_limit").cast(FloatType())) \
    .dropDuplicates()

## Eventos: Separar campos, coalesce e corrigir tipos
transactions = transactions_raw \
    .select(col('account_id'), col('event'), col('time_since_test_start'), col('value.*')) \
    .withColumn("offer_id2", coalesce("offer id", "offer_id")) \
    .withColumn("amount", col("amount").cast(FloatType())) \
    .withColumn("reward", col("reward").cast(FloatType())) \
    .withColumn("time_since_test_start", col("time_since_test_start").cast(IntegerType())) \
    .drop("offer id", "offer_id") \
    .dropDuplicates()


# COMMAND ----------


#print('--- Ofertas ---')
#print(' -- Describe')
#offers.select('discount_value','duration','min_value').describe().show() 
#print(' -- Analise de Missings')
#offers.select([count(when(isnan(c) | col(c).isNull(), c)).alias(c) for c in offers.columns]).show()
#print(' -- Distribuicao: Canal x Tipo de Oferta')
#offers.groupBy('channel','offer_type').count().orderBy('channel', ascending=False).show()
#
#print('--- Clientes ---')
#print(' -- Describe')
#customers.select('age','credit_card_limit','registered_on').describe().show() 
#print(' -- Analise de Missings')
#customers.select([count(when(isnan(c) | col(c).isNull(), c)).alias(c) for c in customers.columns]).show()
#print(' -- Distribuicao: Idade e Genero')
#customers.groupBy('age').count().show()
#customers.groupBy('gender').count().show()
#
#print('--- Eventos ---')
#print(' -- Describe')
#transactions.select('time_since_test_start','amount','reward').describe().show() 
#print(' -- Analise de Missings')
#transactions.select([count(when(isnan(c) | col(c).isNull(), c)).alias(c) for c in transactions.columns]).show()
#print(' -- Distribuicao: Evento x Atributos da Coluna Value')
#transactions.groupBy('event').agg({"offer_id2":"count"
#                                   ,"reward":"count"
#                                   ,"amount":"count"}).show()

# COMMAND ----------

# MAGIC %md
# MAGIC 4) Tratar missing values imputando valores no dataset `Customers.json` utilizando a estratégia do KNNImputer para variáveis numéricas e Moda para variáveis categóricas
# MAGIC

# COMMAND ----------

# MAGIC %run ../src/knn_imputer_pyspark 

# COMMAND ----------

# Pipeline KNN Imputer para varias colunas

# Definir colunas alvo e features para imputação
target_cols = ["gender", "credit_card_limit"]
features_ref = ["age","registered_on"]

#customers_imputed = knn_imputer_pyspark(customers, target_cols, features_ref, k=5)
customers_imputed = KNNImputer(customers, target_cols, features_ref, k=5)


# COMMAND ----------

#print('--- Clientes Apos Imput---')
#print(' -- Describe')
#customers_imputed.select('age','credit_card_limit','registered_on').describe().show() 
#print(' -- Analise de Missings')
#customers_imputed.select([count(when(isnan(c) | col(c).isNull(), c)).alias(c) for c in customers_imputed.columns]).show()
#print(' -- Distribuicao: Idade e Genero')
#customers_imputed.groupBy('age').count().show()
#customers_imputed.groupBy('gender').count().show()

# COMMAND ----------

# MAGIC %md
# MAGIC 6) Cria novas variáveis:
# MAGIC >- `perc_offers_used` - Percentual de ofertas utilizadas
# MAGIC >- `time_offer_received_completed` - Tempo entre uma oferta recebida e utilizada
# MAGIC >- `converted` - Coluna binária que indica se uma oferta foi convertida em transação (se foi completada) ou não
# MAGIC * Dentro da janela de 30 dias:
# MAGIC >- `count_transactions_by_accountid` - Quantidade de transações por cliente
# MAGIC >- `sum_amount_by_accountid` - Soma dos valores das transações realizadas por cliente
# MAGIC >- `sum_reward_by_accountid` - Soma dos descontos utilizados por cliente
# MAGIC

# COMMAND ----------

# Criar coluna auxiliar com ordem dos eventos
transactions = transactions.withColumn(
    "event_order",
    when(col("event") == "offer received", 1)
    .when(col("event") == "offer viewed", 2)
    .when(col("event") == "offer completed", 3)
    .when(col("event") == "transaction", 4)
    .otherwise(5)  # caso apareca outro tipo de evento inesperado
)

# Ordenar primeiro por account_id (agrupamento), depois pela ordem do evento
transactions = transactions.orderBy("account_id", "time_since_test_start", "event_order")

# Opcional: remover a coluna auxiliar depois
transactions = transactions.drop("event_order")

# COMMAND ----------

# Feature engineering
# Total ofertas recebidas por account_id
offers_received = transactions.filter(col("event") == "offer received") \
    .groupBy("account_id") \
    .agg(count("*").alias("total_offers_received"))

# Total ofertas usadas por account_id
offers_used = transactions.filter(col("event") == "offer completed") \
    .groupBy("account_id") \
    .agg(count("*").alias("total_offers_used"))

# Juntar e calcular percentual
offers_stats = offers_received.join(offers_used, "account_id", "left") \
    .withColumn("total_offers_used", coalesce(col("total_offers_used"), lit(0))) \
    .withColumn("perc_offers_used", col("total_offers_used") / col("total_offers_received")) \
    .select("account_id", "perc_offers_used")


transactions_final = transactions.join(
    offers_stats,
    on=["account_id"],
    how="outer"
).withColumn(
    "perc_offers_used",
    coalesce(col("perc_offers_used"), lit(0))
)

transactions_final = transactions_final.withColumn("offer_received_temp", when(col("event") == "offer received", 1).otherwise(0)) \
    .withColumn("received_offer_flag", max("offer_received_temp").over(Window.partitionBy("account_id"))) \
    .drop("offer_received_temp")

# COMMAND ----------

# Separa ofertas recebidas e ofertas completadas
offers_received = transactions.filter(col("event") == "offer received") \
    .select("account_id", "offer_id2", col("time_since_test_start").alias("received_time"))

offers_completed = transactions.filter(col("event") == "offer completed") \
    .select("account_id", "offer_id2", col("time_since_test_start").alias("completed_time"))

# Faz o join cruzado (mesmo cliente e oferta) para parear eventos
offers_pair = offers_completed.alias("c").join(
    offers_received.alias("r"),
    on=[
        col("c.account_id") == col("r.account_id"),
        col("c.offer_id2") == col("r.offer_id2"),
        col("r.received_time") <= col("c.completed_time")  # Apenas recebimentos antes da conclusao
    ],
    how="inner"
)

# Cria ranking para pegar a RECEBIDA mais proxima da COMPLETADA
window_spec = Window.partitionBy("c.account_id", "c.offer_id2", "c.completed_time") \
    .orderBy(col("r.received_time").desc())

offers_ranked = offers_pair.withColumn("rank", row_number().over(window_spec)) \
    .filter(col("rank") == 1) \
    .withColumn("time_offer_received_completed", col("c.completed_time") - col("r.received_time")) \
    .select(
        col("c.account_id").alias("completed_account_id"),
        col("c.offer_id2").alias("completed_offer_id2"),
        col("r.received_time"),
        col("c.completed_time"),
        "time_offer_received_completed"
    )

transactions_final = transactions_final.alias("t").join(
    offers_ranked.alias("o"),
    on=[
        col("t.account_id") == col("o.completed_account_id"),
        col("t.offer_id2") == col("o.completed_offer_id2"),
        (
            # Evento entre received e completed
            (col("t.time_since_test_start") >= col("o.received_time")) &
            (col("t.time_since_test_start") <= col("o.completed_time"))
        ) |
        (
            # Evento "offer viewed" entre received e viewed_time, mesmo que depois de completed
            (col("t.event") == "offer viewed") &
            (col("t.time_since_test_start") >= col("o.received_time")) &
            (col("t.time_since_test_start") >= col("o.completed_time"))
        )
    ],
    how="left"
).withColumn(
    "converted",
    when(col("o.received_time").isNotNull(), lit(1)).otherwise(lit(0))
)

#define que eventos de transacao nao serao utilizados para definir se uma oferta foi convertida ou nao
transactions_final = transactions_final.withColumn(
    "converted",
    when(col("t.event") == "transaction", None).otherwise(col("converted").cast(IntegerType()))
) \
.drop("completed_account_id", "completed_offer_id2", "completed_time", "received_time") \
.withColumn( #define um valor alto constante para indica de forma explicita que não houve conversao da oferta
    "time_offer_received_completed_filled",
    when(col("time_offer_received_completed").isNull(), lit(9999))
    .otherwise(col("time_offer_received_completed"))
) \
.withColumn(#cria uma flag auxiliar para ser usada junto ao valor constante imputado indicando o imput e ponderar a informacao
    "was_imputed_time_offer",
    when(col("time_offer_received_completed").isNull(), lit(1)).otherwise(lit(0))
)

# COMMAND ----------

# MAGIC %run ../src/metrics 

# COMMAND ----------

# Quantidade de transações, Soma dos valores das transações e Soma dos descontos dutilizados por cliente
# Eventos de transação
transactions_event = transactions.filter(col("event") == "transaction")
transactions_event = calc_count(
    transactions_event, 'account_id', 'count_transactions_by_accountid', 'event', 'time_since_test_start'
)
transactions_event = calc_sum(
    transactions_event, 'account_id', 'sum_amount_by_accountid', 'amount', 'time_since_test_start'
)

# Ofertas completadas
transactions_offers = transactions.filter(col("event") == "offer completed")
transactions_offers = calc_sum(
    transactions_offers, 'account_id', 'sum_reward_by_accountid', 'reward', 'time_since_test_start'
)

transactions_event = transactions_event.drop("amount", "event", "reward", "offer_id2")
transactions_offers = transactions_offers.drop("amount", "event", "reward", "offer_id2")

# Une os resultados por account_id e time
from functools import reduce

dfs_to_join = [transactions_event, transactions_offers]

# Usar full outer para preservar todas as linhas
transactions_features = reduce(
    lambda df1, df2: df1.join(df2, on=["account_id","time_since_test_start"], how="outer"),
    dfs_to_join
)

# Junta com a base original (transactions)
transactions_final = transactions_final.join(
    transactions_features, on=["account_id", "time_since_test_start"], how="left"
)

# Faz o tratamento para preencher todos os registros com as variaveis dentro dos 30 dias (o valor maximo em time_since_test_start é 29)
# Lista de colunas com nulls que devem ser preenchidas
cols_to_fill = ["count_transactions_by_accountid_30d","sum_amount_by_accountid_30d","sum_reward_by_accountid_30d"]

# Aplica preenchimento por valor anterior OU 0 se não houver anterior
for col_name in cols_to_fill:
    window_spec = Window.partitionBy("account_id") \
        .orderBy("time_since_test_start") \
        .rowsBetween(Window.unboundedPreceding, 0)

    transactions_final = transactions_final.withColumn(
        f"{col_name}",
        coalesce(
            last(col_name, ignorenulls=True).over(window_spec),
            lit(0)
        )
    )

# COMMAND ----------

# Criar coluna auxiliar com ordem dos eventos
transactions_final = transactions_final.withColumn(
    "event_order",
    when(col("event") == "offer received", 1)
    .when(col("event") == "offer viewed", 2)
    .when(col("event") == "offer completed", 3)
    .when(col("event") == "transaction", 4)
    .otherwise(5)  # caso apareca outro tipo de evento inesperado
)\
.orderBy("account_id", "time_since_test_start", "event_order")\
.drop("event_order")

#transactions_final.show(50)

# COMMAND ----------

# Unir os datasets
df_final = customers_imputed.join(transactions_final, customers_imputed.id == transactions_final.account_id, "left") \
    .drop("id") \
    .join(offers, offers.id == transactions_final.offer_id2, "left") 

df_f.];;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;~2inal = df_final.withColumn(-9666]
    "event_order",

    when(col("event") == "offer received", 1)
    .when(col("event") == "offer viewed", 2)
    .when(col("event") == "offer completed", 3)
    .when(col("event") == "transaction", 4)
    .otherwise(5)  # caso apareca outro tipo de evento inesperado
)\
.orderBy("account_id", "time_since_test_start", "event_order")\
.drop("event_order")

#df_final.show(50)

# COMMAND ----------

# MAGIC %md
# MAGIC 7. Encoding de variáveis categóricas utilizando OneHotEncoder filtrando somente os eventos de ofertas, dados que irão no modelo
# MAGIC * `gender`
# MAGIC * `offer_type`
# MAGIC * `channel`

# COMMAND ----------

df_filled = df_final.filter(col("event") == "offer received")

# COMMAND ----------

# Lista de colunas categoricas para aplicar OHE
categorical_cols = ['gender', 'offer_type', 'channel']

# Gerar nomes para colunas indexadas e codificadas
indexers = [StringIndexer(inputCol=col, outputCol=col + "_idx", handleInvalid="keep") for col in categorical_cols]
encoders = [OneHotEncoder(inputCol=col + "_idx", outputCol=col + "_ohe") for col in categorical_cols]

# Construir pipeline de transformacao
ohe_pipeline = Pipeline(stages=indexers + encoders)

# Aplicar pipeline no DataFrame
ohe_model = ohe_pipeline.fit(df_filled)
df_encoded = ohe_model.transform(df_filled)

# Explodir cada vetor OHE em colunas individuais
for col_name in categorical_cols:
    # Converter vetor para array
    df_encoded = df_encoded.withColumn(f"{col_name}_ohe_array", vector_to_array(f"{col_name}_ohe"))

    # Descobrir número de colunas para cada vetor (usando o metadado do modelo)
    #ohe_size = ohe_model.stages[-len(categorical_cols) + categorical_cols.index(col_name)].getOutputCols()[0]
    num_categories = df_encoded.select(f"{col_name}_ohe_array").head()[f"{col_name}_ohe_array"].__len__()

    # Criar novas colunas com base no vetor
    for i in range(num_categories):
        df_encoded = df_encoded.withColumn(f"{col_name}_ohe_{i}", col(f"{col_name}_ohe_array")[i])

# 5. Remover colunas auxiliares
drop_cols = [f"{col}_idx" for col in categorical_cols] + [f"{col}_ohe" for col in categorical_cols] + [f"{col}_ohe_array" for col in categorical_cols]
df_encoded = df_encoded.drop(*drop_cols)

#.show(50, truncate=False)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Load dos dados unificados

# COMMAND ----------

# Salvar dataset processado em parquet
#df_final.write.mode("overwrite").parquet("FileStore/tables/final_dataset.parquet")

# Salvar dataset final para a modelagem em parquet
df_encoded.write.mode("overwrite").parquet("FileStore/tables/final_dataset.parquet")

print("Pipeline de processamento completo. Dataset salvo em final_dataset.parquet")