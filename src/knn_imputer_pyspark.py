# Databricks notebook source
from pyspark.ml.feature import VectorAssembler, MinMaxScaler
from pyspark.sql.window import Window
from pyspark.sql import functions as F
from pyspark.ml.functions import vector_to_array
from functools import reduce
import operator

def KNNImputer(df, target_cols, features_ref, k=5):
    """
    Imputa valores ausentes nas colunas `target_cols` usando KNN via distancia euclidiana com MinMax scaling.

    - df: DataFrame de entrada
    - target_cols: Lista de colunas a imputar
    - features_ref: Lista de colunas confiaveis (sem missing) para basear a distancia
    - k: Numero de vizinhos mais proximos (default = 5)
    """

    # Vetorizar colunas de referencia
    assembler = VectorAssembler(inputCols=features_ref, outputCol="features_vec")
    df_vec = assembler.transform(df)

    # Normalizar os vetores
    scaler = MinMaxScaler(inputCol="features_vec", outputCol="scaled_features")
    scaler_model = scaler.fit(df_vec)
    df_scaled = scaler_model.transform(df_vec)

    # Converter vetor em array para facilitar o calculo de distancia
    df_scaled = df_scaled.withColumn("scaled_array", vector_to_array("scaled_features"))

    for col_target in target_cols:
        print(f"Imputando coluna: {col_target}")

        df_missing = df_scaled.filter(col(col_target).isNull())
        df_known = df_scaled.filter(col(col_target).isNotNull())

        if df_missing.count() == 0:
            print(f"Coluna {col_target} não tem valores nulos.")
            continue

        # Cross join
        joined = df_missing.alias("a").crossJoin(df_known.alias("b"))

        # Calcular distancia euclidiana
        dist_expr = sqrt(
            reduce(operator.add,
                   [pow(col("a.scaled_array")[i] - col("b.scaled_array")[i], 2) for i in range(len(features_ref))]
            )
        )


        joined = joined.withColumn("distance", dist_expr)

        # Janela para pegar os k vizinhos mais proximos
        window = Window.partitionBy("a.id").orderBy("distance")
        knn_topk = joined.withColumn("rank", row_number().over(window)) \
                         .filter(col("rank") <= k)

        # Verifica tipo da coluna (string = categorica usa moda, senao media usando knn)
        if dict(df.dtypes)[col_target] == "string":
            # Moda (valor mais frequente)
            knn_mode = knn_topk.groupBy("a.id", f"b.{col_target}") \
                .agg(count("*").alias("freq")) \
                .withColumn("rank", row_number().over(Window.partitionBy("id").orderBy(col("freq").desc()))) \
                .filter(col("rank") == 1) \
                .select("id", col(f"b.{col_target}").alias(f"{col_target}_imputed"))

            df_scaled = df_scaled.join(knn_mode, on="id", how="left") \
                .withColumn(col_target, coalesce(col(col_target), col(f"{col_target}_imputed"))) \
                .drop(f"{col_target}_imputed")

        else:
            # Media KNN
            knn_avg = knn_topk.groupBy("a.id") \
                .agg(avg(col(f"b.{col_target}")).alias(f"{col_target}_imputed"))

            df_scaled = df_scaled.join(knn_avg, on="id", how="left") \
                .withColumn(col_target, coalesce(col(col_target), col(f"{col_target}_imputed"))) \
                .drop(f"{col_target}_imputed")

    return df_scaled.drop("features_vec", "scaled_features", "scaled_array")