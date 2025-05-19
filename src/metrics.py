# Databricks notebook source
from pyspark.sql.functions import *
from pyspark.sql.window import Window

# Funcao para calcular transacoes de uma conta de acordo com a janela
def calc_count(df, partition_column, vu_name, count_column, order_column):
    # Definicao das janelas de tempo em dias
    win_spec_1d = Window.partitionBy(partition_column).orderBy(order_column).rangeBetween(-1, 0) # Ultimos 1 dia
    win_spec_7d = Window.partitionBy(partition_column).orderBy(order_column).rangeBetween(-7, 0) # Ultimos 7 dias
    win_spec_30d = Window.partitionBy(partition_column).orderBy(order_column).rangeBetween(-30, 0) # Ultimos 30 dias

    # Cria colunas com o tamanho do conjunto distinto em cada janela
    df = df.withColumn(f'{vu_name}_30d', count(count_column).over(win_spec_30d).cast(IntegerType()))
    return df

# Funcao para calcular ofertas de uma conta de acordo com a janela
def calc_countdist(df, partition_column, vu_name, countdist_column, order_column):
    # Definicao das janelas de tempo em dias
    win_spec_1d = Window.partitionBy(partition_column).orderBy(order_column).rangeBetween(-1, 0) # Ultimos 1 dia
    win_spec_7d = Window.partitionBy(partition_column).orderBy(order_column).rangeBetween(-7, 0) # Ultimos 7 dias
    win_spec_30d = Window.partitionBy(partition_column).orderBy(order_column).rangeBetween(-30, 0) # Ultimos 30 dias

    # Cria colunas com o tamanho do conjunto distinto em cada janela
    df = df.withColumn(f'{vu_name}_30d', size(collect_set(countdist_column).over(win_spec_30d).cast(IntegerType())))
    return df


# Funcao para calcular soma acumulada em janelas temporais
def calc_sum(df, partition_column, vu_name, sum_column, order_column):
    # Definicao das janelas de tempo em dia
    win_spec_1d = Window.partitionBy(partition_column).orderBy(order_column).rangeBetween(-1, 0)
    win_spec_7d = Window.partitionBy(partition_column).orderBy(order_column).rangeBetween(-7, 0)
    win_spec_30d = Window.partitionBy(partition_column).orderBy(order_column).rangeBetween(-30, 0)

    df = df.withColumn(f'{vu_name}_30d', sum(sum_column).over(win_spec_30d).cast(FloatType()))
    return df