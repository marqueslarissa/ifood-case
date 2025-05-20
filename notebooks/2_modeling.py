# Databricks notebook source
# MAGIC %md
# MAGIC # Case Ifood
# MAGIC
# MAGIC ## Modelagem
# MAGIC
# MAGIC Utiliza duas abordagens: 
# MAGIC * Supervisionada (Random Forest e Regressão Logística) 
# MAGIC   - Prever a probabilidade de conversão de um cliente ao receber um cupom.
# MAGIC * Não Supervisionada (KMeans) 
# MAGIC   - Alternativamente, segmentar os clientes em perfis de comportamento semelhantes (clusterização) para ações direcionadas.

# COMMAND ----------

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, roc_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
#from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns

# Importar bibliotecas e configurar Spark
from pyspark.sql import SparkSession
spark = SparkSession.builder.appName("iFoodCase").getOrCreate()


# COMMAND ----------

# MAGIC %md
# MAGIC ### Carregamento do dataset processado

# COMMAND ----------

df_spark = spark.read.parquet("dbfs:/FileStore/tables/final_dataset.parquet")
df = df_spark.toPandas()

# Filtra apenas os eventos relacionados a ofertas para modelagem supervisionada
#df = df[df['event'].isin(['offer received'])]

# COMMAND ----------

df.dtypes

# COMMAND ----------

# MAGIC %md
# MAGIC ### Verificação da construção do target binário

# COMMAND ----------

print("Distribuição do target")
print(df['converted'].value_counts(normalize=True))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Modelos supervisionados

# COMMAND ----------

print("---- Caminho A: Modelos Supervisionados ----")

# 3.1 Seleção de variáveis
features = ['age','credit_card_limit','count_transactions_by_accountid_30d','sum_amount_by_accountid_30d','sum_reward_by_accountid_30d','duration','min_value'
            ,'gender_ohe_0','gender_ohe_1','gender_ohe_2','offer_type_ohe_0','offer_type_ohe_1','offer_type_ohe_2','channel_ohe_0','channel_ohe_1','channel_ohe_2','channel_ohe_3']
X = df[features]
y = df['converted']

# COMMAND ----------

# 3.2 Escalonamento e split
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, stratify=y, random_state=42)

# COMMAND ----------

## 3.3 Aplicação do SMOTE apenas no conjunto de treino
#smote = SMOTE(random_state=42)
#X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# COMMAND ----------

#print("Distribuição do target após SMOTE (conjunto de treino):")
#print(pd.Series(y_train_resampled).value_counts(normalize=True))

# COMMAND ----------

# MAGIC %md
# MAGIC #### Treino e avaliação dos modelos

# COMMAND ----------

# 3.4 Treinamento dos modelos 
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)
#rf.fit(X_train_resampled, y_train_resampled)
rf_pred = rf.predict(X_test)
rf_probs = rf.predict_proba(X_test)[:, 1]

# COMMAND ----------

lr = LogisticRegression()
lr.fit(X_train, y_train)
#lr.fit(X_train_resampled, y_train_resampled)
lr_pred = lr.predict(X_test)
lr_probs = lr.predict_proba(X_test)[:, 1]

# COMMAND ----------

# MAGIC %md
# MAGIC #### Métricas + análise de erro

# COMMAND ----------

# 4. Avaliação dos Modelos
print("\nRandom Forest Metrics:")
print(classification_report(y_test, rf_pred))
print("AUC:", roc_auc_score(y_test, rf_probs))

print("\nLogistic Regression Metrics:")
print(classification_report(y_test, lr_pred))
print("AUC:", roc_auc_score(y_test, lr_probs))

# COMMAND ----------

# 5. Curva ROC Comparativa
fpr_rf, tpr_rf, _ = roc_curve(y_test, rf_probs)
fpr_lr, tpr_lr, _ = roc_curve(y_test, lr_probs)

plt.figure(figsize=(8, 6))
plt.plot(fpr_rf, tpr_rf, label='Random Forest')
plt.plot(fpr_lr, tpr_lr, label='Logistic Regression')
plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve Comparison')
plt.legend()
plt.grid()
plt.show()

# COMMAND ----------


# Gerar relatórios como dicionários
report_rf = classification_report(y_test, rf_pred, output_dict=True)
report_lr = classification_report(y_test, lr_pred, output_dict=True)

# Extrair F1-score para as classes 0 e 1
f1_rf = [report_rf['0']['f1-score'], report_rf['1']['f1-score']]
f1_lr = [report_lr['0']['f1-score'], report_lr['1']['f1-score']]

classes = ['Classe 0', 'Classe 1']
x = np.arange(len(classes))
width = 0.35

fig, ax = plt.subplots()
rects1 = ax.bar(x - width/2, f1_rf, width, label='Random Forest')
rects2 = ax.bar(x + width/2, f1_lr, width, label='Regressão Logística')

ax.set_ylabel('F1-score')
ax.set_title('Comparação de F1-score por classe')
ax.set_xticks(x)
ax.set_xticklabels(classes)
ax.legend()
plt.ylim(0,1)

plt.show()


# COMMAND ----------

# MAGIC %md
# MAGIC ### Modelo não supervisionado

# COMMAND ----------

print("---- Caminho B: Clusterização Não Supervisionada ----")

clustering_features = ['age','credit_card_limit','duration','min_value'
                       ,'gender_ohe_0','gender_ohe_1','gender_ohe_2','offer_type_ohe_0','offer_type_ohe_1','offer_type_ohe_2','channel_ohe_0','channel_ohe_1','channel_ohe_2','channel_ohe_3']
X_cluster = df[clustering_features]#.dropna()


# COMMAND ----------

X_cluster

# COMMAND ----------

scaler = MinMaxScaler()#StandardScaler()
X_scaled = scaler.fit_transform(X_cluster)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Criação dos clusters

# COMMAND ----------

kmeans = KMeans(n_clusters=6, random_state=42)
df['cluster_id'] = kmeans.fit_predict(X_scaled)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Caracterização dos grupos

# COMMAND ----------

# Amostragem dos dados e labels (10%)
sample_frac = 0.1
random_state = 42

np.random.seed(random_state)
sample_indices = np.random.choice(len(X_scaled), size=int(len(X_scaled) * sample_frac), replace=False)

sample_features = X_scaled[sample_indices]
sample_labels = df['cluster_id'].iloc[sample_indices]

# Calcular Silhouette Score com amostra
sil_score = silhouette_score(sample_features, sample_labels)
print(f"Silhouette Score (amostra {sample_frac*100:.0f}%): {sil_score:.4f}")


# COMMAND ----------

cluster_summary = df.groupby('cluster_id')[clustering_features].mean().round(2)
display(cluster_summary)

# COMMAND ----------

sns.countplot(x='cluster_id', data=df)
plt.title("Distribuição dos Clusters")
plt.show()

# COMMAND ----------

# PCA para redução de dimensionalidade
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Criar DataFrame com componentes principais e clusters
df_viz = pd.DataFrame(X_pca, columns=["PC1", "PC2"])
df_viz["cluster"] = df["cluster_id"].values

# Visualizar
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df_viz, x="PC1", y="PC2", hue="cluster", palette="tab10", s=50)
plt.title("Visualização dos Clusters com PCA")
plt.xlabel("Componente Principal 1")
plt.ylabel("Componente Principal 2")
plt.legend(title="Cluster")
plt.grid(True)
plt.show()
