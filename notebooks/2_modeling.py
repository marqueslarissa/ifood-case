# Databricks notebook source
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, roc_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from imblearn.over_sampling import SMOTE  # <--- Novo
import matplotlib.pyplot as plt
import seaborn as sns


# COMMAND ----------

df_spark = spark.read.parquet("dbfs:/FileStore/tables/final_dataset.parquet")
df = df_spark.toPandas()

# COMMAND ----------

# Filtra apenas os eventos relacionados a ofertas para modelagem supervisionada
df_model = df[df['event'].isin(['offer received', 'offer completed'])].copy()

# COMMAND ----------

print("Distribuição do target antes do SMOTE:")
print(df_model['converted'].value_counts(normalize=True))

# COMMAND ----------

print("---- Caminho A: Modelos Supervisionados ----")

# 3.1 Seleção de variáveis
features = ['age', 'credit_card_limit', 'days_since_registration',
            'offer_type_encoded', 'min_value', 'discount_value',
            'time_offer_received_completed_filled',  
            'was_imputed_time_offer']  

X = df_model[features]
y = df_model['converted']

# COMMAND ----------

# 3.2 Escalonamento e split
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, stratify=y, random_state=42)

# COMMAND ----------

# 3.3 Aplicação do SMOTE apenas no conjunto de treino
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# COMMAND ----------

print("Distribuição do target após SMOTE (conjunto de treino):")
print(pd.Series(y_train_resampled).value_counts(normalize=True))

# COMMAND ----------

# 3.4 Treinamento dos modelos com dados balanceados
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train_resampled, y_train_resampled)
rf_pred = rf.predict(X_test)
rf_probs = rf.predict_proba(X_test)[:, 1]

# COMMAND ----------

lr = LogisticRegression()
lr.fit(X_train_resampled, y_train_resampled)
lr_pred = lr.predict(X_test)
lr_probs = lr.predict_proba(X_test)[:, 1]

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



# COMMAND ----------

print("---- Caminho B: Clusterização Não Supervisionada ----")

clustering_features = ['age', 'credit_card_limit', 'days_since_registration', 'num_offers_received', 'num_transactions']
X_cluster = df[clustering_features].dropna()


# COMMAND ----------

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_cluster)

# COMMAND ----------

kmeans = KMeans(n_clusters=4, random_state=42)
df['cluster_id'] = kmeans.fit_predict(X_scaled)

# COMMAND ----------

silhouette = silhouette_score(X_scaled, df['cluster_id'])
print(f"Silhouette Score: {silhouette:.2f}")

# COMMAND ----------

cluster_summary = df.groupby('cluster_id')[clustering_features].mean()
display(cluster_summary)

# COMMAND ----------

sns.countplot(x='cluster_id', data=df)
plt.title("Distribuição dos Clusters")
plt.show()