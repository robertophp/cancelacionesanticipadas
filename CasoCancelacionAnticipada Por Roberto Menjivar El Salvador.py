# Databricks notebook source
# MAGIC %md
# MAGIC # Cancelaciones anticipadas
# MAGIC Caso a resolver

# COMMAND ----------

!pip install openpyxl

# COMMAND ----------

import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
sns.set_theme(color_codes=True)

# COMMAND ----------



dfsilver = pd.read_excel("/dbfs/FileStore/tables/Base_Precancelación_Perú_Ejercicio.xlsx")
dfsilver.head()

# COMMAND ----------

import pandas as pd
pd.DataFrame.iteritems = pd.DataFrame.items

# COMMAND ----------

display(dfsilver)

# COMMAND ----------

dfsilver.shape

# COMMAND ----------

#Variables con el formato correcto
dfsilver.info()

# COMMAND ----------

#El numero de columnas con valores nulos son 0
dfsilver.isnull().sum()

# COMMAND ----------

#Detecto alta cardinalidad en Tienda, evaluare si es necesario incluirla o no.
#Detecto alta cardinalidad en bancarizado, evaluare si es necesario delimitarla.
dfsilver.select_dtypes(include='object').nunique()

# COMMAND ----------

#No detectamos duplicados
numero_duplicados = dfsilver.duplicated().sum()
print(f'Número de filas duplicadas: {numero_duplicados}')

# COMMAND ----------

# MAGIC %md
# MAGIC ## Exploración y limpieza

# COMMAND ----------

continuous_columns, categorical_columns = [], []
for x in dfsilver.columns:
    if dfsilver[x].dtypes=='object':
        categorical_columns.append(x)
    else:
        continuous_columns.append(x)

# COMMAND ----------

continuous_columns

# COMMAND ----------

# MAGIC %md
# MAGIC ## visualizacion

# COMMAND ----------

# MAGIC %md
# MAGIC ### Graficos Generales (visualizacion)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Variables categoricas

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Barras

# COMMAND ----------

# Creamos GRAFICAS CATEGORICAS con subplots
num_cols = len(categorical_columns)
num_rows = (num_cols + 2) // 3
fig, axs = plt.subplots(nrows=num_rows, ncols=3, figsize=(15, 5*num_rows))
axs = axs.flatten()


for i, var in enumerate(categorical_columns):
    top_values = dfsilver[var].value_counts().nlargest(10).index
    filtered_df = dfsilver[dfsilver[var].isin(top_values)]
    order = filtered_df[var].value_counts().index
    sns.countplot(x=var, data=filtered_df, ax=axs[i],order=order)
    axs[i].set_title(var)
    axs[i].tick_params(axis='x', rotation=90)

# quitamos extra empty plots
if num_cols < len(axs):
    for i in range(num_cols, len(axs)):
        fig.delaxes(axs[i])


fig.tight_layout()

plt.show()

# COMMAND ----------

#revisamos uno de los campos con mayor cardinalidad antes de borrarlo
plt.figure(figsize=(20,10))
dfsilver['BANCARIZADO'].value_counts().plot(kind='bar')

# COMMAND ----------

# MAGIC %md
# MAGIC #### Variables numericas

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Histograma

# COMMAND ----------


num_cols = len(continuous_columns)
num_rows = (num_cols + 2) // 3  # To make sure there are enough rows for the subplots
fig, axs = plt.subplots(nrows=num_rows, ncols=3, figsize=(15, 5*num_rows))
axs = axs.flatten()

# Histogramas
for i, var in enumerate(continuous_columns):
    dfsilver[var].plot.hist(ax=axs[i])
    axs[i].set_title(var)


if num_cols < len(axs):
    for i in range(num_cols, len(axs)):
        fig.delaxes(axs[i])

fig.tight_layout()

# Show plot
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Boxplot

# COMMAND ----------

num_cols = len(continuous_columns)
num_rows = (num_cols + 2) // 3
fig, axs = plt.subplots(nrows=num_rows, ncols=3, figsize=(15, 5*num_rows))
axs = axs.flatten()


for i, var in enumerate(continuous_columns):
    sns.boxplot(x=dfsilver[var], ax=axs[i])
    axs[i].set_title(var)


if num_cols < len(axs):
    for i in range(num_cols, len(axs)):
        fig.delaxes(axs[i])


fig.tight_layout()

# Show plot
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Graficos comparando Clientes con precancelamiento

# COMMAND ----------

# MAGIC %md
# MAGIC #### Variables categoricas

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Barras

# COMMAND ----------

# Creamos GRAFICAS CATEGORICAS con subplots
cat_vars = categorical_columns.copy()

num_cols = len(categorical_columns)
num_rows = (num_cols + 2) // 3
fig, axs = plt.subplots(nrows=num_rows, ncols=3, figsize=(15, 5*num_rows))
axs = axs.flatten()

if 'PRECANCELADO' in cat_vars:
    cat_vars.remove('PRECANCELADO')

for i, var in enumerate(cat_vars):
    top_values = dfsilver[var].value_counts().nlargest(10).index
    filtered_df = dfsilver[dfsilver[var].isin(top_values)]
    order = filtered_df[var].value_counts().index
    sns.countplot(x=var, data=filtered_df,hue='PRECANCELADO', ax=axs[i],order=order)
    axs[i].set_title(var)
    axs[i].tick_params(axis='x', rotation=90)

if num_cols < len(axs):
    for i in range(num_cols, len(axs)):
        fig.delaxes(axs[i])


fig.tight_layout()

plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Barras Apliadas

# COMMAND ----------



num_cols = len(categorical_columns)
num_rows = (num_cols + 2) // 3
fig, axs = plt.subplots(nrows=num_rows, ncols=3, figsize=(15, 5*num_rows))
axs = axs.flatten()

for i, var in enumerate(categorical_columns):
    sns.histplot(x=var, hue='PRECANCELADO', data=dfsilver, ax=axs[i], multiple="fill", kde=False, element="bars", fill=True, stat='density')
    axs[i].set_xticklabels(dfsilver[var].unique(), rotation=90)
    axs[i].set_xlabel(var)

for i in range(num_cols, len(axs)):
    fig.delaxes(axs[i])

fig.tight_layout()

plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC #### Variables Numericas

# COMMAND ----------


num_vars = continuous_columns.copy()

if 'PRECANCELADO' in num_vars:
    num_vars.remove('PRECANCELADO')

num_cols = len(num_vars)
num_rows = (num_cols + 2) // 3  
fig, axs = plt.subplots(nrows=num_rows, ncols=3, figsize=(15, 5*num_rows))
axs = axs.flatten()

for i, var in enumerate(num_vars):
    sns.histplot(data=dfsilver, x=var, hue='PRECANCELADO', kde=True, ax=axs[i])
    axs[i].set_title(var)

if num_cols < len(axs):
    for i in range(num_cols, len(axs)):
        fig.delaxes(axs[i])

fig.tight_layout()

plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Procesamiento de la data

# COMMAND ----------

df = dfsilver.copy()
df.head()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Ingenieria de variables

# COMMAND ----------

df['FLAGBANCARIZADO']=(list(
                             map(lambda x: 1 if x in('Bancarizado VIP 1','Bancarizado OBSERVADO','Bancarizado VIP','Bancarizado OK','Bancarizado VIP.',
                                                     'Bancarizado AA','Bancarizado B','Bancarizado D') 
                                 else 0, df['BANCARIZADO'])
                             ) 
)

# COMMAND ----------

# Ratio que me indica la proporción de pagos realizados frente a los pagos vencidos
df['RATIOPAGOS'] = np.where(df['N_CUOTAS_VENCIDAS'] == 0, 
                                      1, 
                                      df['N_CUOTAS_PAGADAS'] / (df['N_CUOTAS_VENCIDAS']+df['N_CUOTAS_PAGADAS']))

# COMMAND ----------

display(df)

# COMMAND ----------

# Eliminames variables


df.drop(columns = ['ESTADO CIVIL', 'Tienda', 'BANCARIZADO','PRIMA','N_CUOTAS_PAGADAS','N_CUOTAS_VENCIDAS'], inplace=True)

# COMMAND ----------

df.columns

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Graficando nuevas variables

# COMMAND ----------

grouped = df.groupby(['FLAGBANCARIZADO', 'PRECANCELADO']).size().unstack(fill_value=0)


fig, axes = plt.subplots(1, grouped.shape[0], figsize=(15, 8))

# Creamos gráficos de pastel para cada categoría de FLAGBANCARIZADO
for i, (flag, counts) in enumerate(grouped.iterrows()):
    axes[i].pie(counts, labels=counts.index, autopct='%1.1f%%', startangle=140)
    axes[i].set_title(f'FLAGBANCARIZADO: {flag}')

plt.tight_layout()
plt.show()


# COMMAND ----------

sns.set_style("whitegrid")
plt.figure(figsize=(10, 6))

# Creamos histograma para el ratio de pago dejando PRECANCELADO COMO hue
sns.histplot(data=df, x='RATIOPAGOS', hue='PRECANCELADO', kde=True, multiple="stack")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Encoder

# COMMAND ----------

from sklearn import preprocessing
#decidi label encoder por simplicidad, elimine previamente columnas con alta cardinalidad y aproveche algunas para crear variables asi que considero que no afectara

for col in df.select_dtypes(include=['object']).columns:
    

    label_encoder = preprocessing.LabelEncoder()
    label_encoder.fit(df[col].unique())
    df[col] = label_encoder.transform(df[col])
    
    # Print 
    print(f"{col}: {df[col].unique()}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Correlacion

# COMMAND ----------

#correlaciones bajas
plt.figure(figsize=(20, 16))
sns.heatmap(df.corr(), fmt='.2g', annot=True)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Manejo de Outlier

# COMMAND ----------

continuous_columns2 = []
for x in df.columns:
    if df[x].dtypes!='object':
        continuous_columns2.append(x)

continuous_columns2

# COMMAND ----------

def removeOutliers (dat, columnName):
    q4 = dat[columnName].quantile(.99)
    dat[columnName]=list(map(lambda x: q4 if x > q4 else x ,dat[columnName]))

    return dat

# COMMAND ----------

#No eliminaremos datos solo definiremos un techo maximo, de esta manera los maximos seguiran siendo maximos sin afectar a la distribucion de los datos y sesgar el modelo 
varoutlier= ['INGRESOS','N_VALOR_CUOTA','AFECTACION']

for var in varoutlier:
    dftransformado = removeOutliers(df, var)

# COMMAND ----------

dftransformado[varoutlier].describe()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Escalado

# COMMAND ----------

scaler = MinMaxScaler()
dfscaled = scaler.fit_transform(dftransformado)
dfscaled = pd.DataFrame(dfscaled, columns=dftransformado.columns)
dfscaled

# COMMAND ----------

display(dfscaled)

# COMMAND ----------

dfscaled['PRECANCELADO'].value_counts()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Modelado
# MAGIC
# MAGIC Trabajremos con Arboles, dado que son mas sencillos de comprender y son versatiles con bases imbalanceadas

# COMMAND ----------

X = dfscaled.drop('PRECANCELADO', axis=1)
y = dfscaled['PRECANCELADO']


from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2,random_state=0)

# COMMAND ----------

# MAGIC %md
# MAGIC ### DecisionTreeClassifier

# COMMAND ----------

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
dtree = DecisionTreeClassifier(class_weight='balanced')


param_grid = {
    'max_depth': [3, 4, 5, 6, 7, 8],
    'min_samples_split': [2, 3, 4],
    'min_samples_leaf': [1, 2, 3, 4],
    'random_state': [0, 42]
}

# grid search con 5-fold cross-validation 
grid_search = GridSearchCV(dtree, param_grid, cv=5)
grid_search.fit(X_train, y_train)

# mejores hyperparameters
print(grid_search.best_params_)

# COMMAND ----------

from sklearn.tree import DecisionTreeClassifier
dtree = DecisionTreeClassifier(max_depth = 8, min_samples_leaf = 1, min_samples_split = 3, random_state = 0, class_weight='balanced')
dtree.fit(X_train, y_train)

# COMMAND ----------

y_pred = dtree.predict(X_test) 
print("Accuracy Score :", round(accuracy_score(y_test, y_pred)*100 ,2), "%")

# COMMAND ----------

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, jaccard_score, log_loss

print('F-1 Score : ',(f1_score(y_test, y_pred, average='micro')))
print('Precision Score : ',(precision_score(y_test, y_pred, average='micro')))
print('Recall Score : ',(recall_score(y_test, y_pred, average='micro')))
print('Jaccard Score : ',(jaccard_score(y_test, y_pred, average='micro')))
print('Log Loss : ',(log_loss(y_test, y_pred)))

# COMMAND ----------

#me hacen mucho sentido las variables de importancia, ratio de pago es una variable de peso ya que al ver la correlación con la variable de PRECANDELADO, se puede ver que el pago es muy importante ya que tenemos un gran numero de clientes con pago al dia, nunca se retrasan y aun asi deciden anticiparse su pago. El plazo largo siempre tiende a cancelarse antes.


imp_df = pd.DataFrame({
    "Feature Name": X_train.columns,
    "Importance": dtree.feature_importances_
})
fi = imp_df.sort_values(by="Importance", ascending=False)

fi2 = fi.head(10)
plt.figure(figsize=(10,8))
sns.barplot(data=fi2, x='Importance', y='Feature Name')
plt.title('Top 10 Feature Importance Each Attributes (Decision Tree)', fontsize=18)
plt.xlabel ('Importance', fontsize=16)
plt.ylabel ('Feature Name', fontsize=16)
plt.show()

# COMMAND ----------

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10,5))
sns.heatmap(data=cm,linewidths=.5, annot=True,  cmap = 'Blues', fmt = 'd')
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
all_sample_title = 'Accuracy Score for Decision Tree: {0}'.format(dtree.score(X_test, y_test))
plt.title(all_sample_title, size = 15)

# COMMAND ----------

#Un AUC de 0.88 es bastante bueno, sabe distinguir bien entre 1 y 0
from sklearn.metrics import roc_curve, roc_auc_score
y_pred_proba = dtree.predict_proba(X_test)[:][:,1]

df_actual_predicted = pd.concat([pd.DataFrame(np.array(y_test), columns=['y_actual']), pd.DataFrame(y_pred_proba, columns=['y_pred_proba'])], axis=1)
df_actual_predicted.index = y_test.index

fpr, tpr, tr = roc_curve(df_actual_predicted['y_actual'], df_actual_predicted['y_pred_proba'])
auc = roc_auc_score(df_actual_predicted['y_actual'], df_actual_predicted['y_pred_proba'])

plt.plot(fpr, tpr, label='AUC = %0.4f' %auc)
plt.plot(fpr, fpr, linestyle = '--', color='k')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve', size = 15)
plt.legend()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Random Forest

# COMMAND ----------

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
rfc = RandomForestClassifier(class_weight='balanced')
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [None, 5, 10],
    'max_features': ['sqrt', 'log2', None],
    'random_state': [0, 42]
}


grid_search = GridSearchCV(rfc, param_grid, cv=5)
grid_search.fit(X_train, y_train)

print(grid_search.best_params_)

# COMMAND ----------

from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(random_state=42, max_depth=None, max_features='sqrt', n_estimators=200, class_weight='balanced')
rfc.fit(X_train, y_train)

# COMMAND ----------

y_pred = rfc.predict(X_test)
print("Accuracy Score :", round(accuracy_score(y_test, y_pred)*100 ,2), "%")

# COMMAND ----------

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, jaccard_score, log_loss

print('F-1 Score : ',(f1_score(y_test, y_pred, average='micro')))
print('Precision Score : ',(precision_score(y_test, y_pred, average='micro')))
print('Recall Score : ',(recall_score(y_test, y_pred, average='micro')))
print('Jaccard Score : ',(jaccard_score(y_test, y_pred, average='micro')))
print('Log Loss : ',(log_loss(y_test, y_pred)))

# COMMAND ----------

imp_df = pd.DataFrame({
    "Feature Name": X_train.columns,
    "Importance": rfc.feature_importances_
})
fi = imp_df.sort_values(by="Importance", ascending=False)

fi2 = fi.head(10)
plt.figure(figsize=(10,8))
sns.barplot(data=fi2, x='Importance', y='Feature Name')
plt.title('Top 10 Feature Importance Each Attributes (Random Forest)', fontsize=18)
plt.xlabel ('Importance', fontsize=16)
plt.ylabel ('Feature Name', fontsize=16)
plt.show()

# COMMAND ----------

from sklearn.metrics import roc_curve, roc_auc_score
y_pred_proba = rfc.predict_proba(X_test)[:][:,1]

df_actual_predicted = pd.concat([pd.DataFrame(np.array(y_test), columns=['y_actual']), pd.DataFrame(y_pred_proba, columns=['y_pred_proba'])], axis=1)
df_actual_predicted.index = y_test.index

fpr, tpr, tr = roc_curve(df_actual_predicted['y_actual'], df_actual_predicted['y_pred_proba'])
auc = roc_auc_score(df_actual_predicted['y_actual'], df_actual_predicted['y_pred_proba'])

plt.plot(fpr, tpr, label='AUC = %0.4f' %auc)
plt.plot(fpr, fpr, linestyle = '--', color='k')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve', size = 15)
plt.legend()
