#!/usr/bin/env python
# coding: utf-8

# # Model: Regression 

# In diesem Notebook werden verschiedene Regressionsmodelle angewendet, um den mittleren Immobilienpreis von Distrikten in Kalifornien vorherzusagen. Dabei ist die Anforderung an die Modelle einen RMSE von kleiner 70 Tsd. USD zu erzielen. Dann wird das Projekt als Erfolg gewertet. 

# Die Regressionsmodelle werden in diesem Notebook mit der Bibliothek Statsmodels und der Bibliothek scikit-learn durchgeführt. Grundlegend ist jedes Modell in *Data preperation*, *Build* und *Metrics und Validation* unterteilt. Das beste Modell der jeweiligen Bibliothek wird am Ende des Abschnitts mit den Testdaten validiert. 
# 

# > Über diesen [Link](https://sh333hdm.github.io/jupyterbooktest/intro.html) ist die Ansicht der Projektarbeit als Jupyter Book möglich. 

# ## Models statsmodels

# In[1]:


import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from patsy import dmatrices
from func import *

import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.tools.eval_measures import mse, rmse
from statsmodels.compat import lzip
from statsmodels.stats.outliers_influence import variance_inflation_factor

get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
sns.set_theme(palette="Pastel2", style="whitegrid") 


# ### Statsmodels OLS

# **Data preperation**

# In[2]:


df = read_data()
df = transform_data(df)
train_dataset, test_dataset = split_data(df)
train_dataset = fill_missingdata(train_dataset)
train_dataset = add_feautures(train_dataset)
train_dataset


# **Model 1 OLS: Build**

# In[3]:


lm1 = smf.ols(formula ='median_house_value ~ median_income', data=train_dataset).fit()


# Als erstes Modell wird eine einfache lineare Regression durchgeführt. Als unabhängige Variable wird *median_income* gewählt, da das mittlere Einkommen eines Bezirks die höchste Korrelation zu *median_house_value* aufweist. 

# In[4]:


train_dataset['y_pred'] = lm1.predict()


# **Model OLS 1: Metrics and Validation**

# In[5]:


lm1.summary()


# Die Kenngrößen sind wie erwartet nicht zielführend. Der Adj. R-squared liegt bei 0.47. Dieser besagt, dass unter 50% der Streuung der Werte durch das Modell korrekt vorhergesagt werden kann. Weitere Kenngrößen werden im Vergleich zum zweiten Modell betrachtet. 

# In[6]:


plt.rcParams['figure.figsize'] = [7, 7]
sns.lmplot(x='median_income', y='median_house_value', data=train_dataset, line_kws={'color':'red'}, height=5, ci=None);


# In[7]:


r2_lm1, rmse_lm1, mse_lm1 = metrics_stats(lm1,train_dataset)


# Ebenfalls liegt der RMSE mit 84.238 USD über dem Zielwert von 70 Tsd. USD. Das Modell kann dementsprechend nicht als Erfolg gewertet werden. 

# **Model 2 OLS: Build**

# In[8]:


lm2 = smf.ols(formula ='median_house_value ~ median_income + total_rooms_total_bedrooms', data=train_dataset).fit()


# In[9]:


# Add the regression predictions (as "pred") to our DataFrame
train_dataset['y_pred'] = lm2.predict()


# **Model 2 OLS: Metrics and Validation**

# In[10]:


lm2.summary()


# In[11]:


fig = sm.graphics.plot_regress_exog(lm2, "median_income")


# In[12]:


fig = sm.graphics.plot_regress_exog(lm2, "total_rooms_total_bedrooms")


# Im Vergleich zu Model 1 erreicht das Model 2 besser statistische Kenngrößen. Der Adj. R-squared liegt bei ca. 3% über dem von Model 1. Sowohl die Kenngröße der F-Statistik, wie auch AIC und BIC sind höher als bei Model 1. 

# > Hinweis: Zu diesem Modell wird eine ausführliche Regression Diagnostics durchgeführt. 

# In[13]:


r2_lm2, rmse_lm2, mse_lm2 = metrics_stats(lm2,train_dataset)


# Der RMSE liegt mit ca. 81,4 Tsd. USD immer noch deutlich über 70 Tsd. USD. Es muss dementsprechend ein geeigneteres Modell gefunden werden. 

# **Model 3 OLS: Build**

# In[14]:


lm3 = smf.ols(formula ='median_house_value ~ median_income + C(ocean_proximity) + households_population + total_rooms_households +C(geohash) + housing_median_age', data=train_dataset).fit()


# Im dritten Modell werden mehr Variablen aus dem ursprünglichen Datensatz berücksichtigt. Nicht miteingeschlossen werden: 
#  
# * *longitude*, *latidue* (siehe Feauture Selection)
# * *total_rooms*, *total__bedrooms*, *population* und *households*, da die Korrelation dieser Merkmale untereinander zu hoch ist (siehe Feauture Selection)
# * *price_category*, da diese aus *median_house_value* abgeleitet wird
# * *total_rooms_total_bedrooms*, da nach einmaliger Durchführung der Klassifikation ein p-Wert von P|z|=0.555 ermittelt wurde
# 
# 

# In[15]:


train_dataset['y_pred'] = lm3.predict()


# **Model OLS 3: Metrics and Validation**

# In[16]:


lm3.summary()


# In[17]:


r2_lm3, rmse_lm3, mse_lm3 = metrics_stats(lm3,train_dataset)


# Durch Model 3 konnte ein RMSE von 63.731 USD und damit unter der Grenze von 70 Tsd. USD erzielt werden. Erst nach Evaluation mit einem Testdatenset kann, dass Model als Erfolg gewertet werden. 

# ### Regression Diagnostics

# **Outliner and High-Leverage**

# In[18]:


fig = sm.graphics.influence_plot(lm2, criterion="cooks")


# In[274]:


lm_cooksd = lm2.get_influence().cooks_distance[0]
n = len(train_dataset["median_house_value"])
print('Anzahl an Datensätzen:', n)
critical_d = 4/n
print('Kritische Cook distance:', critical_d)
out_d = lm_cooksd > critical_d
print("Anzahl Datensätze mit kritischer Cooks distance:",len(lm_cooksd[out_d]))


# 1019 Datensätzen besitzen eine kritische Cooks Distanz. Dabei handelt es sich um vermutliche *Outliner* mit *leverage*. Ein einfaches Vorgehen ist diese Distrikte zu entfernt. 

# In[275]:


train_dataset.index[out_d]


# In[276]:


train_dataset_opt = train_dataset.drop(train_dataset.index[out_d])
train_dataset_opt


# **Non-linearity and heteroscedasticity**

# In[277]:


fig = sm.graphics.plot_partregress_grid(lm2)
fig.tight_layout(pad=1.0)


# In[278]:


model_fitted_y = lm2.fittedvalues

#  Plot
plot = sns.residplot(x=model_fitted_y, y='median_house_value', data=train_dataset, lowess=True, 
                     scatter_kws={'alpha': 0.5}, 
                     line_kws={'color': 'red', 'lw': 1, 'alpha': 0.8})


# * Es liegt keine ungleichmäßige Verteilung der residuals vor: Hinweis auf Heteroscedasticity
# * Vermutlich liegt auch keine lineare Beziehung vor, da die rote Linie sich von der Gestrichelten unterscheidet.

# In[279]:


# Breusch-Pagan Lagrange Multiplier test

name = ['Lagrange multiplier statistic', 'p-value', 'f-value', 'f p-value']
test = sm.stats.het_breuschpagan(lm2.resid, lm3.model.exog)
lzip(name, test)


# Beide p-werte liegen deutlich unter 0.05. Daher wird die Null-Hypothese abgelehnt. 
# Dieses Ergebnis ist ebenfalls ein Indikator für Heteroscedasticity. Eine mögliche Lösung ist in diesem Fall die Anwendung eines Regression Splines. Dies wird in einem der folgenenden Abschnitten untersucht. 

# **Non-normally distributed errors**

# In[280]:


#Jarque-Bera test
name = ['Jarque-Bera', 'Chi^2 two-tail prob.', 'Skew', 'Kurtosis']
test = sm.stats.jarque_bera(lm2.resid)

lzip(name, test)


# Der p-Wert liegt bei 0.0 damit wird die Nullhyptothese abgelehnt. Das ist ein Indikator für nicht normalverteilte *error terms*.

# In[281]:


#Omnibus normtest
name = ['Chi^2', 'Two-tail probability']
test = sm.stats.omni_normtest(lm2.resid)
lzip(name, test)


# Auch bei diesem Test wird die Nullhyptothese abgelehnt. Dies lässt ebenso auf nicht normalverteilte *error terms* schließen. 

# Da die Anzahl an Datensätzen sehr hoch ist, müssen die Fehler nicht normalverteilt sein. Grund dafür ist der zentrale Grenzwertsatz. Es werden keine weiteren Schritte unternommen. 

# **Correlation of error terms**

# In[282]:



sm.stats.durbin_watson(lm2.resid)


# Der Wert liegt zwischen 1 und 2, aber sehr nah an der 2. Im ersten Schritt wird keine Korrelation der *error terms* angenommen.  

# **Collinearity**

# In[283]:


#Variance inflation factor
y, X = dmatrices('median_house_value ~ median_income + total_rooms_total_bedrooms', data=train_dataset, return_type='dataframe')

# For each X, calculate VIF and save in dataframe
vif = pd.DataFrame()
vif["VIF Factor"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif["Feature"] = X.columns

vif.round(2)


# Für dieses Modell liegt keine problematische Kollinearität in den Daten vor. 

# **Model OLS 3 opt: Build**

# 
# In diesem linearen Modell wird das Modell mit multipler linearer Regression auf die bereinigten Trainingsdaten nach der Regression Diagnostics angewendet. 
# 

# In[284]:


lm3opt = smf.ols(formula ='median_house_value ~ median_income + C(ocean_proximity) + households_population + total_rooms_households +C(geohash) + housing_median_age', data=train_dataset_opt).fit()


# In[285]:


train_dataset_opt['y_pred'] = lm3opt.predict()


# **Model OLS 3 opt: Metrics and Validation**

# In[286]:


lm3opt.summary()


# In[287]:


r2_lm3opt, rmse_lm3opt, mse_lm3opt = metrics_stats(lm3opt,train_dataset_opt)


# Für dieses Modell liegt der RMSE unter 50 Tsd. USD.  Wird nur diese Größe betrachtet handelt es hierbei um das beste Modell. 

# ### Statsmodel Lasso

# **Model Lasso: Build**

# Mit dem Parameter `L1_wt` wird eine regularisiertes Modell nach Lasso umgesetzt. Zu dieser Umsetzung sind wenige Quellen verfügbar.  

# In[288]:


lmlasso = smf.ols(formula ='median_house_value ~ median_income + C(ocean_proximity) + households_population + total_rooms_households + total_rooms_total_bedrooms+C(geohash)', data=train_dataset).fit_regularized(L1_wt=1)


# In[289]:


train_dataset['y_pred'] = lmlasso.predict()


# **Model Lasso: Metrics and Validation**

# In[290]:


rmse_lasso= round(rmse(train_dataset['median_house_value'], train_dataset['y_pred']), 2)
mse_lasso = round (mse(train_dataset['median_house_value'], train_dataset['y_pred']), 2)
print("RMSE ",rmse_lasso)
print("MSE: ",mse_lasso)


# Der RMSE liegt mit ca. 66,7 Tsd. USD unter 70 Tsd. USD. 

# ### Natural Spline Statsmodel

# **Model Splines: Build**

# Das Natural Splines Modell wird mit nur einer Variablen umgesetzt. Da *median_income* die höchste Korrelation  mit *median_house_value* aufweist wird dieser in das Modell einbezogen. 

# In[291]:


X_train = train_dataset['median_income']
y_train = train_dataset['median_house_value']


# In[292]:


from patsy import dmatrix
transformed_x3 = dmatrix("cr(train,df = 3)", {"train": X_train}, return_type='dataframe')


# In[293]:


spline = sm.GLM(y_train, transformed_x3).fit()


# In[294]:


y_pred= spline.predict(dmatrix("cr(train, df=3)", {"train": X_train}, return_type='dataframe'))


# **Model Splines: Metrics and Validation**

# In[295]:


spline.summary()


# In[296]:


rmse_spline= round(rmse(train_dataset['median_house_value'], y_pred), 2)
mse_spline = round (mse(train_dataset['median_house_value'], y_pred), 2)
print("RMSE ",rmse_spline)
print("MSE: ",mse_spline)


# Der RMSE liegt bei über 70 Tsd. USD. Grund für diese Ergebnis ist die Beschränkung auf nur eine Variable.  

# In[297]:


sns.scatterplot(x=X_train, y=y_train)
plt.plot(X_train, y_pred, color='orange', label='Natural spline with df=3')
plt.legend()


# ### Evaluation

# Für die Evaluation werden die RMSE der einzelnen Modelle miteinander verglichen. Das Modell mit dem niedrigsten RMSE wird mit den Testdaten validiert. 

# In[298]:


results= {'RMSE': ['RMSE OLS 1', 'RMSE OLS 2', 'RMSE OLS 3', 'RMSE OLS 3 opt', 'RMSE Lasso', 'RMSE Spline'],
        'Werte': [rmse_lm1, rmse_lm2, rmse_lm3, rmse_lm3opt, rmse_lasso, rmse_spline] }
df_results = pd.DataFrame(results)
df_results.sort_values(by ='Werte')


# In[299]:


test_dataset = add_feautures(test_dataset)
test_dataset = fill_missingdata(test_dataset)


# In[300]:


test_dataset['y_pred'] = lm3opt.predict(test_dataset[['median_income', 'ocean_proximity', 'households_population', 'total_rooms_households', 'housing_median_age', 'geohash']])


# In[301]:


r2_test, rmse_test, mse_test = metrics_stats(lm3opt,test_dataset)


# Das Modell kann als Erfolg gewertet werde, das Modell angewandt auf die Testdaten ebenfalls ein RMSE von unter 70 Tsd. USD erreicht.  

# ## Model sklearn

# In[302]:


from sklearn.model_selection import train_test_split

from sklearn.compose import ColumnTransformer, make_column_selector as selector
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression, LassoCV, Lasso, Ridge
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import PolynomialFeatures

import seaborn as sns  


from func import *
from sklearn import set_config

get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
set_config(display="diagram")


# ### sklearn OLS

# **Data Preperation**

# Zur Datenvorbereitung  mit der Bibliothek scikit-learn wird ein *preprocessor* angewendet. Der Aufbau des *preprocessors* ist im Notebook Data genauer beschrieben. In diesem Fall wird dieser über die Funktion erstellt. 

# In[303]:


df = read_data()
df = transform_data(df)


# In[304]:


#Split Test and Trainingsdata
X = df.drop('median_house_value', axis=1)
y = df['median_house_value']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[305]:


X_train = add_feautures(X_train)


# In[306]:


#Feauture Selection 
X_train = X_train[['median_income', 'ocean_proximity','housing_median_age', 'households_population', 'total_rooms_households', 'total_rooms_total_bedrooms', 'geohash']]
X_train


# In[307]:


preprocessor = build_preprocessor()


# **Model OLS**

# Das erste Modell, welches mit scikit-learn umsgesetzt wird, basiert ebenfalls auf der *Ordinary least squares*-Methode. Im Unterschied zu Statsmodel wird mit scikit-learn gleich eine Multiple-Regression durchgeführt, da diese deutlich bessere Ergebnisse erzielen konnte. 

# In[308]:


lm_pipe = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('lm', LinearRegression())
                        ])


# In[309]:


lm_pipe.fit(X_train, y_train)


# In[310]:


y_pred = lm_pipe.predict(X_train)


# **Metrics and Validation**

# In[311]:


r2_lm, rmse_lm, mse_lm = metrics_sk(y_train, y_pred)


# Durch die Wahl geeignter Variablen wird gleich zu Beginn ein Model mit einem RMSE unter 70 Tsd. USD erzielt. 

# ### sklearn Lasso

# **Data preperation**

# In[312]:


df = read_data()
df = transform_data(df)
df


# In[313]:


#Split Test and Trainingsdata
X = df.drop(columns = ['median_house_value', 'price_category'], axis=1)
y = df['median_house_value']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[314]:


X_train = add_feautures(X_train)


# In[315]:


preprocessor = build_preprocessor()


# Bei der Vorbereitung der Daten für das Lasso-Modell werden keine weiteren Variablen ausgeschlossen, da dieses Model eine Feauture Selection durchführt. 

# **Model LassoCV: Build**

# Zur Bestimmung des optimalen Hyperparameters wird hier eine Lasso k-folds Cross Validation angewendet. Dafür werden die Trainingsdaten in fünf Teilsets aufgeteilt. 

# In[316]:


LassoCV_pipe = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('LassoCV', LassoCV(cv=5, random_state=0, max_iter=10000))
                        ])


# In[317]:


LassoCV_pipe.fit(X_train, y_train)


# In[318]:


y_pred = LassoCV_pipe.predict(X_train)


# **Model LassoCV: Metrics and Validation**

# In[319]:


r2_lasso, rmse_lasso, mse_lasso = metrics_sk(y_train, y_pred)


# In[320]:


alpha = LassoCV_pipe.named_steps['LassoCV'].alpha_
alpha


# Durch Auslesen des Attributs `alpha_` wird der am besten geeignte Hyperparameter ausgewählt. Dieser wird im folgenden auf die Lasso-Regression angewendet. 

# **Model Lasso Best Alpha: Build**

# In[321]:


Lasso_alpha_pipe = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('Lasso', Lasso(alpha=alpha))
                        ])


# In[322]:


Lasso_alpha_pipe.fit(X_train, y_train)


# **Model Lasso Best Alpha: Metrics and Validation**

# In[323]:


r2_lasso, rmse_lasso, mse_lasso = metrics_sk(y_train, y_pred)


# In[324]:


importance = np.abs(Lasso_alpha_pipe.named_steps['Lasso'].coef_)
len(importance)


# In[325]:


plt.rcParams['figure.figsize'] = [30, 5]
list = []
for i in range(len(importance)):
    list.append(i)
list

sns.barplot(x=list, y=importance)


# Der RMSE liegt unter 70 Tsd. USD und entspricht demententsprechend den Anforderung an das Modell. 

# ### sklearn Spline

# **Data preperation**

# In[326]:


df = read_data()
df = transform_data(df)


# In[327]:


#Split Test and Trainingsdata
X = df.drop('median_house_value', axis=1)
y = df['median_house_value']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[328]:


X_train = add_feautures(X_train)


# In[329]:


#Feauture Selection 
X_train = X_train[['median_income', 'ocean_proximity','housing_median_age', 'households_population', 'total_rooms_households', 'total_rooms_total_bedrooms', 'geohash']]


# Im Gegensatz zu Statsmodels können bei scikit-lern mehr als eine Variable verwendet werden. 

# In[330]:


preprocessor = build_preprocessor()


# **Splines: Build**

# In[331]:


Splines_pipe = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('polynomialfeatures', PolynomialFeatures()),
    ('ridge', Ridge())
                        ])


# In[332]:


Splines_pipe.fit(X_train, y_train)


# In[333]:


y_pred = Splines_pipe.predict(X_train)


# **Splines: Metrics and Validation**

# In[334]:


r2_spline, rmse_spline, mse_spline = metrics_sk(y_train, y_pred)


# Der RMSE liegt mit 56,3 Tsd. USD deutlich unter 70 Tsd. USD.

# ### Evaluation Sklearn

# **Select best model**

# Wie bei Statsmodel werden die RMSE-Werte zuerst verglichen und dann das beste Modell mit den Testdaten validiert. 

# In[335]:


results= {'RMSE': ['RMSE OLS', 'RMSE Lasso', 'RMSE Spline'],
        'Werte': [rmse_lm, rmse_lasso, rmse_spline] }
df_results = pd.DataFrame(results)
df_results.sort_values(by ='Werte')


# Das beste Modell ist das Spline-Modell. 

# **Predict with test data**

# In[336]:


X_test = add_feautures(X_test)
X_test = X_test[['median_income', 'ocean_proximity','housing_median_age', 'households_population', 'total_rooms_households', 'total_rooms_total_bedrooms', 'geohash']]


# In[337]:


y_test_pred = Splines_pipe.predict(X_test)


# **Final Results**

# In[338]:


y_test.fillna(y_test.median(), inplace= True)
r2_test, rmse_test, mse_test = metrics_sk(y_test, y_test_pred)


# Der RMSE der Testdaten liegt ebenfalls unter 70 Tsd. USD bei 58,7 Tsd. USD. Dieses Ergebnis ist besser als das durch Statsmodel erzielte.

# > Im Rahmen der Anforderungen kann auch die Regression als Erfolg gewertet werden.
