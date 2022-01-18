#!/usr/bin/env python
# coding: utf-8

# # Model: Regression 

# In diesem Notebook werden verschieden Regressionsmodell angewendet, um den mittleren Immobilienpreis von Distrikten in Kalifornien vorherzusagen. Dabei ist die Anforderung an die Model einen RMSE von kleiner 70 Tsd. USD zu erzielen. Dann wird das Projekt als Erfolg gewertet. 
# 

# Die Regressionsmodelle werden in diesem Notebook mit Statsmodels und der Bibliothek scikit-learn durchgeführt. Die Auswahl der Modelle wird beim jeweiligen Modell begründet und beschrieben. Grundlegend ist jedes Modell in *Data preperation*, *Build* und *Metrics und Validation* unterteilt. Für das beste Modell der jeweiligen Bibliothek wird eine Evaluation mit den Testdaten durchgeführt. 

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


# ### Statsmodels OLS

# **Data preperation**

# Zur *Data preperation* bei Nutzung der Bibliothek Statsmodels werden Funktion aus der der Datei func.py verwendet. Diese werden im Notebook Data näher erläutert. 

# In[2]:


df = read_data()
df = transform_data(df)
train_dataset, evaluate_dataset, train_dataset_total, test_dataset = split_data(df)
train_dataset = fill_missingdata(train_dataset)
train_dataset = add_feautures(train_dataset)
train_dataset


# **Model 1 OLS: Build**

# In[3]:


lm1 = smf.ols(formula ='median_house_value ~ median_income', data=train_dataset).fit()


# Als erstes Modell wir eine einfach lineare Regression gewählt. Als unabhängige Variable wir *median_income* gewählt, da das mittlere Einkommen eines Bezirks die höchste Korrelation zu *median_house_value* aufweist. 

# In[4]:


train_dataset['y_pred'] = lm1.predict()


# **Model OLS 1: Metrics and Evaluation**

# In[5]:


lm1.summary()


# Die Kenngrößen in der summary() sind wie erwartet nicht besonders gut. Der Adj. R-squared liegt bei 0.47. Dieser besagt, dass unter 50% der Streuung der Werte durch das Modell korrekt vorhergesagt werden kann. Weitere Kenngrößen werden im Vergleich zum zweiten Modell betrachtet. 

# In[6]:


r2_lm1, rmse_lm1, mse_lm1 = metrics_stats(lm1,train_dataset)


# Ebenfalls liegt der RMSE mit 84.343 USD über dem Zielwert von 70 Tsd. USD. Das Modell kann dementsprechend nicht als Erfolg gewertet werden. 

# **Model 2 OLS: Build**

# In[7]:


lm2 = smf.ols(formula ='median_house_value ~ median_income + total_rooms_total_bedrooms', data=train_dataset).fit()


# In[8]:


# Add the regression predictions (as "pred") to our DataFrame
train_dataset['y_pred'] = lm2.predict()


# **Model 2 OLS: Metrics and Evaluate**

# In[9]:


lm2.summary()


# Im Vergleich zu Model 1 erreicht das Model 2 besser statistische Kenngröße. Der Adj. R-squared liegt bei ca. 3% über dem vom Model 1. Sowohl die Kenngröße der F-Statistik, wie auch AIC und BIC sind höher als bei Model 1. 

# > Hinweis: Zu diesem Modell wird ausführlich Regression Diagnostics durchgeführt. 

# In[10]:


r2_lm2, rmse_lm2, mse_lm2 = metrics_stats(lm2,train_dataset)


# Der RMSE liegt mit ca. 81,6 Tsd. USD immer noch deutlich über 70 Tsd. USD. Es muss dementsprechend ein geeigneteres Modell gefunden werden. 

# **Model 3 OLS: Build**

# In[11]:


lm3 = smf.ols(formula ='median_house_value ~ median_income + C(ocean_proximity) + households_population + total_rooms_households +C(geohash) + housing_median_age', data=train_dataset).fit()


# Im dritten Modell werden mehr Variablen aus dem ursprünglichen Datensatz berücksichtigt. Nicht miteingeschlossen werden: 
# Nicht im Modell berücksichtigt werden folgende Features 
# * *longitude*, *latidue* (siehe Feauture Selection)
# * *total_rooms*, *total__bedrooms*, *population* und *households*, da die Korrelation dieser Merkmale untereinander zu hoch ist (siehe Feauture Selection)
# * *price_category*, da diese aus *median_house_value* abgeleitet wird
# * *total_rooms_total_bedrooms*, da nach einmaliger Durchführung der Klassifikation ein p-Wert von P|z|=0.555 ermittelt wurde
# 
# 

# In[12]:


train_dataset['y_pred'] = lm3.predict()


# **Model OLS 3: Metrics and Evaluation**

# In[13]:


lm3.summary()


# In[14]:


r2_lm3, rmse_lm3, mse_lm3 = metrics_stats(lm3,train_dataset)


# Durch Model 3 konnte ein RMSE von 63.633 USD und damit unter der Grenze von 70 Tsd. USD erzielt werden. Erst nach Evaluation mit einem Testdatenset kann, dass Model Erfolg gewertet werden. 

# ### Regression Diagnostics

# **Outliner and High-Leverage**

# In[15]:


fig = sm.graphics.influence_plot(lm2, criterion="cooks")


# In[16]:


lm_cooksd = lm2.get_influence().cooks_distance[0]
n = len(train_dataset["median_house_value"])
print('Anzahl an Datensätzen:', n)
critical_d = 4/n
print('Kritische Cook distance:', critical_d)
out_d = lm_cooksd > critical_d
print("Anzahl Datensätze mit kritischer Cooks distance:",len(lm_cooksd[out_d]))


# 914 Datensätzen besitzen eine kritische Cooks Distanz. Dabei handelt es sich um vermutliche outliner mit leverage. Um das einfachste Vorgehen zu wählen werden die observation entfernt. 

# In[17]:


train_dataset.index[out_d]


# In[18]:


train_dataset_opt = train_dataset.drop(train_dataset.index[out_d])
train_dataset_opt


# **Non-linearity and heteroscedasticity**

# In[38]:


fig = sm.graphics.plot_partregress_grid(lm2)
fig.tight_layout(pad=1.0)


# In[19]:


model_fitted_y = lm2.fittedvalues

#  Plot
plot = sns.residplot(x=model_fitted_y, y='median_house_value', data=train_dataset, lowess=True, 
                     scatter_kws={'alpha': 0.5}, 
                     line_kws={'color': 'red', 'lw': 1, 'alpha': 0.8})


# * keine ungleichmäßige Verteilung der residuals: Hinweis auf Heteroscedasticity
# * Vermutlich auch keine lineare Beziehung da die rote Linie sich von der gestrichelten unterscheidet

# In[20]:


# Breusch-Pagan Lagrange Multiplier test

name = ['Lagrange multiplier statistic', 'p-value', 'f-value', 'f p-value']
test = sm.stats.het_breuschpagan(lm2.resid, lm3.model.exog)
lzip(name, test)


# Beide p-werte liegen deutlich unter 0.05. Daher wird die Null-Hypothese abgelehnt. 
# Das deutet ist ebenfalls ein Indikator für Heteroscedasticity. Eine mögliche Lösung ist in diesem Fall die Anwendung eines Regression Splines. Dies wird in einem der folgenenden Abschnitten untersucht. 

# **Non-normally distributed errors**

# In[21]:


#Jarque-Bera test
name = ['Jarque-Bera', 'Chi^2 two-tail prob.', 'Skew', 'Kurtosis']
test = sm.stats.jarque_bera(lm2.resid)

lzip(name, test)


# Der p-Wert liegt bei 0.0 damit wird die Nullhyptothese abgelehnt. Das ist ein Indikator für nicht normalverteilte error terms.

# In[22]:


#Omnibus normtest
name = ['Chi^2', 'Two-tail probability']
test = sm.stats.omni_normtest(lm2.resid)
lzip(name, test)


# Auch bei diesem Test wird die Nullhyptothese abgelehnt. Das lässt ebenso auf nicht normalverteilte error terms schließen. 

# Da die Anzahl an N sehr hoch ist, müssen die Fehler nicht normalverteilt sein. Grund dafür ist das Central Limit Theorem. Es werden keine weiteren Schritte unternommen. 

# **Correlation of error terms**

# In[23]:


# Durbin Watson test
#TODO Durbin Watson zu nah an 2?
sm.stats.durbin_watson(lm2.resid)


# Der Wert liegt zwischen 1 und 2. Aus diesem Grund liegend keine Korrelation vor. 

# **Collinearity**

# In[24]:


#Variance inflation factor
y, X = dmatrices('median_house_value ~ median_income + total_rooms_total_bedrooms', data=train_dataset, return_type='dataframe')

# For each X, calculate VIF and save in dataframe
vif = pd.DataFrame()
vif["VIF Factor"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif["Feature"] = X.columns

vif.round(2)


# Für dieses Modell liegt keine problematische collinearity in den Daten vor. 

# **Model OLS 3 opt: Build**

# 
# In diesem linearen Modell wird das Modell mit multipler linearer Regression auf die bereinigten Trainingsdaten nach der Regression Diagnostics angewendet. 
# 

# In[34]:


lm3opt = smf.ols(formula ='median_house_value ~ median_income + C(ocean_proximity) + households_population + total_rooms_households +C(geohash) + housing_median_age', data=train_dataset_opt).fit()


# In[35]:


train_dataset_opt['y_pred'] = lm3opt.predict()


# **Model OLS 3 opt: Metrics and Evaluation**

# In[36]:


lm3opt.summary()


# In[37]:


r2_lm3opt, rmse_lm3opt, mse_lm3opt = metrics_stats(lm3opt,train_dataset_opt)


# Für dieses Modell liegt der RMSE unter 50 Tsd. USD.  Wird nur diese Größe betrachtet handelt es hierbei um das beste Modell. 

# ### Statsmodel Lasso

# **Model Lasso: Build**

# Folgendes Modell basiert auf eine Recherche im Internet zur Anwendung von Lasso mit der Bibliothek Statsmodels. Mit dem Parameter `L1_wt` wird eine regularisiertes Modell nach Lasso umgesetzt. Zu dieser Umsetzung habe ich wenige Quelle gefunden. 

# In[38]:


lmlasso = smf.ols(formula ='median_house_value ~ median_income + C(ocean_proximity) + households_population + total_rooms_households + total_rooms_total_bedrooms+C(geohash)', data=train_dataset).fit_regularized(L1_wt=1)


# In[39]:


train_dataset['y_pred'] = lmlasso.predict()


# **Model Lasso: Metrics and Evaluation**

# In[40]:


rmse_lasso= round(rmse(train_dataset['median_house_value'], train_dataset['y_pred']), 2)
mse_lasso = round (mse(train_dataset['median_house_value'], train_dataset['y_pred']), 2)
print("RMSE ",rmse_lasso)
print("MSE: ",mse_lasso)


# Der RMSE liegt mit ca. 66,7 Tsd. USD unter 70 Tsd. USD. 

# ### Natural Spline Statsmodel

# **Model Splines: Build**

# Das Natural Splines Modell wir mit nur einer Variablen umgesetzt. Da *median_income* die höchste Korrelation  mit *median_house_value* aufweist wird dieser in das Modell einbezogen. 

# In[48]:


X_train = train_dataset['median_income']
y_train = train_dataset['median_house_value']


# In[50]:


from patsy import dmatrix
transformed_x3 = dmatrix("cr(train,df = 3)", {"train": X_train}, return_type='dataframe')


# In[51]:


spline = sm.GLM(y_train, transformed_x3).fit()


# In[44]:


y_pred= spline.predict(dmatrix("cr(train, df=3)", {"train": X_train}, return_type='dataframe'))


# **Model Splines: Metrics and Evaluation**

# In[45]:


spline.summary()


# In[46]:


rmse_spline= round(rmse(train_dataset['median_house_value'], y_pred), 2)
mse_spline = round (mse(train_dataset['median_house_value'], y_pred), 2)
print("RMSE ",rmse_spline)
print("MSE: ",mse_spline)


# Da ein Ergebnis der Regression Diagnostic ist, dass Spline für diese Problem besser geeignet sind, entspricht das Ergebnis nicht den Erwartungen. Grund kann dafür das Einbeziehen nur einer Variable sein. 

# In[47]:


sns.scatterplot(x=X_train, y=y_train)
plt.plot(X_train, y_pred, color='orange', label='Natural spline with df=3')
plt.legend()


# ### Evaluation

# Für die Evaluation werden die RMSE der einzelnen Modelle miteinander verglichen. Das Modell mit dem niedrigsten RMSE wird mit den Testdaten validiert. 

# In[52]:


results= {'RMSE': ['RMSE OLS 1', 'RMSE OLS 2', 'RMSE OLS 3', 'RMSE OLS 3 opt', 'RMSE Lasso', 'RMSE Spline'],
        'Werte': [rmse_lm1, rmse_lm2, rmse_lm3, rmse_lm3opt, rmse_lasso, rmse_spline] }
df_results = pd.DataFrame(results)
df_results.sort_values(by ='Werte')


# In[74]:


test_dataset = add_feautures(test_dataset)
test_dataset = fill_missingdata(test_dataset)


# In[73]:


test_dataset['y_pred'] = lm3opt.predict(test_dataset[['median_income', 'ocean_proximity', 'households_population', 'total_rooms_households', 'housing_median_age', 'geohash']])


# In[72]:


r2_test, rmse_test, mse_test = metrics_stats(lm3opt,test_dataset)


# In einem realen Umfeld wäre nach Erstellung des Modells eine vollständige Regression Diagnostics notwendig gewesen, um das Ergebnis richtig einordnen zu können. 

# ## Model sklearn

# In[53]:


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

# In[55]:


df = read_data()
df = transform_data(df)


# In[56]:


#Split Test and Trainingsdata
X = df.drop('median_house_value', axis=1)
y = df['median_house_value']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[57]:


X_train = add_feautures(X_train)


# In[58]:


#Feauture Selection 
X_train = X_train[['median_income', 'ocean_proximity','housing_median_age', 'households_population', 'total_rooms_households', 'total_rooms_total_bedrooms', 'geohash']]
X_train


# In[59]:


preprocessor = build_preprocessor()


# **Model OLS**

# Das erste Modell, welches mit scikit-learn umsgesetzt wird, basiert ebenfalls auf der *Ordinary least squares*-Methode. Im Unterschied zu Statsmodel wird mit scikit-learn gleich eine Multiple-Regression durchgeführt, da diese deutlich bessere Ergebnisse erzielen konnte. 

# In[60]:


lm_pipe = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('lm', LinearRegression())
                        ])


# In[61]:


lm_pipe.fit(X_train, y_train)


# In[62]:


y_pred = lm_pipe.predict(X_train)


# **Metrcis and Evaluation**

# In[63]:


r2_lm, rmse_lm, mse_lm = metrics_sk(y_train, y_pred)


# Durch die Wahl geeignter Variablen wird gleich zu Beginn ein Model mit einem RMSE unter 70 Tsd. USD erzielt. 

# In[161]:


#TODO
#sns.residplot(x=y_pred, y=y_train, scatter_kws={"s": 80})


# ### sklearn Lasso

# **Data preperation**

# In[64]:


df = read_data()
df = transform_data(df)
df


# In[71]:


#Split Test and Trainingsdata
X = df.drop(columns = ['median_house_value', 'price_category'], axis=1)
y = df['median_house_value']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[72]:


X_train = add_feautures(X_train)


# In[73]:


preprocessor = build_preprocessor()


# Bei der Vorbereitung der Daten für das Lasso-Modell werden keine weiteren Variablen ausgeschlossen, da das Model eine Feauture Selection durchführt. 

# **Model LassoCV: Build**

# Zur Bestimmung des optimalen Hyperparameters wird hier eine Lasso Cross Validation angewendet. Dafür werden die Trainingsdaten in 5 Teilsets aufgeteilt. 

# In[74]:


LassoCV_pipe = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('LassoCV', LassoCV(cv=5, random_state=0, max_iter=10000))
                        ])


# In[85]:


LassoCV_pipe.fit(X_train, y_train)


# In[76]:


y_pred = LassoCV_pipe.predict(X_train)


# **Model LassoCV: Metrics and Evaluation**

# In[80]:


r2_lasso, rmse_lasso, mse_lasso = metrics_sk(y_train, y_pred)


# In[81]:


alpha = LassoCV_pipe.named_steps['LassoCV'].alpha_
alpha


# Durch Auslesen des Attributs `alpha_` wird der am besten geeignte Hyperparameter ausgewählt. Dieser wird im folgenden auf die Lasso-Regression angewendet. 

# **Model Lasso Best Alpha: Build**

# In[82]:


Lasso_alpha_pipe = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('Lasso', Lasso(alpha=alpha))
                        ])


# In[83]:


Lasso_alpha_pipe.fit(X_train, y_train)


# **Model Lasso Best Alpha: Metrics**

# In[84]:


r2_lasso, rmse_lasso, mse_lasso = metrics_sk(y_train, y_pred)


# Der RMSE liegt unter 70 Tsd. USd. und entspricht demententsprechend den Anforderung an das Modell. 

# ### sklearn Spline

# **Data preperation**

# In[235]:


df = read_data()
df = transform_data(df)


# In[275]:


#Split Test and Trainingsdata
X = df.drop('median_house_value', axis=1)
y = df['median_house_value']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[237]:


X_train = add_feautures(X_train)


# In[238]:


#Feauture Selection 
X_train = X_train[['median_income', 'ocean_proximity','housing_median_age', 'households_population', 'total_rooms_households', 'total_rooms_total_bedrooms', 'geohash']]


# In[239]:


preprocessor = build_preprocessor()


# **Splines: Build**

# In[240]:


Splines_pipe = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('polynomialfeatures', PolynomialFeatures()),
    ('ridge', Ridge())
                        ])


# In[242]:


Splines_pipe.fit(X_train, y_train)


# In[244]:


y_pred = Splines_pipe.predict(X_train)


# In[281]:


r2_spline, rmse_spline, mse_spline = metrics_sk(y_train, y_pred)


# In[147]:


#TODO Graph?#TODO Frage Scipy statt sklearn
#from scipy.interpolate import CubicSpline
#natural_spline = CubicSpline(x, y, bc_type='natural')


# ### Evaluation Sklearn

# **Select best model**

# In[262]:


results= {'RMSE': ['RMSE OLS', 'RMSE Lasso', 'RMSE Spline'],
        'Werte': [rmse_lm, rmse_lasso, rmse_spline] }
df_results = pd.DataFrame(results)
df_results.sort_values(by ='Werte')


# **Predict with test data**

# In[282]:


X_test = add_feautures(X_test)
X_test = X_test[['median_income', 'ocean_proximity','housing_median_age', 'households_population', 'total_rooms_households', 'total_rooms_total_bedrooms', 'geohash']]


# In[285]:


y_test_pred = Splines_pipe.predict(X_test)


# **Final Results**

# In[286]:


y_test.fillna(y_test.median(), inplace= True)
r2_test, rmse_test, mse_test = metrics_sk(y_test, y_test_pred)

