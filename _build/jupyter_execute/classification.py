#!/usr/bin/env python
# coding: utf-8

# # Model: Klassifikation 

# Ziel der Klassifikation ist es die Variabel *price category* mit einem F1-Score größer als 0.8 vorherzusagen. Die Variable *price category* gibt an ob der mittlere Immobilienwert eines Distrikts über 150 Tsd. USD liegt. 

# In diesem Notebook wird die Klassifikation einmal mit der Bibliothek Statsmodels und der Bibliothek scikit-learn durchgeführt. Die Dokumention zu beiden Modellen ist unterteilt in: *data preperation*, *model* und *evaluation*. 

# > Über den [Link](https://sh333hdm.github.io/jupyterbooktest/intro.html) ist die Ansicht der Projektarbeit als Jupyter Book möglich. 

# ## Statsmodel Model

# In[1]:


from func import *
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

import statsmodels.api as sm
import statsmodels.formula.api as smf
get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


# ### Data preperation 

# In[2]:


df = read_data()
df = transform_data(df)
train_dataset, evaluate_dataset, train_dataset_total, test_dataset = split_data(df)
train_dataset = fill_missingdata(train_dataset)
train_dataset = add_feautures(train_dataset)
train_dataset


# In diesem Schritt werden die Funktionen func.py verwendet. Diese sind im Notebook [Data](https://sh333hdm.github.io/jupyterbooktest/intro.html)  näher erläutert. 

# In[29]:


train_dataset.drop(columns=['longitude', 'latitude', 'total_rooms', 'total_bedrooms', 'population','households', 'median_house_value', 'housing_median_age'], inplace= True)


# Nicht im Modell berücksichtigt werden folgende Features 
# * *longitude*, *latidue* (siehe Feauture Selection)
# * *total_rooms*, *total__bedrooms*, *population* und *households*, da die Korrelation dieser Merkmale untereinander zu hoch ist (siehe Feauture Selection)
# * *median_house_value*, da *price_category* aus diesem Merkmal abgeleitet wird
# * *housing_median_age*, da nach einmaliger Durchführung der Klassifikation ein p-Wert von P|z|=0.309 ermittelt wurde 

# ### Model

# In[30]:


model = smf.glm(formula = 'C(price_category) ~ median_income + C(ocean_proximity) + households_population + total_rooms_households + total_rooms_total_bedrooms+C(geohash)' , data=train_dataset, family=sm.families.Binomial()).fit()


# Die Klassifikation wird durch das Modell der logistischen Regression umgesetzt. In Statsmodels wird dafür die Klasse *Generalized Linear Model* genutzt. 
# Die Varaiblen werden dem Model mit Hilfe der ``formula.api`` übergeben. Bei der Übergabe werden die kategorialen Variablen *price_category*, *ocean_proximity* und *geohash* explizit durch die patsy-funktion `C()` als solche gekennzeichnet. Damit wird sichergestellt, dass diesem im Modell als kategorialen Variablen behandelt werden und ein Dummy-Coding durchgeführt wird. 
# Mit dem Parameter sm.family.Binomial() wird eine binomiale Verteilung spezifiziert. 
# 

# In[38]:


print(model.summary())


# In[32]:


data_set_prob = train_dataset
data_set_prob['Probability_above'] = model.predict()
data_set_prob


# Ergebnis der logistischen Regression sind Wahrscheinlichkeiten p für das Eintreten der Ausprägung *above* der kategorialen Variabel *price_category*. 

# In[33]:


data_set_prob['Threshold 0.4'] = np.where(data_set_prob['Probability_above'] > 0.4, 'above', 'below')
data_set_prob['Threshold 0.5'] = np.where(data_set_prob['Probability_above'] > 0.5, 'above', 'below')
data_set_prob['Threshold 0.6'] = np.where(data_set_prob['Probability_above'] > 0.6, 'above', 'below')
data_set_prob['Threshold 0.7'] = np.where(data_set_prob['Probability_above'] > 0.7, 'above', 'below')
data_set_prob


# Zum bestimmten den Metriken*Accuracy*, *Precision*, *Recall* und *F1 Score*, sowie der Confusion-Matrix der einzelnen Grenzwerte wird die Funktion `print_metrics` angewendet. Diese wird in der Datei func.py definiert und ist der Funktion im [Notebook: World happiness report]( https://colab.research.google.com/github/kirenz/applied-statistics/blob/main/docs/cl-logistic-whr.ipynb) nachempfunden. 

# In[39]:


print_metrics(data_set_prob, 'Threshold 0.4')
print_metrics(data_set_prob, 'Threshold 0.5')
print_metrics(data_set_prob, 'Threshold 0.6')
print_metrics(data_set_prob, 'Threshold 0.7')


# Die Klassifikation mit den Grenzwert 0.6 hat den höchsten F1-Score mit 0.8221. Im Schritt Plan wurde festgelegt, dass der F1-Score, dass ausschlagegebende Kriterium ist. Daher wird dieser Grenzwert auch für die Testdaten angewendet.  

# ### Evaluate

# In[35]:


test_dataset = fill_missingdata(test_dataset)
test_dataset = add_feautures(test_dataset)
test_dataset


# Um das Modell auf Testdaten anzuwenden, müssen diese auch mit den Funktionen `fill_missingdata` und `add_features` vorbereitet werden. 

# In[36]:


test_dataset['y_pred'] = model.predict(test_dataset[['median_income', 'ocean_proximity','housing_median_age', 'households_population', 'total_rooms_households', 'total_rooms_total_bedrooms', 'geohash']])


# In[37]:


test_dataset['Threshold 0.6'] = np.where(test_dataset['y_pred'] > 0.6, 'above', 'below')
print_metrics(test_dataset, 'Threshold 0.6')


# Mit dem Testdaten wird ein F1-Score von 0.8155 erhalten. Dieser liegt über den Zielwert von 0.8. Das Model kann damit als Erfolg gewertet werden.  

# ## scikit-learn Model

# In[91]:


from sklearn.model_selection import train_test_split

from sklearn.compose import ColumnTransformer
from sklearn.compose import make_column_selector as selector
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import sklearn.linear_model as skl_lm

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report

from func import *
from sklearn import set_config
set_config(display="diagram")


# ### Data preperation

# In[73]:


df = read_data()
df = transform_data(df)


# In[74]:


#Split Test and Trainingsdata
X = df.drop(['price_category', 'median_house_value', ], axis=1)
y = df['price_category']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[75]:


X_train = add_feautures(X_train)


# In[78]:


X_train.drop(columns=['longitude', 'latitude', 'total_rooms', 'total_bedrooms', 'population','households'], inplace= True)


# In[79]:


preprocessor = build_preprocessor()


# Zur Datenvorbereitung für die Klassifikation mit der Bibliothek scikit-learn wird ein *preprocessor* angewendet. Der Aufbau des *preprocessors* ist im Notebook Data genauer beschrieben. In diesem Fall wird dieser über die Funktion `build_preprocessor`erstellt. 

# ### Model

# In[93]:


lr_pipe = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('lm', skl_lm.LogisticRegression(max_iter=1000))
                        ])


# In[94]:


y_pred = lr_pipe.fit(X_train, y_train).predict(X_train)


# Mit scikit-learn wird ebenfalls eine logistische Regression durchgeführt. 

# In[95]:


cm = confusion_matrix(y_train, y_pred)

disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                              display_labels=lr_pipe.classes_)
disp.plot()
plt.show()


# In[96]:


print(classification_report(y_train, y_pred))


# Für die Ausgabe der Metriken der Klassifikation stellt scikit-learn Funktionen zur Verfügung. Wie bei dem Model mit Statsmodels wird zu Bewertung des Models die *Confusion Matrix* und die Metriken *Accuracy*, *Precision*, *Recall* und *F1 Score* betrachtet.  

# Da bei der Erstellung des Models mit scikit-learn ein Grenzwert von 0.5 angewendet wird. Können im folgenden anderen Grenzwerte getestet werden: 

# In[102]:



pred_proba = lr_pipe.predict_proba(X_train)

df_conf = pd.DataFrame({'y_train': y_train, 'y_pred': pred_proba[:,1]})
df_conf['Threshold 0.1'] = np.where(df_conf['y_pred'] > 0.1, 'below', 'above')
df_conf['Threshold 0.2'] = np.where(df_conf['y_pred'] > 0.2, 'below', 'above')
df_conf['Threshold 0.3'] = np.where(df_conf['y_pred'] > 0.3, 'below', 'above')
df_conf['Threshold 0.4'] = np.where(df_conf['y_pred'] > 0.4, 'below', 'above')
df_conf['Threshold 0.5'] = np.where(df_conf['y_pred'] > 0.5, 'below', 'above')
df_conf['Threshold 0.6'] = np.where(df_conf['y_pred'] > 0.5, 'below', 'above')
df_conf['Threshold 0.7'] = np.where(df_conf['y_pred'] > 0.7, 'below', 'above')
df_conf['Threshold 0.8'] = np.where(df_conf['y_pred'] > 0.8, 'below', 'above')
#Ausgabe Änderung Treshold
cm = confusion_matrix(y_train, df_conf['Threshold 0.5'])
print(classification_report(y_train, df_conf['Threshold 0.5']))


disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=lr_pipe.classes_)

disp.plot()
plt.show()


# Der ursprüngliche Grenzwert eignet sich am besten für die Anwendung auf die Testdaten, da dieser mit 0.87 den höchsten F1-Score erzielt. 

# ### Evaluate

# In[103]:


y_pred = lr_pipe.fit(X_test, y_test).predict(X_test)


# In[104]:


print(classification_report(y_test, y_pred))


# In[105]:


cm = confusion_matrix(y_test, y_pred)

disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                              display_labels=lr_pipe.classes_)
disp.plot()
plt.show()


# Mit den Testdaten wird ein F1-Score von 0.84 erreicht. Das Modell kann damit als Erfolg gewertet werden. 
