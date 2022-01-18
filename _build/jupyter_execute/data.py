#!/usr/bin/env python
# coding: utf-8

# # Data

# > Über den [Link](https://sh333hdm.github.io/jupyterbooktest/intro.html) ist die Ansicht der Projektarbeit als Jupyter Book möglich. 

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')
import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.stats import descriptivestats

sns.set_theme(palette="Pastel2", style="whitegrid") 


# ## Data ingestion

# In[2]:


# Einlesen der Datei
# func.py: read_data
path = "data\project_data.csv"
df = pd.read_csv(path)


# In[3]:


df.info()


# Eine Änderung der Datentypen der Variablen *median_house_value*, *housing_median_age*, *ocean_proximity* und *price_category* ist notwendig. 

# In[4]:


# func.py: transform_data
#median_house_value -> float
#housing_median_age -> float
#ocean_proximity -> category
#price_category -> category

df['ocean_proximity'] = df['ocean_proximity'].astype("category")
df['price_category'] = df['price_category'].astype("category")

df['median_house_value'] = pd.to_numeric(df['median_house_value'], errors='coerce')
df['housing_median_age'] = pd.to_numeric(df['housing_median_age'], errors='coerce')


# Da Anomalien ausschließlich in der ersten Zeile vorkommen, werde diese schon bei der Umwandlung zum Datentyp "numeric" entfernt. Durch Setzen des Parameter `errors='coerce'` werden die ungültigen Werte mit NAN aufgefüllt. 

# In[5]:


df[(df['median_house_value'] <150000) & (df['price_category'] != 'below')]


# Die Variable  *price_category* ist nicht korrekt auf den *median_house_value* gemappt. Aus diesem Grund wird die Spalte *price_category* zur Korrektur angepasst. 

# In[6]:


# func.py: transform_data
df['price_category'] = np.where(df['median_house_value'] > 150000, 'above', 'below')
df['price_category'] = df['price_category'].astype("category")


# In[7]:


df[(df['median_house_value'] <150000) & (df['price_category'] != 'below')]


# ## Data Spliting

# Die vorliegenden Daten werden in Test- und Trainingsdaten aufgespaltet. 
# 

# In[8]:


# func.py: split_data
train_dataset = df.sample(frac=0.8, random_state=0)
test_dataset = df.drop(train_dataset.index)


# ## Clean Data

# In[9]:


train_dataset.isnull().sum()


# In[10]:


sns.heatmap(train_dataset.isnull(),yticklabels=False,cbar=False, cmap='flare')


# NAN tritt insgesamt 162mal auf. Auf Grund der geringen Anzahl werden die Datensätze entfernt.  

# In[11]:


#func.py: fill_missingdata
median_total_bedrooms = train_dataset["total_bedrooms"].median()
train_dataset.dropna(inplace=True)
train_dataset.isnull().sum()


# Für die Modell mit scikit-learn wird der `SimpleImputer` genutzt. Die numerischen NAN-Werte, werden dafür mit dem Median aufgefüllt. 

# ## Analyse Data

# ### Geographischer Daten- Mapping

# Im ersten Schritt wird der *median_house_value* auf Kartendarstellung visualisert. 

# In[12]:


import plotly.express as px
import plotly.graph_objects as go
import plotly.express as px


fig = go.Figure(data=go.Scattergeo(
        locationmode = 'USA-states',
        lon = df['longitude'],
        lat = df['latitude'],
        mode = 'markers',
        marker = dict(
            size = 8,
            opacity = 0.8,
            reversescale = True,
            autocolorscale = False,
            line = dict(
                width=1,
                color='rgba(102, 102, 102)'
            ),
            colorscale = 'AgSunset',
            cmin = 0,
            color = df['median_house_value'],
            cmax = df['median_house_value'].max(),
            colorbar_title="Median house value"
        )))

fig.update_layout(
        title = 'Map Median House Value',
        geo_scope='usa',
    )
fig.show()


# In[13]:



import plotly.express as px

fig = px.scatter_mapbox(train_dataset, lat="latitude", lon="longitude", size= "median_house_value",color="median_house_value",color_continuous_scale = 'Sunset',zoom=3, height=300)
fig.update_layout(mapbox_style="open-street-map")
fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
fig.show()


# ### Geographische Variablen - Scatterplot

# Um die Beziehung der Variablen *Longitude/Latitude* zu *median_house_value* und *ocean_proximity* besser nachvollziehen zu können, werden diese Werte im zweiten Schritt als Streudiagramm analysiert. 

# In[14]:


plt.rcParams['figure.figsize'] = [16, 16]


# In[15]:


sns.regplot(data = train_dataset, x="longitude", y="latitude",  fit_reg=False)


# Im Scatterplot ist der US-Bundesstaat Kalifornien mit den Ballungsgebieten San Francisco und Los Angeles abgebildet. 

# In[16]:


sns.scatterplot(data=train_dataset, x="longitude", y="latitude", size="population", legend=True, sizes=(20, 2000))


# Die Größe der Population-Kreis bestätigt die Annahme, dass es sich um Ballungsgebiete handelt. 

# In[17]:



sns.scatterplot(data = train_dataset, x="longitude", y="latitude",  hue = "price_category")


# Die Hervorhebung der Ausprägungen der kategorialen Variable *price_category* lässt vermuten, dass es eine Zusammenhang zwischen der Lage und *median_house_value*/*price_category* gibt. 

# In[18]:


sns.scatterplot(data=train_dataset, x="longitude", y="latitude", size="median_house_value", hue="price_category", legend=True, sizes=(20, 2000))


# In[19]:


df_below = train_dataset[train_dataset['price_category'] == 'below']
sns.scatterplot(data=df_below, x="longitude", y="latitude", size="median_house_value", legend=True, sizes=(20, 2000))


# * Die Distrikte mit einem *median_house_value* von unter 150 Tsd. sind über die gesamt Kalifornien verteilt.

# In[20]:


df_above = train_dataset[train_dataset['price_category'] == 'above']
sns.scatterplot(data=df_above, x="longitude", y="latitude", size="median_house_value", legend=True, sizes=(20, 2000))


# * Die Distrikte  mit einem *median_house_value* von über 150 Tsd. scheinen sich auf bestimmte Flächen zu konzentrieren. 
# * Aus diesem Grund kann es sinnvoll sein *longitude*/ *latitude* mit in der Erstellung der Modelle zu berücksichtigen. Beim Feauture Engineering sollte dafür ein geeigneter Weg gefunden werden. 

# ### Numerischen Variablen -Übersicht

# In[21]:


train_dataset.describe()


# In[22]:


sns.pairplot(train_dataset)


# Folgende erste Erkenntnisse können aus den `pairplot` geschlossen werden: 
# * *median_house_value* hat die deutlichste Beziehung mit *median_income*. Die Beziehung ist positiv. 
# * Zu den anderen numerischen Größen (in Bezug auf *median_house_value*) ist im `pairplot` kein eindeutiger Zusammenhang identifizierbar. 
# * Die Variablen *households*, *population*, *total_bedrooms* und *total_rooms* scheinen untereinander in Beziehung (positiv)zu stehen. 

# In[23]:


fig, axs = plt.subplots(1, 7, figsize=(30,6))
sns.histplot(data=train_dataset, x="population", ax=axs[0])
sns.histplot(data=train_dataset, x="households", ax=axs[1])
sns.histplot(data=train_dataset, x="median_income", ax=axs[2])
sns.histplot(data=train_dataset, x="housing_median_age", ax=axs[3])
sns.histplot(data=train_dataset, x="total_rooms", ax=axs[4])
sns.histplot(data=train_dataset, x="total_bedrooms", ax=axs[5])
sns.histplot(data=train_dataset, x="median_house_value", ax=axs[6])


# Folgende Erkenntnisse lassen sich aus dem Diagrammen ableiten: 
# 
# * Die Verteilungen von *population, households, total_rooms, total_bedrooms, median_income, median_house_value* sind links schief. 
# * Auffällig sind Ausschläge am Maximum-Punkt bestimmter Variablen: 
#   * Peak bei *median_house_value* 50000 
#   * Peak bei *house_median_age* 50 Jahre
#   * Peak bei *median_income* bei 15,0 
#   * Schlussfolgerung: Werte scheinen eine obere Grenze zu haben. Alle Distrikte oberhalb der jeweiligen Grenze scheinen zu einem Datenpunkt zusammengefasst sein. 
#   * Für die Anwendung eines linearen Modells kann es sinnvoll sein diese Werte zu entfernen. 

# In[24]:


#Mögliche Funktion
#id_income = train_dataset[train_dataset['median_income'] >= 15.0].index
#id_housing = train_dataset[train_dataset['housing_median_age'] >= 52.0].index
#train_dataset.drop(index=id_income, inplace =True)
#train_dataset.drop(index=id_housing, inplace =True)


# ### Numerische Variablen -Korrelation 

# In[25]:


#Korellation untersuchen 
train_dataset_corr = train_dataset.drop(columns=['longitude', 'latitude'])
corr_matrix = train_dataset_corr.corr()
corr_matrix


# In[26]:


# Simple heatmap
plt.rcParams['figure.figsize'] = [8, 8]
cmap = sns.diverging_palette(220, 10, as_cmap=True)
mask = np.zeros_like(corr_matrix, dtype=bool)
mask[np.triu_indices_from(mask)] = True

heatmap = sns.heatmap(corr_matrix, mask = mask, square = True, cmap= cmap,  annot=True)


# * Wie nach Auswertung des `pairplots` vermutet, bestätigt sich, dass *median_house_value* mit *median_income* am stärksten korreliert. 
# * Die Korrelation zu den anderen Variablen ist gering. 
# * Es besteht eine sehr hohe Korrelation zwischen *total_rooms, total_bedrooms*, *households* und *population*. Bei gemeinsamer Verwendung in einem Modell deutet dies auf Multikollinearität hin. 

# ### Numerische Variablen- Deskriptive Statistik

# In[27]:


round(train_dataset.describe(),2).transpose()


# Die deskriptive Statistik der numerischen Variablen erfolgt auf Grundlage der Lagemaße Median (median), Mittelwert (mean) und Modus (mode), sowie der Streuungsmaße Standardabweichung (std), Spannweite (range) und Interquartilabstand. 

# In[28]:


descriptivestats.describe(data = train_dataset[['median_house_value', 'median_income', 'housing_median_age', 'population', 'households']],stats = ["mean", "median", "mode", "std", "range", "min", "max", "percentiles","iqr"], categorical= False).transpose()


# * Die obere Grenze der Werte *housing_median_age, median_income* und *median_house_value* wird auch in der deskriptiven Statistik deutlich in dem der Modus gleichhoch ist wie der maximal Wert. 
# * Die weiteren ermittelten Größen werden in die EDA einbezogen. 

# ### Numerische Variabeln- EDA

# **EDA Median Income**

# In[29]:


sns.regplot(x=train_dataset["median_house_value"], y=train_dataset["median_income"], line_kws={"color":"r","alpha":0.7,"lw":5})


# * Die positive Beziehung wird durch das Streuungsdiagramm visuell verdeutlicht. 
# 

# In[30]:


fig, axs = plt.subplots(1, 3, figsize=(20,5))
sns.histplot(data=train_dataset, x="median_income", ax=axs[0])
sns.boxplot(data=train_dataset, x="median_income",ax=axs[1])
sns.violinplot(y=train_dataset["median_income"],ax=axs[2])


# * Im Boxplot wird deutlich, dass die Interquartilrange im Vergleich zur Spannweite gering ist. 
# * Da im Boxplot Ausreißer als oberhalb der Grenze von 1,5 des Interquartilsabstands definiert  werden, ist eine Vielzahl an Ausreißer zu erkennen. 
# * Aus diesem Grund wurde noch ein Violin-Plot zur Darstellung gewählt. 

# In[31]:


fig, axs = plt.subplots(1, 2, figsize=(15,5))
sns.boxplot( x=train_dataset["price_category"], y=train_dataset["median_income"], ax=axs[0])
sns.kdeplot(data=train_dataset, x="median_income", hue="price_category",ax=axs[1])


# * Sowohl in der Boxplot- wie auch in der Densitiy-Darstellung wird unterschiedliche Verteilung des *median_incomes* deutlich. 
# * Der Median liegt bei above-Distrikten bei einem mittleren Einkommen von 45 Tsd USD und bei below-Distrikten bei 26 Tsd USD. 

# > Das Feature *median_income* sollte in den statistischen Modell (Klassifikation und Regression) mit berücksichtigt werden.

# **EDA housing_age**

# In[32]:


sns.regplot(x=train_dataset["median_house_value"], y=train_dataset["housing_median_age"], line_kws={"color":"r","alpha":0.7,"lw":5})


# * Im Streudiagramm ist kein deutliche Beziehung zwischen *housing_median_age* und *median_house_value* erkennbar. 
# * Vielmehr scheint das mittlere Immobilienalter über die Gesamte Fläche zu streuen. 
# 

# In[33]:


fig, axs = plt.subplots(1, 3, figsize=(20,5))
sns.kdeplot(data=train_dataset, x="housing_median_age", shade=True, bw=0.25, ax=axs[0])
sns.boxplot(data=train_dataset, x="housing_median_age",ax=axs[1])
sns.violinplot(y=train_dataset["housing_median_age"],ax=axs[2])


# Um weitere Visualisierung der Beziehung zwischen housing_median_age und median_house_value darzustellen, wird aus Erstem eine kategoriale Variable erstellt. 

# In[34]:


train_dataset['housingage_cat'] = '00'
train_dataset['housingage_cat'][train_dataset['housing_median_age'] <18] = '25'
train_dataset['housingage_cat'][(train_dataset['housing_median_age'] >=18) & (train_dataset['housing_median_age'] <=29)] = '50'
train_dataset['housingage_cat'][(train_dataset['housing_median_age'] >29) & (train_dataset['housing_median_age'] <=37)] = '75'
train_dataset['housingage_cat'][(train_dataset['housing_median_age'] >37)] = '100'


# In[35]:


train_dataset


# In[36]:


sns.boxplot( x=train_dataset["housingage_cat"], y=train_dataset["median_house_value"])


# In[37]:


train_dataset.drop(columns= 'housingage_cat', inplace = True)


# Auch in dieser Darstellung ist kein eindeutige Beziehung erkennbar und die generierte Variable wird wieder entfernt. 

# In[38]:


fig, axs = plt.subplots(1, 2, figsize=(15,5))
sns.boxplot( x=train_dataset["price_category"], y=train_dataset["housing_median_age"], ax=axs[0])
sns.kdeplot(data=train_dataset, x="housing_median_age", hue="price_category",ax=axs[1])


# Auch bei Betrachtung der Beziehung zwischen *price_category* und *housing_median_age*, ist kein eindeutiger Zusammenhang erkennbar. 

# > Bei *housing_median_age* ist keine eindeutige Beziehung zu *median_house_value* oder *price_category* erkennbar.
# > Der positive Effekt auf das Modell wird als gering eingeschätzt. 

# ### Kategoriale Variablen- Deskriptive Statistik

# In[39]:


from statsmodels.stats import descriptivestats
descriptivestats.describe(data = train_dataset, categorical= True).ocean_proximity.dropna()


# In[40]:


train_dataset.ocean_proximity[train_dataset['ocean_proximity'] == 'ISLAND'].count()


# Die kategoriale Variable *ocean_proximity* hat 5 Ausprägung: "<1H OCEAN", "INLAND", "NEAR BAY", "NEAR OCEAN" und "ISLAND". 
# 
# Durch die Analyse der Verteilung wird deutlich, dass die Ausprägungen <1H OCEAN (44%)und Inland (32%) am häufigsten vorkommen. Distrikte "NEAR BAY" und "NEAR OCEAN" kommen seltenere vor. Sehr selten sind Distrikte auf einer Insel. In dem Trainingsdatensatz sind es nur 4 Distrikte.

# ### Kategoriale Variable- EDA

# Im folgenden wird die Variable "ocean_proximity" detailiert betrachtet. 

# In[41]:


sns.scatterplot(data = train_dataset, x="longitude", y="latitude",  hue = "ocean_proximity")


# Zum Nachvollziehen der kategorialen Variable *ocean_proximity* ist diese im Scatterplot durch `hue` dargestellt. 

# In[42]:


sns.histplot(data=train_dataset, x="ocean_proximity")


# * Die meisten Distrikte liegen weniger als eine Stunde vom Ozean weg.
# * Sehr wenige Distrikte befinden sich auf Insel.
# 

# In[43]:


from pySankey.sankey import sankey
sankey(train_dataset["price_category"], train_dataset["ocean_proximity"], aspect=20, fontsize=12)


# * Die meisten Distrikte, die zur Preiskategegorie "below" gehören, liegen im Inland. 
# * Die meisten Distrikte, die zur Preiskategorie "above" gehören, liegen weniger als eine Stunde vom Ozean entfernt. 
# * Dementsprechend kann es sinnvoll sein, die kategoriale Variable *ocean_proximity* bei der Klassifikation mit einzubeziehen. 

# In[44]:


sns.relplot(data = train_dataset, 
            x = "median_house_value",
            y= "median_income", 
            hue = "ocean_proximity")


# In[45]:


sns.histplot(data = train_dataset, x="median_house_value", hue="ocean_proximity")


# * Das Histogramm bestätigt, dass die Nähe zum Ozean Einfluss auf *median_house_value*"* hat. 
# * Für Distrikten im Inland ist die Verteilung im Vergleich zum Beispiel zu "<1h Ocean" linksverschoben. 
# * Da die Variable *ocean_proximity* 5 Ausprägungen hat, ist es sinnvoll sich die Verteilung in separaten Plots anzuschauen.

# In[46]:


sns.displot(train_dataset, x="median_house_value", col="ocean_proximity")


# * Die unterscheidliche Verteilung scheint die Vermutung zu Bestätigen. 
# 

# > Die Variable *ocean_proximity* sollte in dem statistischen  Modell (Klassifikation und Regression) mit berücksichtigt werden.

# ## Feature Engineering 

# ### Feature extraction 

# Wie in der EDA festgestellt, besitzen die Variablen *total_rooms, total_bedrooms, population* und *households* eine hohe Korrelation untereinander und eine niedrige Korrelation zum *median_house_value*. Für das Modell wird versucht aus der Kombination der Variablen neue Merkmale zu generieren, welche eine höhere Korrelation zu der zu bestimmenden Größe aufweisen.  

# In[47]:


train_dataset['households_population'] = train_dataset['households']/train_dataset['population']
train_dataset['total_rooms_households'] = train_dataset['total_rooms']/train_dataset['households']
train_dataset['total_rooms_total_bedrooms'] = train_dataset['total_rooms']/train_dataset['total_bedrooms']


# In[48]:


corr = train_dataset.corr()
corr["median_house_value"].sort_values(ascending=False)


# Die Variablen * households_population, total_rooms_households* und *total_rooms_total_bedrooms* wurden iterativ bestimmt. Grundlage für die Auswahl dieser Features ist die Korrelation zu *median_house_value*. Wie oben gezeigt weisen die konstruierten Features eine höhere Korrelation auf, als die ursprünglichen Variablen im Datensatz.

# In[49]:


fig, axs = plt.subplots(1, 3, figsize=(15,5))
sns.regplot(x=train_dataset["median_house_value"], y=train_dataset["households_population"], line_kws={"color":"r","alpha":0.7,"lw":5}, ax=axs[0])
sns.regplot(x=train_dataset["median_house_value"], y=train_dataset["total_rooms_households"], line_kws={"color":"r","alpha":0.7,"lw":5}, ax=axs[1])
sns.regplot(x=train_dataset["median_house_value"], y=train_dataset["total_rooms_total_bedrooms"], line_kws={"color":"r","alpha":0.7,"lw":5}, ax=axs[2])


# ### Feature creation

# Wie bei der Analyse der geographischen Daten festgestellt, kann es sinnvoll zu sein diese in dem statistischen Modell zu berücksichtigen. Es hat sich gezeigt, dass in bestimmten geographischen Bereichen verstärkt Distrikte mit hohem *median_house_value* liegen. Um Koordinatenangaben in Bereiche zu gliedern, können verschiedene Verfahren angewendet werden, z.B. Clustering. Ein anderer Ansatz ist die Verwendung von geohashes mit der Python-Bibliothek geohash. Im zu Grunde liegenden Konzept werden GPS-Daten in eine Kombination aus Buchstaben und Ziffern kodiert. Die Erde wird dabei in ein Schema aus Rechtecken unterteilt. Die Größe des Gitters wird über die Anzahl der Buchstaben bestimmt.[Quelle: [Wikpedia](https://en.wikipedia.org/wiki/Geohash)]
# 
# <img src="doc\geohash_grid.JPG" alt="drawing" width="400"/>

# In[50]:


import geohash as gh
train_dataset_geo = train_dataset[['latitude', 'longitude']]
train_dataset['geohash']=train_dataset_geo.apply(lambda x: gh.encode(x.latitude, x.longitude, precision=3), axis=1)
train_dataset['geohash'] = train_dataset['geohash'].astype("category")

train_dataset


# In[51]:


train_dataset['geohash'].describe(include=['category'])


# Die Größe der Gitter ist auf drei Stellen des geohashs auslegt. Dies entspricht einem Gitter von einer Länge und Höhe von 156km. Das neue Feature "geohash" hat eine Ausprägung von 32 Werten. Mit einer 4stelligen Auflösung wäre es 493 Werte. Damit würde das Risiko für overfitting erhöht und die Anzahl der Spalten bei der Bildung von Dummy-Variabln aufgebläht werden. 
# Da es sich um eine künstlich erzeugte kategoriale Variable handelt, besteht ebenfalls das Risiko, dass in den Testdaten Ausprägungen vorkommen mit welchen das Modell nicht trainiert wurde. 

# In[52]:


sns.scatterplot(data=train_dataset, x="longitude", y="latitude", hue="geohash", legend= False)


# > Weiter Feature Selection findet modellspezifisch im nächsten Schritt statt. 
