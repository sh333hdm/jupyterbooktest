# Plan

## Problembeschreibung 
Wir untersuchen den mittleren Immobilienwerte von Distrikten in Kaliforniern. Ziel ist es die Preise möglichst präzise vorauszusagen in welchen Distrikten sich ein Kauf lohnt. Ziel ist es dadurch ein Expertenteam eines Immobilienbüros zu ersetzen.  Um bewerten zu können, ob das Expertenteam ersetzt werden kann, muss ein Regressions-Modell den Immobilienpreis mit einer RMSE von 70.000USD vorhersagen können. Zusätzlich sollte ein Klassifikationsmodell mit einem F1-Score > 0.8 vorhersagen können, ob der mittlere Immobilienwert in einem Distrikt über 150 Tsd. USD liegt. 
## Identifikation der Variablen 
Im ersten Schritt werden dafür die benötigten und möglichen Variablen für die Modell festgelegt. Grundlage dafür ist ein Paper mit dem Titel „Modeling House Price Prediction using Regression Analysis and Particle Swarm Optimization“ erschienen im “International Journal of Advanced Computer Science and Applications, Vol. 8, No. 10, 2017”. In der Veröffentlichung werden die Einflussfaktoren in 3 Kategorien unterteilt: „Location“, „Physical condition“ und „Concept“. Dien „Location“ bezieht sich auf die geographische Lage eines Objekts, kann aber auch Eigenschaften, wie z.B. die Nähe zu Natur einschließen.  Unter „Physical condition“ fassen die Autoren Merkmale, wie z.B. die Anzahl von Schlafzimmer, die Verfügbarkeit eines Gartens und das Alter der Immobilie zusammen. „Concept“ bezieht sich auf die Idee, welchen den Käufer mit einer Immobilie vermittelt wird z.B. durch ein elitäres Umfeld. 
Auf Basis der Recherche besitzt der Datensatz folgende Features: 
- **„Physical condition“**: housing_median_age, total_rooms, total_bedrooms
- **„Concepts“**: population, households, median_income
- **“Location”**: ocean_proximity, longitude, latitude
## Festlegung von Metriken
Das Projekt ist ein Erfolg, wenn der mittlere Immobilienpreise eines Distrikts mit einem RMSE von 70 Tsd. USD oder mit einem F1-Score über 0.8 vorhergesagt werden kann, ob der mittlere Immobilienpreis über 150 Tsd. USD liegt. 
Das Projekt ist fehlgeschlagen, wenn der RMSE über 85 Tsd. USD liegt oder der F1-Score unter 0.75. 

![Image](data/map.JPG)