import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from warnings import simplefilter
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier

simplefilter(action='ignore', category=FutureWarning)

bank = pd.read_csv("./DataSets/Bank Marketing/bank-full.csv")
bank.job = bank.job.replace(['management', 'technician', 'entrepreneur', 'blue-collar',
       'unknown', 'retired', 'admin.', 'services', 'self-employed',
       'unemployed', 'housemaid', 'student'], [0,1,2,3,4,5,6,7,8,9,10,11])
bank.marital = bank.marital.replace(['married', 'single', 'divorced'], [0,1,2])
bank.education = bank.education.replace(['tertiary', 'secondary', 'unknown', 'primary'], [0,1,2,3])
bank.default = bank.default.replace(["no","yes"],[0,1])
bank.housing = bank.housing.replace(["no","yes"],[0,1])
bank.loan = bank.loan.replace(["no","yes"],[0,1])
bank.contact = bank.contact.replace(['unknown', 'cellular', 'telephone'],[0,1,2])
bank.month = bank.month.replace(['may', 'jun', 'jul', 'aug', 'oct', 'nov', 'dec', 'jan', 'feb',
       'mar', 'apr', 'sep'], [0,1,2,3,4,5,6,7,8,9,10,11])
bank.poutcome = bank.poutcome.replace(['unknown', 'failure', 'other', 'success'],[0,1,2,3])
bank.y = bank.y.replace(["no","yes"],[0,1])
rangos = [0, 8, 15, 18, 25, 40, 60, 100]
nombres = ['1', '2', '3', '4', '5', '6', '7']
bank.age = pd.cut(bank.age, rangos, labels=nombres)
bank.dropna(axis=0,how='any', inplace=True)

data_train = bank[:36168]
data_test = bank[36168:]

x = np.array(data_train.drop(['y'], 1))
y = np.array(data_train.y)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

x_test_out = np.array(data_test.drop(['y'], 1))
y_test_out = np.array(data_test.y)

def metricas(modelo, nombre):
    print('*'*50)
    print(f'Modelo {nombre}')
    
    modelo.fit(x_train, y_train)
    
    # Accuracy de Entrenamiento de Entrenamiento
    print(f'accuracy de Entrenamiento de Entrenamiento: {modelo.score(x_train, y_train)}')

    # Accuracy de Test de Entrenamiento
    print(f'accuracy de Test de Entrenamiento: {modelo.score(x_test, y_test)}')

    # Accuracy de Validación
    print(f'accuracy de Validación: {modelo.score(x_test_out, y_test_out)}')

random = RandomForestClassifier()
logreg = LogisticRegression(solver='lbfgs', max_iter = 7600)
arbol = DecisionTreeClassifier()
red = MLPClassifier()
probabilistico = GaussianNB()

print("Dataset bank")
metricas(red, "Multilayer perceptron")
metricas(logreg, "Regresión logística")
metricas(arbol, "Arbol de decisión")
metricas(red, "Naive bayes")
metricas(random, "Random Forest")

diabetes = pd.read_csv("./DataSets/Prima Indians Diabetes Database/diabetes.csv")
rangos = [0, 8, 15, 18, 25, 40, 60, 100]
nombres = ['1', '2', '3', '4', '5', '6', '7']
diabetes.Age = pd.cut(diabetes.Age, rangos, labels=nombres)
diabetes.dropna(axis=0,how='any', inplace=True)

data_train = diabetes[:614]
data_test = diabetes[614:]

x = np.array(data_train.drop(['Outcome'], 1))
y = np.array(data_train.Outcome)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

x_test_out = np.array(data_test.drop(['Outcome'], 1))
y_test_out = np.array(data_test.Outcome)

random = RandomForestClassifier()
logreg = LogisticRegression(solver='lbfgs', max_iter = 7600)
arbol = DecisionTreeClassifier()
red = MLPClassifier()
probabilistico = GaussianNB()

print("Dataset diabetes")
metricas(red, "Multilayer perceptron")
metricas(logreg, "Regresión logística")
metricas(arbol, "Arbol de decisión")
metricas(red, "Naive bayes")
metricas(random, "Random Forest")

weather = pd.read_csv("./DataSets/weatherAUS/weatherAUS.csv")
weather = weather.drop(['Date'], 1)
weather.Location = weather.Location.replace(['Albury', 'BadgerysCreek', 'Cobar', 'CoffsHarbour', 'Moree', 'Newcastle', 'NorahHead', 'NorfolkIsland', 'Penrith', 'Richmond', 'Sydney', 'SydneyAirport', 'WaggaWagga', 'Williamtown', 'Wollongong', 'Canberra', 'Tuggeranong', 'MountGinini', 'Ballarat', 'Bendigo', 'Sale', 'MelbourneAirport', 'Melbourne', 'Mildura', 'Nhil', 'Portland', 'Watsonia', 'Dartmoor', 'Brisbane', 'Cairns', 'GoldCoast', 'Townsville', 'Adelaide', 'MountGambier', 'Nuriootpa', 'Woomera', 'Albany', 'Witchcliffe', 'PearceRAAF', 'PerthAirport', 'Perth', 'SalmonGums', 'Walpole', 'Hobart', 'Launceston', 'AliceSprings', 'Darwin', 'Katherine', 'Uluru'] , [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48])
weather.WindGustDir = weather.WindGustDir.replace(['SSW', 'S', 'NNE', 'WNW', 'N', 'SE', 'ENE', 'NE', 'E', 'SW', 'W', 'WSW', 'NNW', 'ESE', 'SSE', 'NW'], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15])
weather.WindDir9am = weather.WindDir9am.replace(['ENE', 'SSE', 'NNE', 'WNW', 'NW', 'N', 'S', 'SE', 'NE', 'W', 'SSW', 'E', 'NNW', 'ESE', 'WSW', 'SW'],[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15])
weather.WindDir3pm = weather.WindDir3pm.replace(['SW', 'SSE', 'NNW', 'WSW', 'WNW', 'S', 'ENE', 'N', 'SE', 'NNE', 'NW', 'E', 'ESE', 'NE', 'SSW', 'W'],[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15])
weather.RainToday = weather.RainToday.replace(['No', 'Yes'],[0, 1])
weather.RainTomorrow = weather.RainTomorrow.replace(['No', 'Yes'],[0, 1])
weather.dropna(axis=0,how='any', inplace=True)

data_train = weather[:45136]
data_test = weather[45136:]

x = np.array(data_train.drop(['RainTomorrow'], 1))
y = np.array(data_train.RainTomorrow)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

x_test_out = np.array(data_test.drop(['RainTomorrow'], 1))
y_test_out = np.array(data_test.RainTomorrow)

random = RandomForestClassifier()
logreg = LogisticRegression(solver='lbfgs', max_iter = 7600)
arbol = DecisionTreeClassifier()
red = MLPClassifier()
probabilistico = GaussianNB()

print("Dataset weather")
metricas(red, "Multilayer perceptron")
metricas(logreg, "Regresión logística")
metricas(arbol, "Arbol de decisión")
metricas(red, "Naive bayes")
metricas(random, "Random Forest")
