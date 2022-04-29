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