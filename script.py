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
