import pandas as pd
from random import randint, uniform
from age_pmf import getRandomAge
from gender_pmf import getRandomGender
from medicalrequirments_pmf import getRandomMedical
from sklearn.linear_model import LinearRegression

genderToInteger = {'F':1, 'M':2}
applicantTypeToInteger = {'A': 1, 'B': 2, 'C': 3}
olderPersonAssessmentToInteger = {'Urgent': 1, 'General': 2, 'Homeless': 3, 'Reserve': 4}
mobilityAssessemntToInteger = {'Urgent': 1, 'General': 2, 'Homeless': 3, 'Homeless / Priority': 4, 'Reserve': 5}

medicalRequirmentsToInteger = {'N': 1, 'Y': 2}

# Load the dataset
data = pd.read_csv('sample.csv')

# Split the data into features and label
# features = data.drop('weeksToHouse', axis=1).values
# labels = data['weeksToHouse'].values

def fillMissingFeatures(f):
    features = list(map(lambda x : [
        genderToInteger[x[0]], #gender
        getRandomAge(), #age
        x[1], # min beds
        x[2], # maxbeds
        medicalRequirmentsToInteger[getRandomMedical()] # medical requirments none / yes
    ], f))
    
    return features

def generateData(n):
    features = data.drop('weeksToHouse', axis=1).values
    labels = data['weeksToHouse'].values
    features = fillMissingFeatures(features)

    
    reg = LinearRegression().fit(features, labels)
    print(reg.score(features, labels))
    X = []
    Y = []

    for i in range(n):
        instance = [
            genderToInteger[getRandomGender()],
            getRandomAge(),
            randint(1,5),
            randint(1,5), 
            medicalRequirmentsToInteger[getRandomMedical()]
        ]
        X.append(instance)
        Y.append(reg.predict([instance])[0])
    return X,Y


X,Y = generateData(100000)
    
# generateData(1)

data = pd.DataFrame(X)
data['weeksToHouse'] = Y
data.to_csv('generated.csv', index=False)
