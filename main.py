import pandas as pd
from random import randint, uniform

genderToInteger = {'F':1, 'M':2}
applicantTypeToInteger = {'A': 1, 'B': 2, 'C': 3}
olderPersonAssessmentToInteger = {'Urgent': 1, 'General': 2, 'Homeless': 3, 'Reserve': 4}
mobilityAssessemntToInteger = {'Urgent': 1, 'General': 2, 'Homeless': 3, 'Homeless / Priority': 4, 'Reserve': 5}

# Load the dataset
data = pd.read_csv('sample.csv')

# Split the data into features and label
features = data.drop('weeksToHouse', axis=1).values
labels = data['weeksToHouse'].values

features = list(map(lambda x : [
    1,
    genderToInteger[x[0]],
    x[1], 
    applicantTypeToInteger[x[2]],
    olderPersonAssessmentToInteger[x[3]],
    mobilityAssessemntToInteger[x[4]]
], features))

def fitLinear(features, labels):
    alpha = 0.001    
    w = [0] * len(features[0])
    y = lambda x : sum([w[i] * x[i] for i in range(len(x))])
    count = 0
    while True:
        for i in range(len(features)):
            change = 0
            for j in range(len(features[i])):
                change += abs(alpha * (labels[i] - y(features[i])) * features[i][j])
                w[j] = w[j] + alpha * (labels[i] - y(features[i])) * features[i][j]        
            count += 1
        if count > 100000:
            break
    return w

# def score(w, features, labels):
#     error = 0 
#     y = lambda x : sum([w[i] * x[i] for i in range(len(x))])
#     for i in range(len(features)):
#         error += abs(labels[i] - y(features[i]))
#     return error 

# w = fitLinear(features, labels)
# # print(score(w, features, labels))

# reg = LinearRegression().fit(features, labels)
# print(score(reg.coef_, features, labels))

def generateData(n):
    w = fitLinear(features, labels)
    y = lambda x : sum([w[i] * x[i] for i in range(len(x))])

    bounds = [[1,2], [1,3], [1,6], [1,4], [1,5]]
    X = []
    Y = []
    for i in range(n):
        instance = []
        for l,r in bounds:
            instance.append(randint(l,r))
        X.append(instance)
        Y.append(y(instance) * uniform(0.9, 1.1))
    return X,Y

X,Y = generateData(1)
print(X,Y)
