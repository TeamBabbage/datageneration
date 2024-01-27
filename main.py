import pandas as pd
# {gender, applicantType, minBedSize, olderPersonAssessment, mobilityAssessemnt} -> {timeToHouse}

# applicantType in {A, B, C} |-> {1,2,3}
# olderPersonAssessment in {Urgent, General, Homeless} |-> {1,2,3}
# mobilityAssessemnt in {Urgent, General, Homeless, Homeless / Priority} |-> {1,2,3,4}

genderToInteger = {'F':1, 'M':2}
applicantTypeToInteger = {'A': 1, 'B': 2, 'C': 3}
olderPersonAssessmentToInteger = {'Urgent': 1, 'General': 2, 'Homeless': 3, 'Reserve': 4}
mobilityAssessemntToInteger = {'Urgent': 1, 'General': 2, 'Homeless': 3, 'Homeless / Priority': 4, 'Reserve': 5}

p_male = 139304 / 279554
p_female = 1 - p_male
# Load the dataset
data = pd.read_csv('sample.csv')

# Split the data into features and label
features = data.drop('weeksToHouse', axis=1).values
labels = data['weeksToHouse'].values

features = list(map(lambda x : [
    genderToInteger[x[0]],
    x[1], 
    applicantTypeToInteger[x[2]],
    olderPersonAssessmentToInteger[x[3]],
    mobilityAssessemntToInteger[x[4]]
], features))

# def fitLinear(features, labels):
    
#     for feature in features:


