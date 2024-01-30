import pandas as pd
import numpy as np
data = pd.read_csv('age_stats.csv')

ageRanges = data['age'].values
pop = data['pop'].values

ageRangeToPop = [i for i in map(lambda x,y: [x,y], ageRanges, pop) if int(i[0].split('-')[0]) >= 18]

totalPop = sum([i[1] for i in ageRangeToPop])
ageRangeToPop = [[i[0], i[1] / totalPop] for i in ageRangeToPop]
age_ranges = [i[0] for i in ageRangeToPop]

probabilities = [i[1] for i in ageRangeToPop]
# def getAgeRange(x):

# return an age range from the pmf.
def getAgeRange():
    return np.random.choice(age_ranges, p=probabilities)

