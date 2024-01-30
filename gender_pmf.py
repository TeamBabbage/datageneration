import pandas as pd
import numpy as np

data = pd.read_csv('gender_stats.csv')

gender = data['gender'].values
count = data['pop'].values

totalCount = sum(count)

probabilities = [i / totalCount for i in count]    

def getRandomGender():
    return np.random.choice(gender, p = probabilities)

