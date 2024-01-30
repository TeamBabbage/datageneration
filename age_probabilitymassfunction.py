import pandas as pd

data = pd.read_csv('age_stats.csv')

ageRanges = data['age'].values
pop = data['pop'].values

ageRangeToPop = [i for i in map(lambda x,y: [x,y], ageRanges, pop) if int(i[0].split('-')[0]) >= 18]