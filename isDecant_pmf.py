104,510

import numpy as np

# probabily of being decant = 104510/ 28.2 10^6 according to https://www.gov.uk/government/statistics/statutory-homelessness-in-england-january-to-march-2023/statutory-homelessness-in-england-january-to-march-2023

def getRandomIsDecant():
    return np.random.choice(['N', 'Y'], p=[0.996, 0.004])