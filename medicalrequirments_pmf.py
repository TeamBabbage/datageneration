# https://drive.google.com/file/d/1_KsPGfaHANgewA3YMsmUEyU3HhYzHatX/view
# according to link above 14.6% of the population in hackney is disabled or has long term illness.
import numpy as np
def getRandomMedical():
    return np.random.choice(['N', 'Y'], p=[0.854, 0.146])