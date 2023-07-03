import numpy as np
import pandas as pd
import os

import matplotlib.pyplot as plt
import seaborn as sns

MTAT_TAGS = [ "guitar", "classical", "slow", "techno", "strings", 
    "drums", "electronic", "rock", "fast", "piano",
    "ambient", "beat", "violin", "vocal", "synth", 
    "female", "indian", "opera", "male", "singing", 
    "vocals", "no vocals", "harpsichord", "loud", "quiet", 
    "flute", "woman", "male vocal", "no vocal", "pop", 
    "soft", "sitar", "solo", "man", "classic", 
    "choir", "voice", "new age", "dance", "male voice",
    "female vocal", "beats", "harp", "cello", "no voice", 
    "weird", "country", "metal", "female voice", "choral" ]

DCASE_TAGS = [
    "Train horn", "Air horn, truck horn", "Car alarm", 
    "Reversing beeps", "Ambulance (siren)", "Police car (siren)", 
    "Fire engine, fire truck (siren)", "Civil defense siren", "Screaming",
    "Bicycle", "Skateboard", "Car", 
    "Car passing by", "Bus", "Truck", 
    "Motorcycle", "Train" ]

KEYWORD_TAGS = [ "right", "eight", "cat", "tree", "backward",
    "learn", "bed", "happy", "go", "dog", 
    "no", "wow", "follow", "nine", "left", 
    "stop", "three", "sheila", "one", "bird", 
    "zero", "seven", "up", "visual", "marvin", 
    "two", "house", "down", "six", "yes", 
    "on", "five", "forward", "off", "four" ]

DATAPATH = "/home/jaehwlee/DATASET/JETATAG/"

binary = np.load(os.path.join(DATAPATH, "mtat", "binary.npy"))
print(binary.shape)

df = pd.DataFrame(binary, columns = MTAT_TAGS)

corr_df = df.corr()
corr_df = corr_df.apply(lambda x: round(x, 2))

for tag_type in ['classical', 'country', 'rock']:
    corr_type = corr_df.nlargest(10, tag_type)
    corr_type = corr_type[list(corr_type.index)] 
    plt.figure(figsize = (10,10))
    ax = sns.heatmap(corr_type, annot=True, annot_kws=dict(color='g'), cmap='Greys')
    plt.savefig('mtat_' + tag_type + '_output.png')
