# Step 1: Imports

    # general imports
import warnings
warnings.filterwarnings("ignore")

    # Data Handling/Transformation
import pandas as pd
import numpy as np


    # Data Visualization
import matplotlib.pyplot as plt
from matplotlib import gridspec

    # Machine/Deep Learning



# Step 2: Read Data
    # Data was seperated in a sample file (1% of orig data) to explore it without too long loading
df = pd.read_pickle('./data/sample.pkl')


# Step 3: Prepare / Manipulate Data
df = df.drop(['waferIndex'], axis = 1)

def find_dim(x):
    dim0=np.size(x,axis=0)
    dim1=np.size(x,axis=1)
    return dim0,dim1
df['waferMapDim']=df.waferMap.apply(find_dim)

df['failureNum']=df.failureType
mapping_type={'Center':0,'Donut':1,'Edge-Loc':2,'Edge-Ring':3,'Loc':4,'Random':5,'Scratch':6,'Near-full':7,'none':8}

df=df.replace({'failureNum':mapping_type})
new_df = df[df['failureNum'] <= 7]
new_df = new_df[new_df['waferMapDim'] == (32,32)]


X = np.array([np.ravel(x) for x in new_df['waferMap']])
print(X)
print(X.shape)