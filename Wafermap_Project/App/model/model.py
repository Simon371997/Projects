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
print(df.head())

a = df['waferMap'][0]
print(a, a.shape)