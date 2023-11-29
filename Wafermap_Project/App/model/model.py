# Step 1: Imports
    # Data Handling
import pandas as pd
import numpy as np

    # Data Visualization
import matplotlib.pyplot as plt
from matplotlib import gridspec

    # Machine/Deep Learning
from sklearn.model_selection import train_test_split

# Step 2: Read Data
    # Data was seperated in a sample file (1% of orig data) to explore it without too long loading
df = pd.read_pickle('./data/sample.pkl')


# Step 3: Prepare / Manipulate Data
df = df.drop(['waferIndex'], axis = 1)

    # Function to identify the size of the Wafermaps
def find_dim(column):
    dim0 = np.size(column, axis=0)
    dim1 = np.size(column, axis=1)
    return (dim0, dim1)

df['waferMapDim'] = df.waferMap.apply(find_dim)

df['failureNum'] = df.failureType
df['trainTestNum'] = df.trianTestLabel
mapping_type = {'Center':0,'Donut':1,'Edge-Loc':2,'Edge-Ring':3,'Loc':4,'Random':5,'Scratch':6,'Near-full':7,'none':8}
mapping_traintest = {'Training':0, 'Test':1}
df = df.replace({'failureNum':mapping_type, 'trainTestNum':mapping_traintest})

df_withlabel = df[(df['failureNum']>=0) & (df['failureNum']<=8)]
df_withlabel = df_withlabel.reset_index()

df_withpattern = df[(df['failureNum']>=0) & (df['failureNum']<=7)]
df_withpattern = df_withpattern.reset_index()

df_nonpattern = df[df['failureNum']== 8]
df_nonpattern = df_nonpattern.reset_index()

def plot_failureType():
    fig = plt.figure(figsize=(30, 5))
    gs = gridspec.GridSpec(1, 2, width_ratios=[1, 2.5])
    ax1 = plt.subplot(gs[0])
    ax2 = plt.subplot(gs[1])

    no_wafers = [df.shape[0] - df_withlabel.shape[0], df_withpattern.shape[0], df_nonpattern.shape[0]]

    colors = ['silver', 'orange', 'gold']
    explode = (0.1, 0, 0)
    labels = ['no-label', 'label&pattern', 'label&non-pattern']
    plt.setp(ax1.pie(no_wafers, labels=labels, explode=explode, colors=colors, autopct='%1.1f%%', shadow=True, startangle=20)[1], size=8)

    uni_pattern=np.unique(df_withpattern.failureNum, return_counts=True)
    labels2 = ['','Center','Donut','Edge-Loc','Edge-Ring','Loc','Random','Scratch','Near-full']
    ax2.bar(uni_pattern[0],uni_pattern[1]/df_withpattern.shape[0], color='gold', align='center', alpha=0.9)
    ax2.set_title("failure type frequency")
    ax2.set_ylabel("% of pattern wafers")
    ax2.set_xticklabels(labels2)
    plt.show()


def plot_waferMaps():
    fig, ax = plt.subplots(nrows=5, ncols=5, figsize = (8,8))
    ax = ax.ravel(order='c')
    for i in range(25):
        img = df_withpattern.waferMap[i]
        ax[i].imshow(img)
        ax[i].set_title(df_withpattern.failureType[i][0][0], fontsize=10)
        ax[i].set_xlabel(df_withpattern.index[i], fontsize=8)
        ax[i].set_xticks([])
        ax[i].set_yticks([])
    plt.tight_layout()
    plt.show()


df = df[df['waferMapDim'] == (26, 26)] #7084 Stück
df = df[(df['failureNum']>=0) & (df['failureNum']<=8)] # 6267 Stück
new_df = df.drop(['dieSize', 'lotName', 'trianTestLabel', 'failureType', 'waferMapDim', 'trainTestNum'], axis=1)

print(new_df['waferMap'].values.shape)