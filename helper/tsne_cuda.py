import numpy as np
from sklearn.manifold import TSNE
# from tsnecuda import TSNE
import os, shutil
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import torch

# import matplotlib as mpl
# mpl.rcParams['figure.dpi'] = 1000
sns.set(style="darkgrid")

save_dir = 'tsne_plots_cuda'
# store = np.load('saved_features/amazon_features.npy', allow_pickle=True)

### [[feat_1,label_1], [feat_2,label_2] ... [feat_n, label_n]] ---> Everything except label is np array


# shutil.rmtree(save_dir)
# os.mkdir(save_dir)


use_model = 'RC'
use_arch = 'deit_s'
load_stored_pt = torch.load(f'save_feats/C_{use_model}_{use_arch}.pth')
X = load_stored_pt['features']
label = load_stored_pt['labels']


# X = np.array(store[:,0].tolist())
# label = store[:,1] 

data = {}


lr = [5,10,18,20,25,30]
for i in range(5,25,3):
    f = plt.figure(figsize=(50,50))
    for count,j in enumerate(lr):
        print(f'TSNE Running for perplexity={i} and lr={j}')
        X_embedded = TSNE().fit_transform(X)
        x,y = X_embedded[:,0],X_embedded[:,1]
        data['x'], data['y'], data['label'] = x,y, label
        df=pd.DataFrame(data)
        ax = f.add_subplot(round(len(lr)**(0.5))+1,round(len(lr)**(0.5))+1,count+1)
        sns.scatterplot(data=df, x="x", y="y", hue="label", legend=False,s=50)
        # break
    plt.savefig(f'{save_dir}/tsne_per_{i}_{use_arch}.png')
    plt.clf()
    print('Saved! ' + f'{save_dir}/tsne_per_{i}_{use_arch}.png')
    # break
    # break
print('Completed all')
