import matplotlib.pyplot as plt 
import seaborn as sns
import pandas as pd
from matplotlib.pyplot import figure
import numpy as np
sns.set_style("darkgrid")

def a_dist(acc):
    "Given the accuracy get the A-Distance"
    return abs(2*(0.02*acc - 1))

df = pd.read_csv('a_dist.csv')
df['A-Distance'] = df['accuracy'].apply(a_dist)
# df = df.drop([12,13, 18, 19])
print(df)

figure(figsize=(7,4))
y1 = 1.0
y2 = 1.8

plt.ylim(y1, y2)
plt.yticks(np.arange(y1, y2, 0.1))

sns.barplot(x="Adaptation", y="A-Distance", hue="model", data=df)
plt.legend(bbox_to_anchor=(0.5, 1.05), loc='upper center', borderaxespad=0., ncol=2)

plt.savefig('a_dist.pdf')