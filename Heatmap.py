import pandas as pd
import matplotlib.pyplot as plt
#matplotlib inline
import seaborn as sns
df=pd.read_csv("IBD5.csv")

##特征两两相关性分析
df.corr()
plt.figure(figsize=(30,30))
# sns.heatmap(df.corr(),annot=True,fmt=".lf",square=True)
#annot=True:把数字写在图标上，fmt=".1f：保留一位小数，square=True：图是方形的
sns.heatmap(df.corr(),annot=False,fmt=".1f",square=True)
plt.show()


