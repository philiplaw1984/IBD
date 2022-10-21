import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn import svm, linear_model
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import precision_score, roc_auc_score, average_precision_score
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from scipy import interp
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import confusion_matrix

from sklearn.metrics import classification_report
from xgboost import XGBClassifier
from sklearn.model_selection import learning_curve

data = pd.read_csv('./IBD.csv')
# data = pd.read_csv(r'C:\IBD\example.csv') # 数据存放的文件路径
np.set_printoptions(suppress=True)
y = data["case"].values  # y值“case”表示结局事件，要放在最后一列，根据数据实际情况修改
X = data.iloc[:, 0:32].values # 影响因素：第一列为0，41表示一共有多少影响因素，需要根据实际情况修改。IBD1(49);IBD2(31)。

skf=StratifiedKFold(n_splits=5, random_state=None, shuffle=False) # 做多少折交叉验证，这里是5折交叉。
finalAuc=0

# 调用模型，但是并未经过任何调参操作，使用默认值 一共八个模型，自己逐一调用。
# lr_model = XGBClassifier()                                 #1
# lr_model = LogisticRegression()                            #2
# lr_model = tree.DecisionTreeClassifier()                   #3
# lr_model = KNeighborsClassifier()                          #4
# lr_model = GaussianNB()                                    #5
# lr_model = MultinomialNB()                                 #6
lr_model = MLPClassifier()                                 #7
# lr_model = RandomForestClassifier()                        #8


tprs = []
aucs = []
mean_fpr = np.linspace(0, 1, 100)

i = 0
accuracy=[]
precision=0
recall=0
tpr2=[]
tnr=[]
ppv=[]
npv=[]
avePrecision=[]

list=[]
id = np.array(list)
id2 = np.array(list)
probality = np.array(list)

for train, test in skf.split(X, y):
    id = np.hstack((id, test))
    id2 = np.hstack((id2, train))

    probas_ = lr_model.fit(X[train], y[train]).predict_proba(X[test])
    fpr, tpr, thresholds = roc_curve(y[test], probas_[:, 1])
    probality = np.hstack((probality, probas_[:, 1]))

    y_preds = lr_model.fit(X[train], y[train]).predict(X[test])

    accuracy.append(accuracy_score(y[test], y_preds))
    precision = precision + precision_score(y[test], y_preds)

    # recall = recall + recall_score(y[test], y_preds)
    confusionMatrix = confusion_matrix(y[test], y_preds)
    print(confusionMatrix)

    tp = confusionMatrix[1][1]
    fp = confusionMatrix[0][1]
    fn = confusionMatrix[1][0]
    tn = confusionMatrix[0][0]

    tpr2.append(tp/(tp+fn))
    tnr.append(tn/(tn+fp))
    ppv.append(tp/(tp+fp))
    npv.append(tn/(tn+fn))

    avePrecision.append(average_precision_score(y[test], y_preds))


    tprs.append(interp(mean_fpr, fpr, tpr))
    tprs[-1][0] = 0.0
    roc_auc = auc(fpr, tpr)
    aucs.append(roc_auc)
    plt.plot(fpr, tpr, lw=1, alpha=0.3,
             label='ROC fold %d (AUC = %0.3f)' % (i+1, roc_auc))

    i += 1
plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
         label='Chance', alpha=.8)

mean_tpr = np.mean(tprs, axis=0)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
std_auc = np.std(aucs)
plt.plot(mean_fpr, mean_tpr, color='b',
         label=r'Mean ROC (AUC = %0.3f $\pm$ %0.3f)' % (mean_auc, std_auc),
         lw=2, alpha=.8)

std_tpr = np.std(tprs, axis=0)
tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                 label=r'$\pm$ 1 std. dev.')

print('准确率accuracy：', accuracy)
print('Average precision：', avePrecision)
print("敏感度TPR:", tpr2)
print("特异度TNR:", tnr)
print("PPV:", ppv)
print("NPV:", npv)

plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()


