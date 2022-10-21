import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from xgboost import XGBClassifier
from xgboost import plot_importance
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from matplotlib import pyplot
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
import shap
# import lime                # 这里我把lime隐去了，因为lime要在vscode上用矩池云运算
# import lime.lime_tabular



data = pd.read_csv('./IBD.csv')
# data = pd.read_csv(r'C:\IBD\example.csv')  # 数据存放的文件路径
np.set_printoptions(suppress=True)
y = data["case"].values                 # y值“case”表示结局事件，要放在最后一列，根据数据实际情况修改
X = data.iloc[:, 0:53].values           # 影响因素：第一列为0，49表示一共有多少影响因素，需要根据实际情况修改。IBD1(49);IBD2(31)。

skf = StratifiedKFold(n_splits=5, random_state=None, shuffle=False)   # 做多少折交叉验证，这里是5折交叉。

# 调用模型，但是并未经过任何调参操作，使用默认值
lr_model = XGBClassifier()                                 #1
# lr_model = LogisticRegression()                            #2
# lr_model = tree.DecisionTreeClassifier()                   #3
# lr_model = KNeighborsClassifier()                          #4
# lr_model = GaussianNB()                                    #5
# lr_model = MultinomialNB()                                 #6
# lr_model = MLPClassifier()                                 #7
# lr_model = RandomForestClassifier()                        #8

# 这里是影响因素的罗列，自己根据需求更改。
col = ['age1', 'age2', 'sex', 'abdominal pain', 'diarrhea', 'hematochezia', 'arthralgia',
       'constipation', 'fatigue', 'fever', 'course1',
       'course2', 'OB', 'ALT', 'AST', 'BUN', 'Cr', 'UA', 'PLT', 'RBC', 'WBC',
       'Hb', 'HCT', 'ESR', 'CRP', 'HsCRP', 'ALB', 'Na', 'K', 'Tspot', 'ANCA',
       'ANA', 'colonoscopy0', 'colonoscopy1', 'colonoscopy2', 'colonoscopy3', 'colonoscopy4',
       'colonoscopy5', 'colonoscopy6', 'endoscopic performance1', 'endoscopic performance2',
       'endoscopic performance3', 'endoscopic performance4', 'endoscopic performance5',
       'pathology1', 'pathology2', 'pathology3', 'pathology4', 'pathology5', 'pathology6', 'tuberculosis',
       'Intestinal complications', 'hematochezia']


# 下面是shap的解释器
importance = np.zeros(shape=(231, 53))   #一共多少个病例（行），多少列影响因素
for train, test in skf.split(X, y):
    lr_model.fit(X[train], y[train])

    explainer = shap.TreeExplainer(lr_model)
    # explainer = shap.KernelExplainer(lr_model.predict_proba, X[train])
    shap_values = explainer.shap_values(X)
    # shap.summary_plot(shap_values, df, plot_type="bar")

    # shapV = shap_values[0]
    # shapV = shap_values[1]
    for i in range(0, 231):     # 病例（行）
        for j in range(0, 53):   # 影响因素（列）
            importance[i][j] += shap_values[i][j]
            # importance[i][j] += shapV[i][j]

    # LIME ed by philip 09/03/23
    # visual_ind = 5  # TP fold1 5 21 22 26 29 32 34 35 39 50 51 64 70 71 72 93 103
    # visual_ind = 0  # TN 我们看第一个病人，可以看任意病人
    #FN fold1 13 20 40 63 112 115 118
    # visual_ind = 40  # FN fold2 35 38 39 69 82 92 95 110 117
    # visual_ind = 2  # FP

    test_pred = lr_model.predict(X[test])
    print(test_pred)
    print(y[test])
    print(test)

    a = y[test]
    for i in range(0, len(test_pred)):
        if(test_pred[i]==1 and a[i]==0):
            print(i)

    errors = test_pred - y[test]
    sorted_errors = np.argsort(abs(errors))

# 解释器2是lime
    train_X = X[train]
    test_X = X[test]
    # explainer2 = lime.lime_tabular.LimeTabularExplainer(train_X, feature_names=col, class_names=['case'],
    #                                                   verbose=True, mode='regression')
    # print('Error =', errors[visual_ind])
    # exp = explainer2.explain_instance(test_X[0], lr_model.predict, num_features=41)               # 影响因素（列）
    # exp.show_in_notebook(show_table=True)
    # exp.as_pyplot_figure()
    a = 1

for i in range(0, 231):              # 病例（行）
    for j in range(0, 53):            # 影响因素（列）
        importance[i][j] = importance[i][j] / 10

df = pd.DataFrame(X, columns=col)
# shap.summary_plot(importance, df)                      #对重要性（影响因素）做点图
# shap.summary_plot(importance, df, plot_type="bar")     #对重要性（影响因素）做柱状图

# 对单个影响因素做点状分布图
shap.dependence_plot("colonoscopy0", shap_values, df, interaction_index=None)                  #可以看交互作用
shap.dependence_plot("age2", shap_values, df, interaction_index=None)
shap.dependence_plot("pathology2", shap_values, df, interaction_index=None)
shap.dependence_plot("hematochezia", shap_values, df, interaction_index=None)
shap.dependence_plot("CRP", shap_values, df, interaction_index=None)
shap.dependence_plot("diarrhea", shap_values, df, interaction_index=None)
shap.dependence_plot("AST", shap_values, df, interaction_index=None)
shap.dependence_plot("ALT", shap_values, df, interaction_index=None)
shap.dependence_plot("PLT", shap_values, df, interaction_index=None)
shap.dependence_plot("UA", shap_values, df, interaction_index=None)
shap.dependence_plot("K", shap_values, df, interaction_index=None)
shap.dependence_plot("endoscopic performance5", shap_values, df, interaction_index=None)
shap.dependence_plot("course1", shap_values, df, interaction_index=None)
shap.dependence_plot("colonoscopy1", shap_values, df, interaction_index=None)
shap.dependence_plot("ESR", shap_values, df, interaction_index=None)
shap.dependence_plot("pathology6", shap_values, df, interaction_index=None)
shap.dependence_plot("HsCRP", shap_values, df, interaction_index=None)
shap.dependence_plot("Na", shap_values, df, interaction_index=None)
shap.dependence_plot("endoscopic performance4", shap_values, df, interaction_index=None)
shap.dependence_plot("age1", shap_values, df, interaction_index=None)

# shap.force_plot(explainer.expected_value, shap_values, df)                                  #js文件需要用notebook来实现



# data1 = pd.DataFrame(importance)
# data1.to_csv(r"C:\Users\Administrator\Desktop\gt\7.28\xgbShapValue.csv", header=False, index=False)




