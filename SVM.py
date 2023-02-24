import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# 读取数据集
df = pd.read_csv("data2.csv")
X = df.drop("label", axis=1).values
y = df["label"].values

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0)

# 数据预处理（标准化）
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 训练支持向量机模型
clf = SVC(kernel="linear")
clf.fit(X_train, y_train)

# 预测结果
y_pred = clf.predict(X_test)

# 评估准确率
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)