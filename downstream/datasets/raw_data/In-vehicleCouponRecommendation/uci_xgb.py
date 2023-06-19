import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 设置参数使用GPU
params = {
    "tree_method": "gpu_hist",
    "gpu_id": 1
}

# 读取数据
data = pd.read_csv('in-vehicle-coupon-recommendation.csv')

# 将非数值型的特征转换为数值型
for column in data.columns:
    if data[column].dtype == 'object':
        data[column] = pd.Categorical(data[column]).codes


# 将特征和标签分离
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 将数据集转换为DMatrix格式
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

# 定义XGBoost模型
model = xgb.XGBClassifier(
    max_depth=7,
    learning_rate=0.1,
    n_estimators=100,
    min_child_weight=3,
    subsample=0.8,
    colsample_bytree=0.8,
    objective='binary:logistic',
    seed=27,
    **params)


# 训练模型
model.fit(X_train, y_train)

# 在测试集上进行预测
y_pred = model.predict(X_test)

# 计算准确率
accuracy = accuracy_score(y_test, y_pred)
print("准确率：", accuracy)
