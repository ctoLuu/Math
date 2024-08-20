import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, classification_report

def train_decision_tree_with_pca(data, oversample=True, pca_components=20, random_state=None):
    X = data.iloc[:, 3:]
    y = data['y']

    if oversample:
        smote = SMOTE(random_state=random_state)
        X_resampled, y_resampled = smote.fit_resample(X, y)
    else:
        X_resampled, y_resampled = X, y

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_resampled)

    pca = PCA(n_components=pca_components, random_state=random_state)
    X_pca = pca.fit_transform(X_scaled)

    dt_classifier = DecisionTreeClassifier(random_state=random_state)
    dt_classifier.fit(X_pca, y_resampled)

    train_accuracy = accuracy_score(y_resampled, dt_classifier.predict(X_pca))

    return dt_classifier, pca, scaler, train_accuracy

def predict_customer_intent(customer_data, model, pca, scaler):
    X_customer = customer_data.iloc[:, 3:]

    X_customer_scaled = scaler.transform(X_customer)
    X_customer_pca = pca.transform(X_customer_scaled)

    intent_probabilities = model.predict_proba(X_customer_pca)[:, 1]

    results = []
    for i, prob in enumerate(intent_probabilities):
        results.append({
            '客户ID': customer_data.iloc[i, 1],
            '原始意向': customer_data.iloc[i, 2],
            '预测概率': prob
        })
    return results

def create_dataset(data):
    train, test = train_test_split(data, test_size=0.2, random_state=0)
    return train, test

def evaluate_model(model, test_data, test_labels, pca, scaler):
    X_test = test_data.iloc[:, 3:]
    test_labels = test_labels

    X_test_scaled = scaler.transform(X_test)
    X_test_pca = pca.transform(X_test_scaled)

    predictions = model.predict(X_test_pca)
    test_accuracy = accuracy_score(test_labels, predictions)
    f1 = f1_score(test_labels, predictions, average='weighted')
    recall = recall_score(test_labels, predictions, average='weighted')
    precision = precision_score(test_labels, predictions, average='weighted')
    report = classification_report(test_labels, predictions, output_dict=True)

    return test_accuracy, f1, recall, precision, report

# 读取数据
customer_data = pd.read_excel('customer_data.xlsx')
data = pd.read_excel('handledData.xlsx')

# 创建数据集
train, test = create_dataset(data)

# 训练模型
model, pca, scaler, train_accuracy = train_decision_tree_with_pca(data=train, oversample=True, pca_components=20, random_state=0)
print(f"训练集准确率：{train_accuracy}")

# 预测客户意向
customer_intent = predict_customer_intent(customer_data, model, pca, scaler)
for result in customer_intent:
    print(f"客户ID: {result['客户ID']}, 原始意向: {result['原始意向']}, 预测概率: {result['预测概率']}")

# 评估模型
test_accuracy, f1_score, recall, precision, report = evaluate_model(model, test, test['y'], pca, scaler)
print(f"测试集准确率：{test_accuracy}")
print(f"测试集F1分数：{f1_score}")
print(f"测试集召回率：{recall}")
print(f"测试集精确率：{precision}")