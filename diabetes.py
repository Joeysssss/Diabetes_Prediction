############ Import the required Libraries ##################
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    auc,
    precision_score,
    recall_score,
    f1_score,
)

@st.cache_data
def load_data():
    """載入糖尿病資料集"""
    data = pd.read_csv("diabetes.csv")
    return data

@st.cache_data
def split_data(df, train_ratio):
    """依照使用者設定之比例分割訓練集與測試集"""
    y = df.Outcome
    x = df.drop(columns=['Outcome'])
    return train_test_split(x, y, train_size=train_ratio, random_state=0)

@st.cache_resource
def train_model(classifier_name, train_ratio, params):
    """
    模型訓練函式 (使用 @st.cache_resource 快取)
    只要模型名稱、訓練資料比例與超參數未改變，就不會重新訓練模型。
    """
    df = load_data()
    x_train, _, y_train, _ = split_data(df, train_ratio)
    
    if classifier_name == 'Decision Tree':
        criterion, splitter = params
        model = DecisionTreeClassifier(criterion=criterion, splitter=splitter, random_state=0)
    elif classifier_name == 'Logistic Regression':
        C, max_iter = params
        model = LogisticRegression(C=C, penalty='l2', max_iter=max_iter, random_state=0)
    elif classifier_name == 'Random Forest':
        n_estimators, max_depth, bootstrap = params
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            bootstrap=(bootstrap == 'True' or bootstrap is True),
            n_jobs=-1,
            random_state=0
        )
    else:
        raise ValueError(f"未知的分類器: {classifier_name}")
        
    model.fit(x_train, y_train)
    return model

def plot_metrics(metrics_list, model, x_test, y_test, class_names):
    """繪製選取的評估圖表 (僅保留 Confusion Matrix 與 ROC Curve)"""
    if 'Confusion Matrix' in metrics_list:
        st.subheader("Confusion Matrix (混淆矩陣)")
        fig, ax = plt.subplots()
        cm = confusion_matrix(y_test, model.predict(x_test))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
        disp.plot(cmap="Blues", ax=ax)
        st.pyplot(fig)
        plt.close(fig)

    if 'ROC Curve' in metrics_list:
        st.subheader("ROC Curve (受試者工作特徵曲線)")
        y_prob = model.predict_proba(x_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)
        
        fig, ax = plt.subplots()
        ax.plot(fpr, tpr, color='#1f77b4', lw=2, label=f"ROC curve (AUC = {roc_auc:.2f})")
        ax.plot([0, 1], [0, 1], "k--", lw=1.5, label="Random guess")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("Receiver Operating Characteristic")
        ax.legend(loc="lower right")
        st.pyplot(fig)
        plt.close(fig)

def main():
    st.set_page_config(page_title="Predicting Diabetes Web App", layout="wide")
    st.title("Predicting Diabetes Web App")
    st.markdown("針對糖尿病進行機器學習模型訓練、評估與單筆資料即時預測")

    df = load_data()
    class_names = ['Non-Diabetes', 'Diabetes']

    # 側邊欄：模型與資料設定
    st.sidebar.title("Model Selection Panel")
    st.sidebar.markdown("請設定模型參數與訓練資料比例")

    # 1. 設定訓練資料比例
    st.sidebar.subheader("1. 資料集設定")
    train_ratio = st.sidebar.slider(
        "訓練資料比例 (Train Ratio)",
        min_value=0.5,
        max_value=0.9,
        value=0.7,
        step=0.05,
        format="%.2f"
    )
    test_ratio = round(1.0 - train_ratio, 2)
    st.sidebar.caption(f"訓練集: {int(train_ratio * 100)}% | 測試集: {int(test_ratio * 100)}%")

    # 分割資料
    x_train, x_test, y_train, y_test = split_data(df, train_ratio)

    # 2. 選擇模型
    st.sidebar.subheader("2. 選擇模型")
    classifier = st.sidebar.selectbox(
        "選擇分類器 (Classifier)",
        ("Decision Tree", "Logistic Regression", "Random Forest")
    )

    # 模型超參數
    if classifier == 'Decision Tree':
        st.sidebar.subheader("模型超參數")
        criterion = st.sidebar.radio("Criterion (split quality)", ("gini", "entropy"), key='dt_criterion')
        splitter = st.sidebar.radio("Splitter", ("best", "random"), key='dt_splitter')
        params = (criterion, splitter)

    elif classifier == 'Logistic Regression':
        st.sidebar.subheader("模型超參數")
        C = st.sidebar.number_input("C (正規化參數)", 0.01, 10.0, value=1.0, step=0.01, key='lr_c')
        max_iter = st.sidebar.slider("最大迭代次數 (max_iter)", 100, 1000, value=500, step=50, key='lr_iter')
        params = (C, max_iter)

    elif classifier == 'Random Forest':
        st.sidebar.subheader("模型超參數")
        n_estimators = st.sidebar.number_input("樹的數量 (n_estimators)", 10, 1000, value=100, step=10, key='rf_n')
        max_depth = st.sidebar.number_input("樹的最大深度 (max_depth)", 1, 20, value=5, step=1, key='rf_depth')
        bootstrap = st.sidebar.radio("Bootstrap 取樣", ('True', 'False'), key='rf_boot')
        params = (n_estimators, max_depth, bootstrap)

    # 3. 評估圖表選項 (已刪除 Precision-Recall Curve)
    st.sidebar.subheader("3. 評估圖表")
    metrics = st.sidebar.multiselect(
        "選擇要繪製的圖表",
        ('Confusion Matrix', 'ROC Curve'),
        default=['Confusion Matrix', 'ROC Curve']
    )

    # 取得模型 (利用快取，避免重複訓練)
    model = train_model(classifier, train_ratio, params)

    # 側邊欄附加功能
    if st.sidebar.button("清除快取並強制重訓"):
        st.cache_resource.clear()
        st.rerun()

    if st.sidebar.checkbox("顯示原始資料集 (Show raw data)", False):
        st.subheader("Diabetes Raw Dataset")
        st.write(df)

    # 主畫面區域：分為「模型評估結果」與「單筆即時預測」
    st.header(f"📊 {classifier} 模型評估結果")
    st.caption(f"使用訓練集比例: {int(train_ratio * 100)}% (共 {len(x_train)} 筆) | 測試集: {int(test_ratio * 100)}% (共 {len(x_test)} 筆)")

    # 模型在測試集上的表現
    accuracy = model.score(x_test, y_test)
    y_pred = model.predict(x_test)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    # 輸出指標 (含 F1 指標)
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("準確率 (Accuracy)", f"{accuracy * 100:.2f}%")
    col2.metric("精確率 (Precision)", f"{prec:.2f}")
    col3.metric("召回率 (Recall)", f"{rec:.2f}")
    col4.metric("F1 分數 (F1-Score)", f"{f1:.2f}")

    # 繪製選取的圖表
    plot_metrics(metrics, model, x_test, y_test, class_names)

    st.markdown("---")

    # 單筆資料即時預測
    st.header("🩺 單筆資料即時預測 (Single Sample Prediction)")
    st.markdown("請輸入患者的各項生理數據，系統將使用已訓練的模型直接進行預測（**模型不需重新訓練**）：")

    # 8 個特徵輸入欄位
    input_col1, input_col2, input_col3, input_col4 = st.columns(4)
    with input_col1:
        pregnancies = st.number_input("懷孕次數 (Pregnancies)", min_value=0, max_value=20, value=1, step=1)
        insulin = st.number_input("胰島素 (Insulin, mu U/ml)", min_value=0.0, max_value=900.0, value=79.8, step=1.0)
    with input_col2:
        glucose = st.number_input("血糖 (Glucose, mg/dl)", min_value=0.0, max_value=300.0, value=120.9, step=1.0)
        bmi = st.number_input("BMI (體重/身高^2)", min_value=0.0, max_value=70.0, value=32.0, step=0.1)
    with input_col3:
        blood_pressure = st.number_input("血壓 (BloodPressure, mm Hg)", min_value=0.0, max_value=200.0, value=69.1, step=1.0)
        dpf = st.number_input("糖尿病遺傳函數 (DPF)", min_value=0.0, max_value=3.0, value=0.47, step=0.01)
    with input_col4:
        skin_thickness = st.number_input("皮膚厚度 (SkinThickness, mm)", min_value=0.0, max_value=100.0, value=20.5, step=1.0)
        age = st.number_input("年齡 (Age)", min_value=1, max_value=120, value=33, step=1)

    # 構造單筆資料 DataFrame
    single_sample = pd.DataFrame([{
        'Pregnancies': pregnancies,
        'Glucose': glucose,
        'BloodPressure': blood_pressure,
        'SkinThickness': skin_thickness,
        'Insulin': insulin,
        'BMI': bmi,
        'DiabetesPedigreeFunction': dpf,
        'Age': age
    }])

    # 進行單筆預測 (使用已快取的 model)
    single_pred = model.predict(single_sample)[0]
    single_proba = model.predict_proba(single_sample)[0]
    prob_non = single_proba[0]
    prob_diab = single_proba[1]

    pred_res_col, pie_col = st.columns([1, 1])

    with pred_res_col:
        st.subheader("預測結果")
        if single_pred == 1:
            st.error("⚠️ **診斷結果：糖尿病陽性 (Diabetes)**")
        else:
            st.success("✅ **診斷結果：糖尿病陰性 (Non-Diabetes)**")

        st.write(f"- **未患糖尿病機率 (Non-Diabetes)**: `{prob_non * 100:.2f}%`")
        st.write(f"- **患糖尿病機率 (Diabetes)**: `{prob_diab * 100:.2f}%`")
        st.caption(f"目前使用模型: `{classifier}` (直接推論，不需重訓)")

    with pie_col:
        st.subheader("機率分布圓餅圖")
        fig_pie, ax_pie = plt.subplots(figsize=(4, 4))
        pie_labels = [f"Non-Diabetes\n({prob_non * 100:.1f}%)", f"Diabetes\n({prob_diab * 100:.1f}%)"]
        pie_colors = ['#4CAF50', '#F44336']
        explode = (0.05, 0) if prob_non >= prob_diab else (0, 0.05)
        
        ax_pie.pie(
            [prob_non, prob_diab],
            labels=pie_labels,
            autopct='%1.1f%%',
            startangle=140,
            colors=pie_colors,
            explode=explode,
            shadow=False,
            textprops={'fontsize': 11}
        )
        ax_pie.axis('equal')
        st.pyplot(fig_pie)
        plt.close(fig_pie)

if __name__ == '__main__':
    main()
