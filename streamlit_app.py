import streamlit as st
import joblib
import pandas as pd
import numpy as np
from sklearn.utils import resample
import matplotlib.pyplot as plt

# 定义一个函数用于预测和展示结果
# ========== 预测新患者的生存曲线 ==========
def predict_survival(new_patient, target_time):
    """ 使用已训练好的 RSF 模型预测新患者的生存曲线，并计算置信区间 """
    # 加载模型
    rsf = joblib.load('rsf_model.pkl')
    X_train = joblib.load('X_train.pkl')
    train_yt_merge_y = joblib.load('train_yt_merge_y.pkl')

    # 计算新患者的生存函数
    surv_fn = rsf.predict_survival_function(new_patient)

    # 获取时间点和对应的生存概率
    time_points = surv_fn[0].x  # 提取时间点
    survival_probs = surv_fn[0].y  # 提取对应的生存概率

    # ========== 计算置信区间 ==========
    n_bootstrap = 2  # 你可以调大，比如 50 以提高稳定性
    survival_curves = []

    for _ in range(n_bootstrap):
        X_resampled, y_resampled = resample(X_train, train_yt_merge_y, random_state=_)
        rsf_boot = rsf
        rsf_boot.fit(X_resampled, y_resampled)
        surv_fn_boot = rsf_boot.predict_survival_function(new_patient)
        survival_prob_interp = np.interp(time_points, surv_fn_boot[0].x, surv_fn_boot[0].y)  # 统一时间点
        survival_curves.append(survival_prob_interp)

    survival_curves = np.array(survival_curves)  # 现在所有曲线长度一致

    # 计算均值和置信区间
    mean_survival = survival_curves.mean(axis=0)
    se_survival = survival_curves.std(axis=0) / np.sqrt(n_bootstrap)
    ci_lower = mean_survival - 1.96 * se_survival
    ci_upper = mean_survival + 1.96 * se_survival

    fig, ax = plt.subplots()
    ax.plot(time_points, mean_survival, label="Survival Probability", color='blue')
    ax.fill_between(time_points, ci_lower, ci_upper, color='blue', alpha=0.2, label="95% CI")
    ax.set_xlabel("Time")
    ax.set_ylabel("Survival Probability")
    ax.set_title("Survival Curve with 95% CI")
    ax.legend()


    survival_prob_at_t = np.interp(target_time, time_points, mean_survival)
    death_risk_at_t = 1 - survival_prob_at_t
    # 查找对应时间点的置信区间
    lower_ci_at_t = np.interp(target_time, time_points, ci_lower)
    upper_ci_at_t = np.interp(target_time, time_points, ci_upper)

    # 显示预测结果
    st.write(f"新患者在 {target_time} 个月后的死亡风险: {death_risk_at_t:.4f}")
    st.write(f"新患者在 {target_time} 个月后的生存概率: {survival_prob_at_t:.4f}")
    st.write(f"95% 置信区间: ({lower_ci_at_t:.4f}, {upper_ci_at_t:.4f})")
    st.pyplot(fig)

############################################
# Streamlit 页面设置
st.title("第二原发乳腺癌患者生存预测模型")
st.header("请输入新患者的信息:")

# 输入表单
age = st.slider('年龄(岁)', min_value=20, max_value=90, value=50, step=1)
latency = st.slider('两次肿瘤的确诊间隔时间(月)', min_value=2, max_value=280, value=50, step=1)
node_examined = st.slider('区域淋巴结检测数(个)', min_value=0, max_value=50, value=10, step=1)

# first_site
first_site_options = {1: "乳腺", 2: "女性生殖系统", 3: "消化系统", 4: "皮肤",
                      5: "内分泌系统", 6: "泌尿系统", 7: "呼吸系统", 8: "淋巴", 9: "其他位点"}
first_site = st.selectbox('第一原发恶性肿瘤位点', options=list(first_site_options.keys()),
                          format_func=lambda x: first_site_options[x])

# marital
marital_options = {1: "已婚", 2: "丧偶", 3: "单身", 4: "离异"}
marital = st.selectbox('婚姻状况', options=list(marital_options.keys()),
                       format_func=lambda x: marital_options[x])

# histology_type
histology_type_options = {1: "8500/3: Infiltrating duct carcinoma, NOS",
                          2: "8520/3: Lobular carcinoma, NOS",
                          3: "8522/3: Infiltrating duct and lobular carcinoma",
                          4: "Other"}
histology_type = st.selectbox('SPBC组织学类型', options=list(histology_type_options.keys()),
                              format_func=lambda x: histology_type_options[x])

# surgery_2nd
surgery_2nd_options = {0: "未接受手术", 1: "接受手术"}
surgery_2nd = st.selectbox('SPBC手术情况', options=list(surgery_2nd_options.keys()),
                           format_func=lambda x: surgery_2nd_options[x])

# radiotherapy_2nd
radiotherapy_2nd_options = {1: "未接受放疗", 2: "接受放疗"}
radiotherapy_2nd = st.selectbox('SPBC放疗情况', options=list(radiotherapy_2nd_options.keys()),
                                format_func=lambda x: radiotherapy_2nd_options[x])

# chemotherapy_1st
chemotherapy_1st_options = {0: "未接受化疗", 1: "接受化疗"}
chemotherapy_1st = st.selectbox('第一原发恶性肿瘤化疗情况', options=list(chemotherapy_1st_options.keys()),
                                format_func=lambda x: chemotherapy_1st_options[x])

# chemotherapy_2nd
chemotherapy_2nd_options = {0: "未接受化疗", 1: "接受化疗"}
chemotherapy_2nd = st.selectbox('SPBC化疗情况', options=list(chemotherapy_2nd_options.keys()),
                                format_func=lambda x: chemotherapy_2nd_options[x])

# systemic_therapy
systemic_therapy_options = {0: "未接受全身/系统治疗", 1: "接受全身/系统治疗"}
systemic_therapy = st.selectbox('SPBC全身/系统治疗情况', options=list(systemic_therapy_options.keys()),
                                format_func=lambda x: systemic_therapy_options[x])

# nodes_positive
nodes_positive_options = {0: "未检测到淋巴结", 1: "淋巴结阳性", 2: "淋巴结阴性"}
nodes_positive = st.selectbox('SPBC区域淋巴结阳性检测', options=list(nodes_positive_options.keys()),
                              format_func=lambda x: nodes_positive_options[x])
# ER
er_options = {0: "ER-", 1: "ER+"}
er = st.selectbox('雌激素受体（ER）', options=list(er_options.keys()), format_func=lambda x: er_options[x])

# PR
pr_options = {0: "PR-", 1: "PR+"}
pr = st.selectbox('孕激素受体（PR）', options=list(pr_options.keys()), format_func=lambda x: pr_options[x])

# T
T_options = {1: "T1期", 2: "T2期", 3: "T3期", 4: "T4期"}
T = st.selectbox('第二原发乳腺癌T分期', options=list(T_options.keys()), format_func=lambda x: T_options[x])

# N
N_options = {0: "N0期", 1: "N1期", 2: "N2期", 3: "N3期"}
N = st.selectbox('第二原发乳腺癌N分期', options=list(N_options.keys()), format_func=lambda x: N_options[x])

# T
M_options = {0: "M0期", 1: "M1期"}
M = st.selectbox('第二原发乳腺癌M分期', options=list(M_options.keys()), format_func=lambda x: M_options[x])

# Stage
stage_options = {1: "Stage I期", 2: "Stage II期", 3: "Stage III期", 4: "Stage IV期"}
stage = st.selectbox('SPBC临床综合分期', options=list(stage_options.keys()), format_func=lambda x: stage_options[x])

# Stage
grade_options = {1: "Grade I: Well differentiated",
                 2: "Grade II: Moderately differentiated",
                 3: "Grade III: Poorly differentiated",
                 4: "Grade IV: Undifferentiated (Anaplastic)"}
grade = st.selectbox('SPBC肿瘤分级', options=list(grade_options.keys()), format_func=lambda x: grade_options[x])

# 输入目标时间
target_time = st.slider('请输入您想要预测的时间点(月)', min_value=2, max_value=280, value=60, step=1)


new_patient = pd.DataFrame({"Age": [age],
                            "Latency": [latency],
                            "nodes_examined_2nd": [node_examined],
                            "First_site": [first_site],
                            "Marital": [marital],
                            "Histology": [histology_type],
                            "Surgery_2nd": [surgery_2nd],
                            "Radio_2nd": [radiotherapy_2nd],
                            "Chemo_1st": [chemotherapy_1st],
                            "Chemo_2nd": [chemotherapy_2nd],
                            "Systemic_therapy": [systemic_therapy],
                            "nodes_positive_2nd": [nodes_positive],
                            "ER": [er],
                            "PR": [pr],
                            "T": [T],
                            "N": [N],
                            "M": [M],
                            "Stage": [stage],
                            "Grade": [grade]})

# 按钮触发预测
if st.button('输出模型预测结果'):
    predict_survival(new_patient, target_time)
