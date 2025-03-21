import pandas as pd
import numpy as np
from sksurv.ensemble import RandomSurvivalForest
from sklearn.utils import resample
import joblib
import matplotlib.pyplot as plt
from scipy.stats import randint
from sklearn.model_selection import RandomizedSearchCV

# 特征选择集
data_train = pd.read_csv(r"D:\1.研究设计\4.初稿修改\6. 模型工具开发\SPBC_patient_SEER(1998-2015)_train.csv")
data_test = pd.read_csv(r"D:\1.研究设计\4.初稿修改\6. 模型工具开发\SPBC_patient_SEER(1998-2015)_test.csv")

# 定义自变量与因变量
yt_train = data_train['Survival_time']
y_train = data_train['Status']
yt_test = data_test['Survival_time']
y_test = data_test['Status']

X_train = data_train.drop(["Survival_time", "Status"], axis=1)
X_test = data_test.drop(["Survival_time", "Status"], axis=1)

# 因变量转为多维数组
yt_train = np.array(yt_train)
y_train = np.array(y_train)
yt_test = np.array(yt_test)
y_test = np.array(y_test)
# 把c变成布尔值
y_train = y_train == 1
y_test = y_test == 1

# 变成list
yt_train = yt_train.reshape((14333,))
y_train = y_train.reshape((14333,))
yt_train = yt_train.tolist()
y_train = y_train.tolist()
yt_test = yt_test.reshape((6143,))
y_test = y_test.reshape((6143,))
yt_test = yt_test.tolist()
y_test = y_test.tolist()

# 合并b和c，并转变为list
yt_merge_y_train = list(zip(y_train, yt_train))
train_yt_merge_y = np.array(yt_merge_y_train, dtype=[('Status', 'bool'), ('Survival_time', 'int')])
yt_merge_y_test = list(zip(y_test, yt_test))
test_yt_merge_y = np.array(yt_merge_y_test, dtype=[('Status', 'bool'), ('Survival_time', 'int')])

# # 构建置信区间
# dataframe_y = pd.DataFrame(data=test_yt_merge_y)
# dataframe_y['new'] = list(zip(dataframe_y['Status'].to_list(), dataframe_y['Survival_time'].to_list()))
# new_dataframe = dataframe_y['new']
# all_data = pd.concat([X_test, new_dataframe], axis=1)
#
# # bootstrap
# bootstrap_total = []
# n = 50
# for i in range(n):
#     bootstrap_aa = resample(all_data, n_samples=1000, replace=1, stratify=all_data['new'])
#     bootstrap_total.append(bootstrap_aa)
# # 拆分x
# ax = []
# for df in bootstrap_total:
#     x_testbs = df.drop(['new'], axis=1)
#     ax.append(x_testbs)
# # 拆分y
# c_name = 'new'
# c_value = []
# for df in bootstrap_total:
#     c_value.append(df[c_name].tolist())
# # y变为多维数组
# at = []
# for i_list in c_value:
#     it = np.array(i_list, dtype=[('status', 'bool'), ('Survival_time', 'int')])
#     at.append(it)
#
# # 把x重新设置索引
# ax_index = []
# for i in ax:
#     index = i.reset_index(drop=True)
#     ax_index.append(index)


# ========== 训练 RSF 模型并保存 ==========
def train_and_save_rsf():
    """ 训练 Random Survival Forest (RSF) 并保存模型 """
    # 拟合RSF模型
    rsf = RandomSurvivalForest()
    rsf.fit(X_train, train_yt_merge_y)
    # 训练模型
    # rsf.score(X_test, test_yt_merge_y)

    # 保存模型和训练数据
    joblib.dump(rsf, 'rsf_model.pkl')
    # joblib.dump(X_train, 'X_train.pkl')  # 也保存训练数据，以便 Bootstrap 计算
    # joblib.dump(train_yt_merge_y, 'train_yt_merge_y.pkl')
    print("模型已保存为 'rsf_model.pkl'")


# ========== 预测新患者的生存曲线 ==========
def predict_survival(new_patient, target_time):
    """ 使用已训练好的 RSF 模型预测新患者的生存曲线，并计算置信区间 """
    # 加载模型
    rsf = joblib.load('rsf_model.pkl')
    # X_train = joblib.load('X_train.pkl')
    # train_yt_merge_y = joblib.load('train_yt_merge_y.pkl')

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
    fig.show()
    # # ========== 绘制生存曲线 ==========
    # plt.plot(time_points, mean_survival, label="Survival Probability", color='blue')
    # plt.fill_between(time_points, ci_lower, ci_upper, color='blue', alpha=0.2, label="95% CI")
    # plt.xlabel("Time")
    # plt.ylabel("Survival Probability")
    # plt.title("Survival Curve with 95% CI")
    # plt.legend()
    # plt.show()

    survival_prob_at_t = np.interp(target_time, time_points, mean_survival)
    death_risk_at_t = 1 - survival_prob_at_t
    # 查找对应时间点的置信区间
    lower_ci_at_t = np.interp(target_time, time_points, ci_lower)
    upper_ci_at_t = np.interp(target_time, time_points, ci_upper)
    print(f"新患者在 {target_time} 个月后的死亡风险: {death_risk_at_t:.4f}")
    print(f"该时间点死亡风险的95%置信区间: ({1 - upper_ci_at_t:.4f}, {1 - lower_ci_at_t:.4f})")

    # 返回计算结果，方便后续分析
    return survival_prob_at_t, death_risk_at_t, upper_ci_at_t, lower_ci_at_t

# ========== 仅在直接运行 rsf.py 时才执行训练 ==========
if __name__ == "__main__":
    train_and_save_rsf()