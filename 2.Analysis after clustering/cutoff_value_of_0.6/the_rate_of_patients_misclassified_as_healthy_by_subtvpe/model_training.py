import numpy as np
import pandas as pd
import os
import pickle
import sys
import logging
from copy import deepcopy
from tqdm import tqdm
from scipy.stats import binomtest, fisher_exact
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from optuna.integration import SkoptSampler
import optuna
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

# --------------------- 日志设置 ---------------------
os.makedirs('./metrics', exist_ok=True)
log_file = './metrics/training_and_evaluation_results.txt'

logger = logging.getLogger()
logger.setLevel(logging.INFO)

# 创建文件处理器，写入日志文件（清空原内容）
fh = logging.FileHandler(log_file, mode='w')
fh.setLevel(logging.INFO)

# 创建控制台处理器，输出到控制台
ch = logging.StreamHandler(sys.stdout)
ch.setLevel(logging.INFO)

# 设置日志格式
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)
ch.setFormatter(formatter)

# 添加处理器到 logger
if not logger.handlers:
    logger.addHandler(fh)
    logger.addHandler(ch)
else:
    # 避免重复添加handler
    logger.handlers.clear()
    logger.addHandler(fh)
    logger.addHandler(ch)

os.makedirs('./model_history_outer', exist_ok=True)
os.makedirs('./model_history_followup', exist_ok=True)

file_path = 'rawdata.xlsx'

# --------------------- 数据预处理 ---------------------
# 训练数据：Baseline 包含患者亚型（0和1）以及健康标签（4）
baseline_df = pd.read_excel(file_path, sheet_name='Baseline')
baseline_df = baseline_df[baseline_df['Label-0.65'].isin([0, 1, 4])]

# 将标签重新编码为二分类：健康（原标签4）设为1，其余（患者，原标签0和1）设为0
baseline_df['binary_label'] = baseline_df['Label-0.65'].apply(lambda x: 1 if x == 4 else 0)

features = ['Stroop_incongruent_rt', 'Stroop_interference effect_rt', 'Nogo_acc', 
            'Switch_cost', 'RM-1,750_acc', 'RM-750_acc', 'DSBT_Span']
X = baseline_df[features]
y = baseline_df['binary_label']

logger.info("[类别验证] 训练集原始标签: %s", np.unique(baseline_df['Label-0.65']))
logger.info("[类别验证] 训练集二分类标签 (0: 患者, 1: 健康): %s", np.unique(y))

# 随访数据：选择患者样本（亚型），即标签为0和1（原始亚型信息保留，后续用于统计比较）
followup_df = pd.read_excel(file_path, sheet_name='Followup') 
followup_patients = followup_df[followup_df['Label-0.65'].isin([0, 1])]
X_followup = followup_patients[features]
y_followup_true = followup_patients['Label-0.65']  # 原始患者亚型（0 或 1）
logger.info("[类别验证] 随访数据患者亚型标签: %s", np.unique(y_followup_true))

def optuna_opti(trial, model_type, x, y):
    if model_type == 'RandomForest':
        params = {'n_jobs': -1}
        params['n_estimators'] = trial.suggest_int('n_estimators', 50, 500)  # 树的个数
        params['max_depth'] = trial.suggest_int('max_depth', 2, 50)  # 树的最大深度
        params['random_state'] = trial.number  # 使用 trial 编号作为随机种子
        clf = RandomForestClassifier(**params)
        
    clf.fit(x, y)
    train_acc = balanced_accuracy_score(y, clf.predict(x))
    return train_acc

models = {"RandomForest": RandomForestClassifier}

def get_best_rf(best_params):
    params = {
        'n_estimators' : best_params['n_estimators'], 
        'max_depth' : best_params['max_depth'],
        'random_state': None  # 不固定随机状态
    }
    return RandomForestClassifier(**params)

epoch = 1

inner_metrics_list = []
out_metrics_list = []
followup_metrics_list = []  
followup_params_list = []
global_best_item = {'acc': 0, 'clf': None}  # 用于记录全局最佳模型

# 用于分别记录随访数据中患者亚型（0 和 1）被错误预测为健康的统计数据
misclassification_stats = {
    'subtype0_rate': [],
    'subtype1_rate': [],
    'subtype0_count': [],
    'subtype1_count': [],
    'subtype0_total': [],
    'subtype1_total': [],
    'subtype0_p_value': [],
    'subtype1_p_value': []
}

epoch_results = []
inner_auc_list = []
outer_auc_list = []
epoch_misclassification_rates = []

for model_name, model_type in models.items():
    best_hyperparameters = []
    global_best_auc = 0
    
    for epoch_idx in tqdm(range(epoch), desc="Epoch"):
        epoch_seed = 42 + epoch_idx  
        np.random.seed(epoch_seed)
        
        best_auc = 0
        best_params = {}
        best_params_list = []
        
        outer_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=epoch_seed)

        # 外层交叉验证
        for fold_idx, (outer_train_idx, outer_test_idx) in enumerate(outer_cv.split(X, y)):
            inner_seed = epoch_seed * 100 + fold_idx
            inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=inner_seed)
            
            X_outer_train, X_outer_test = X.iloc[outer_train_idx], X.iloc[outer_test_idx]
            y_outer_train, y_outer_test = y.iloc[outer_train_idx], y.iloc[outer_test_idx]

            # 内层交叉验证用于超参数调优
            for inner_idx, (inner_train_idx, inner_test_idx) in enumerate(inner_cv.split(X_outer_train, y_outer_train)):
                X_inner_train = X_outer_train.iloc[inner_train_idx]
                y_inner_train = y_outer_train.iloc[inner_train_idx]
                X_inner_test = X_outer_train.iloc[inner_test_idx]
                y_inner_test = y_outer_train.iloc[inner_test_idx]

                algo = SkoptSampler(skopt_kwargs={'base_estimator': 'GP', 'n_initial_points': 10, 'acq_func': 'EI'})
                study = optuna.create_study(sampler=algo, direction="maximize")
                study.optimize(lambda trial: optuna_opti(trial, model_name, X_inner_train, y_inner_train), 
                               n_trials=50,
                               show_progress_bar=True)

                best_params_list.append(study.best_trial)
                best_trial = study.best_trial
                
                if best_trial.value > best_auc:
                    best_auc = best_trial.value
                    best_params = best_trial.params

                clf = get_best_rf(best_params)
                clf.fit(X_inner_train, y_inner_train)
                y_pred_prob = clf.predict_proba(X_inner_test)[:, 1]
                auc_inner = roc_auc_score(y_inner_test, y_pred_prob)
                inner_auc_list.append(auc_inner)

            clf = get_best_rf(best_params)
            clf.fit(X_outer_train, y_outer_train)
            y_pred_prob = clf.predict_proba(X_outer_test)[:, 1]
            auc_outer = roc_auc_score(y_outer_test, y_pred_prob)
            outer_auc_list.append(auc_outer)

            out_metrics_list.append(np.array([y_outer_test, y_pred_prob]))

            current_acc = balanced_accuracy_score(y_outer_test, (y_pred_prob > 0.5).astype(int))
            if current_acc > global_best_item['acc']:
                global_best_item = {
                    "params": best_params,
                    "clf": deepcopy(clf),
                    "eval": out_metrics_list[-1],
                    "index": epoch_idx,
                    'acc': current_acc
                }
        
        # 每个 epoch 结束后，对随访数据（患者亚型）进行评估
        if global_best_item["clf"] is not None:
            clf = global_best_item["clf"]
            clf.fit(X, y) 

            followup_pred = clf.predict(X_followup)

            # 计算亚组差异检验：构造 2x2 列联表（行：亚型0 与亚型1；列：误判为健康与非误判）
            subtype0_idx = (y_followup_true == 0)
            total0 = np.sum(subtype0_idx)
            misclassified0 = np.sum(followup_pred[subtype0_idx] == 1)
            rate0 = misclassified0 / total0 if total0 > 0 else np.nan

            subtype1_idx = (y_followup_true == 1)
            total1 = np.sum(subtype1_idx)
            misclassified1 = np.sum(followup_pred[subtype1_idx] == 1)
            rate1 = misclassified1 / total1 if total1 > 0 else np.nan

            table = [[misclassified0, total0 - misclassified0],
                     [misclassified1, total1 - misclassified1]]
            _, p_value_diff = fisher_exact(table)

            # 保存统计数据到 misclassification_stats
            misclassification_stats['subtype0_rate'].append(rate0)
            misclassification_stats['subtype0_count'].append(misclassified0)
            misclassification_stats['subtype0_total'].append(total0)
            misclassification_stats['subtype0_p_value'].append(p_value_diff)

            misclassification_stats['subtype1_rate'].append(rate1)
            misclassification_stats['subtype1_count'].append(misclassified1)
            misclassification_stats['subtype1_total'].append(total1)
            misclassification_stats['subtype1_p_value'].append(p_value_diff)

            # 记录每个 epoch 的结果（分别记录两个亚型，但 p 值均为差异检验结果）
            epoch_results.append({
                'Epoch': epoch_idx + 1,
                '亚型': '亚型1',  # 原标签0对应为亚型1
                '误判为健康标签的比例': rate0,
                'p值': f"{p_value_diff:.2e}",
                '误判为健康标签的数量': misclassified0,
                '样本总数': total0
            })
            epoch_results.append({
                'Epoch': epoch_idx + 1,
                '亚型': '亚型2',  # 原标签1对应为亚型2
                '误判为健康标签的比例': rate1,
                'p值': f"{p_value_diff:.2e}",
                '误判为健康标签的数量': misclassified1,
                '样本总数': total1
            })

            followup_metrics_list.append(np.array([y_followup_true, followup_pred]))
            followup_params_list.append(global_best_item["params"])

            with open(f'./model_history_followup/{model_name}_best_epoch_{epoch_idx}.pkl', 'wb') as f:
                pickle.dump({'clf': clf}, f)

            overall_misclassification_rate = np.sum(followup_pred == 1) / len(followup_pred)
            epoch_misclassification_rates.append({
                'Epoch': epoch_idx + 1,
                '整体误判为健康标签的比例': overall_misclassification_rate
            })

# 保存 AUC 结果到 Excel 文件
inner_auc_df = pd.DataFrame(inner_auc_list, columns=['Inner AUC'])
outer_auc_df = pd.DataFrame(outer_auc_list, columns=['Outer AUC'])
auc_results_df = pd.concat([inner_auc_df, outer_auc_df], axis=1)
auc_results_df.to_excel('./metrics/auc_results.xlsx', index=False)

# 保存每个 epoch 的整体误判率
epoch_misclassification_df = pd.DataFrame(epoch_misclassification_rates)
epoch_misclassification_df.to_excel('./metrics/epoch_misclassification_rates.xlsx', index=False)

# 输出最终随访患者误判统计
logger.info("\n[最终随访患者误判统计]")
logger.info("亚型1（标签0）平均误判为健康标签的比例: %.2f%% (±%.2f%%)",
            np.mean(misclassification_stats['subtype0_rate']) * 100,
            np.std(misclassification_stats['subtype0_rate']) * 100)
logger.info("亚型2（标签1）平均误判为健康标签的比例: %.2f%% (±%.2f%%)",
            np.mean(misclassification_stats['subtype1_rate']) * 100,
            np.std(misclassification_stats['subtype1_rate']) * 100)
logger.info("亚型1（标签0）误判为健康标签的平均数量: %.2f", np.mean(misclassification_stats['subtype0_count']))
logger.info("亚型2（标签1）误判为健康标签的平均数量: %.2f", np.mean(misclassification_stats['subtype1_count']))

avg_p_value_subtype0 = (np.min(misclassification_stats['subtype0_p_value']), np.max(misclassification_stats['subtype0_p_value']))
avg_p_value_subtype1 = (np.min(misclassification_stats['subtype1_p_value']), np.max(misclassification_stats['subtype1_p_value']))
logger.info("亚型1（标签0）显著性检验p值范围: %.4f - %.4f", avg_p_value_subtype0[0], avg_p_value_subtype0[1])
logger.info("亚型2（标签1）显著性检验p值范围: %.4f - %.4f", avg_p_value_subtype1[0], avg_p_value_subtype1[1])

num_people = len(X_followup)
logger.info("随访数据总人数: %d", num_people)

epoch_results_df = pd.DataFrame(epoch_results)
epoch_results_df.to_excel('./metrics/followup_misclassification_epoch_results.xlsx', index=False)

with open('./metrics/misclassification_stats.pkl', 'wb') as f:
    pickle.dump(misclassification_stats, f)

with open('./metrics/inner_metrics_list.pkl', 'wb') as f:
    pickle.dump(inner_metrics_list, f)

with open('./metrics/out_metrics_list.pkl', 'wb') as f:
    pickle.dump(out_metrics_list, f)

logger.info("\n模型训练与评估全部完成!")