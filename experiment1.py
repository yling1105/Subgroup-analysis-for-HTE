'''
Author: Yaobin Ling
Date: Dec 16, 2022
Version 1
'''

import numpy as np
import pickle as pkl
import pandas as pd

from sklearn.model_selection import train_test_split
from rulefit_uplift_forest import CausalRuleEnsembling
from uplift_forest_customed import UpliftTreeNew
from sklift.metrics import qini_auc_score
from sklearn.model_selection import KFold

df = pd.read_csv('data/matched_cohort_3.csv')

pre_treatment_var = ['age', 'DEMO_MALE', 'DEMO_HISPANIC', 'DEMO_ASIAN', 'DEMO_BLACK','DEMO_OTHERRACE', 'DEMO_WHITE', 
                     'sbp','dbp', 'gcs', 'nihss', 'pp', 'map',
                     
                     'HIS_HYPERTENSION', 'HIS_HYPERLIPIDEMIA', 'HIS_DM2',
                     'HIS_ANTIPLATELET', 'HIS_ANTICOAGULANT', 'HIS_ALCOHOL',
                     'HIS_DM1', 'HIS_HF', 'HIS_AF', 'HIS_PTCA', 'HIS_PV','HIS_MYOCARDIAL', 
                     'HIS_ANTIDIABETIC', 'HIS_ANTIHYPERTENSIVES', 'HIS_COCAINE','HIS_SMK', 
                       
                     'CT1_ICHSIDE', 'CT1_ICHLOC_Cerebellum', 'CT1_ICHLOC_Lobar', 'CT1_ICHLOC_Pons',
                     'CT1_ICHLOC_Thalamus', 'CT1_LOGICHVOL', 'CT1_LOGIVHVOL',
                     
                     'LB_LOGWBC', 'LB_LOGAPTT', 'LB_LOGCREATININE', 'LB_LOGINR', 'hemoglobin', 'hematocrit', 
                     'pc', 'glucose', 'sodium', 'chloride', 'potassium', 'cd', 'bun']

df['map'] = df['map'].apply(lambda x: np.log(x + 1))
df['pp'] = df['pp'].apply(lambda x: np.log(x + 1))

# Data prepare
X=np.array(df[pre_treatment_var])
treatment=(df['TREATMENT']!=0).astype(int).values
y=df['OUTCOME_mRS90'].values

X_train, X_test, treatment_train, treatment_test, y_train, y_test = train_test_split(X, treatment, y, random_state = 222, test_size=0.3)

trt_train = [0] * len(treatment_train)
for i in range(len(treatment_train)):
    if treatment_train[i] != 0:
        trt_train[i] = 'treatment'
    else:
        trt_train[i] = 'control'

y_train_bin = np.array([0] * len(y_train))
y_train_bin[np.where(y_train <= 3)] = 1

trt_test = [0] * len(treatment_test)
for i in range(len(treatment_test)):
    if treatment_test[i] != 0:
        trt_test[i] = 'treatment'
    else:
        trt_test[i] = 'control'
        
y_test_bin = np.array([0] * len(y_test))
y_test_bin[np.where(y_test <= 3)] = 1
trt_train = np.array(trt_train)

dist_func = ['KL', 'DDP', 'CTS']
depth = [2, 3]
n_reg = [3, 5, 10]
min_samples = [60, 100, 150]

res = {}
ttl_exp = len(dist_func) * len(depth) * len(n_reg) * len(min_samples)
for i in range(ttl_exp):
    res[i] = [[], [], []]


round_idx = 1
ylseed = 1105
kf = KFold(n_splits=5)

X=np.array(df[pre_treatment_var])
treatment=(df['TREATMENT']!=0).astype(int).values
y=df['OUTCOME_mRS90'].values

y_bin = np.array([0] * len(y))
y_bin[np.where(y<=3)] = 1

# Experiment start
f = open('results/CV_results_all.txt', 'w')
for train_index, test_index in kf.split(X):
    print('Round:', round_idx)
    idx = 1
    f.write('Round:'+str(idx))
    f.write('\n')
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y_bin[train_index], y_bin[test_index]
    treatment_train, treatment_test = treatment[train_index], treatment[test_index]
    if sum(treatment_test) == 0:
        continue
    
    trt_train = [0] * len(treatment_train)
    for i in range(len(treatment_train)):
        if treatment_train[i] != 0:
            trt_train[i] = 'treatment'
        else:
            trt_train[i] = 'control'
    trt_train = np.array(trt_train)
    for metric in dist_func:
        for d in depth:
            for r in n_reg:
                for j in range(len(min_samples)):
                    n_samples = min_samples[j]
                    n_trts = n_samples / 2
            
                # train the model
                tree_model = UpliftTreeNew(tree_depth = d, n_reg=r, min_samples_leaf = n_samples, min_samples_treatment = n_trts, model_type='rl', tree_eval_func = metric,random_state = ylseed)
                tree_model.fit(X_train, trt_train, y_train)
                pred_tree = tree_model.predict(X_test)
                hte_tree = pred_tree[:, 1] - pred_tree[:, 0]
                qini_tree = qini_auc_score(y_true=y_test, uplift=hte_tree, treatment=treatment_test)

                model = CausalRuleEnsembling(tree_depth = d, n_reg=r, min_samples_leaf = n_samples, min_samples_treatment = n_trts, model_type='rl', tree_eval_func = metric,random_state = ylseed)
                model.fit(X_train, trt_train, y_train, pre_treatment_var)
                qini_rf, qini_last = model.eval_qini(X_test, y_test, treatment_test)

            
                res[idx][0].append(qini_tree)
                res[idx][1].append(qini_rf)
                res[idx][2].append(qini_last)
            
                res_temp = ['Metric:', metric, 'Depth:' + str(d), 'n_reg:'+str(r),'n_samples:'+str(n_samples), 'n_trt:'+str(), 'Qini tree:'+str(qini_tree), 'Qini rf:', qini_rf ,'qini_final:'+str(qini_last)]
                f.write(', '.join(res_temp))
                f.write("\n")
                print(','.join(res_temp))
                idx += 1
    round_idx += 1

# Experiment end

mu_qini_final = []
mu_qini_tree = []
mu_qini_rf = []
for i in range(ttl_exp):
    idx = i + 1
    mu_qini_final.append(np.mean(res[idx][2]))
    mu_qini_tree.append(np.mean(res[idx][0]))
    mu_qini_rf.append(np.mean(res[idx][1]))

f.write("\n")
best_tree = [indice for indice, item in enumerate(mu_qini_tree) if item == max(mu_qini_tree)]
best_rf = [indice for indice, item in enumerate(mu_qini_rf) if item == max(mu_qini_rf)]
best_final = [indice for indice, item in enumerate(mu_qini_final) if item == max(mu_qini_final)]

f.write('Best tree idx:', best_tree, "\n")
f.write('Best randomz forest idx:', best_rf,"\n")
f.write('Best Ensembling model idx:', best_final, "\n")


f.close()
