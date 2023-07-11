import pandas as pd
import numpy as np
import random
from rulefit_uplift_forest import CausalRuleEnsembling
from sklearn.model_selection import KFold
from sdv.sampling import Condition
import torch
import pickle
from causalml.match import NearestNeighborMatch, create_table_one
from sdv.tabular import CTGAN
from sklearn.linear_model import LogisticRegression

# Load real data

directory = 'temp_data/'
df = pd.read_csv(directory + 'imputed_all.csv')
df.drop(columns=['Unnamed: 0'], inplace =True)

obs = pd.read_csv(directory + 'matched_erich.csv')
obs.drop(columns=['Unnamed: 0', 'Unnamed: 0.1', 'score', 'source', 'group'], inplace=True)

pre_treatment_var = ['age', 'ich_vol', 'ivh_vol', 'gcs', 'nihss', 'sbp', 'dbp', 'pp', 'map', 
                     'RACE_Asian', 'RACE_Black', 'RACE_Other', 'RACE_White', 'GENDER_Male', 'GENDER_Female',
                     'ETHNICITY_Hispanic','ETHNICITY_Non-Hispanic',
                     
                     'HIS_HYPERTENSION', 'HIS_HYPERLIPIDEMIA', 'HIS_DM2',
                     'HIS_DM1', 'HIS_HF', 'HIS_AF', 'HIS_PTCA', 'HIS_PV','HIS_MYOCARDIAL', 
                     'HIS_ANTIDIABETIC', 'HIS_ANTIHYPERTENSIVES',
                     
                     'CT1_ICHSIDE','ICHLOC_Basal Ganglia', 'ICHLOC_Lobar', 'ICHLOC_Thalamus', 'ICHLOC_Other',
                     
                     'wbc', 'hemoglobin','hematocrit', 'pc', 'aptt', 'inr', 'glucose',
                     'sodium', 'potassium', 'chloride', 'cd', 'bun','creatinie']


# This experiment is used to compare how many data samples would give a best performance in terms of qini coefficient

num_samples = [100, 200, 500, 800, 1000, 1500, 2000]


ctgan_model = CTGAN.load('temp_data/ctgan_comb_v2.pkl')

def transform_var(df):
	df['age'] = (df['DEMO_AGE'] - np.mean(df['DEMO_AGE'])) / np.std(df['DEMO_AGE'])
	df['ich_vol'] = np.log(df['CT1_ICHVOL'] + 1)
	df['ivh_vol'] = np.log(df['CT1_IVHVOL'] + 1)
	df['gcs'] = (df['GCS_TTL'] - np.mean(df['GCS_TTL'])) / np.std(df['GCS_TTL'])
	df['nihss'] = np.log(df['NIHSS_TTL'] + 1)
	df['sbp'] = (df['BP_S0'] - np.mean(df['BP_S0'])) / np.std(df['BP_S0'])
	df['dbp'] = (df['BP_D0'] - np.mean(df['BP_D0'])) / np.std(df['BP_D0'])
	df['PP'] = df['BP_S0'] - df['BP_D0']
	df['MAP'] = df['BP_S0']/3 + df['BP_D0']*2/3
	df['pp'] = (df['PP'] -np.mean(df['PP'])) / np.std(df['PP'])
	df['map'] = (df['MAP'] - np.mean(df['MAP']))/np.std(df['MAP'])
    
	df['pc'] = (df['LB_PC'] - np.mean(df['LB_PC'])) / np.std(df['LB_PC'])
	df['glucose'] = (df['LB_GLUCOSE'] - np.mean(df['LB_GLUCOSE'])) / np.std(df['LB_GLUCOSE'])
	df['sodium'] = (df['LB_SODIUM'] - np.mean(df['LB_SODIUM'])) / np.std(df['LB_SODIUM'])
	df['potassium'] = (df['LB_POTASSIUM'] - np.mean(df['LB_POTASSIUM'])) / np.std(df['LB_POTASSIUM'])
	df['chloride'] = (df['LB_CHLORIDE'] - np.mean(df['LB_CHLORIDE'])) / np.std(df['LB_CHLORIDE'])
	df['cd'] = (df['LB_CD'] - np.mean(df['LB_CD'])) / np.std(df['LB_CD'])
	df['bun'] = (df['LB_BUN'] - np.mean(df['LB_BUN'])) / np.std(df['LB_BUN'])
	df['hemoglobin'] = (df['LB_HEMOGLOBIN'] - np.mean(df['LB_HEMOGLOBIN'])) / np.std(df['LB_HEMOGLOBIN'])
	df['hematocrit'] = (df['LB_HEMATOCRIT'] - np.mean(df['LB_HEMATOCRIT'])) / np.std(df['LB_HEMATOCRIT'])
	df['wbc'] = (df['LB_WBC'] - np.mean(df['LB_WBC'])) / np.std(df['LB_WBC'])
	df['creatinie'] = (df['LB_CREATINIE'] - np.mean(df['LB_CREATINIE'])) / np.std(df['LB_CREATINIE'])
	df['aptt'] = (df['LB_APTT'] - np.mean(df['LB_APTT'])) / np.std(df['LB_APTT'])
	df['inr'] = (df['LB_INR'] - np.mean(df['LB_INR'])) / np.std(df['LB_INR'])
    
	return df


def prepare_cohort(num_syn_samples, syn_model, df_all):
	SEED_VALUE = 20
	np.random.seed(SEED_VALUE)
	torch.manual_seed(SEED_VALUE)
	cond1 = Condition({'TREATMENT':1}, num_syn_samples)
	cond2 = Condition({'TREATMENT':0}, num_syn_samples)

	new_data_t = syn_model.sample_conditions(conditions=[cond1])
	new_data_c = syn_model.sample_conditions(conditions=[cond2])

	syn = pd.concat([new_data_t, new_data_c])
	syn.reset_index(drop=True, inplace=True)
	syn = pd.get_dummies(syn)
	syn['source'] = ['syn'] * len(syn)
	syn['group'] = ['train'] * len(syn)

	df_new = df_all.copy()
	df_new.drop(columns=['source', 'group'], inplace=True)
	df_new = pd.get_dummies(df_new)
	df_new['source'] = df['source']
	df_new['group'] = df['group']

	merge1 = pd.concat([df_new, syn])
	merge1 = transform_var(merge1)

	trial_train = merge1[(merge1['source'] == 'atach2')&(merge1['group']=='train')].copy()
	trial_train.reset_index(inplace=True, drop=True)
	df_test = merge1[merge1['group'] == 'test'].copy()
	syn = merge1[merge1['source'] == 'syn'].copy()
	matched_ids = list(obs['index'])
	matched_obs = merge1[merge1['index'].isin(matched_ids)]
	matched_obs.shape

	merge_temp1 = pd.concat([trial_train, matched_obs, syn])
	X = np.array(merge_temp1[pre_treatment_var])
	y = np.array(merge_temp1['TREATMENT'])
	pm_lgr = LogisticRegression(penalty='none', max_iter=3000)
	pm_lgr.fit(X, y)
	clip_bounds = (1e-3, 1-1e-3)
	score_lgr1 = np.clip(pm_lgr.predict_proba(X)[:, 1], *clip_bounds)
	
	merge_temp1['score'] = score_lgr1
	t_syn = merge_temp1[(merge_temp1['TREATMENT'] == 1) & (merge_temp1['source'] == 'syn')].copy()
	c_syn = merge_temp1[(merge_temp1['TREATMENT'] == 0) & (merge_temp1['source'] == 'syn')].copy()
	t_real = merge_temp1[(merge_temp1['TREATMENT'] == 1) & (merge_temp1['source'] != 'syn')].copy()
	c_real = merge_temp1[(merge_temp1['TREATMENT'] == 0) & (merge_temp1['source'] != 'syn')].copy()

	temp1 = pd.concat([t_syn, c_real])
	temp1['treatment'] = 1 - temp1['TREATMENT']
	temp1.index = np.arange(0, len(temp1))
	temp2 = pd.concat([c_syn, t_real])
	temp2.index = np.arange(0, len(temp2))


	# PSM1: use real control to find matched treatment
	psm1 = NearestNeighborMatch(caliper=0.2, replace=False, ratio=1, random_state=2205)
	matched_syn1 = psm1.match(data=temp1, treatment_col='treatment',score_cols=['score'])

	# PSM2: use real treatment to find matched control
	psm2 = NearestNeighborMatch(caliper=0.2, replace=False, ratio=1, random_state=2205)
	matched_syn2 = psm2.match(data=temp2, treatment_col='TREATMENT',score_cols=['score'])

	matched_syn = pd.concat([matched_syn1[matched_syn1['TREATMENT'] == 1], matched_syn2[matched_syn2['TREATMENT'] == 0]])
	
	merge = pd.concat([trial_train, matched_obs, matched_syn])
	merge = transform_var(merge)
	
	eval_merge0 = create_table_one(merge_temp1, 'TREATMENT', pre_treatment_var)
	a = 0
	b = 0
	# Calculate average smd on ERICH + ATACH2
	for i in range(len(eval_merge0['SMD'])):
		if (eval_merge0['SMD'][i] !='') & ~(pd.isna(eval_merge0['SMD'][i])):
			a += abs(eval_merge0['SMD'][i])
			b += 1
	smd0 = a/b

	eval_merge1 = create_table_one(merge, 'TREATMENT', pre_treatment_var)
	a = 0
	b = 0
	# Calculate average smd on ERICH + ATACH2
	for i in range(len(eval_merge1['SMD'])):
		if (eval_merge1['SMD'][i] !='') & ~(pd.isna(eval_merge1['SMD'][i])):
			a += abs(eval_merge1['SMD'][i])
			b += 1
	smd1 = a/b


	with open('result.txt', 'a') as f:
		f.write('Number of generated samples:{}\n'.format(num_syn_samples))
		f.write('Matched synthetic sample size:{}\n'.format(len(matched_syn)))
		f.write('SMD before matching:{}\n'.format(smd0))
		f.write('SMD after matching:{}\n'.format(smd1))
	return merge, df_test

# Start the experiments
for num in num_samples:
	merge, df_test = prepare_cohort(num, ctgan_model, df)
	test = df_test.copy()
	train_X = np.array(merge[pre_treatment_var])
	test_X = np.array(test[pre_treatment_var])
	train_treatment=(merge['TREATMENT']!=0).astype(int).values
	test_treatment=(test['TREATMENT']!=0).astype(int).values
	y_train = (merge['OUTCOME_mRS90']).values
	y_test = (test['OUTCOME_mRS90']).values
	y_bin_train = np.array([0] * len(y_train))
	y_bin_train[np.where(y_train <= 3)] = 1
	y_bin_test = np.array([0] * len(y_test))
	y_bin_test[np.where(y_test <= 3)] = 1
	treatment_train = ['control'] * len(train_treatment)
	for i in range(len(train_treatment)):
		if train_treatment[i] == 1:
			treatment_train[i] = 'treatment'
	treatment_train = np.array(treatment_train)

	res = {'res_tree':[], 'res_lasso':[]}
	for seed in range(0, 30):
		model_temp = CausalRuleEnsembling(
         tree_depth = 3, 
         tree_eval_func = 'KL', 
         n_reg=10, 
         n_estimator = 100,   
         min_samples_leaf = 80, 
         min_samples_treatment = 30, 
         model_type='rl', 
         random_state = seed)
		model_temp.fit(train_X, treatment_train, y_bin_train, pre_treatment_var)
		a, b = model_temp.eval_qini(test_X, y_bin_test, test_treatment)
		res['res_tree'].append(a)
		res['res_lasso'].append(b)
		with open('result.txt', 'a') as f:
			f.write('Seed round:{}, {}\n'.format(seed, b))
	with open('result.txt', 'a') as f:
		f.write('Qini tree:{}\n'.format(np.mean(res['res_tree'])))
		f.write('Qini tree:{}\n'.format(np.mean(res['res_lasso'])))
	



	

	

	