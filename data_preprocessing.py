from rulefit_uplift_forest import CausalRuleEnsembling
from sklearn.model_selection import KFold
from causalml.propensity import ElasticNetPropensityModel
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
import miceforest as mf
from causalml.match import create_table_one

df_atach2=pd.read_pickle("../data_explore/atach2.pkl")
df_erich=pd.read_pickle("../data_explore/erich.pkl")

df_atach2['source'] = 'atach2'
df_erich['source'] = 'erich'

df_atach2['SUBJECT_ID'] = df_atach2['SUBJECT_ID'].apply(lambda x: 'atach2'+str(x))
df_erich['SUBJECT_ID'] = df_erich['SUBJECT_ID'].apply(lambda x: 'erich'+str(x))

df_erich.drop(columns=['OUTCOME_mRS180', 'OUTCOME_mRS365'], inplace=True)
df = pd.concat([df_atach2, df_erich], axis=0)

# Organize the data distribution
def str2loc(loc):
    """
    Convert free text brain location to 
    - (Side) (Location)
    where
    - Side: L or R
    - Location: 
        - Basal Ganglia #deep
        - Thalamic #deep
        - Lobar
        - Cerebellum
    """
    if not loc:
        return ""
    else:
        #Side
        side=""
        if loc[-5:] == 'Right':
            loc = 'R ' + loc[:-6]
        if loc[-4:] == 'Left':
            loc = 'L ' + loc[:-5]
            
        if loc[:2]=='R ':
            loc=loc[2:]
            side='R'
        if loc[:2]=='L ':
            loc=loc[2:]
            side='L'

        #Location
        #Thalamus
        loc = loc.replace("Thalamic", "Thalamus")

        # Basal Ganglia
        loc = loc.replace("Caudate Nucleus", "Basal Ganglia")
        loc = loc.replace("Putaminal", "Basal Ganglia")
        loc = loc.replace("BG", "Basal Ganglia")

        #Lobar
        loc = loc.replace("Temporal-Parietal-Occipital", "Lobar")
        loc = loc.replace("Temporal-parietal", "Lobar")
        loc = loc.replace("Temporal", "Lobar")
        loc = loc.replace("Frontal", "Lobar")
        loc = loc.replace("Frontoparietal", "Lobar")
        loc = loc.replace("Parietal", "Lobar")
        loc = loc.replace("parietal", "Lobar")
        loc = loc.replace("Occipital", "Lobar")
        loc = loc.replace("Lobar Lobar", "Lobar")
        #Cerebellum
        loc = loc.replace("Cerebellar", "Cerebellum")
        
        #Thalamus and Basal Ganglia are deep
        loc = loc.replace("Deep ", "")

        loc = loc.replace("Parieto-Lobar", "Lobar")
        loc = loc.replace("Lobar (L Hemisphere)", "Lobar")

        loc = loc.replace("Brainstem", "Pons")

        return loc, side
    
temp_ichloc=df.apply(lambda row: str2loc(row['CT1_ICHLOC']), axis=1, result_type='expand').rename(columns={0:'CT1_ICHLOC', 1:'CT1_ICHSIDE'})
df = pd.concat([df.drop('CT1_ICHLOC', axis = 1), temp_ichloc], axis=1)
others=['Periventricular', 'Primary IVH', 'Multiple Hemorrhages - Deep', 'Corona Radiata']
df['CT1_ICHLOC']=df['CT1_ICHLOC'].apply(lambda x: 'Others' if x in others else x)
df=pd.concat([df, pd.get_dummies(df['CT1_ICHLOC'], prefix='CT1_ICHLOC').drop(columns=['CT1_ICHLOC_', 'CT1_ICHLOC_Others'])], axis=1)
df.drop(columns=['CT1_ICHLOC'],inplace=True)
df['CT1_ICHSIDE']=df['CT1_ICHSIDE'].apply(lambda x: 1 if x=='R' else 0 if x=='L' else None)
df.loc[df['source']=='atach2','CT1_ICHVOL']=df.loc[df['source']=='atach2','CT1_ICHVOL']/1000
df.loc[df['source']=='atach2','CT1_IVHVOL']=df.loc[df['source']=='atach2','CT1_IVHVOL']/1000
df.loc[df['BP_S0']==888, 'BP_S0']=np.nan
df.loc[df['BP_D0']==888, 'BP_D0']=np.nan
df.loc[df['HIS_HYPERTENSION']==98, 'HIS_HYPERTENSION']=np.nan
df.loc[df['HIS_HYPERLIPIDEMIA']==98, 'HIS_HYPERLIPIDEMIA']=np.nan
df.loc[df['HIS_DM2']==98, 'HIS_DM2']=np.nan
df.loc[df['HIS_DM1']==98, 'HIS_DM1']=np.nan
df.loc[df['HIS_HF']==98, 'HIS_HF']=np.nan
df.loc[df['HIS_AF']==98, 'HIS_AF']=np.nan
df.loc[df['HIS_PTCA']==98, 'HIS_PTCA']=np.nan
df.loc[df['HIS_PV']==98, 'HIS_PV']=np.nan
df.loc[df['HIS_MYOCARDIAL']==98, 'HIS_MYOCARDIAL']=np.nan
df.reset_index(inplace=True, drop=True)
df_erich_format = df[df['source']=='erich']

df_new = df.copy()
df_new = df_new.drop(columns=['DEMO_MALE', 'DEMO_HISPANIC', 'DEMO_BLACK', 'DEMO_WHITE', 'DEMO_ASIAN', 'DEMO_OTHERRACE'])
df_new['RACE'] = ['Other'] * len(df_new)
df_new['GENDER'] = ['Female'] * len(df_new)
df_new['ETHNICITY'] = ['Non-Hispanic'] * len(df_new)
for i in range(len(df)):
    if df.loc[i, 'DEMO_MALE'] == 1:
        df_new.loc[i, 'GENDER'] = 'Male'
    
    if df.loc[i, 'DEMO_HISPANIC'] == 1:
        df_new.loc[i, 'ETHNICITY'] = 'Hispanic'
        
    
    if df.loc[i, 'DEMO_BLACK'] == 1:
        df_new.loc[i, 'RACE'] = 'Black'
    elif df.loc[i, 'DEMO_WHITE'] == 1:
        df_new.loc[i, 'RACE'] = 'White'
    elif df.loc[i, 'DEMO_ASIAN'] == 1:
        df_new.loc[i, 'RACE'] = 'Asian'

res = [''] * len(df_new)
for i in range(len(df_new)):
    if df_new.loc[i, 'CT1_ICHLOC_Basal Ganglia'] == 1:
        res[i] = 'Basal Ganglia'
    elif df_new.loc[i, 'CT1_ICHLOC_Lobar'] == 1:
        res[i] = 'Lobar'
    elif df_new.loc[i, 'CT1_ICHLOC_Thalamus'] == 1:
        res[i] = 'Thalamus'
    else:
        res[i] = 'Other'
df_new['ICHLOC'] = res

ids = list(range(1000))
random.Random(11).shuffle(ids)
train_idx, test_idx = ids[:int(1000*0.8)], ids[int(1000*0.8):]

var_imputed = ['TREATMENT', 'DEMO_AGE', 'OUTCOME_mRS90', 'CT1_ICHVOL', 'CT1_IVHVOL', 'GCS_TTL',
       'NIHSS_TTL', 'BP_S0', 'BP_D0', 'HIS_HYPERTENSION', 'HIS_HYPERLIPIDEMIA',
       'HIS_DM1', 'HIS_DM2', 'HIS_HF', 'HIS_AF', 'HIS_PTCA', 'HIS_PV',
       'HIS_MYOCARDIAL', 'HIS_ANTIDIABETIC', 'HIS_ANTIHYPERTENSIVES', 'LB_WBC',
       'LB_HEMOGLOBIN', 'LB_HEMATOCRIT', 'LB_PC', 'LB_APTT', 'LB_INR',
       'LB_GLUCOSE', 'LB_SODIUM', 'LB_POTASSIUM', 'LB_CHLORIDE', 'LB_CD',
       'LB_BUN', 'LB_CREATINIE', 'RACE', 'GENDER', 'ETHNICITY', 'CT1_ICHSIDE', 'ICHLOC']
cate_columns = ['TREATMENT', 'HIS_HYPERTENSION', 'HIS_HYPERLIPIDEMIA',
               'HIS_DM1', 'HIS_DM2', 'HIS_HF', 'HIS_AF', 'HIS_PTCA', 'HIS_PV',
               'HIS_MYOCARDIAL', 'HIS_ANTIDIABETIC', 'HIS_ANTIHYPERTENSIVES', 
                'RACE', 'GENDER', 'ETHNICITY', 'CT1_ICHSIDE', 'ICHLOC']
df_new[cate_columns] = df_new[cate_columns].astype('category')
trial = df_new[df_new['source'] == 'atach2'].copy()
trial.reset_index(inplace=True)
trial_train = trial.iloc[train_idx, :]
trial_test = trial.iloc[test_idx, :]
erich = df_new[df_new['source'] == 'erich'].copy()
df_train = pd.concat([trial_train, erich], axis=0)

# Imputation
SEED_VALUE=100
kernel1 = mf.ImputationKernel(df_train[var_imputed], datasets=1,random_state=SEED_VALUE)
kernel1.mice(iterations=3, boosting='gbdt', min_data_in_leaf=40)
kernel1.save_kernel('temp_data/imputation_kernel.pkl',n_threads=1)
df_imputed = kernel1.complete_data(inplace=False)
df_imputed['source'] = df_train['source']
df_test = kernel1.impute_new_data(trial_test[var_imputed], random_state = SEED_VALUE).complete_data(inplace=False)
df_test['source'] = ['atach2']*len(df_test)
df_imputed['group'] = ['train']*len(df_imputed)
df_test['group'] = ['test']*len(df_test)
df_all = pd.concat([df_imputed, df_test], axis=0)
df_all.to_csv('temp_data/imputed_all.csv')