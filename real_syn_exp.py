from rulefit_uplift_forest import CausalRuleEnsembling
from sklearn.model_selection import KFold
import pandas as pd
import numpy as np
import random
import sdv
from sdv.sampling import Condition
import torch
import pickle
from causalml.match import NearestNeighborMatch, create_table_one
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score as auc
from sklearn import preprocessing
from causalml.propensity import ElasticNetPropensityModel
from sdmetrics.reports.single_table import QualityReport
import seaborn as sns
import matplotlib.pyplot as plt

directory = 'temp_data/'
df = pd.read_csv(directory + 'imputed_all.csv')
df.drop(columns=['Unnamed: 0'], inplace =True)
obs = pd.read_csv(directory + 'matched_erich.csv')
obs.drop(columns=['Unnamed: 0', 'Unnamed: 0.1', 'score', 'source', 'group'], inplace=True)

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

df_new = df.copy()
df_new.drop(columns=['source', 'group'])
df_new = pd.get_dummies(df_new)
df_new = transform_var(df_new)
df_new.shape
df_new['source'] = df['source']
df_new['group'] = df['group']
df_train = df_new[df_new['group'] == 'train'].copy()
df_test = df_new[df_new['group'] == 'test'].copy()
trial_train = df_new[(df_new['source'] == 'atach2')&(df_new['group']=='train')].copy()
trial_train.reset_index(inplace=True, drop=True)
obs = pd.get_dummies(obs)
obs = transform_var(obs)

from sdv.tabular import TVAE
from sdv.tabular import CTGAN