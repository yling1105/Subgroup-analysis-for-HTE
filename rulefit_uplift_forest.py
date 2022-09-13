import re
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier, RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LassoCV,LogisticRegressionCV
from functools import reduce
import kmeans1d
from uplift_forest_customed import UpliftRfNewClassifier
#from causalml.inference.tree.causaltree import CausalForestRegressor


class RuleCondition():
    """Class for binary rule condition
    Warning: this class should not be used directly.
    """

    def __init__(self,
                 feature_index,
                 threshold,
                 operator,
                 support,
                 n_treatment,
                 n_control,
                 feature_name = None):
        self.feature_index = feature_index
        self.threshold = threshold
        self.operator = operator
        self.support = support
        self.n_treatment = n_treatment,
        self.n_control = n_control,
        self.feature_name = feature_name


    def __repr__(self):
        return self.__str__()

    def __str__(self):
        if self.feature_name:
            feature = self.feature_name
        else:
            feature = self.feature_index
        return "%s %s %s" % (feature, self.operator, self.threshold)

    def transform(self, X):
        """Transform dataset.
        Parameters
        ----------
        X: array-like matrix, shape=(n_samples, n_features)
        Returns
        -------
        X_transformed: array-like matrix, shape=(n_samples, 1)
        """
        if self.operator == "<":
            res =  1 * (X[:,self.feature_index] <= self.threshold)
        elif self.operator == ">=":
            res = 1 * (X[:,self.feature_index] > self.threshold)
        return res

    def __eq__(self, other):
        return self.__hash__() == other.__hash__()

    def __hash__(self):
        return hash((self.feature_index, self.threshold, self.operator, self.feature_name))


class Winsorizer():
    """Performs Winsorization 1->1*
    Warning: this class should not be used directly.
    """
    def __init__(self,trim_quantile=0.0):
        self.trim_quantile=trim_quantile
        self.winsor_lims=None

    def train(self,X):
        # get winsor limits
        self.winsor_lims=np.ones([2,X.shape[1]])*np.inf
        self.winsor_lims[0,:]=-np.inf
        if self.trim_quantile>0:
            for i_col in np.arange(X.shape[1]):
                lower=np.percentile(X[:,i_col],self.trim_quantile*100)
                upper=np.percentile(X[:,i_col],100-self.trim_quantile*100)
                self.winsor_lims[:,i_col]=[lower,upper]

    def trim(self,X):
        X_=X.copy()
        X_=np.where(X>self.winsor_lims[1,:],np.tile(self.winsor_lims[1,:],[X.shape[0],1]),np.where(X<self.winsor_lims[0,:],np.tile(self.winsor_lims[0,:],[X.shape[0],1]),X))
        return X_

class FriedScale():
    """Performs scaling of linear variables according to Friedman et al. 2005 Sec 5
    Each variable is first Winsorized l->l*, then standardised as 0.4 x l* / std(l*)
    Warning: this class should not be used directly.
    """
    def __init__(self, winsorizer = None):
        self.scale_multipliers=None
        self.winsorizer = winsorizer

    def train(self,X):
        # get multipliers
        if self.winsorizer != None:
            X_trimmed= self.winsorizer.trim(X)
        else:
            X_trimmed = X

        scale_multipliers=np.ones(X.shape[1])
        for i_col in np.arange(X.shape[1]):
            num_uniq_vals=len(np.unique(X[:,i_col]))
            if num_uniq_vals>2: # don't scale binary variables which are effectively already rules
                scale_multipliers[i_col]=0.4/(1.0e-12 + np.std(X_trimmed[:,i_col]))
        self.scale_multipliers=scale_multipliers

    def scale(self,X):
        if self.winsorizer != None:
            return self.winsorizer.trim(X)*self.scale_multipliers
        else:
            return X*self.scale_multipliers


class Rule():
    """Class for binary Rules from list of conditions
    Warning: this class should not be used directly.
    """
    def __init__(self,
                 rule_conditions,prediction_value):
        self.conditions = set(rule_conditions)
        self.support = min([x.support for x in rule_conditions])
        self.prediction_value=prediction_value
        self.rule_direction=None
    def transform(self, X):
        """Transform dataset.
        Parameters
        ----------
        X: array-like matrix
        Returns
        -------
        X_transformed: array-like matrix, shape=(n_samples, 1)
        """
        rule_applies = [condition.transform(X) for condition in self.conditions]
        return reduce(lambda x,y: x * y, rule_applies)

    def __str__(self):
        return  " & ".join([x.__str__() for x in self.conditions])

    def __repr__(self):
        return self.__str__()

    def __hash__(self):
        return sum([condition.__hash__() for condition in self.conditions])

    def __eq__(self, other):
        return self.__hash__() == other.__hash__()



def extract_rules_from_uplift_tree(tree, alpha = 0.05, feature_names=None):
    rules = set()
    ttl_sample_size = int(re.findall(r"\d+", tree.summary['group_size'])[0]) + int(re.findall(r"\d+", tree.summary['group_size'])[1])
    def traverse(node, sample_size=ttl_sample_size, alpha = alpha , conditions=[], branch=None, parent_col=None, parent_val=None):
        cnt = 1

        node_col = node.col
        node_col = feature_names[node_col]
        node_val = node.value
        node_sentence = None
        feature_name = parent_col
        if branch != None:                             
            if branch == 'l':
                operator = '>='
            else:
                operator = '<'
            threshold = parent_val
            ctrl = int(re.findall(r"\d+", node.summary['group_size'])[0])
            treatment = int(re.findall(r"\d+", node.summary['group_size'])[1])
            support = (ctrl + treatment) / ttl_sample_size
            feature_index = feature_names.index(feature_name)
            rule_condition = RuleCondition(feature_index=feature_index,
                                           threshold=threshold,
                                           operator=operator,
                                       support = support,
                                       n_treatment = treatment,
                                       n_control = ctrl,                                   
                                       feature_name=feature_name)  
            new_conditions = conditions + [rule_condition]
        else:
            new_conditions = []
        if node.trueBranch == None and node.falseBranch == None:
            if node.upliftScore[1] < alpha:
                print(cnt)
                new_rule = Rule(new_conditions, node.matchScore)
                rules.update([new_rule])
                #print(rules)
                cnt += 1
        else:
            if node.trueBranch != None:
                traverse(node.trueBranch, ttl_sample_size,alpha,
                         conditions=new_conditions, branch='l',
                         parent_col=node_col,
                         parent_val=node_val)
            if node.falseBranch != None:
                traverse(node.falseBranch, ttl_sample_size,alpha,
                         conditions = new_conditions, branch = 'r',
                         parent_col = node_col, 
                         parent_val = node_val)
    
    traverse(tree, ttl_sample_size, alpha)
    return rules            



class RuleEnsemble():
    """Ensemble of binary decision rules
    This class implements an ensemble of decision rules that extracts rules from
    an ensemble of decision trees.
    Parameters
    ----------
    tree_list: List or array of DecisionTreeClassifier or DecisionTreeRegressor
        Trees from which the rules are created
    feature_names: List of strings, optional (default=None)
        Names of the features
    Attributes
    ----------
    rules: List of Rule
        The ensemble of rules extracted from the trees
    """
    def __init__(self,
                 tree_list,
                 feature_names=None):
        self.tree_list = tree_list
        self.feature_names = feature_names
        self.rules = set()
        ## TODO: Move this out of __init__
        self._extract_rules()
        self.rules=list(self.rules)
        self.rule_clusters = {}

    def _extract_rules(self):
        """Recursively extract rules from each tree in the ensemble
        """
        for tree in self.tree_list:
            rules = extract_rules_from_uplift_tree(tree[0].fitted_uplift_tree,feature_names=self.feature_names)
            self.rules.update(rules)

    def filter_rules(self, func):
        self.rules = filter(lambda x: func(x), self.rules)

    def filter_short_rules(self, k):
        self.filter_rules(lambda x: len(x.conditions) > k)

    def transform(self, X,coefs=None):
        """Transform dataset.
        Parameters
        ----------
        X:      array-like matrix, shape=(n_samples, n_features)
        coefs:  (optional) if supplied, this makes the prediction
                slightly more efficient by setting rules with zero
                coefficients to zero without calling Rule.transform().
        Returns
        -------
        X_transformed: array-like matrix, shape=(n_samples, n_out)
            Transformed dataset. Each column represents one rule.
        """
        rule_list=list(self.rules)
        if coefs is None:
            return np.array([rule.transform(X) for rule in rule_list]).T
        else: # else use the coefs to filter the rules we bother to interpret
            res= np.array([rule_list[i_rule].transform(X) for i_rule in np.arange(len(rule_list)) if coefs[i_rule]!=0]).T
            res_=np.zeros([X.shape[0],len(rule_list)])
            res_[:,coefs!=0]=res
            return res_
    def __str__(self):
        return (map(lambda x: x.__str__(), self.rules)).__str__()
    
    def emseble_compression(self, k = 3):
        rule_list = list(self.rules)
        res = []

        for i in range(len(rule_list)):
            single_conditions = rule_list[i].split(' & ')
            var_split = re.split(r'>=|<', single_conditions)
            direction = re.findall(r'>=|<', single_conditions)
            if direction == '>=':
                var_split.append(1)
            else:
                var_split.append(0)
            var_split.append(i)
            res_temp = pd.DataFrame(var_split, columns=['feature', 'split_val', 'direction', 'rule_idx'])
            res[i] = res_temp

        out = pd.concat(res)
        f_space = list(out['feature'].unique())
        new_res = []
        
        for f in f_space:
            df_temp = out[out['feature'] == f]
            df_temp['cluster_id'], cent = self.kmeans_clustering(df_temp, 'split_val', 2)
            new_res.append(df_temp)
            self.rule_clusters[f] = cent
        
        clustered_out = pd.concat(new_res)
        

    def kmeans_clustering(self, df, f_name,k):
        n_un = len(list(df[f_name].unique()))
        if n_un > 10:
            clusters, cent = kmeans1d.cluster(df[f_name], k)
            return pd.Series(clusters, index=df.index), cent
        else:
            return df['split_val'], list(df[f_name].unique())
        return clusters
        
        







class RuleFit(BaseEstimator, TransformerMixin):
    """Rulefit class
    Parameters
    ----------
        tree_size:      Number of terminal nodes in generated trees. If exp_rand_tree_size=True,
                        this will be the mean number of terminal nodes.
        sample_fract:   fraction of randomly chosen training observations used to produce each tree.
                        FP 2004 (Sec. 2)
        max_rules:      approximate total number of rules generated for fitting. Note that actual
                        number of rules will usually be lower than this due to duplicates.
        memory_par:     scale multiplier (shrinkage factor) applied to each new tree when
                        sequentially induced. FP 2004 (Sec. 2)
        lin_standardise: If True, the linear terms will be standardised as per Friedman Sec 3.2
                        by multiplying the winsorised variable by 0.4/stdev.
        lin_trim_quantile: If lin_standardise is True, this quantile will be used to trim linear
                        terms before standardisation.
        exp_rand_tree_size: If True, each boosted tree will have a different maximum number of
                        terminal nodes based on an exponential distribution about tree_size.
                        (Friedman Sec 3.3)
        model_type:     'r': rules only; 'l': linear terms only; 'rl': both rules and linear terms
        random_state:   Integer to initialise random objects and provide repeatability.
        tree_generator: Optional: this object will be used as provided to generate the rules.
                        This will override almost all the other properties above.
                        Must be GradientBoostingRegressor or GradientBoostingClassifier, optional (default=None)
        tol:            The tolerance for the optimization for LassoCV or LogisticRegressionCV:
                        if the updates are smaller than `tol`, the optimization code checks the dual
                        gap for optimality and continues until it is smaller than `tol`.
        max_iter:       The maximum number of iterations for LassoCV or LogisticRegressionCV.
        n_jobs:         Number of CPUs to use during the cross validation in LassoCV or
                        LogisticRegressionCV. None means 1 unless in a joblib.parallel_backend
                        context. -1 means using all processors.
    Attributes
    ----------
    rule_ensemble: RuleEnsemble
        The rule ensemble
    feature_names: list of strings, optional (default=None)
        The names of the features (columns)
    """
    def __init__(
            self,
            tree_depth=5,
            min_samples_leaf=50,
            min_samples_treatment=25,
            tree_eval_func = 'KL',
            sample_fract='default',
            max_rules=2000,
            memory_par=0.01,
            tree_generator=None,  
            lin_trim_quantile=0.025,
            lin_standardise=True,
            exp_rand_tree_size=True,
            model_type='rl',
            Cs=None,
            cv=3,
            tol=0.0001,
            n_jobs=None,
            random_state=None):
        # Parameters for random forest classifier  
        self.tree_depth=tree_depth
        self.min_samples_leaf = min_samples_leaf
        self.min_samples_treatment = min_samples_treatment
        self.tree_eval_func = tree_eval_func
        self.tree_generator = tree_generator
        self.lin_trim_quantile=lin_trim_quantile
        self.lin_standardise=lin_standardise
        self.winsorizer=Winsorizer(trim_quantile=lin_trim_quantile)
        self.friedscale=FriedScale(self.winsorizer)
        self.stddev = None
        self.mean = None
        self.exp_rand_tree_size=exp_rand_tree_size
        self.max_rules=max_rules
        self.sample_fract=sample_fract
        self.max_rules=max_rules
        self.memory_par=memory_par
        self.random_state=random_state
        self.model_type=model_type
        self.cv=cv
        self.tol=tol
        # LassoCV default max_iter is 1000 while LogisticRegressionCV 100.
        self.max_iter=1000 if 'regress' else 1000
        self.n_jobs=n_jobs
        self.Cs=Cs


    def delete_duplicates(self, ensemble_rules, rule_matrix, alpha= 0.01):
        '''
        Remove the redundant and colinear rules from the ensembling

        ensemble_rules: RuleEnsemble class
        rule_matrix: Transformed rule matrix
        alpha: threshold of covariance to be removed
        '''
        
        ncol = rule_matrix.shape[1]
        rule_matrix = rule_matrix.T

        cov_x = np.cov(rule_matrix)
        idx = np.where(((cov_x <=1+alpha)&(cov_x >= 1-alpha))|((cov_x <= -1+alpha)&(cov_x >= -1-alpha)))

        # determine the index of rule to be removed
        remove_lst = []
        for i in range(len(idx[0])):
            x = idx[0][i]
            y = idx[1][i]

            if x > y:
                if y not in remove_lst:
                    remove_lst.append(y)
        return self

        

    def fit(self, X, treatment, y=None, feature_names=None):
        """Fit and estimate linear combination of rule ensemble
        """
        ## Enumerate features if feature names not provided
        N=X.shape[0]
        if feature_names is None:
            self.feature_names = ['feature_' + str(x) for x in range(0, X.shape[1])]
        else:
            self.feature_names=feature_names
        if 'r' in self.model_type:
            ## initialise tree generator
            if self.tree_generator is None:
                self.sample_fract_=min(0.5,(100+6*np.sqrt(N))/N)
                self.tree_generator = UpliftRfNewClassifier(control_name='control', n_estimators=100,
                                                            max_features=30,
                                                            max_depth=self.tree_depth, # reduce
                                                            min_samples_leaf=self.min_samples_leaf,
                                                            min_samples_treatment=self.min_samples_treatment,
                                                            n_reg=10,
                                                            random_state=self.random_state,
                                                            evaluationFunction=self.tree_eval_func)
                ## fit tree generator
                self.tree_generator.fit(X, treatment, y)
            tree_list = [[x] for x in self.tree_generator.uplift_forest]

            ## extract rules
            self.rule_ensemble = RuleEnsemble(tree_list = tree_list,
                                              feature_names=self.feature_names)

            ## concatenate original features and rules
            X_rules = self.rule_ensemble.transform(X)
        

        if 'l' in self.model_type:
            ## standard deviation and mean of winsorized features
            self.winsorizer.train(X)
            winsorized_X = self.winsorizer.trim(X)
            self.stddev = np.std(winsorized_X, axis = 0)
            self.mean = np.mean(winsorized_X, axis = 0)

            if self.lin_standardise:
                self.friedscale.train(X)
                X_regn=self.friedscale.scale(X)
            else:
                X_regn=X.copy()

        ## Compile Training data
        X_concat=np.zeros([X.shape[0],0])
        if 'l' in self.model_type:
            X_concat = np.concatenate((X_concat,X_regn), axis=1)
        if 'r' in self.model_type:
            if X_rules.shape[0] >0:
                X_concat = np.concatenate((X_concat, X_rules), axis=1)

        """
        In HTE modeling, the response variable is estimated HTE score. 
        Then it will always be a regression model.
        """
        if self.Cs is None: # use defaultshasattr(self.Cs, "__len__"):
            n_alphas= 100
            alphas=None
        elif hasattr(self.Cs, "__len__"):
            n_alphas= None
            alphas=1./self.Cs
        else:
            n_alphas= self.Cs
            alphas=None
        self.lscv = LassoCV(
            n_alphas=n_alphas, alphas=alphas, cv=self.cv,
            max_iter=self.max_iter, tol=self.tol,
            n_jobs=self.n_jobs,
            random_state=self.random_state)
        hte = self.tree_generator.predict(X)
        self.lscv.fit(X_concat, hte.ravel())
        self.coef_=self.lscv.coef_
        self.intercept_=self.lscv.intercept_

        return self

    def predict(self, X):
        """Predict outcome for X
        """
        X_concat=np.zeros([X.shape[0],0])
        if 'l' in self.model_type:
            if self.lin_standardise:
                X_concat = np.concatenate((X_concat,self.friedscale.scale(X)), axis=1)
            else:
                X_concat = np.concatenate((X_concat,X), axis=1)
        if 'r' in self.model_type:
            rule_coefs=self.coef_[-len(self.rule_ensemble.rules):]
            if len(rule_coefs)>0:
                X_rules = self.rule_ensemble.transform(X,coefs=rule_coefs)
                if X_rules.shape[0] >0:
                    X_concat = np.concatenate((X_concat, X_rules), axis=1)
        return self.lscv.predict(X_concat)

    def predict_proba(self, X):
        """Predict outcome probability for X, if model type supports probability prediction method
        """

        if 'predict_proba' not in dir(self.lscv):

            error_message = '''
            Probability prediction using predict_proba not available for
            model type {lscv}
            '''.format(lscv=self.lscv)
            raise ValueError(error_message)

        X_concat=np.zeros([X.shape[0],0])
        if 'l' in self.model_type:
            if self.lin_standardise:
                X_concat = np.concatenate((X_concat,self.friedscale.scale(X)), axis=1)
            else:
                X_concat = np.concatenate((X_concat,X), axis=1)
        if 'r' in self.model_type:
            rule_coefs=self.coef_[-len(self.rule_ensemble.rules):]
            if len(rule_coefs)>0:
                X_rules = self.rule_ensemble.transform(X,coefs=rule_coefs)
                if X_rules.shape[0] >0:
                    X_concat = np.concatenate((X_concat, X_rules), axis=1)
        return self.lscv.predict_proba(X_concat)

    def transform(self, X=None, y=None):
        """Transform dataset.
        Parameters
        ----------
        X : array-like matrix, shape=(n_samples, n_features)
            Input data to be transformed. Use ``dtype=np.float32`` for maximum
            efficiency.
        Returns
        -------
        X_transformed: matrix, shape=(n_samples, n_out)
            Transformed data set
        """
        return self.rule_ensemble.transform(X)

    def get_rules(self, exclude_zero_coef=False, subregion=None):
        """Return the estimated rules
        Parameters
        ----------
        exclude_zero_coef: If True (default), returns only the rules with an estimated
                           coefficient not equalt to  zero.
        subregion: If None (default) returns global importances (FP 2004 eq. 28/29), else returns importance over
                           subregion of inputs (FP 2004 eq. 30/31/32).
        Returns
        -------
        rules: pandas.DataFrame with the rules. Column 'rule' describes the rule, 'coef' holds
               the coefficients and 'support' the support of the rule in the training
               data set (X)
        """

        n_features= len(self.coef_) - len(self.rule_ensemble.rules)
        rule_ensemble = list(self.rule_ensemble.rules)
        output_rules = []
        ## Add coefficients for linear effects
        for i in range(0, n_features):
            if self.lin_standardise:
                coef=self.coef_[i]*self.friedscale.scale_multipliers[i]
            else:
                coef=self.coef_[i]
            if subregion is None:
                importance = abs(coef)*self.stddev[i]
            else:
                subregion = np.array(subregion)
                importance = sum(abs(coef)* abs([ x[i] for x in self.winsorizer.trim(subregion) ] - self.mean[i]))/len(subregion)
            output_rules += [(self.feature_names[i], 'linear',coef, 1, importance)]

        ## Add rules
        for i in range(0, len(self.rule_ensemble.rules)):
            rule = rule_ensemble[i]
            coef=self.coef_[i + n_features]

            if subregion is None:
                importance = abs(coef)*(rule.support * (1-rule.support))**(1/2)
            else:
                rkx = rule.transform(subregion)
                importance = sum(abs(coef) * abs(rkx - rule.support))/len(subregion)

            output_rules += [(rule.__str__(), 'rule', coef,  rule.support, importance)]
        rules = pd.DataFrame(output_rules, columns=["rule", "type","coef", "support", "importance"])
        if exclude_zero_coef:
            rules = rules.loc[rules.coef != 0]
        return rules
    
    def get_feature_importance(self, exclude_zero_coef=False, subregion=None, scaled=False):
        """
        Returns feature importance for input features to RuleFit model.
        Parameters:
        -----------
            exclude_zero_coef: If True, returns only the rules with an estimated
                           coefficient not equalt to  zero.
            subregion: If None (default) returns global importances (FP 2004 eq. 28/29), else returns importance over
                           subregion of inputs (FP 2004 eq. 30/31/32).
                           
            scaled: If True, will scale the importances to have a max of 100.
            
        Returns:
        --------
            return_df (pandas DataFrame): DataFrame for feature names and feature importances (FP 2004 eq. 35)
        """
        
        def find_mk(rule:str):
            """
            Finds the number of features in a given rule from the get_rules method.
            Parameters:
            -----------
                rule (str): 
                
            Returns:
            --------
                var_count (int): 
            """
            
            ## Count the number of features found in a rule
            feature_count = 0
            for feature in self.feature_names:
                if feature in rule:
                    feature_count += 1
            return(feature_count)
        
        feature_set = self.feature_names
        rules = self.get_rules(exclude_zero_coef, subregion)
        
        # Return an array of counts for features found in rules
        features_in_rule = rules.rule.apply(lambda x: find_mk(x))

        feature_imp = list()
        for feature in feature_set:
            # Rules where feature is found
            feature_rk = rules.rule.apply(lambda x: feature in x)
            # Linear importance array for feature
            linear_imp = rules[(rules.type=='linear')&(rules.rule==feature)].importance.values
            # Rule importance array
            rule_imp = rules[rules.type!='linear'].importance[feature_rk]
            # Total count of features in each rule feature is found
            mk_array = features_in_rule[feature_rk]
            feature_imp.append(float(linear_imp + (rule_imp/mk_array).sum())) # (FP 2004 eq. 35)

        # Scaled output
        if scaled:
            feature_imp = 100*(feature_imp/np.array(feature_imp).max())
           
        return_df = pd.DataFrame({'feature':self.feature_names, 'importance':feature_imp})
        return(return_df)