from collections import defaultdict
import cython
from joblib import Parallel, delayed
import multiprocessing as mp
import numpy as np
from packaging import version
import pandas as pd
import scipy.stats as stats
import sklearn
from sklearn.utils import check_array, check_random_state, check_X_y
from typing import List

from causalml.inference.tree import UpliftTreeClassifier, UpliftRandomForestClassifier
from causalml.inference.tree import uplift_tree_string, uplift_tree_plot

if version.parse(sklearn.__version__) >= version.parse('0.22.0'):
    from sklearn.utils._testing import ignore_warnings
else:
    from sklearn.utils.testing import ignore_warnings

MAX_INT = np.iinfo(np.int32).max


class UpliftTreeNew(UpliftTreeClassifier):
    def __init__(self, control_name, max_features=None, max_depth=4, min_samples_leaf=100,
                 min_samples_treatment=20, n_reg=100, evaluationFunction='KL',
                 normalization=True, random_state=None):
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.min_samples_treatment = min_samples_treatment
        self.n_reg = n_reg
        self.max_features = max_features
        if evaluationFunction == 'KL':
            self.evaluationFunction = self.evaluate_KL
        elif evaluationFunction == 'ED':
            self.evaluationFunction = self.evaluate_ED
        elif evaluationFunction == 'Chi':
            self.evaluationFunction = self.evaluate_Chi
        elif evaluationFunction == 'DDP':
            self.evaluationFunction = self.evaluate_DDP
        else:
            self.evaluationFunction = self.evaluate_CTS
        self.fitted_uplift_tree = None

        assert control_name is not None and isinstance(control_name, str), \
            f"control_group should be string but {control_name} is passed"
        self.control_name = control_name
        self.classes_ = [self.control_name]
        self.n_class = 1

        self.normalization = normalization
        self.random_state = random_state
        
    def fill(self, X, treatment, y):
        """ Fill the data into an existing tree.
        This is a higher-level function to transform the original data inputs
        into lower level data inputs (list of list and tree).
        Args
        ----
        X : ndarray, shape = [num_samples, num_features]
            An ndarray of the covariates used to train the uplift model.
        treatment : array-like, shape = [num_samples]
            An array containing the treatment group for each unit.
        y : array-like, shape = [num_samples]
            An array containing the outcome of interest for each unit.
        Returns
        -------
        self : object
        """

        X, y = check_X_y(X, y)
        treatment = np.asarray(treatment)
        assert len(y) == len(treatment), 'Data length must be equal for X, treatment, and y.'

        # Get treatment group keys. self.classes_[0] is reserved for the control group.
        treatment_idx = np.zeros_like(treatment, dtype=int)
        for i, tr in enumerate(self.classes_[1:], 1):
            treatment_idx[treatment == tr] = i
        
        #print(treatment_idx)

        self.fillTree(X, treatment_idx, y, tree=self.fitted_uplift_tree)
        return self

    def fillTree(self, X, treatment_idx, y, tree):
        """ Fill the data into an existing tree.
        This is a lower-level function to execute on the tree filling task.
        Args
        ----
        X : ndarray, shape = [num_samples, num_features]
            An ndarray of the covariates used to train the uplift model.
        treatment_idx : array-like, shape = [num_samples]
            An array containing the treatment group index for each unit.
        y : array-like, shape = [num_samples]
            An array containing the outcome of interest for each unit.
        tree : object
            object of DecisionTree class
        Returns
        -------
        self : object
        """
        # Current Node Summary for Validation Data Set
        currentNodeSummary = self.tree_node_summary(treatment_idx, y,
                                                    min_samples_treatment=0,
                                                    n_reg=0,
                                                    parentNodeSummary=None)
        tree.nodeSummary = currentNodeSummary

        # Divide sets for child nodes
        if tree.trueBranch or tree.falseBranch:
            X_l, X_r, w_l, w_r, y_l, y_r = self.divideSet(X, treatment_idx, y, tree.col, tree.value)

            # recursive call for each branch
            if tree.trueBranch is not None:
                self.fillTree(X_l, w_l, y_l, tree.trueBranch)
            if tree.falseBranch is not None:
                self.fillTree(X_r, w_r, y_r, tree.falseBranch)

        # Update Information

        # matchScore
        matchScore = (currentNodeSummary[1][0] - currentNodeSummary[0][0])
        tree.matchScore = round(matchScore, 4)
        tree.summary['matchScore'] = round(matchScore, 4)

        # Samples, Group_size
        tree.summary['samples'] = len(y)
        tree.summary['group_size'] = ''
        for treatment_group, summary in zip(self.classes_, currentNodeSummary):
            tree.summary['group_size'] += ' ' + treatment_group + ': ' + str(summary[1])
        # classProb
        if tree.results is not None:
            tree.results = self.uplift_classification_results(treatment_idx, y)
        return self    
    
    def group_uniqueCounts(self, treatment_idx, y):
        '''
        Count sample size by experiment group.
        Args
        ----
        treatment_idx : array-like, shape = [num_samples]
            An array containing the treatment group index for each unit.
        y : array-like, shape = [num_samples]
            An array containing the outcome of interest for each unit.
        Returns
        -------
        results : list of list
            The negative and positive outcome sample sizes for each of the control and treatment groups.
        '''
        results = []
        for i in range(self.n_class):
            #print(treatment_idx)
            filt = treatment_idx == i
            n_pos = y[filt].sum()

            # [N(Y = 0, T = 1), N(Y = 1, T = 1)]
            results.append([filt.sum() - n_pos, n_pos])

        return results

class UpliftRfNewClassifier(UpliftRandomForestClassifier):
    """ Uplift Random Forest for Classification Task.
    Parameters
    ----------
    n_estimators : integer, optional (default=10)
        The number of trees in the uplift random forest.
    evaluationFunction : string
        Choose from one of the models: 'KL', 'ED', 'Chi', 'CTS', 'DDP'.
    max_features: int, optional (default=10)
        The number of features to consider when looking for the best split.
    random_state: int, RandomState instance or None (default=None)
        A random seed or `np.random.RandomState` to control randomness in building the trees and forest.
    max_depth: int, optional (default=5)
        The maximum depth of the tree.
    min_samples_leaf: int, optional (default=100)
        The minimum number of samples required to be split at a leaf node.
    min_samples_treatment: int, optional (default=10)
        The minimum number of samples required of the experiment group to be split at a leaf node.
    n_reg: int, optional (default=10)
        The regularization parameter defined in Rzepakowski et al. 2012, the
        weight (in terms of sample size) of the parent node influence on the
        child node, only effective for 'KL', 'ED', 'Chi', 'CTS' methods.
    control_name: string
        The name of the control group (other experiment groups will be regarded as treatment groups)
    normalization: boolean, optional (default=True)
        The normalization factor defined in Rzepakowski et al. 2012,
        correcting for tests with large number of splits and imbalanced
        treatment and control splits
    n_jobs: int, optional (default=-1)
        The parallelization parameter to define how many parallel jobs need to be created.
        This is passed on to joblib library for parallelizing uplift-tree creation and prediction.
    joblib_prefer: str, optional (default="threads")
        The preferred backend for joblib (passed as `prefer` to joblib.Parallel). See the joblib
        documentation for valid values.
    Outputs
    ----------
    df_res: pandas dataframe
        A user-level results dataframe containing the estimated individual treatment effect.
    """
    def __init__(self,
                 control_name,
                 n_estimators=10,
                 max_features=30,
                 random_state=None,
                 max_depth=4,
                 min_samples_leaf=100,
                 min_samples_treatment=10,
                 n_reg=10,
                 evaluationFunction='KL',
                 normalization=True,
                 n_jobs=-1,
                 joblib_prefer: str = "threads"):

        """
        Initialize the UpliftRandomForestClassifier class.
        """
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.random_state = random_state
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.min_samples_treatment = min_samples_treatment
        self.n_reg = n_reg
        self.evaluationFunction = evaluationFunction
        self.control_name = control_name
        self.normalization = normalization
        self.n_jobs = n_jobs
        self.joblib_prefer = joblib_prefer

        assert control_name is not None and isinstance(control_name, str), \
            f"control_group should be string but {control_name} is passed"
        self.control_name = control_name
        self.classes_ = [control_name]
        self.n_class = 1

        if self.n_jobs == -1:
            self.n_jobs = mp.cpu_count()

    def fit(self, X, treatment, y):
        """
        Fit the UpliftRandomForestClassifier.
        Args
        ----
        X : ndarray, shape = [num_samples, num_features]
            An ndarray of the covariates used to train the uplift model.
        treatment : array-like, shape = [num_samples]
            An array containing the treatment group for each unit.
        y : array-like, shape = [num_samples]
            An array containing the outcome of interest for each unit.
        """
        random_state = check_random_state(self.random_state)

        # Create forest
        self.uplift_forest = [
            UpliftTreeNew(
                max_features=self.max_features, max_depth=self.max_depth,
                min_samples_leaf=self.min_samples_leaf,
                min_samples_treatment=self.min_samples_treatment,
                n_reg=self.n_reg,
                evaluationFunction=self.evaluationFunction,
                control_name=self.control_name,
                normalization=self.normalization,
                random_state=random_state.randint(MAX_INT))
            for _ in range(self.n_estimators)
        ]

        # Get treatment group keys. self.classes_[0] is reserved for the control group.
        treatment_groups = sorted([x for x in list(set(treatment)) if x != self.control_name])
        self.classes_ = [self.control_name]
        for tr in treatment_groups:
            self.classes_.append(tr)
        self.n_class = len(self.classes_)

        self.uplift_forest = (
            Parallel(n_jobs=self.n_jobs, prefer=self.joblib_prefer)
            (delayed(self.bootstrap)(X, treatment, y, tree) for tree in self.uplift_forest)
        )

        all_importances = [tree.feature_importances_ for tree in self.uplift_forest]
        self.feature_importances_ = np.mean(all_importances, axis=0)
        self.feature_importances_ /= self.feature_importances_.sum()  # normalize to add to 1

    @staticmethod
    def bootstrap(X, treatment, y, tree):
        random_state = check_random_state(tree.random_state)
        bt_index = random_state.choice(len(X), len(X))
        x_train_bt = X[bt_index]
        y_train_bt = y[bt_index]
        treatment_train_bt = treatment[bt_index]
        tree.fit(X=x_train_bt, treatment=treatment_train_bt, y=y_train_bt)
        return tree

    @ignore_warnings(category=FutureWarning)
    def predict(self, X, full_output=False):
        '''
        Returns the recommended treatment group and predicted optimal
        probability conditional on using the recommended treatment group.
        Args
        ----
        X : ndarray, shape = [num_samples, num_features]
            An ndarray of the covariates used to train the uplift model.
        full_output : bool, optional (default=False)
            Whether the UpliftTree algorithm returns upliftScores, pred_nodes
            alongside the recommended treatment group and p_hat in the treatment group.
        Returns
        -------
        y_pred_list : ndarray, shape = (num_samples, num_treatments])
            An ndarray containing the predicted treatment effect of each treatment group for each sample
        df_res : DataFrame, shape = [num_samples, (num_treatments * 2 + 3)]
            If `full_output` is `True`, a DataFrame containing the predicted outcome of each treatment and
            control group, the treatment effect of each treatment group, the treatment group with the
            highest treatment effect, and the maximum treatment effect for each sample.
        '''
        # Make predictions with all trees and take the average

        if self.n_jobs != 1:
            y_pred_ensemble = sum(
                Parallel(n_jobs=self.n_jobs, prefer=self.joblib_prefer)
                (delayed(tree.predict)(X=X) for tree in self.uplift_forest)
            ) / len(self.uplift_forest)
        else:
            y_pred_ensemble = sum([tree.predict(X=X) for tree in self.uplift_forest]) / len(self.uplift_forest)

        # Summarize results into dataframe
        df_res = pd.DataFrame(y_pred_ensemble, columns=self.classes_)
        df_res['recommended_treatment'] = df_res.apply(np.argmax, axis=1)

        # Calculate delta
        delta_cols = [f'delta_{treatment_group}' for treatment_group in self.classes_[1:]]
        for i_tr in range(1, self.n_class):
            treatment_group = self.classes_[i_tr]
            df_res[f'delta_{treatment_group}'] = df_res[treatment_group] - df_res[self.control_name]

        df_res['max_delta'] = df_res[delta_cols].max(axis=1)

        if full_output:
            return df_res
        else:
            return df_res[delta_cols].values