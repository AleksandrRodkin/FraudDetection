import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_curve
from sklearn.model_selection import BaseCrossValidator
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, TargetEncoder, StandardScaler, RobustScaler
from sklearn.utils.validation import check_is_fitted


class TimeSeriesMonthSplit(BaseCrossValidator):
    """
    Custom class for splits for cross validation by the value of the Month column
    """

    def __init__(self, ):
        pass

    def get_n_splits(self, X, y=None, groups=None):
        unique_months = sorted(X['month'].unique())
        return len(unique_months) - 1

    def split(self, X, y=None, groups=None):
        cv = []

        unique_months = sorted(X['month'].unique())
        for i in range(len(unique_months) - 1):
            train_months = unique_months[:i + 1]
            val_month = unique_months[i + 1]

            cv.append([X[X['month'].isin(train_months)].index, X[X['month'] == val_month].index])

        return cv


class NoneHandler(BaseEstimator, TransformerMixin):
    """
    Custom class for handling missing values
    """

    def __init__(self):
        self.base_mode = 'mode'
        self.base_mode_numeric = 'median'
        self.max_missing_prop = 0.15
        self.handler_params = {'prev_address_months_count': 'median',
                               'current_address_months_count': 0,
                               'bank_months_count': 0,
                               'device_distinct_emails_8w': 0,
                               'velocity_6h': None,  # distribution covers -1
                               'credit_risk_score': None,  # distribution covers -1
                               }
        self.num_features = ['session_length_in_minutes', 'bank_branch_count_8w', 'velocity_24h',
                             'current_address_months_count',
                             'credit_risk_score', 'velocity_6h', 'name_email_similarity', 'days_since_request',
                             'prev_address_months_count', 'intended_balcon_amount', 'bank_months_count',
                             'device_fraud_count',
                             'date_of_birth_distinct_emails_4w', 'velocity_4w', 'zip_count_4w']
        self.computed_values_ = dict()
        self.missing_indicators_ = dict()

    def _compute_value(self, series, mode):
        "Compute the values to replace missing values"
        clean_values = series.loc[series != -1].dropna()
        if mode == "median":
            return clean_values.median()
        elif mode == "mean":
            return clean_values.mean()
        elif mode == "mode":
            return clean_values.mode().iloc[0] if not clean_values.empty else np.nan
        else:
            raise ValueError(f"{mode} not in ['median', 'mean', 'mode']")

    def fit(self, X, y=None):
        X = X.copy()
        if 'intended_balcon_amount' in X.columns:
            X.loc[X.intended_balcon_amount < 0, 'intended_balcon_amount'] = -1

        for col in X.columns:
            self.missing_indicators_[col] = ((X[col] == -1) | (X[col].isna())).mean() > self.max_missing_prop

            if col in self.handler_params:
                rule = self.handler_params[col]
                if isinstance(rule, str):  # median/mean/mode
                    self.computed_values_[col] = self._compute_value(X[col], rule)
                elif rule is not None:  # fixed value
                    self.computed_values_[col] = rule
                # if None → skip
            else:
                # use base_modes
                if col in self.num_features:
                    self.computed_values_[col] = self._compute_value(X[col], self.base_mode_numeric)
                else:
                    self.computed_values_[col] = self._compute_value(X[col], self.base_mode)
        self._is_fitted = True
        return self

    def transform(self, X, y=None):
        check_is_fitted(self)
        X = X.copy()
        if 'intended_balcon_amount' in X.columns:
            X.loc[X.intended_balcon_amount < 0, 'intended_balcon_amount'] = -1
        for col in X.columns:
            if col in self.handler_params and self.handler_params[col] is None:
                continue
            elif col not in self.computed_values_:
                continue
            else:
                replacement = self.computed_values_[col]
                missing_mask = (X[col] == -1) | (X[col].isna())

                if self.missing_indicators_.get(col, False):
                    X[f"{col}_miss"] = missing_mask.astype(int)
                X.loc[missing_mask, col] = replacement
        return X

    def __sklearn_is_fitted__(self):
        """
        Check fitted status and return a Boolean value.
        """
        return hasattr(self, "_is_fitted") and self._is_fitted


class DataHandler(BaseEstimator, TransformerMixin):
    """
    Custom class for dataset processing, like in the EDA-notebook
    """

    def __init__(self):
        pass

    def fit(self, X, y=None):
        self._is_fitted = True
        return self

    def transform(self, X, y=None):
        check_is_fitted(self)
        X = X.copy()

        if 'device_distinct_emails' in X.columns:
            X.loc[X.device_distinct_emails_8w != 1, 'device_distinct_emails_8w'] = 2

        if 'device_os' in X.columns:
            X.loc[X.device_os.isin(['x11']), 'device_os'] = 'other'

        if 'housing_status' in X.columns:
            X.loc[X.housing_status.isin(['BD', 'BF', 'BG']), 'housing_status'] = 'other'

        if 'employment_status' in X.columns:
            X.loc[X.employment_status.isin(['CD', 'CE', 'CF', 'CG']), 'employment_status'] = 'other'

        if 'name_email_similarity' in X.columns and 'has_other_cards' in X.columns:
            X['EDA'] = np.zeros(len(X))
            X.loc[(X.name_email_similarity < 0.35) & (X.has_other_cards == 1), 'EDA'] = 1

        if 'proposed_credit_limit' in X.columns:
            X.loc[X.proposed_credit_limit < 1000, 'proposed_credit_limit'] = 0
            X.loc[(X.proposed_credit_limit >= 1000) & (X.proposed_credit_limit < 1900), 'proposed_credit_limit'] = 1
            X.loc[X.proposed_credit_limit >= 1900, 'proposed_credit_limit'] = 2

        return X

    def __sklearn_is_fitted__(self):
        """
        Check fitted status and return a Boolean value.
        """
        return hasattr(self, "_is_fitted") and self._is_fitted


class DataPreprocessor(BaseEstimator, TransformerMixin):
    """
    Custom class for feature selection, adding features based on functions and PCA,
    applying ColumnTransformer (Scaler, OHE, MTE).
    """

    def __init__(self,
                 cat_features=None,
                 num_features=None,
                 pca_features=None,
                 pca_n_components=None,
                 custom_functions=None,
                 num_ohe_mte_threshold=5,
                 scaler=RobustScaler()
                 ):
        self.cat_features = cat_features
        self.num_features = num_features
        self.pca_features = pca_features
        self.pca_n_components = pca_n_components
        self.custom_functions = custom_functions
        self.num_ohe_mte_threshold = num_ohe_mte_threshold
        self.scaler = scaler

    def fit(self, X, y=None):
        X = X.copy()

        self._cat_features = self.cat_features.copy() if self.cat_features else []
        self._num_features = self.num_features.copy() if self.num_features else []

        self.similarity_storage = {}
        self.excess_ratio_storage = {}

        if self.pca_features is not None:
            # 1. Add PCA
            self._fit_pca(X)

            # 2. Add PCA
            self.pca_columns = [f'pca_feature_{i}' for i in range(self.pca_n_components)]
            self._num_features += self.pca_columns

            # Modify df
            pca_values = self.pca.transform(X[self.pca_features])
            for i, col in enumerate(self.pca_columns):
                X[col] = pca_values[:, i]

        # 3. Process custom features
        if self.custom_functions is not None:
            self.generated_columns = []
            for func, cols in self.custom_functions:
                if func == 'similarity':
                    main_feature, base_features = cols
                    X, new_feature_name = self._fit_similarity(main_feature, base_features, X)
                    self._cat_features.append(new_feature_name)
                    self.generated_columns.append((new_feature_name, func, (main_feature, base_features)))
                elif func == 'excess_ratio':
                    main_feature, base_features = cols
                    X, new_feature_name = self._fit_excess_ratio(
                        main_feature=main_feature,
                        base_features=base_features,
                        X=X)
                    self._cat_features.append(new_feature_name)
                    self.generated_columns.append((new_feature_name, func, (main_feature, base_features)))

                else:
                    if func == 'not_round':
                        func = self.not_round
                    for col in cols:
                        new_col = f"{col}_{func.__name__}"
                        X[new_col] = func(X[col])
                        self.generated_columns.append((new_col, func, col))
                        if np.issubdtype(X[new_col].dtype, np.number):
                            self._num_features.append(new_col)
                        else:
                            self._cat_features.append(new_col)

        # 4. Fit ColumnTransformer
        self._fit_transformer(X, y)

        # 5. List of all final features
        ohe_feature_names = self._get_ohe_feature_names()
        self.transformed_feature_names_ = (
                self._num_features + ohe_feature_names + self.mte_features
        )

        return self

    def transform(self, X, y=None):
        X = X.copy()

        # Shortcut-mask
        self.shortcut_mask = X['device_fraud_count'] > 0

        if self.pca_features is not None:
            # PCA
            pca_values = self.pca.transform(X[self.pca_features])
            for i, col in enumerate(self.pca_columns):
                X[col] = pca_values[:, i]

        # Custom features
        if self.custom_functions is not None:
            for new_col, func, base_col in self.generated_columns:
                if func == 'similarity':
                    main_feature, base_features = base_col
                    X = self._similarity(new_col, main_feature, base_features, X)
                elif func == 'excess_ratio':
                    main_feature, base_features = base_col
                    X = self._excess_ratio(
                        new_feature_name=new_col,
                        main_feature=main_feature,
                        base_features=base_features,
                        X=X
                    )

                else:
                    if func == 'not_round':
                        func = self.not_round
                    X[new_col] = func(X[base_col])

        # Remove unnecessary features
        keep_cols = set(self._num_features + self._cat_features)
        X = X[[col for col in X.columns if col in keep_cols]]

        # Transform
        transformed = self.column_transformer.transform(X)

        # Wrap in DataFrame
        return pd.DataFrame(transformed, columns=self.transformed_feature_names_, index=X.index)

    def predict(self, X, model):
        X_processed = self.transform(X)

        preds = model.predict(X_processed)

        # shortcut
        preds = np.where(self.shortcut_mask.values, 1, preds)

        return preds

    def _fit_pca(self, X):
        self.pca = PCA(n_components=self.pca_n_components)
        self.pca.fit(X[self.pca_features])

    def _fit_transformer(self, X, y):
        self.ohe_features = [x for x in self._cat_features if X[x].nunique() < self.num_ohe_mte_threshold]
        self.mte_features = [x for x in self._cat_features if X[x].nunique() >= self.num_ohe_mte_threshold]

        num_transformer = Pipeline([
            ('scaler', self.scaler)
        ])

        most_frequent = [X[col].value_counts().idxmax() for col in self.ohe_features]
        ohe_transformer = Pipeline([
            ('ohe', OneHotEncoder(drop=most_frequent, sparse_output=False, handle_unknown='warn'))
        ])

        mte_transformer = Pipeline([
            ('mte', TargetEncoder(cv=5, shuffle=True, random_state=42))
        ])

        self.column_transformer = ColumnTransformer(transformers=[
            ('num', num_transformer, self._num_features),
            ('ohe', ohe_transformer, self.ohe_features),
            ('mte', mte_transformer, self.mte_features)
        ], remainder='drop', n_jobs=-1)

        self.column_transformer.fit(X, y)

    def _get_ohe_feature_names(self):
        ohe = self.column_transformer.named_transformers_['ohe'].named_steps['ohe']
        feature_names = ohe.get_feature_names_out(self.ohe_features)
        return list(feature_names)

    def _fit_similarity(self, main_feature: str, base_features: list, X: pd.DataFrame):
        counts = (
            X[[*base_features, main_feature]].groupby([*base_features, main_feature], observed=True)
            .size()
            .reset_index(name='count')
        )

        new_feature_name = f"{main_feature}_similarity_rank_by_{'_'.join(base_features)}"

        counts[new_feature_name] = -1 * (
            counts
            .sort_values([*base_features, 'count'], ascending=[True, True, False])
            .groupby(base_features, observed=True)
            .cumcount()
        )

        counts.loc[counts[new_feature_name] <= -3, new_feature_name] = -3
        max_rank = counts[new_feature_name].min()
        self.similarity_storage[new_feature_name] = (counts, max_rank)

        X = self._similarity(new_feature_name, main_feature, base_features, X)
        return X, new_feature_name

    def _similarity(self, new_feature_name: str, main_feature: str, base_features: list, X: pd.DataFrame):

        counts, max_rank = self.similarity_storage[new_feature_name]

        X = X.merge(
            counts[[*base_features, main_feature, new_feature_name]],
            on=[*base_features, main_feature],
            how='left'
        )
        X[new_feature_name] = X[new_feature_name].fillna(max_rank)

        return X

    def _fit_excess_ratio(self, main_feature: str, base_features: list, X: pd.DataFrame):
        counts = (
            X[[*base_features, main_feature]]
            .groupby([*base_features], observed=True)
            .median()
            .rename(columns={main_feature: f'{main_feature}_median'})
            .reset_index()
        )
        new_feature_name = f"{main_feature}_excess_ratio_by_{'_'.join(base_features)}"

        self.excess_ratio_storage[new_feature_name] = counts

        X = self._excess_ratio(new_feature_name, main_feature, base_features, X)
        return X, new_feature_name

    def _excess_ratio(self, new_feature_name: str, main_feature: str, base_features: list, X: pd.DataFrame):

        counts = self.excess_ratio_storage[new_feature_name]

        X = X.merge(
            counts[[*base_features, f'{main_feature}_median']],
            on=base_features,
            how='left'
        )

        X[new_feature_name] = X[main_feature] / X[f'{main_feature}_median']
        X = X.drop(f'{main_feature}_median', axis=1)

        return X

    def not_round(self, ser):
        def _nr(num):
            if (num >= 1000 and num % 500 == 0) or (num < 1000 and num % 100 == 0):
                return False
            else:
                return True

        return ser.apply(_nr)


class IsoForest(BaseEstimator, TransformerMixin):
    """
    Class to add IsolationForest mark to the dataset
    """

    def __init__(self, n_estimators=200, bootstrap=True, contamination='by_sample', column_name=None):
        self.n_estimators = n_estimators
        self.bootstrap = bootstrap
        self.contamination = contamination
        self.column_name = column_name

    def fit(self, X, y=None):

        if self.contamination == 'by_sample':
            con = np.sum(y) / len(y)
        else:
            con = self.contamination

        self.iso = IsolationForest(
            n_estimators=self.n_estimators,
            bootstrap=self.bootstrap,
            contamination=con,
            random_state=42,
            n_jobs=-1,
        )

        self.iso.fit(X)
        self._is_fitted = True
        return self

    def transform(self, X, y=None):
        check_is_fitted(self)
        X = X.copy()

        if self.column_name is None:
            col_name = 'IsolationForestMark'
        else:
            col_name = self.column_name

        X[col_name] = np.where(self.iso.predict(X) == -1, 1, 0)
        return X

    def __sklearn_is_fitted__(self):
        """
        Check fitted status and return a Boolean value.
        """
        return hasattr(self, "_is_fitted") and self._is_fitted


class ClassifierWrapper(ClassifierMixin, BaseEstimator):
    def __init__(self, model, target_fpr=0.05):
        self.base_model = model
        self.target_fpr = target_fpr

    def fit(self, X, y):

        self.cv_thresholds = []
        self.max_threshold = 0

        for train_idx, val_idx in TimeSeriesMonthSplit().split(X, y):
            X_train, y_train = X.loc[train_idx], y.loc[train_idx]
            X_val, y_val = X.loc[val_idx], y.loc[val_idx]

            model = clone(self.base_model)
            model.fit(X_train, y_train)

            y_score = model.predict_proba(X_val)[:, 1]
            fpr, tpr, thresholds = roc_curve(y_val, y_score)
            self.cv_thresholds.append(thresholds[np.where(fpr <= self.target_fpr)[0].max()])
            self.max_threshold = np.max([self.max_threshold, np.sort(thresholds)[-2]])  # the last threshold = inf

        self.threshold_fpr = np.median(self.cv_thresholds)

        self.model = clone(self.base_model)
        self.model.fit(X, y)

        self._is_fitted = True

        return self

    def predict(self, X):
        check_is_fitted(self)
        # Shortcut: if device_fraud_count > 0,return 1
        probas = self.predict_proba(X)[:, 1]
        return (probas >= self.threshold_fpr).astype(int)

    def predict_proba(self, X):
        check_is_fitted(self)
        probas = self.model.predict_proba(X)
        if 'device_fraud_count' in X.columns:
            fraud_idx = X['device_fraud_count'] > 0
            probas[fraud_idx, 0] = 1 - self.max_threshold  # class 0
            probas[fraud_idx, 1] = self.max_threshold  # class 1
        return probas

    def __sklearn_is_fitted__(self):
        """
        Check fitted status and return a Boolean value.
        """
        return hasattr(self, "_is_fitted") and self._is_fitted

    @property
    def classes_(self):
        check_is_fitted(self)
        return self.model.classes_
