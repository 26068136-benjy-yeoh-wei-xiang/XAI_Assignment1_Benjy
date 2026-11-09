"""Data loading helpers adapted from the supplied course utils.py.

preprocess_data() retains the original zero-filled EDA demonstration.
For modelling, split first, then call preprocess_split() and oversample().
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler

pd.set_option("display.max_columns", None)


class DataLoader:
    def __init__(self):
        self.data = None
        self.bmi_median = None
        self.feature_columns = None
        self.sampler = None

    def load_dataset(self, path="data/healthcare-dataset-stroke-data.csv"):
        self.data = pd.read_csv(path)

    def preprocess_data(self):
        """Retain the supplied preprocessing for the Task 1 demonstration."""
        categorical_cols = [
            "gender", "ever_married", "work_type", "Residence_type", "smoking_status"
        ]
        encoded = pd.get_dummies(
            self.data[categorical_cols], prefix=categorical_cols
        )
        self.data = pd.concat([encoded, self.data], axis=1)
        self.data.drop(categorical_cols, axis=1, inplace=True)
        self.data.bmi = self.data.bmi.fillna(0)
        self.data.drop(["id"], axis=1, inplace=True)

    def get_data_split(self, test_size=0.20, random_state=2021, stratify=False):
        """Split records before fitting any modelling preprocessing."""
        X = self.data.drop(columns=["id", "stroke"], errors="ignore")
        y = self.data["stroke"]
        return train_test_split(
            X, y, test_size=test_size, random_state=random_state,
            stratify=y if stratify else None,
        )

    def preprocess_split(self, X_train, X_test):
        """Fit BMI imputation and indicator columns on the training split."""
        X_train = X_train.copy()
        X_test = X_test.copy()
        self.bmi_median = X_train["bmi"].median()
        X_train["bmi"] = X_train["bmi"].fillna(self.bmi_median)
        X_test["bmi"] = X_test["bmi"].fillna(self.bmi_median)
        X_train = pd.get_dummies(X_train, dtype=float)
        X_test = pd.get_dummies(X_test, dtype=float)
        X_test = X_test.reindex(columns=X_train.columns, fill_value=0)
        self.feature_columns = X_train.columns.tolist()
        return X_train, X_test

    def oversample(self, X_train, y_train, random_state=None):
        """Duplicate minority training records; never pass the test set here."""
        self.sampler = RandomOverSampler(
            sampling_strategy="minority", random_state=random_state
        )
        return self.sampler.fit_resample(X_train, y_train)
