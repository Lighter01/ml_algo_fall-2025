from typing import Union
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.model_selection import train_test_split
from src.knn_model import KNearestNeighbors, SklearnParzenKNN, Kernel, return_kernel
from src.k_param_optim import optimize_k_with_cv
from src.utils import utils

#=========================================================================#

__all__ = [
    "run_tests"
]

#=========================================================================#

def read_alzheimer_dataset(
    input_dir: str | Path
):
    df = pd.read_csv(input_dir, header=0)
    df = df.drop(columns=['PatientID', 'DoctorInCharge'])
    return df

def classify_cat_cols(X: pd.DataFrame, int_cols: list) -> tuple[list, list]:
    bin_cols = []
    cat_cols = []
    for col in int_cols:
        n_unique = X[col].nunique()
        if n_unique == 2:
            bin_cols.append(col)
        elif n_unique > 2 and n_unique < 10:
            cat_cols.append(col)

    for col in (bin_cols + cat_cols):
        int_cols.remove(col)

    print('int cols:         ', len(int_cols))
    print('catecorical cols: ', len(cat_cols))
    print('binary cols:      ', len(bin_cols), "\n")

    return bin_cols, cat_cols

def return_prepocessor(
    int_cols: list, 
    float_cols: list, 
    bin_cols: list, 
    cat_cols: list
) -> ColumnTransformer:
    preprocessing = ColumnTransformer(
        transformers = [
            ('float_cols', StandardScaler(), float_cols + int_cols),
            ('cat_cols', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), [cat_cols[0]]),
            ('ord_cols', OrdinalEncoder(), [cat_cols[1]]),
            ('bin_cols', 'passthrough', bin_cols),
        ],
        remainder = 'passthrough'
    )
    return preprocessing


def run_tests(
    input_dir: str | Path,
    results_dir: str | Path
):  
    df_alzh = read_alzheimer_dataset(input_dir)

    X, y = df_alzh.drop(columns=['Diagnosis']), df_alzh['Diagnosis']
    print(f"Dataset shape: {df_alzh.shape}")
    print(f"Number of classes: {df_alzh['Diagnosis'].nunique()}\n")
    print(f"Classes distribution: ")
    print(y.value_counts(normalize=True), "\n")

    int_cols = X.select_dtypes(include=['int']).columns.tolist()
    float_cols = X.select_dtypes(include=['float']).columns.tolist()

    print('Int cols: ', len(int_cols))
    print('Float cols: ', len(float_cols))

    bin_cols, cat_cols = classify_cat_cols(X, int_cols)

    preprocessor = return_prepocessor(int_cols, float_cols, bin_cols, cat_cols)

    X_prep = preprocessor.fit_transform(X)

    # ------------------------------------

    per_kernel_results = defaultdict(dict)

    
