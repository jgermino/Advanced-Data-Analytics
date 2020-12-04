#!/usr/bin/env python3

from project import ModelBuilder as MB
from project import DataSlicer as DS
import pandas as pd
import numpy as np


if __name__ == '__main__':
    file = pd.read_csv("qualified_hitting_zscores.csv")

    # Generate training and validation sets
    ds = DS.DataSlicer(file, 'zscores')
    one_year_basic_z_scores_train, one_year_basic_z_scores_validate, _ = ds.get_one_year_basic()
    one_year_advanced_z_scores_train, one_year_advanced_z_scores_validate, _ = ds.get_one_year_advanced()
    one_year_all_z_scores_train, one_year_all_z_scores_validate, _ = ds.get_one_year_all()
    two_year_z_scores_train, two_year_z_scores_validate, _ = ds.get_two_year()
    # three_year_z_scores_train, three_year_z_scores_validate, _ = ds.get_three_year()
    one_year_z_scores_dtree_imputed_train, one_year_z_scores_dtree_imputed_validate, _ = ds.get_one_year_dtree_imputed()
    two_year_z_scores_dtree_imputed_train, two_year_z_scores_dtree_imputed_validate, _ = ds.get_two_year_dtree_imputed()
    two_year_z_scores_dtree_partial_imputed_train, two_year_z_scores_dtree_partial_imputed_validate, _ = ds.get_two_year_dtree_partial_imputed()
    three_year_z_scores_dtree_imputed_train, three_year_z_scores_dtree_imputed_validate, _ = ds.get_three_year_dtree_imputed()
    three_year_z_scores_dtree_partial_imputed_train, three_year_z_scores_dtree_partial_imputed_validate, _ = ds.get_three_year_dtree_partial_imputed()
    one_year_z_scores_linear_imputed_train, one_year_z_scores_linear_imputed_validate, _ = ds.get_one_year_linear_imputed()
    two_year_z_scores_linear_imputed_train, two_year_z_scores_linear_imputed_validate, _ = ds.get_two_year_linear_imputed()
    two_year_z_scores_linear_partial_imputed_train, two_year_z_scores_linear_partial_imputed_validate, _ = ds.get_two_year_linear_partial_imputed()
    three_year_z_scores_linear_imputed_train, three_year_z_scores_linear_imputed_validate, _ = ds.get_three_year_linear_imputed()
    three_year_z_scores_linear_partial_imputed_train, three_year_z_scores_linear_partial_imputed_validate, _ = ds.get_three_year_linear_partial_imputed()

    trains = [one_year_basic_z_scores_train, one_year_advanced_z_scores_train, one_year_all_z_scores_train,
              two_year_z_scores_train, one_year_z_scores_dtree_imputed_train,
              two_year_z_scores_dtree_imputed_train, two_year_z_scores_dtree_partial_imputed_train,
              three_year_z_scores_dtree_imputed_train, three_year_z_scores_dtree_partial_imputed_train,
              one_year_z_scores_linear_imputed_train, two_year_z_scores_linear_imputed_train,
              two_year_z_scores_linear_partial_imputed_train, three_year_z_scores_linear_imputed_train,
              three_year_z_scores_linear_partial_imputed_train]
    names = ['one_year_basic_z_scores_train', 'one_year_advanced_z_scores_train', 'one_year_all_z_scores_train',
             'two_year_z_scores_train', 'one_year_z_scores_dtree_imputed_train',
             'two_year_z_scores_dtree_imputed_train', 'two_year_z_scores_dtree_partial_imputed_train',
             'three_year_z_scores_dtree_imputed_train', 'three_year_z_scores_dtree_partial_imputed_train',
             'one_year_z_scores_linear_imputed_train', 'two_year_z_scores_linear_imputed_train',
             'two_year_z_scores_linear_partial_imputed_train', 'three_year_z_scores_linear_imputed_train',
             'three_year_z_scores_linear_partial_imputed_train']
    validations = [one_year_basic_z_scores_validate, one_year_advanced_z_scores_validate, one_year_all_z_scores_validate,
                   two_year_z_scores_validate, one_year_z_scores_dtree_imputed_validate,
                   two_year_z_scores_dtree_imputed_validate, two_year_z_scores_dtree_partial_imputed_validate,
                   three_year_z_scores_dtree_imputed_validate, three_year_z_scores_dtree_partial_imputed_validate,
                   one_year_z_scores_linear_imputed_validate, two_year_z_scores_linear_imputed_validate,
                   two_year_z_scores_linear_partial_imputed_validate, three_year_z_scores_linear_imputed_validate,
                   three_year_z_scores_linear_partial_imputed_validate]

    ks = [20, 15, 10, 5, 5, 10, 10, 10, 5, 10, 20, 20, 10, 15]  # Picked based on plots in parameter_testing
    depths = [5, 5, 5, 3, 5, 5, 5, 7, 7, 5, 5, 7, 5, 5]  # Picked based on plots in parameter_testing
    pca_dims = [6, 10, 10, 4, 7, 5, 5, 10, 5, 5, 9, 8, 10, 13]  # Picked based on plots in pca_testing

    assert len(pca_dims) == len(depths) == len(ks) == len(trains)
    # Create dictionary of final validation results
    results = {}
    for i in range(len(trains)):
        model = MB.ModelBuilder(trains[i].iloc[:, 4:-1], validations[i].iloc[:, 4:-1],
                                trains[i].iloc[:, -1], validations[i].iloc[:, -1])
        _, err = model.knn(ks[i], pca_dims[i])  # RMSE of K-Nearest Neighbor model
        results[(names[i], 'knn')] = err
        _, err = model.dt(depths[i], pca_dims[i])  # RMSE of Decision Tree model
        results[(names[i], 'dt')] = err
        _, err = model.lr(pca_dims[i])  # RMSE of Linear Regression model
        results[(names[i], 'lr')] = err
        params = {'learning_rate': np.linspace(.01, .25, 13), 'n_estimators': np.linspace(50, 250, 6, dtype=int)}
        _, err = model.gridsearch('gbr', params, pca_dims[i], False)  # RMSE of Gradient Boosting Regression model
        results[(names[i], 'gbr')] = err

    print(results)
