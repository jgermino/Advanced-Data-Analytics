#!/usr/bin/env python3

from project import ModelBuilder as MB
from project import DataSlicer as DS
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    file = pd.read_csv("qualified_hitting_zscores.csv")
    # Generate training sets
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

    assert len(depths) == len(ks) == len(trains)
    for i in range(len(trains)):
        model = MB.ModelBuilder(trains[i].iloc[:, 4:-1], validations[i].iloc[:, 4:-1],
                                trains[i].iloc[:, -1], validations[i].iloc[:, -1])
        rmse_knn = []
        rmse_dt = []
        rmse_lr = []
        rmse_gbr = []
        params = {'learning_rate': np.linspace(.01, .25, 13), 'n_estimators': np.linspace(50, 250, 6, dtype=int)}

        for n in range(1, min(model.num_columns(), model.num_rows(), 30)):  # Cap dimensions at 30 to cut down on processing time
            _, err = model.knn(ks[i], n)  # RMSE of K-Nearest Neighbor model
            rmse_knn.append(err)
            _, err = model.dt(depths[i], n)  # RMSE of Decision Tree model
            rmse_dt.append(err)
            _, err = model.lr(n)  # RMSE of Linear Regression model
            rmse_lr.append(err)
            _, err = model.gridsearch('gbr', params, n)  # RMSE of Gradient Boosting model
            rmse_gbr.append(err)
        dims = np.linspace(1, n, n)

        plt.title(f"{names[i]}")
        plt.xlabel("PCA Dimensions")
        plt.ylabel("RMSE")
        plt.plot(dims, rmse_knn, color='xkcd:blue', label='knn')
        plt.plot(rmse_knn.index(min(rmse_knn))+1, min(rmse_knn), 'ro')
        plt.plot(dims, rmse_dt, color='xkcd:red', label='dt')
        plt.plot(rmse_dt.index(min(rmse_dt))+1, min(rmse_dt), 'ro')
        plt.plot(dims, rmse_lr, color='xkcd:green', label='lr')
        plt.plot(rmse_lr.index(min(rmse_lr))+1, min(rmse_lr), 'ro')
        plt.plot(dims, rmse_gbr, color='xkcd:orange', label='gbr')
        plt.plot(rmse_gbr.index(min(rmse_gbr))+1, min(rmse_gbr), 'ro')
        plt.legend()
        plt.show()
