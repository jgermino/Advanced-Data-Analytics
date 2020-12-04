#!/usr/bin/env python3

from project import ModelBuilder as MB
from project import DataSlicer as DS
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    file = pd.read_csv("qualified_hitting_zscores.csv")
    # Generate training, validation sets
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

    for i in range(len(trains)):
        model = MB.ModelBuilder(trains[i].iloc[:, 4:-1], validations[i].iloc[:, 4:-1],
                                trains[i].iloc[:, -1], validations[i].iloc[:, -1])
        for n in range(1, min(model.num_columns(), model.num_rows())):  # PCA Dimensions
            rmse_knn = []

            for k in range(1, 101):  # k neighbors
                _, err = model.knn(k, n)  # RMSE of K-Nearest Neighbor model
                rmse_knn.append(err)
                if k == model.num_rows():  # Cannot have more neighbors than samples
                    break
            ks = np.linspace(1, k, k)

            plt.title(f"{names[i]}")
            plt.xlabel("K")
            plt.ylabel("RMSE")
            plt.plot(ks, rmse_knn)
            plt.plot(rmse_knn.index(min(rmse_knn))+1, min(rmse_knn), 'ro')
        plt.show()

        for n in range(1, min(model.num_columns(), model.num_rows())):  # PCA dimensions
            rmse_dt = []

            for depth in range(1, 31):  # max Decision Tree depth
                _, err = model.dt(depth, n)  # RMSE of Decision Tree model
                rmse_dt.append(err)
            depths = np.linspace(1, depth, depth)

            plt.title(f"{names[i]}")
            plt.xlabel("Max Depth")
            plt.ylabel("RMSE")
            plt.plot(depths, rmse_dt)
            plt.plot(rmse_dt.index(min(rmse_dt))+1, min(rmse_dt), 'ro')
        plt.show()
