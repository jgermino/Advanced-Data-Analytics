#!/usr/bin/env python3

from project import ModelBuilder as MB
from project import DataSlicer as DS
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


if __name__ == '__main__':
    file = pd.read_csv("qualified_hitting_zscores.csv")

    ds = DS.DataSlicer(file, 'zscores')
    two_year_z_scores_linear_imputed_train, _, two_year_z_scores_linear_imputed_test = ds.get_two_year_linear_imputed()
    
    mb = MB.ModelBuilder(two_year_z_scores_linear_imputed_train.iloc[:, 4:-1], two_year_z_scores_linear_imputed_test.iloc[:, 4:-1],
                         two_year_z_scores_linear_imputed_train.iloc[:, -1], two_year_z_scores_linear_imputed_test.iloc[:, -1])
    # Get final test predictions
    lr_pred, _ = mb.lr(9)
    dt_pred, _ = mb.dt(5, 9)
    gbr_pred, _ = mb.gbr(.05, 150, 9)
    knn_pred, _ = mb.knn(10, 9)
    actual = two_year_z_scores_linear_imputed_test.loc[:, 'y']
    prev_year = two_year_z_scores_linear_imputed_test.loc[:, 'wOBA']

    mean = ds.get_means().loc[2019, 'wOBA']
    sd = ds.get_sds().loc[2019, 'wOBA']
    # Convert z-score predictions back to wOBA
    lr_pred = lr_pred * sd + mean
    dt_pred = dt_pred * sd + mean
    gbr_pred = gbr_pred * sd + mean
    knn_pred = knn_pred * sd + mean
    actual = actual * sd + mean
    prev_year = prev_year * sd + mean
    # Calculate error
    lr_rmse = (np.sqrt(sum((lr_pred - actual) ** 2) / len(lr_pred)))
    dt_rmse = (np.sqrt(sum((dt_pred - actual) ** 2) / len(dt_pred)))
    gbr_rmse = (np.sqrt(sum((gbr_pred - actual) ** 2) / len(gbr_pred)))
    knn_rmse = (np.sqrt(sum((knn_pred - actual) ** 2) / len(knn_pred)))
    # Calculate R**2
    lr_r_squared = np.corrcoef(lr_pred, actual)[0, 1] ** 2
    dt_r_squared = np.corrcoef(dt_pred, actual)[0, 1] ** 2
    gbr_r_squared = np.corrcoef(gbr_pred, actual)[0, 1] ** 2
    knn_r_squared = np.corrcoef(knn_pred, actual)[0, 1] ** 2

    print(f"Linear Regression: {lr_rmse}, {lr_r_squared}")
    print(f"Decision Tree: {dt_rmse}, {dt_r_squared}")
    print(f"Gradient Boosting Regression: {gbr_rmse}, {gbr_r_squared}")
    print(f"K-Nearest Neighbors: {knn_rmse}, {knn_r_squared}")
    # Calculate error and R**2 for baseline Marcel projections
    marcel = pd.read_csv('marcel.csv')
    marcel = file.merge(marcel, how='right', on=['playerid', 'Season', 'Name', 'Team', 'Age'])
    marcel_pred = marcel.loc[:, 'proj_wOBA']
    marcel_actual = marcel.loc[:, 'wOBA']
    marcel_rmse = (np.sqrt(sum((marcel_pred - marcel_actual) ** 2) / len(marcel_pred)))
    marcel_r_squared = np.corrcoef(marcel_pred, marcel_actual)[0, 1] ** 2
    print(f"Marcel: {marcel_rmse}, {marcel_r_squared}")
    # Plot predicted wOBA vs actual value wOBA
    plt.title(f"Linear Regression v. Actual")
    plt.xlabel("Projected wOBA")
    plt.ylabel("Actual wOBA")
    plt.scatter(lr_pred, actual, color="xkcd:green")
    plt.plot(actual, actual, label="Exact match", color="xkcd:black")  # exact match line
    plt.show()

    plt.title(f"Decision Tree v. Actual")
    plt.xlabel("Projected wOBA")
    plt.ylabel("Actual wOBA")
    plt.scatter(dt_pred, actual, color="xkcd:red")
    plt.plot(actual, actual, label="Exact match", color="xkcd:black")  # exact match line
    plt.show()

    plt.title(f"Gradient Boosting Regression v. Actual")
    plt.xlabel("Projected wOBA")
    plt.ylabel("Actual wOBA")
    plt.scatter(gbr_pred, actual, color="xkcd:orange")
    plt.plot(actual, actual, label="Exact match", color="xkcd:black")  # exact match line
    plt.show()

    plt.title(f"K-Nearest Neighbors v. Actual")
    plt.xlabel("Projected wOBA")
    plt.ylabel("Actual wOBA")
    plt.scatter(knn_pred, actual, color="xkcd:blue")
    plt.plot(actual, actual, label="Exact match", color="xkcd:black")  # exact match line
    plt.show()

    plt.title(f"Marcel v. Actual")
    plt.xlabel("Projected wOBA")
    plt.ylabel("Actual wOBA")
    plt.scatter(marcel_pred, marcel_actual, color="xkcd:purple")
    plt.plot(marcel_actual, marcel_actual, label="Exact match", color="xkcd:black")  # exact match line
    plt.show()
    # Plot predicted wOBA vs previous year wOBA
    plt.title(f"Linear Regression v. Prev Year")
    plt.xlabel("Projected wOBA")
    plt.ylabel("Prev Year wOBA")
    plt.scatter(lr_pred, prev_year, color="xkcd:green")
    plt.plot(actual, actual, label="Exact match", color="xkcd:black")  # exact match line
    plt.show()

    plt.title(f"Decision Tree v. Prev Year")
    plt.xlabel("Projected wOBA")
    plt.ylabel("Prev Year wOBA")
    plt.scatter(dt_pred, prev_year, color="xkcd:red")
    plt.plot(actual, actual, label="Exact match", color="xkcd:black")  # exact match line
    plt.show()

    plt.title(f"Gradient Boosting Regression v. Prev Year")
    plt.xlabel("Projected wOBA")
    plt.ylabel("Prev Year wOBA")
    plt.scatter(gbr_pred, prev_year, color="xkcd:orange")
    plt.plot(actual, actual, label="Exact match", color="xkcd:black")  # exact match line
    plt.show()

    plt.title(f"K-Nearest Neighbors v. Prev Year")
    plt.xlabel("Projected wOBA")
    plt.ylabel("Prev Year wOBA")
    plt.scatter(knn_pred, prev_year, color="xkcd:blue")
    plt.plot(actual, actual, label="Exact match", color="xkcd:black")  # exact match line
    plt.show()
