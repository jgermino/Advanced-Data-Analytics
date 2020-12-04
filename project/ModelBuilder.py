import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV


class ModelBuilder:
    def __init__(self, train: pd.DataFrame, test: pd.DataFrame, train_y: pd.DataFrame, test_y: pd.DataFrame):
        """
        :param train: Pandas Dataframe of training data
        :param test: Pandas Dataframe of test data
        :param train_y: Pandas Dataframe of actual prediction values for training data
        :param test_y: Pandas Dataframe of actual prediction values for test data
        """
        self.train_X = train
        self.test_X = test
        self.train_y = train_y
        self.test_y = test_y

    def knn(self, k: int, dims=None):
        """
        Fit a K-Nearest Neighbor Regression model on training data and return prediction and error of test data

        :param k: k neighbors
        :param dims: PCA dimensions - if None PCA will not be run
        :return: numpy array of predictions, rmse
        """
        train_X = self.train_X.copy()  # create copy so changes are not made in place
        test_X = self.test_X.copy()

        if dims is not None:
            pca = PCA(dims)
            pca.fit(train_X, self.train_y)
            train_X = pca.transform(train_X)
            test_X = pca.transform(test_X)

        model = KNeighborsRegressor(n_neighbors=k, weights='distance')
        model.fit(train_X, self.train_y)
        prediction = model.predict(test_X)
        rmse = (np.sqrt(sum((np.array(prediction) - self.test_y) ** 2) / len(prediction)))

        return prediction, rmse

    def lr(self, dims=None):
        """
        Fit a Linear Regression model on training data and return prediction and error of test data

        :param dims: PCA dimensions - if None PCA will not be run
        :return: numpy array of predictions, rmse
        """
        train_X = self.train_X.copy()  # create copy so changes are not made in place
        test_X = self.test_X.copy()

        if dims is not None:
            pca = PCA(dims)
            pca.fit(train_X, self.train_y)
            train_X = pca.transform(train_X)
            test_X = pca.transform(test_X)

        model = LinearRegression()
        model.fit(train_X, self.train_y)
        prediction = model.predict(test_X)
        rmse = (np.sqrt(sum((np.array(prediction) - self.test_y) ** 2) / len(prediction)))

        return prediction, rmse

    def dt(self, depth=None, dims=None):
        """
        Fit a Decision Tree Regression model on training data and return prediction and error of test data

        :param depth: max depth of Decision Tree
        :param dims: PCA dimensions - if None PCA will not be run
        :return: numpy array of predictions, rmse
        """
        train_X = self.train_X.copy()  # create copy so changes are not made in place
        test_X = self.test_X.copy()

        if dims is not None:
            pca = PCA(dims)
            pca.fit(train_X, self.train_y)
            train_X = pca.transform(train_X)
            test_X = pca.transform(test_X)

        model = DecisionTreeRegressor(max_depth=depth)
        model.fit(train_X, self.train_y)
        prediction = model.predict(test_X)
        rmse = (np.sqrt(sum((np.array(prediction) - self.test_y) ** 2) / len(prediction)))

        return prediction, rmse

    def gbr(self, learning_rate: float, n_estimators: int, dims=None):
        """
        Fit a Gradient Boosting Regression model on training data and return prediction and error of test data

        :param learning_rate: learning rate for Gradient Descent
        :param n_estimators: number of estimators
        :param dims: PCA dimensions - if None PCA will not be run
        :return: numpy array of predictions, rmse
        """
        # No option to adjust subsample - for purposes of this project want to avoid stochastic
        train_X = self.train_X.copy()  # create copy so changes are not made in place
        test_X = self.test_X.copy()

        if dims is not None:
            pca = PCA(dims)
            pca.fit(train_X, self.train_y)
            train_X = pca.transform(train_X)
            test_X = pca.transform(test_X)

        model = GradientBoostingRegressor(learning_rate=learning_rate, n_estimators=n_estimators, subsample=.5)
        model.fit(train_X, self.train_y)
        prediction = model.predict(test_X)
        rmse = (np.sqrt(sum((np.array(prediction) - self.test_y) ** 2) / len(prediction)))

        return prediction, rmse

    def gridsearch(self, model: str, params: dict, dims=None, print_results=True):
        """
        Fit a specified model on training data using Gridsearch to find the optimal hyperparameters and return
        prediction and error of test data

        :param model: string inidicating
        :param params: dictionary of parameters to use for Gridsearch
        :param dims: PCA dimensions - if None PCA will not be run
        :param print_results: boolean indicating if the optimal hyperparamter values should be printed
        :return: numpy array of predictions, rmse
        """
        train_X = self.train_X.copy()
        test_X = self.test_X.copy()

        if dims is not None:
            pca = PCA(dims)
            pca.fit(train_X, self.train_y)
            train_X = pca.transform(train_X)
            test_X = pca.transform(test_X)

        if model == 'gbr':
            mod = GradientBoostingRegressor(subsample=.5)
        elif model == 'lr':
            mod = DecisionTreeRegressor()
        elif model == 'dt':
            mod = LinearRegression()
        elif model == 'knn':
            mod = KNeighborsRegressor()
        else:
            raise NameError("Model must be one of ['lr', 'dt', 'gbr', 'knn']")
        clf = GridSearchCV(mod, params, scoring='neg_mean_squared_error')
        clf.fit(train_X, self.train_y)
        if print_results:
            print(clf.best_params_)

        prediction = clf.predict(test_X)  # Uses best_params_
        rmse = (np.sqrt(sum((np.array(prediction) - self.test_y) ** 2) / len(prediction)))

        return prediction, rmse

    def num_rows(self):
        """Get number of samples in training data"""
        return self.train_X.shape[0]

    def num_columns(self):
        """Get number of columns in training data"""
        return self.train_X.shape[1]

    def pca(self):
        """Get the explained variance ratio of PCA on max dimensions"""
        pca = PCA(min(self.train_X.shape[1], self.train_X.shape[0]))
        pca.fit(self.train_X, self.train_y)
        return pca.explained_variance_ratio_
