import pandas as pd
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression


class DataManipulator:
    def __init__(self, dat: pd.DataFrame, validation_year: int = 2017, test_year: int = 2018):
        """
        :param dat: Pandas Dataframe of Fangraphs data
        :param validation_year: Integer value of season which should be set aside as validation data. All years prior to
        validation_year will be included as part of training data.
        :param test_year: Integer value of season which should be set aside as test data. All years after test_year
        will be thrown out.
        """
        try:  # Separate data into training, validation, test
            dat2 = dat.copy()
            dat2.Season -= 1
            self.dat = dat.copy()
            dat2 = dat.merge(dat2.loc[:, ['playerid', 'Season']])
            self.train = dat2[dat2['Season'] < validation_year]
            self.validation = dat2[dat2['Season'] == validation_year]
            self.test = dat2[dat2['Season'] == test_year]
        except KeyError:
            raise KeyError("Data must contain 'Season' field")

    def get_data(self, y: str = 'wOBA'):
        """
        Return train, validation, and test data after making desired changes

        :param y: string indicating which column should be the prediction variable
        :return: DataFrames of train, validation, test
        """
        try:  # Add y value to end of data set and return training, validation, and test
            res = self.dat.loc[:, ['playerid', 'Season', y]]
            res = res.rename(columns={y: 'y'})
            res.Season -= 1
            train_out = pd.merge(self.train, res)
            validate_out = pd.merge(self.validation, res)
            test_out = pd.merge(self.test, res)
            return train_out, validate_out, test_out
        except KeyError:
            raise KeyError(f"Data does not contain {y}. Set parameter y to name of predicted column")

    def calculate_z_scores(self, needed_z_scores: list):
        """
        Calculate z-scores of the desired features grouped by season

        :param needed_z_scores: list of strings indicating which columns should be calculated to z-scores
        :return: None
        """
        self.means = self.dat.groupby('Season')[needed_z_scores].mean()
        self.sds = self.dat.groupby('Season')[needed_z_scores].std()
        extra_cols = [col for col in self.dat.columns if col not in needed_z_scores]

        temp = self.train.apply(lambda x: (x[needed_z_scores] - self.means.loc[x[1]]) / self.sds.loc[x[1]], axis=1)
        self.train = pd.concat([self.train[extra_cols], temp], ignore_index=False, axis=1)

        temp = self.validation.apply(lambda x: (x[needed_z_scores] - self.means.loc[x[1]]) / self.sds.loc[x[1]], axis=1)
        self.validation = pd.concat([self.validation[extra_cols], temp], ignore_index=False, axis=1)

        temp = self.test.apply(lambda x: (x[needed_z_scores] - self.means.loc[x[1]]) / self.sds.loc[x[1]], axis=1)
        self.test = pd.concat([self.test[extra_cols], temp], ignore_index=False, axis=1)

        temp = self.dat.apply(lambda x: (x[needed_z_scores] - self.means.loc[x[1]]) / self.sds.loc[x[1]], axis=1)
        self.dat = pd.concat([self.dat[extra_cols], temp], ignore_index=False, axis=1)

    def impute_data(self, train_cols: list, columns_to_impute: list,  model: str, dims: int = None):
        """
        Impute data for the specified columns using Linear Regression or Decision Trees

        :param train_cols: list of strings of column names which should be used to impute data
        :param columns_to_impute: list of strings of columns names which should be imputed
        :param model: string indicating if linear regression or decision trees should be used
        :param dims: PCA dimensions - if None PCA will not be run
        :return: None
        """
        if model == 'dt':
            model = DecisionTreeRegressor()
        elif model == 'lr':
            model = LinearRegression()
        else:
            raise NameError("Model must be one of ['lr', 'dt']")

        train = self.train.copy()  # Avoid modifying in place
        cols = train_cols.copy()
        cols.extend(columns_to_impute)
        train = train.loc[:, cols].dropna()

        if dims is None:
            model.fit(train.loc[:, train_cols], train.loc[:, columns_to_impute])
            self.train.loc[self.train[columns_to_impute[0]].isna(), columns_to_impute] = model.predict(  # Impute data to rows missing imputed columns
                self.train.loc[self.train[columns_to_impute[0]].isna(), train_cols])
            if self.validation[columns_to_impute[0]].isna().sum() > 0:
                self.validation.loc[self.validation[columns_to_impute[0]].isna(), columns_to_impute] = model.predict(
                    self.validation.loc[self.validation[columns_to_impute[0]].isna(), train_cols])
            if self.test[columns_to_impute[0]].isna().sum() > 0:
                self.test.loc[self.test[columns_to_impute[0]].isna(), columns_to_impute] = model.predict(
                    self.test.loc[self.test[columns_to_impute[0]].isna(), train_cols])
            self.dat.loc[self.dat[columns_to_impute[0]].isna(), columns_to_impute] = model.predict(
                self.dat.loc[self.dat[columns_to_impute[0]].isna(), train_cols])

        else:
            pca = PCA(dims)
            pca.fit(self.train.loc[:, train_cols], train.loc[:, columns_to_impute])  # Run PCA on training columns for imputation columns
            model.fit(pd.DataFrame(pca.transform(train.loc[:, train_cols])), train.loc[:, columns_to_impute])
            self.train.loc[self.train[columns_to_impute[0]].isna(), columns_to_impute] = model.predict(pd.DataFrame(  # Impute data to rows missing imputed columns
                pca.transform(self.train.loc[self.train[columns_to_impute[0]].isna(), train_cols])))
            if self.validation[columns_to_impute[0]].isna().sum() > 0:
                self.validation.loc[self.validation[columns_to_impute[0]].isna(), columns_to_impute] = model.predict(pd.DataFrame(
                    pca.transform(self.validation.loc[self.validation[columns_to_impute[0]].isna(), train_cols])))
            if self.test[columns_to_impute[0]].isna().sum() > 0:
                self.test.loc[self.test[columns_to_impute[0]].isna(), columns_to_impute] = model.predict(pd.DataFrame(
                    pca.transform(self.test.loc[self.test[columns_to_impute[0]].isna(), train_cols])))
            self.dat.loc[self.dat[columns_to_impute[0]].isna(), columns_to_impute] = model.predict(pd.DataFrame(
                pca.transform(self.dat.loc[self.dat[columns_to_impute[0]].isna(), train_cols])))

    def generate_past_years(self, years_back: int):
        """
        Add columns for multiyear data

        :param years_back: Number of years to be added to the training set
        :return: None
        """
        temp = self.dat.copy()
        columns = self.dat.columns
        for year in range(years_back):
            rename_dict = {}
            if year == 0:
                temp['Season'] = temp['Season'] + 1
                for col in columns:
                    rename_dict[col] = col + f'-{year + 1}'  # Add columns for each old season to rename dict
            else:
                temp[f'Season-{year}'] = temp[f'Season-{year}'] + 1
                for col in columns:
                    rename_dict[f'{col}-{year}'] = f'{col}-{year + 1}'

            temp = temp.rename(columns=rename_dict)
            self.train = self.train.merge(temp, how="left", left_on=["playerid", "Season"],
                                          right_on=[f"playerid-{year+1}", f"Season-{year+1}"])
            self.validation = self.validation.merge(temp, how="left", left_on=["playerid", "Season"],
                                                    right_on=[f"playerid-{year+1}", f"Season-{year+1}"])
            self.test = self.test.merge(temp, how="left", left_on=["playerid", "Season"],
                                        right_on=[f"playerid-{year+1}", f"Season-{year+1}"])
            self.dat = self.dat.merge(temp, how="left", left_on=["playerid", "Season"],
                                      right_on=[f"playerid-{year+1}", f"Season-{year+1}"])

    def get_means(self):
        """Get means used to calculate z-scores"""
        try:
            return self.means
        except NameError:
            raise RuntimeError("Must run calculate_z_scores first.")

    def get_sds(self):
        """Get standard deviations used to calculate z-scores"""
        try:
            return self.sds
        except NameError:
            raise RuntimeError("Must run calculate_z_scores first.")

    def pca(self, train_cols: list, y_cols: list):
        """Get the explained variance ratio of PCA on max dimensions"""
        pca = PCA(len(train_cols))
        pca.fit(self.train.loc[:, train_cols], self.train.loc[:, y_cols])
        return pca.explained_variance_ratio_
