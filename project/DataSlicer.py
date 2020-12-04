import pandas as pd
from project import DataManipulator as DM


class DataSlicer:
    def __init__(self, file: pd.DataFrame, dat_type: str, print_pca=False):
        """
        Used to generate the specific training sets as described in the project report

        :param file: Pandas Dataframe of Fangraphs data
        :param dat_type: string indicating which data set is being used. Only zscores were submitted with the report
        :param print_pca: Boolean indicating if explained variance ratio of PCA should be printed to stdout
        """
        self.file = file

        if dat_type == 'zscores':
            ylab = 'wOBA'
        elif dat_type == 'plus':
            ylab = 'wRC+'
        else:
            raise NameError("dat_type must be in [zscores, plus]")
        # Create a DataManipulator instance for each of the different data imputation methods. DM imputes in place
        dm1 = DM.DataManipulator(self.file)  # no imputing
        self.raw_data_train, self.raw_data_validate, self.raw_data_test = dm1.get_data(ylab)

        dm2 = DM.DataManipulator(self.file)  # DT imputing
        dm3 = DM.DataManipulator(self.file)  # LR imputing

        if dat_type == 'zscores':
            self.zscore_setter()  # sets private variables to use with DM to generate necessary training sets
            dm1.calculate_z_scores(self._needed_z_scores)
            dm2.calculate_z_scores(self._needed_z_scores)
            dm3.calculate_z_scores(self._needed_z_scores)
            self.means = dm1.get_means()
            self.sds = dm1.get_sds()
        elif dat_type == 'plus':
            self.plus_setter()

        dm1.generate_past_years(2)
        train, validate, test = dm1.get_data(ylab)

        # Set up 3 years of training data imputed with Decision Trees
        if print_pca:  # Used to analyze explained variance ratio of PCA prior to imputing
            print(dm2.pca(self._train_cols_1, self._impute_cols_1))
        dm2.impute_data(self._train_cols_1, self._impute_cols_1, model='dt', dims=self._dims[0])  # Impute "Advanced" Cols
        if dat_type == 'zscores':
            if print_pca:
                print(dm2.pca(self._train_cols_2, self._impute_cols_2))
            dm2.impute_data(self._train_cols_2, self._impute_cols_2, model='dt', dims=self._dims[1])  # Impute "All" Cols
        dm2.generate_past_years(2)
        dtree_partial_imputed_train, dtree_partial_imputed_validate, dtree_partial_imputed_test = dm2.get_data(ylab)
        if print_pca:
            print(dm2.pca(self._train_cols_3, self._impute_cols_3))
        dm2.impute_data(self._train_cols_3, self._impute_cols_3, model='dt', dims=self._dims[2])  # Impute Season X-1 cols
        if print_pca:
            print(dm2.pca(self._train_cols_4, self._impute_cols_4))
        dm2.impute_data(self._train_cols_4, self._impute_cols_4, model='dt', dims=self._dims[3])  # Impute Season X-2 cols
        dtree_imputed_train, dtree_imputed_validate, dtree_imputed_test = dm2.get_data(ylab)

        # Set up 3 years of training data imputed with Linear Regression
        dm3.impute_data(self._train_cols_1, self._impute_cols_1, model='lr', dims=self._dims[0])  # Impute "Advanced" Cols
        if dat_type == 'zscores':
            dm3.impute_data(self._train_cols_2, self._impute_cols_2, model='lr', dims=self._dims[1])  # Impute "All" Cols
        dm3.generate_past_years(2)
        linear_partial_imputed_train, linear_partial_imputed_validate, linear_partial_imputed_test = dm3.get_data(ylab)
        dm3.impute_data(self._train_cols_3, self._impute_cols_3, model='lr', dims=self._dims[2])  # Impute Season X-1 cols
        dm3.impute_data(self._train_cols_4, self._impute_cols_4, model='lr', dims=self._dims[3])  # Impute Season X-2 cols
        linear_imputed_train, linear_imputed_validate, linear_imputed_test = dm3.get_data(ylab)

        # Remove null values from all datasets
        self.one_year_basic_train = train.loc[:, self._basic_cols].dropna()

        if dat_type == 'zscores':
            self.one_year_advanced_train = train.loc[:, self._advanced_cols].dropna()
        self.one_year_all_train = train.loc[:, self._one_year_cols].dropna()
        self.two_year_train = train.loc[:, self._two_year_cols].dropna()
        self.three_year_train = train.loc[:, self._three_year_cols].dropna()
        self.one_year_dtree_imputed_train = dtree_imputed_train.loc[:, self._one_year_cols].dropna()
        self.two_year_dtree_imputed_train = dtree_imputed_train.loc[:, self._two_year_cols].dropna()
        self.two_year_dtree_partial_imputed_train = dtree_partial_imputed_train.loc[:, self._two_year_cols].dropna()
        self.three_year_dtree_imputed_train = dtree_imputed_train.loc[:, self._three_year_cols]
        self.three_year_dtree_partial_imputed_train = dtree_partial_imputed_train.loc[:, self._three_year_cols].dropna()
        self.one_year_linear_imputed_train = linear_imputed_train.loc[:, self._one_year_cols].dropna()
        self.two_year_linear_imputed_train = linear_imputed_train.loc[:, self._two_year_cols].dropna()
        self.two_year_linear_partial_imputed_train = linear_partial_imputed_train.loc[:, self._two_year_cols].dropna()
        self.three_year_linear_imputed_train = linear_imputed_train.loc[:, self._three_year_cols]
        self.three_year_linear_partial_imputed_train = linear_partial_imputed_train.loc[:, self._three_year_cols].dropna()

        self.one_year_basic_validate = validate.loc[:, self._basic_cols].dropna()
        if dat_type == 'zscores':
            self.one_year_advanced_validate = validate.loc[:, self._advanced_cols].dropna()
        self.one_year_all_validate = validate.loc[:, self._one_year_cols].dropna()
        self.two_year_validate = validate.loc[:, self._two_year_cols].dropna()
        self.three_year_validate = validate.loc[:, self._three_year_cols].dropna()
        self.one_year_dtree_imputed_validate = dtree_imputed_validate.loc[:, self._one_year_cols].dropna()
        self.two_year_dtree_imputed_validate = dtree_imputed_validate.loc[:, self._two_year_cols].dropna()
        self.two_year_dtree_partial_imputed_validate = dtree_partial_imputed_validate.loc[:, self._two_year_cols].dropna()
        self.three_year_dtree_imputed_validate = dtree_imputed_validate.loc[:, self._three_year_cols]
        self.three_year_dtree_partial_imputed_validate = dtree_partial_imputed_validate.loc[:, self._three_year_cols].dropna()
        self.one_year_linear_imputed_validate = linear_imputed_validate.loc[:, self._one_year_cols].dropna()
        self.two_year_linear_imputed_validate = linear_imputed_validate.loc[:, self._two_year_cols].dropna()
        self.two_year_linear_partial_imputed_validate = linear_partial_imputed_validate.loc[:, self._two_year_cols].dropna()
        self.three_year_linear_imputed_validate = linear_imputed_validate.loc[:, self._three_year_cols]
        self.three_year_linear_partial_imputed_validate = linear_partial_imputed_validate.loc[:,self._three_year_cols].dropna()

        self.one_year_basic_test = test.loc[:, self._basic_cols].dropna()
        if dat_type == 'zscores':
            self.one_year_advanced_test = test.loc[:, self._advanced_cols].dropna()
        self.one_year_all_test = test.loc[:, self._one_year_cols].dropna()
        self.two_year_test = test.loc[:, self._two_year_cols].dropna()
        self.three_year_test = test.loc[:, self._three_year_cols].dropna()
        self.one_year_dtree_imputed_test = dtree_imputed_test.loc[:, self._one_year_cols].dropna()
        self.two_year_dtree_imputed_test = dtree_imputed_test.loc[:, self._two_year_cols].dropna()
        self.two_year_dtree_partial_imputed_test = dtree_partial_imputed_test.loc[:, self._two_year_cols].dropna()
        self.three_year_dtree_imputed_test = dtree_imputed_test.loc[:, self._three_year_cols]
        self.three_year_dtree_partial_imputed_test = dtree_partial_imputed_test.loc[:,self._three_year_cols].dropna()
        self.one_year_linear_imputed_test = linear_imputed_test.loc[:, self._one_year_cols].dropna()
        self.two_year_linear_imputed_test = linear_imputed_test.loc[:, self._two_year_cols].dropna()
        self.two_year_linear_partial_imputed_test = linear_partial_imputed_test.loc[:, self._two_year_cols].dropna()
        self.three_year_linear_imputed_test = linear_imputed_test.loc[:, self._three_year_cols]
        self.three_year_linear_partial_imputed_test = linear_partial_imputed_test.loc[:,self._three_year_cols].dropna()

    def zscore_setter(self):
        """
        Creates various list of column headers for Data Manipulator pertaining to the z-score data set

        :return: None
        """
        self._dims = [6, 14, 16, 30]  # dims set to 95% threshold
        self._needed_z_scores = ['Age', 'PA', 'AVG', 'OBP', 'SLG', 'OPS', 'ISO', 'BABIP', 'BB%', 'K%', 'LD%', 'GB%',
                                 'FB%', 'GB/FB', 'IFFB%', 'HR/FB', 'O-Swing%', 'Z-Swing%', 'Swing%', 'O-Contact%',
                                 'Z-Contact%', 'Contact%', 'Pull%', 'Cent%', 'Oppo%', 'Soft%', 'Med%', 'Hard%',
                                 'EV', 'LA', 'Barrel%', 'maxEV', 'HardHit%', 'wOBA']

        self._train_cols_1 = ['Age', 'PA', 'AVG', 'OBP', 'SLG', 'OPS', 'ISO', 'BABIP', 'BB%', 'K%']
        self._impute_cols_1 = ['LD%', 'GB%', 'FB%', 'GB/FB', 'IFFB%', 'HR/FB', 'O-Swing%', 'Z-Swing%', 'Swing%',
                               'O-Contact%', 'Z-Contact%', 'Contact%', 'Pull%', 'Cent%', 'Oppo%', 'Soft%', 'Med%',
                               'Hard%']

        self._train_cols_2 = ['Age', 'PA', 'AVG', 'OBP', 'SLG', 'OPS', 'ISO', 'BABIP', 'BB%', 'K%', 'LD%', 'GB%',
                              'FB%', 'GB/FB', 'IFFB%', 'HR/FB', 'O-Swing%', 'Z-Swing%', 'Swing%', 'O-Contact%',
                              'Z-Contact%', 'Contact%', 'Pull%', 'Cent%', 'Oppo%', 'Soft%', 'Med%', 'Hard%']
        self._impute_cols_2 = ['EV', 'LA', 'Barrel%', 'maxEV', 'HardHit%']

        self._train_cols_3 = ['Age', 'PA', 'AVG', 'OBP', 'SLG', 'OPS', 'ISO', 'BABIP', 'BB%', 'K%', 'LD%', 'GB%',
                              'FB%', 'GB/FB', 'IFFB%', 'HR/FB', 'O-Swing%', 'Z-Swing%', 'Swing%', 'O-Contact%',
                              'Z-Contact%', 'Contact%', 'Pull%', 'Cent%', 'Oppo%', 'Soft%', 'Med%', 'Hard%', 'EV',
                              'LA', 'Barrel%', 'maxEV', 'HardHit%', 'wOBA']
        self._impute_cols_3 = ['AVG-1', 'OBP-1', 'SLG-1', 'OPS-1', 'ISO-1', 'BABIP-1', 'BB%-1', 'K%-1', 'LD%-1',
                               'GB%-1', 'FB%-1', 'GB/FB-1', 'IFFB%-1', 'HR/FB-1', 'O-Swing%-1', 'Z-Swing%-1',
                               'Swing%-1', 'O-Contact%-1', 'Z-Contact%-1', 'Contact%-1', 'Pull%-1', 'Cent%-1',
                               'Oppo%-1', 'Soft%-1', 'Med%-1', 'Hard%-1', 'EV-1', 'LA-1', 'Barrel%-1', 'maxEV-1',
                               'HardHit%-1', 'wOBA-1']

        self._train_cols_4 = ['Age', 'PA', 'AVG', 'OBP', 'SLG', 'OPS', 'ISO', 'BABIP', 'BB%', 'K%', 'LD%', 'GB%',
                              'FB%', 'GB/FB', 'IFFB%', 'HR/FB', 'O-Swing%', 'Z-Swing%', 'Swing%', 'O-Contact%',
                              'Z-Contact%', 'Contact%', 'Pull%', 'Cent%', 'Oppo%', 'Soft%', 'Med%', 'Hard%', 'EV',
                              'LA', 'Barrel%', 'maxEV', 'HardHit%', 'wOBA', 'AVG-1', 'OBP-1', 'SLG-1', 'OPS-1',
                              'ISO-1', 'BABIP-1', 'BB%-1', 'K%-1', 'LD%-1', 'GB%-1', 'FB%-1', 'GB/FB-1', 'IFFB%-1',
                              'HR/FB-1', 'O-Swing%-1', 'Z-Swing%-1', 'Swing%-1', 'O-Contact%-1', 'Z-Contact%-1',
                              'Contact%-1', 'Pull%-1', 'Cent%-1', 'Oppo%-1', 'Soft%-1', 'Med%-1', 'Hard%-1', 'EV-1',
                              'LA-1', 'Barrel%-1', 'maxEV-1', 'HardHit%-1', 'wOBA-1']
        self._impute_cols_4 = ['AVG-2', 'OBP-2', 'SLG-2', 'OPS-2', 'ISO-2', 'BABIP-2', 'BB%-2', 'K%-2', 'LD%-2',
                               'GB%-2', 'FB%-2', 'GB/FB-2', 'IFFB%-2', 'HR/FB-2', 'O-Swing%-2', 'Z-Swing%-2',
                               'Swing%-2', 'O-Contact%-2', 'Z-Contact%-2', 'Contact%-2', 'Pull%-2', 'Cent%-2',
                               'Oppo%-2', 'Soft%-2', 'Med%-2', 'Hard%-2', 'EV-2', 'LA-2', 'Barrel%-2', 'maxEV-2',
                               'HardHit%-2', 'wOBA-2']

        self._basic_cols = ['playerid', 'Season', 'Name', 'Team', 'Age', 'PA', 'AVG', 'OBP', 'SLG', 'OPS', 'ISO',
                            'BABIP', 'BB%', 'K%', 'wOBA', 'y']
        self._advanced_cols = ['playerid', 'Season', 'Name', 'Team', 'Age', 'PA', 'AVG', 'OBP', 'SLG', 'OPS', 'ISO',
                               'BABIP', 'BB%', 'K%', 'LD%', 'GB%', 'FB%', 'GB/FB', 'IFFB%', 'HR/FB', 'O-Swing%',
                               'Z-Swing%', 'Swing%', 'O-Contact%', 'Z-Contact%', 'Contact%', 'Pull%', 'Cent%',
                               'Oppo%', 'Soft%', 'Med%', 'Hard%', 'wOBA', 'y']
        self._one_year_cols = ['playerid', 'Season', 'Name', 'Team', 'Age', 'PA', 'AVG', 'OBP', 'SLG', 'OPS', 'ISO',
                               'BABIP', 'BB%', 'K%', 'LD%', 'GB%', 'FB%', 'GB/FB', 'IFFB%', 'HR/FB', 'O-Swing%',
                               'Z-Swing%', 'Swing%', 'O-Contact%', 'Z-Contact%', 'Contact%', 'Pull%', 'Cent%',
                               'Oppo%', 'Soft%', 'Med%', 'Hard%', 'EV', 'LA', 'Barrel%', 'maxEV', 'HardHit%',
                               'wOBA', 'y']
        self._two_year_cols = ['playerid', 'Season', 'Name', 'Team', 'Age', 'PA', 'AVG', 'OBP', 'SLG', 'OPS', 'ISO',
                               'BABIP', 'BB%', 'K%', 'LD%', 'GB%', 'FB%', 'GB/FB', 'IFFB%', 'HR/FB', 'O-Swing%',
                               'Z-Swing%', 'Swing%', 'O-Contact%', 'Z-Contact%', 'Contact%', 'Pull%', 'Cent%',
                               'Oppo%', 'Soft%', 'Med%', 'Hard%', 'EV', 'LA', 'Barrel%', 'maxEV', 'HardHit%',
                               'wOBA', 'AVG-1', 'OBP-1', 'SLG-1', 'OPS-1', 'ISO-1', 'BABIP-1', 'BB%-1', 'K%-1',
                               'LD%-1', 'GB%-1', 'FB%-1', 'GB/FB-1', 'IFFB%-1', 'HR/FB-1', 'O-Swing%-1',
                               'Z-Swing%-1', 'Swing%-1', 'O-Contact%-1', 'Z-Contact%-1', 'Contact%-1', 'Pull%-1',
                               'Cent%-1', 'Oppo%-1', 'Soft%-1', 'Med%-1', 'Hard%-1', 'EV-1', 'LA-1', 'Barrel%-1',
                               'maxEV-1', 'HardHit%-1', 'wOBA-1', 'y']
        self._three_year_cols = ['playerid', 'Season', 'Name', 'Team', 'Age', 'PA', 'AVG', 'OBP', 'SLG', 'OPS',
                                 'ISO', 'BABIP', 'BB%', 'K%', 'LD%', 'GB%', 'FB%', 'GB/FB', 'IFFB%', 'HR/FB',
                                 'O-Swing%', 'Z-Swing%', 'Swing%', 'O-Contact%', 'Z-Contact%', 'Contact%', 'Pull%',
                                 'Cent%', 'Oppo%', 'Soft%', 'Med%', 'Hard%', 'EV', 'LA', 'Barrel%', 'maxEV',
                                 'HardHit%', 'wOBA', 'AVG-1', 'OBP-1', 'SLG-1', 'OPS-1', 'ISO-1', 'BABIP-1',
                                 'BB%-1', 'K%-1', 'LD%-1', 'GB%-1', 'FB%-1', 'GB/FB-1', 'IFFB%-1', 'HR/FB-1',
                                 'O-Swing%-1', 'Z-Swing%-1', 'Swing%-1', 'O-Contact%-1', 'Z-Contact%-1',
                                 'Contact%-1', 'Pull%-1', 'Cent%-1', 'Oppo%-1', 'Soft%-1', 'Med%-1', 'Hard%-1',
                                 'EV-1', 'LA-1', 'Barrel%-1', 'maxEV-1', 'HardHit%-1', 'wOBA-1', 'AVG-2', 'OBP-2',
                                 'SLG-2', 'OPS-2', 'ISO-2', 'BABIP-2', 'BB%-2', 'K%-2', 'LD%-2', 'GB%-2', 'FB%-2',
                                 'GB/FB-2', 'IFFB%-2', 'HR/FB-2', 'O-Swing%-2', 'Z-Swing%-2', 'Swing%-2',
                                 'O-Contact%-2', 'Z-Contact%-2', 'Contact%-2', 'Pull%-2', 'Cent%-2', 'Oppo%-2',
                                 'Soft%-2', 'Med%-2', 'Hard%-2', 'EV-2', 'LA-2', 'Barrel%-2', 'maxEV-2',
                                 'HardHit%-2', 'wOBA-2', 'y']

    def plus_setter(self):
        """
        Creates various list of column headers for Data Manipulator pertaining to the plus data set

        :return: None
        """
        self._dims = [4, None, 7, 13]  # dims set to 95% threshold
        self._train_cols_1 = ['Age', 'PA', 'AVG+', 'BB%+', 'K%+', 'OBP+', 'SLG+', 'ISO+', 'BABIP+']
        self._impute_cols_1 = ['LD+%', 'GB%+', 'FB%+', 'HR/FB%+', 'Pull%+', 'Cent%+', 'Oppo%+', 'Soft%+', 'Med%+',
                               'Hard%+']

        self._train_cols_3 = ['Age', 'PA', 'AVG+', 'BB%+', 'K%+', 'OBP+', 'SLG+', 'ISO+', 'BABIP+', 'LD+%', 'GB%+', 
                              'FB%+', 'HR/FB%+', 'Pull%+', 'Cent%+', 'Oppo%+', 'Soft%+', 'Med%+', 'Hard%+', 'wRC+']
        self._impute_cols_3 = ['AVG+-1', 'BB%+-1', 'K%+-1', 'OBP+-1', 'SLG+-1', 'ISO+-1', 'BABIP+-1', 'LD+%-1', 
                               'GB%+-1', 'FB%+-1', 'HR/FB%+-1', 'Pull%+-1', 'Cent%+-1', 'Oppo%+-1', 'Soft%+-1', 
                               'Med%+-1', 'Hard%+-1', 'wRC+-1']

        self._train_cols_4 = ['Age', 'PA', 'AVG+', 'BB%+', 'K%+', 'OBP+', 'SLG+', 'ISO+', 'BABIP+', 'LD+%', 'GB%+',
                              'FB%+', 'HR/FB%+', 'Pull%+', 'Cent%+', 'Oppo%+', 'Soft%+', 'Med%+', 'Hard%+', 'wRC+',
                              'AVG+-1', 'BB%+-1', 'K%+-1', 'OBP+-1', 'SLG+-1', 'ISO+-1', 'BABIP+-1', 'LD+%-1',
                              'GB%+-1', 'FB%+-1', 'HR/FB%+-1', 'Pull%+-1', 'Cent%+-1', 'Oppo%+-1', 'Soft%+-1',
                              'Med%+-1', 'Hard%+-1', 'wRC+-1']
        self._impute_cols_4 = ['AVG+-2', 'BB%+-2', 'K%+-2', 'OBP+-2', 'SLG+-2', 'ISO+-2', 'BABIP+-2', 'LD+%-2', 
                               'GB%+-2', 'FB%+-2', 'HR/FB%+-2', 'Pull%+-2', 'Cent%+-2', 'Oppo%+-2', 'Soft%+-2', 
                               'Med%+-2', 'Hard%+-2', 'wRC+-2']

        self._basic_cols = ['playerid', 'Season', 'Name', 'Team', 'Age', 'PA', 'AVG+', 'BB%+', 'K%+', 'OBP+', 'SLG+',
                            'ISO+', 'BABIP+', 'wRC+', 'y']
        self._one_year_cols = ['playerid', 'Season', 'Name', 'Team', 'Age', 'PA', 'AVG+', 'BB%+', 'K%+', 'OBP+', 'SLG+',
                               'ISO+', 'BABIP+', 'LD+%', 'GB%+', 'FB%+', 'HR/FB%+', 'Pull%+', 'Cent%+', 'Oppo%+',
                               'Soft%+', 'Med%+', 'Hard%+', 'wRC+', 'y']
        self._two_year_cols = ['playerid', 'Season', 'Name', 'Team', 'Age', 'PA', 'AVG+', 'BB%+', 'K%+', 'OBP+', 'SLG+',
                               'ISO+', 'BABIP+', 'LD+%', 'GB%+', 'FB%+', 'HR/FB%+', 'Pull%+', 'Cent%+', 'Oppo%+',
                               'Soft%+', 'Med%+', 'Hard%+', 'wRC+', 'AVG+-1', 'BB%+-1', 'K%+-1', 'OBP+-1', 'SLG+-1',
                               'ISO+-1', 'BABIP+-1', 'LD+%-1', 'GB%+-1', 'FB%+-1', 'HR/FB%+-1', 'Pull%+-1', 'Cent%+-1',
                               'Oppo%+-1', 'Soft%+-1', 'Med%+-1', 'Hard%+-1', 'wRC+-1', 'y']
        self._three_year_cols = ['playerid', 'Season', 'Name', 'Team', 'Age', 'PA', 'AVG+', 'BB%+', 'K%+', 'OBP+',
                                 'SLG+', 'ISO+', 'BABIP+', 'LD+%', 'GB%+', 'FB%+', 'HR/FB%+', 'Pull%+', 'Cent%+',
                                 'Oppo%+', 'Soft%+', 'Med%+', 'Hard%+', 'wRC+', 'AVG+-1', 'BB%+-1', 'K%+-1', 'OBP+-1',
                                 'SLG+-1', 'ISO+-1', 'BABIP+-1', 'LD+%-1', 'GB%+-1', 'FB%+-1', 'HR/FB%+-1', 'Pull%+-1',
                                 'Cent%+-1', 'Oppo%+-1', 'Soft%+-1', 'Med%+-1', 'Hard%+-1', 'wRC+-1', 'AVG+-2',
                                 'BB%+-2', 'K%+-2', 'OBP+-2', 'SLG+-2', 'ISO+-2', 'BABIP+-2', 'LD+%-2', 'GB%+-2',
                                 'FB%+-2', 'HR/FB%+-2', 'Pull%+-2', 'Cent%+-2', 'Oppo%+-2', 'Soft%+-2', 'Med%+-2',
                                 'Hard%+-2', 'wRC+-2', 'y']

    def get_raw_data(self):
        """Get train, validation, and test sets of raw data"""
        return self.raw_data_train, self.raw_data_validate, self.raw_data_test

    def get_one_year_basic(self):
        """Get train, validation, and test sets of 1-Year Basic"""
        return self.one_year_basic_train, self.one_year_basic_validate, self.one_year_basic_test

    def get_one_year_advanced(self):
        """Get train, validation, and test sets of 1-Year Advanced"""
        return self.one_year_advanced_train, self.one_year_advanced_validate, self.one_year_advanced_test

    def get_one_year_all(self):
        """Get train, validation, and test sets of 1-Year All"""
        return self.one_year_all_train, self.one_year_all_validate, self.one_year_all_test

    def get_two_year(self):
        """Get train, validation, and test sets of 2-Year All"""
        return self.two_year_train, self.two_year_validate, self.two_year_test

    def get_three_year(self):
        """Get train, validation, and test sets of 3-Year All"""
        return self.three_year_train, self.three_year_validate, self.three_year_test

    def get_one_year_dtree_imputed(self):
        """Get train, validation, and test sets of 1-Year DT Imputed"""
        return self.one_year_dtree_imputed_train, self.one_year_dtree_imputed_validate, self.one_year_dtree_imputed_test

    def get_two_year_dtree_imputed(self):
        """Get train, validation, and test sets of 2-Year DT Imputed"""
        return self.two_year_dtree_imputed_train, self.two_year_dtree_imputed_validate, self.two_year_dtree_imputed_test

    def get_two_year_dtree_partial_imputed(self):
        """Get train, validation, and test sets of 2-Year DT Partial Imputed"""
        return self.two_year_dtree_partial_imputed_train, self.two_year_dtree_partial_imputed_validate, self.two_year_dtree_partial_imputed_test

    def get_three_year_dtree_imputed(self):
        """Get train, validation, and test sets of 3-Year DT Imputed"""
        return self.three_year_dtree_imputed_train, self.three_year_dtree_imputed_validate, self.three_year_dtree_imputed_test

    def get_three_year_dtree_partial_imputed(self):
        """Get train, validation, and test sets of 3-Year DT Partial Imputed"""
        return self.three_year_dtree_partial_imputed_train, self.three_year_dtree_partial_imputed_validate, self.three_year_dtree_partial_imputed_test

    def get_one_year_linear_imputed(self):
        """Get train, validation, and test sets of 1-Year LR Imputed"""
        return self.one_year_linear_imputed_train, self.one_year_linear_imputed_validate, self.one_year_linear_imputed_test

    def get_two_year_linear_imputed(self):
        """Get train, validation, and test sets of 2-Year LR Imputed"""
        return self.two_year_linear_imputed_train, self.two_year_linear_imputed_validate, self.two_year_linear_imputed_test

    def get_two_year_linear_partial_imputed(self):
        """Get train, validation, and test sets of 2-Year LR Partial Imputed"""
        return self.two_year_linear_partial_imputed_train, self.two_year_linear_partial_imputed_validate, self.two_year_linear_partial_imputed_test

    def get_three_year_linear_imputed(self):
        """Get train, validation, and test sets of 3-Year LR Imputed"""
        return self.three_year_linear_imputed_train, self.three_year_linear_imputed_validate, self.three_year_linear_imputed_test

    def get_three_year_linear_partial_imputed(self):
        """Get train, validation, and test sets of 3-Year LR Partial Imputed"""
        return self.three_year_linear_partial_imputed_train, self.three_year_linear_partial_imputed_validate, self.three_year_linear_partial_imputed_test

    def get_means(self):
        """Get means used to calculate z-scores"""
        return self.means

    def get_sds(self):
        """Get standard deviations used to calculate z-scores"""
        return self.sds
