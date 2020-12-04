The code in this project is divided into 7 different files: 3 classes which assisted with model building and data manipulation and 4 scripts which were run sequentially at each step of the analysis. All of the code in this project was written by me. Below is a brief description of each of the files and their uses:

Data Manipulator (167 lines) - This class takes a pandas Dataframe of baseball stats as input. It provides methods to manipulate the data - imputation, multiyear,
remove nulls, etc. - which were used to test a variety of models for best results. It was written specifically for data downloaded from Fangraphs but is general enough to be expanded for pitching and fielding models.

Data Slicer (320 lines) - This class takes a pandas Dataframe as input. This class works with the Data Manipulator and was used to generate the training sets as described in the project report.

Model Builder (174 lines) - This class takes pandas Dataframes for the train and test/validation data as input. It provides methods to easily test each of the different models using the methodology described in the report and return the RMSE.

Z-score Paramter Testing (85 lines) - This script is the first script to run sequentially. It generates the graphs which can be used to select the k and depth parameters of the K-Nearest Neighbors and Decision Tree models. 

Z Scores PCA Testing (88 lines) - After running the parameter testing script, the ks and depths values were selected and set in the PCA Testing file. This script was used to generate the graphs for selecting the number of dimensions PCA should reduce to in the final models.

Z Scores RMSEs (73 lines) - Using the dimensions from the PCA Testing analysis, this script generated the final validation scores. It prints a dictionary of RMSEs for each model and training set which was used for most of the analysis in the report.

Z Scores Comparison (121 lines) - This final script runs the test data for each of the four models and prints their results. It also created scatter plots for each comparing predicted vs actual and predicted vs previous year.