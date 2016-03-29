%% Introduction to Data Mathematics Classification Project
% Company Name: SWD Inc.
% Nominal CEO: Takuya(Not in project group)
% Employee: Shuang Guan, Shutong Luo, Zhengneng Chen, Ziao Yan

% First, let's clean any open figures.
clear;
close all;

% Now read raw data.
data_a_initial = csvread('DatasetA.csv')

%% 1. The size of data set "DatasetA.csv"
% number of attributes

% number of points in each class

%% 2. Mean and covariance of class 1
% Hint: the Matlab imagesc command can be a good way to display covariances as images

%% 3. Mean and covariance of class -1
% Hint: the Matlab imagesc command can be a good way to display covariances as images

%% 4. Normal and threshold of Mean mathod

%% 5. Normal and threshold of Fisher LDA

%% 6. The predictive model(Fisher LDA)

% normal_pred = ; uncomment if finished
% threshold_pred = ; uncomment if finished

%% 7. Error in "DatasetA.csv"
% Error of class 1

% Error of class -1

% Total error

%% 8. Estimate our model's performance

%% 9. Test data in "DatasetV.csv" using our model
% Let's read data in 'DatasetV.csv' into a matrix
data_v_initial = csvread('DatasetV.csv')

% Number of points of class 1

% Number of points of class -1
