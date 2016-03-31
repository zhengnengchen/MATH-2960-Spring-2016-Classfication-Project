%% Introduction to Data Mathematics Classification Project
% Company Name: SWD Inc.
% Nominal CEO: Takuya(Not in project group)
% Employee: Shuang Guan, Shutong Luo, Zhengneng Chen, Ziao Yan

% First, let's clean any open figures.
clear;
clc;
close all;

% Now read raw data.
a_raw = csvread('DatasetA.csv');
a = a_raw(:,2:(end - 1));
label = a_raw(:, (end));

%% 1. The size of data set "DatasetA.csv"
% Number of attributes
attributes = size(a, 2);
disp('Number of attributes in DatasetA is ');
disp(attributes);

% Number of points in each class
points = size(a, 1);
disp('Number of points in DatasetA is ');
disp(points);

%% 2. The mean and covariance of each class in 'DatasetA.csv'
% Hint: the Matlab imagesc command can be a good way to display covariances as images
format long;

ap = a(label == 1, :);
ap_mean = mean(ap);
ap_size = size(ap, 1);
ap_var = ap - ones(ap_size,1) * ap_mean;
ap_cov = 1/(ap_size - 1) * (ap_var' * ap_var);
disp('Mean of class 1 in DatasetA is ');
disp(ap_mean);

% Create a grayscale heatmap of class 1's covariances
figure
imagesc(ap_cov)
hold on
title('Heatmap of covariance of class 1');
colormap(gray)
colorbar
hold off

am = a(label == -1, :);
am_mean = mean(am);
am_size = size(am, 1);
am_var = am - ones(am_size,1) * am_mean;
am_cov = 1/(am_size - 1) * (am_var' * am_var);
disp('Mean of class -1 in DatasetA is ');
disp(am_mean);

% Create a grayscale heatmap of class -1's covariances
figure
imagesc(am_cov)
hold on
title('Heatmap of covariance of class -1');
colormap(gray)
colorbar
hold off

%% 3. The mean and covariance of all of 'DatasetV.csv'
v_raw = csvread('DatasetV.csv');
v = v_raw(:, 2:(end));
format long;

v_mean = mean(v);
v_size = size(v, 1);
v_var = v - ones(v_size,1) * v_mean;
v_cov = 1/(v_size - 1) * (v_var * v_var');
disp('Mean of DatasetV is ');
disp(v_mean);

% Create a grayscale heatmap of class -1's covariances
figure
imagesc(v_cov)
hold on
title('Heatmap of covariance of DatasetV');
colormap(gray)
colorbar
hold off

%% 4. Create train and test set
% Classp_train  := Class 1 training data
% Classm_train  := Class -1 training data
% Classp_test   := Class 1 testing data
% Classm_test   := Class -1 testing  data

% Set random number to an initial seed
seed = RandStream('mt19937ar', 'Seed', 550);

% Generate a permutation of the data
permutation = randperm(seed, points);
a = a(permutation,:);
label = label(permutation);

% 90% of DatasetA is used for training and 10% for testing
train_percent = 0.9;
train_size = ceil(points * train_percent);

% Grab training and testing data
a_train = a(1:train_size, :);
a_test = a(train_size + 1:end, :);

label_train = label(1:train_size, :);
label_test = label(train_size + 1:end, :);

% Break them up into Class 1 and Class -1
ap_train = a_train(label_train == 1, :);
am_train = a_train(label_train == -1, :);

ap_test = a_test(label_test == 1, :);
am_test = a_test(label_test == -1, :);

%% 5. Normal and threshold of Mean method
format long;

% Calculate w as difference of the class means
ap_train_mean = mean(ap_train);
am_train_mean = mean(am_train);
w_mean = (ap_train_mean - am_train_mean)';
w_mean = w_mean / norm(w_mean);
disp('Normal of Mean method is ');
disp(w_mean);

% Calculate threshold t
t_mean = (ap_train_mean + am_train_mean) / 2 * w_mean;
disp('Threshold of Mean method is ');
disp(t_mean);

%% 6. Normal and threshold of Fisher LDA
format long;

ap_train_var = ap_train - ones(size(ap_train,1), 1) * ap_mean;
am_train_var = am_train - ones(size(am_train,1), 1) * am_mean;
w_fisher = ap_train_var' * ap_train_var + am_train_var' * am_train_var;
w_fisher = w_fisher \ (ap_mean - am_mean)';

% Normal of fisher LDA
w_fisher = w_fisher / norm(w_fisher);
disp('Normal of Fisher LDA method is ');
disp(w_fisher);

% Threshold of Fisher LDA
t_fisher = (ap_mean + am_mean) ./ 2 * w_fisher;
disp('Threshold of Fisher LDA method is ');
disp(t_fisher);

%% 7. Error analysis for data of "DatasetA.csv"
% Training error of mean method
MeanPosErrorTrain = sum(ap_train * w_mean <= t_mean);
MeanNegErrorTrain = sum(am_train * w_mean >= t_mean);
MeanTrainError = (MeanPosErrorTrain + MeanNegErrorTrain) / (size(ap_train, 1) + size(am_train, 1));
disp('Training error using Mean method is ');
disp(MeanTrainError);
HistClass(ap_train, am_train, w_mean, t_mean, 'Mean Method Training Results', MeanTrainError);

% Training error of Fisher LDA method
FisherPosErrorTrain = sum(ap_train * w_fisher <= t_fisher);
FisherNegErrorTrain = sum(am_train * w_fisher >= t_fisher);
FisherTrainError = (FisherPosErrorTrain + FisherNegErrorTrain) / (size(ap_train, 1) + size(am_train, 1));
disp('Training error using Fisher LDA method is ');
disp(FisherTrainError);
HistClass(ap_train, am_train, w_fisher, t_fisher, 'Fisher LDA Method Training Results', FisherTrainError);

% Testing error of mean method
MeanPosErrorTest = sum(ap_test * w_mean <= t_mean);
MeanNegErrorTest = sum(am_test * w_mean >= t_mean);
MeanTestError = (MeanPosErrorTest + MeanNegErrorTest) / (size(ap_test, 1) + size(am_test, 1));
disp('Testing error using Mean method is ');
disp(MeanTestError);
HistClass(ap_test, am_test, w_mean, t_mean, 'Mean Method Testing Results', MeanTestError);

% Testing error of Fisher LDA method
FisherPosErrorTest = sum(ap_test * w_fisher <= t_fisher);
FisherNegErrorTest = sum(am_test * w_fisher >= t_fisher);
FisherTestError = (FisherPosErrorTest + FisherNegErrorTest) / (size(ap_test, 1) + size(am_test, 1));
disp('Testing error using Fisher LDA method is ');
disp(FisherTestError);
HistClass(ap_test, am_test, w_fisher, t_fisher, 'Fisher LDA Method Testing Results', FisherTestError);

%% 8. The predictive model(Fisher LDA)
% Because using Mean method has an total error of 33.37% on training data
% and 41.90% on testing data and using Fisher LDA method has an total error of
% 13.79% on training data and 16.19% on test data.
% We use Fisher LDA method as our predictive model
normal_pred = w_fisher;
threshold_pred = t_fisher;
disp('Normal of our predictive model is ');
disp(normal_pred);
disp('Threshold of our predictive model is ');
disp(threshold_pred);

%% 9. Performance of our model doing on 'DatasetA.csv'
% Class 1 error
ap_error = (FisherPosErrorTrain + FisherPosErrorTest) / (size(ap_train, 1) + size(ap_test, 1));
disp('Error of class 1 using our predictive model is ')
disp(ap_error);

% Class -1 error
am_error = (FisherNegErrorTrain + FisherNegErrorTest) / (size(am_train, 1) + size(am_test, 1));
disp('Error of class -1 using our predictive model is ')
disp(am_error);

% Total error
total_error = (FisherPosErrorTrain + FisherPosErrorTest + FisherNegErrorTrain + FisherNegErrorTest) / (size(a, 1));
disp('Total error using our predictive model is ')
disp(total_error);

%% 10. Estimate our model's performance
% As we tested a set of data in 'DatasetA.csv' by using Mean method and
% Fisher LDA method, we have a conclusion that Mean method have an total error in
% range of 30-40% varing due to data set's size. Meanwhile, Fisher LDA
% method give us an total error of 10-20% varing due to data set's size.

% Let's have a look about the test data's size in 'DatasetA.csv' and the to-be-test
% data's size in 'DatasetV.csv'
a_test_size = size(a_test, 1);
v_size = size(v, 1);

% As we tested before, training set has a size of 1055 - 105 = 950 and has an error of
% 13.79%. Data in 'DatasetV.csv's size is 400 which is less than 950 but
% greater than 105. It's reasonable to estimate that the total error of using
% Fisher LDA method to test 'DatasetV.csv' is around 15% which between
% 13.79% and 16.19%

% As we already tested that class 1's error is 14.3% and class -1's error
% is 13.9%. We can reasonable induct that class 1 and class -1's error on
% 'DatasetV.csv' is around 14% by using Fisher LDA method.
disp('Estimate error of class 1 and class -1 in DatasetV is around 14% and estimate total error is around 15%.')
%% 11. Test data in "DatasetV.csv" using our model
% Number of points of class 1
vp_size = sum(v * w_fisher > t_fisher);
disp('Number of points of class 1 in DatasetV is ')
disp(vp_size);

% Number of points of class -1
vm_size = sum(v * w_fisher < t_fisher);
disp('Number of points of class -1 in DatasetV is ')
disp(vm_size);

%% 12. Write into 'DatasetV.csv'

v_label = (v * w_fisher > t_fisher) - (v * w_fisher < t_fisher);
v_bas = [v_raw, v_label];
csvwrite('DatasetV using Fisher LDA by SWD Inc.csv', v_bas);

%% 13. Some improvement besides our predictive model
% As linear SVM is also a good classifier, we check if it's better than
% Fisher LDA method
SVMStruct = svmtrain(a_train, label_train);
Group = svmclassify(SVMStruct, a_test);
svm_test_error = sum(Group ~= label_test) / size(Group, 1);
disp('Test error using Linear SVM Classifier is ');
disp(svm_test_error);
disp(' which is better than 16.19% of Fisher LDA method.')

% Here we use Matlab build-in SVM to produce a reference label for DatasetV.
SVMStruct = svmtrain(a, label);
Group = svmclassify(SVMStruct, v);
v_ref = [v_raw, Group];
csvwrite('DatasetV using SVM by SWD Inc.csv', v_ref);
