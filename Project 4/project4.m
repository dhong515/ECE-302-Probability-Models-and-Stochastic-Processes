%% Danny Hong, Arthur Skok, Kenny Huang
% ECE 302 Project 4: Detection

%% Part 1: Radar Detection
clc
clear
close all

%% Question 1a

iterations1 = 1000; %The number of iterations

A_magnitude1 = 1; %Mean difference
variance1 = 1; %Variance
SNR1 = A_magnitude1/variance1; %SNR (Signal-to-Noise) Ratio

p0 = 0.8; %Probability that the target is present
p1 = 0.2; %Probability that the target is not present (p1 = 1 - p0) 
eta1 = p0/p1; %eta

%Generating a Y vector for 2 distributions of variance and mean difference
X1 = sqrt(variance1) * randn(iterations1, 1);
target1 = (rand(iterations1, 1) > p0);

A1 = A_magnitude1 * (target1); 
% initially we thought that we would have to somehow convert this to make it
% work with a regular number (since we're multiplying by a logical data
% type) but matlab does the conversion here already and magnitude is 1 an
Y1 = A1 + X1;

%Obtaining the gamma value from using the MAP decision boundary
gamma1 = (A_magnitude1./2 + variance1 * log(eta1)./(A_magnitude1));

%Finding the Experimental Probability of Error
p_err_Experimental = round((1 - (sum(or(and(Y1 > gamma1, target1), and(Y1 <= gamma1, ~target1)))/iterations1)),2);

%Finding the Theoretical Probability of Error
p10 = (1 - normcdf(gamma1, 0, sqrt(variance1)));
p01 = normcdf(gamma1, A_magnitude1, sqrt(variance1));
p_err_Theoretical = ((p0*p10) + (p1*p01));

%Printing out the Experimental and Theoretical Probabilities of Error (full
%number is ugly)
fprintf("1a.) Experimental Probability of Error = %.3f\n", p_err_Experimental);
fprintf("1a.) Theoretical Probability of Error = %.3f\n", p_err_Theoretical);

% Experimental and Theoretical Probability of Error seem to almost match
% (within one hundredth of each other), which is pretty good. Hovers at around
% 18-20%ish. This number goes down with the number of iterations, as
% expected.
%% Question 1b

%%Applying the same procedure from part A

iterations2 = 10000; %iterations

A_magnitude2 = [0.2, 1, 2, 5]; %Mean differences
variance2 = 1; %variance

eta2 = logspace(-7, 7, iterations2); %eta

%Declaring zero matrices for the true positive probability, and the false positive probability. 
false_positive = zeros(max(size(A_magnitude2)), 1, iterations2);
true_positive = zeros(max(size(A_magnitude2)), 1, iterations2);

%Looping over the set of mean difference values.
for i = 1:max(size(A_magnitude2))
    
    %Generating a Y vector for 2 distributions of variance and mean difference
    X2 = sqrt(variance2) * randn(iterations2, 1);
    target2 = (rand(iterations2, 1) > p0);
    A2 = A_magnitude2(i) * double(target2);
    Y2 = A2 + X2;
    
    %Obtaining the gamma value from using the MAP decision boundary for each mean difference value
    gamma2 = (A_magnitude2(i)./2 + variance2 * log(eta2)./(A_magnitude2(i)));
    
    %Getting the true/false positive probabilities and storing them in their respective arrays
    true_positive(i, :, :) = sum(and(Y2 > gamma2, target2))./sum(target2);
    false_positive(i, :, :) = sum(and(Y2 > gamma2, ~target2))/sum(~target2);
    
end

%Declaring a zero matrix for the SNR
SNR2 = zeros(max(size(A_magnitude2)), 1, iterations2);

figure;

%Looping over the set of mean difference values
for j = 1:max(size(A_magnitude2))
    SNR2(j) = A_magnitude2(j)/variance2; %Calculating the SNR for each mean difference value
    
    %Plotting the ROC plot
    plot(reshape(false_positive(j, :, :), [1, iterations2]), reshape(true_positive(j, :, :), [1, iterations2]), "DisplayName", ['SNR = ', num2str(SNR2(j))])
    hold on
end

%Labeling
legend
title("Receiving Operating Curve (ROC) Plot For Same \sigma^2 Different \mu")
xlabel("false positive rate")
ylabel("true positive rate")

% Based off the plot, we can see that the higher the SNR ratio, the better
% the results are (as in, the higher the rate of true positives and the
% lower the rate of false negatives), which makes perfect sense. The lowest
% SNR we try here results in what would basically be a random guess, as
% it's near a 50-50 chance at that point as it goes along the diagonal (any
% lower than the 45 degree diagonal and we would actually get a better
% predictor than the main diagonal by just inverting our guesses)
%% Question 1c

%Plotting the corresponding ROC first using a = 2 and SNR = 2
figure;
plot(reshape(false_positive(3, :, :), [1, iterations2]), reshape(true_positive(3, :, :), [1, iterations2]), "DisplayName", ['SNR = ',num2str(SNR2(3))])
hold on

iterations3 = 10000; %Number of iterations

A_magnitude3 = 2; %Mean difference
variance3 = 1; %Variance

%Generating a Y vector for 2 distributions of variance and mean difference
X3 = sqrt(variance3) * randn(iterations3, 1);
target3 = (rand(iterations3, 1) > p0);
A3 = A_magnitude3 * double(target3);
Y3 = A3 + X3;

eta3 = 0.1 * (p0/p1); %eta

%Obtaining the gamma value from using the MAP decision boundary for each mean difference value
gamma3 = (A_magnitude3/2 + variance3 * log(eta3)/A_magnitude3);

%Getting the true/false positive probabilities
falsepositive2 = sum(and(Y3 > gamma3, target3))./sum(target3);
truepositive2 = sum(and(Y3 > gamma3, ~target3))/sum(~target3);

%Plotting and Marking the point 
plot(truepositive2, falsepositive2, "*", "DisplayName", "\eta = 0.4")

%Labeling
legend;
title(['Receiving Operating Curve (ROC) Plot with C_{01} = 10*C_{10} (\eta = 0.4) and SNR = ', num2str(SNR2(3))])
xlabel("false positive rate")
ylabel("true positive rate")

%% Question 1e
% For this part we got a lot of help from watching the code review of
% Allister's projected to make the code neater, we were having trouble
% synthesizing the stuff together and the general outline of the code was
% very helpful in pushing through the block we were having.
clear;

%Redo Part a

iterations = 10000; %Number of iterations

A_magnitude = 1; %Mean difference
variance_X = 1; %X variance
variance_Z = 25; %Z variance

p0 = 0.8; %Probability that the target is present
p1 = 0.2; %Probability that the target is not present (p1 = 1 - p0)
eta = p0/p1; %eta 

%Generating a Y vector for the distributions of X and Z variances and mean difference
X = sqrt(variance_X) * randn(iterations, 1); 
Z = sqrt(variance_Z) * randn(iterations, 1);
target = (rand(iterations, 1) > p0);
A = A_magnitude * double(target);
Y = A_magnitude + X.*target + Z.*(~target);

%Obtaining the gamma value from using the MAP decision boundary
gamma = (sqrt(2 * ((variance_X * variance_Z)/(variance_X - variance_Z)) * log(eta * (sqrt(variance_X/variance_Z)))));

%Using anonymous functions to represent P(y|H0) and P(y|H1) for convenience of coding
p_yH1 = @(variance_X, A, Y) (1/sqrt(variance_X *2 * pi)) * (exp(-((Y - A).^2)/(2 * variance_X)));
p_yH0 = @(variance_Z, A, Y) (1/sqrt(variance_Z *2 * pi)) * (exp(-((Y - A).^2)/(2 * variance_Z)));

%Finding the Experimental Probability of Error
p_err_Experimental = sum(or(and(p_yH1(variance_X, A, Y) * p1 >= p_yH0(variance_Z, A, Y) * p0, target), and(p_yH1(variance_X, A, Y) * p1 >= p_yH0(variance_Z, A, Y) * p0, ~target)))/iterations;

%Finding the Theoretical Probability of Error
p10 = (normcdf(gamma, 0, sqrt(variance_Z)) - normcdf(-gamma, 0, sqrt(variance_Z)));
p01 = (2 * (1 - normcdf(gamma, 0, sqrt(variance_X))));
p_err_Theoretical = ((p0*p10) + (p1*p01));

%Printing out the Experimental and Theoretical Probabilites of Error
fprintf("1e.) Experimental Probability of Error = %.3f\n", p_err_Experimental);
fprintf("1e.) Theoretical Probability of Error = %.3f\n", p_err_Theoretical);
% The results were even closer comparing the theoretical and experimental for 
% this part with the values chosen. The ROC was more skewed towards the
% middle though because of the addition of a different parameter to test
% for a not present target. 

%Redo Part b

iterations2 = 10000; %Number of iterations

variance_Z2 = [4, 9, 16, 25]; %Variances of Z
sigma_Z2 = sqrt(variance_Z2); %Standard deviations of Z

eta2 = logspace(-5, 3, 500); %eta

%Declaring zero matrices for the true positive probability, and Pf, the false positive probability. 
false_positive = zeros(max(size(variance_Z2)), 1, 500);
true_positive = zeros(max(size(variance_Z2)), 1, 500);

%Looping over the set of variances of Z
for i = 1:max(size(variance_Z2))
    
       %Generating a Y vector for distributions of X and Z variances and mean difference
       X2 = sqrt(variance_X) * randn(iterations2, 1);
       Z2 = sigma_Z2(i) * randn(iterations2, 1);
       target2 = (rand(iterations2, 1) > p0);
       A2 = A_magnitude * double(target2);
       Y2 = A_magnitude + X2 .* target2 + Z2 .* (~target2);
       
       %Getting the true/false positive probabilities and storing them in their respective arrays
       false_positive(i, :, :) = sum(and(p_yH1(variance_X, A2, Y2) >= p_yH0(variance_Z2(i), A2, Y2) * eta2, ~target2))/sum(~target2);
       true_positive(i, :, :) = sum(and(p_yH1(variance_X, A2, Y2) >= p_yH0(variance_Z2(i), A2, Y2) * eta2, target2))/sum(target2);
       
end

%Declaring a zero matrix for the sigma_z^2/sigma_x^2 ratio
s2_Z_over_s2_X = zeros(max(size(variance_Z2)), 1, iterations);

figure;

%Looping over the set of Z variance values
for j = 1:max(size(variance_Z2))
    s2_Z_over_s2_X(j) = variance_Z2(j)/variance_X; %Calculating the sigma_z^2/sigma_x^2 ratio for each Z variance value
    
    %Plotting the ROC plot
    plot(reshape(false_positive(j, :, :), [1, 500]), reshape(true_positive(j, :, :), [1, 500]), "DisplayName", ['\sigma_z^2/\sigma_x^2= ', num2str(s2_Z_over_s2_X(j))])
    hold on
    
end

%Labeling
legend;
title("Receiver Operating Curve (ROC) Plot For Same \sigma^2 Different \mu")
xlabel("false positive rate");
ylabel("true positive rate");

%% Part 2 Machine Learning : O back again 
clear;

%Loading in the Iris Dataset (good times)
data = load('Iris.mat');

%Shuffling Data So That the Training Set and Testing Set Would Be Chosen Randomly
shuffled = randperm(size(data.features,1));

%Shuffling the Features and Labels
shuffled_Features = data.features(shuffled,:);
shuffled_Labels = data.labels(shuffled,:);

%Splitting the Data (150 Labels, 150 Features) in Half
training_Features = shuffled_Features(1:75, :);
training_Labels = shuffled_Labels(1:75);
testing_Features = shuffled_Features(76:end,:);
testing_Labels = shuffled_Labels(76:end);

%Finding the Labels for Each Class (1, 2, 3)
p1 = training_Features(training_Labels == 1, :);
p2 = training_Features(training_Labels == 2, :);
p3 = training_Features(training_Labels == 3, :);

%Computing the Means
mean1 = [mean(p1(:, 1)), mean(p1(:, 2)), mean(p1(:, 3)), mean(p1(:, 4))];
mean2 = [mean(p2(:, 1)), mean(p2(:, 2)), mean(p2(:, 3)), mean(p2(:, 4))];
mean3 = [mean(p3(:, 1)), mean(p3(:, 2)), mean(p3(:, 3)), mean(p3(:, 4))];

%Computing the Variances
covariance1 = cov(p1);
covariance2 = cov(p2);
covariance3 = cov(p3);

%Using mvnpdf to Help Compute the Max Likelihood with the Means and
%Covariances
likelihood = [mvnpdf(testing_Features, mean1, covariance1), mvnpdf(testing_Features, mean2, covariance2), mvnpdf(testing_Features, mean3, covariance3)];
[~, result] = max(likelihood, [], 2);

%Computing the Probability of Error and printing it
p_Error = (sum(testing_Labels ~= result))/(size(testing_Labels,1));
fprintf("2.) Probability of Error = %.3f\n", p_Error);

%Creating and Displaying the Confusion Matrix
confusion_Matrix = confusionmat(testing_Labels, result);
% (Have to learn more about how confusion matrices actually work on our
% ends)
figure;
confusionchart(confusion_Matrix);
title("Confusion Matrix");
% The probability of error on classification here was very low, I think
% this is because of the generally "high quality" data that the IRIS
% dataset contains, and that it is generally optimized (or at least a quite
% oftenly used set for training models because the data happened to be optimized, not
% necessarily made with it in mind, it just works and that's why people use it haha).
