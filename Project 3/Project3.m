%% Danny Hong, Arthur Skok, Kenny Huang
% ECE 302 Project 3: ML Estimation

clc;
clear;
close all;

%% Part 1

iterations = 1000;

observations = 2:100;
lambda = [0.25 0.5 1];


% Preallocating matrices to make it run faster:
exponential_MSE_1 = zeros(size(observations));
exponential_bias_1 = zeros(size(observations));
exponential_variance_1 = zeros(size(observations));

exponential_MSE_2 = zeros(size(observations));
exponential_bias_2 = zeros(size(observations));
exponential_variance_2 = zeros(size(observations));

exponential_MSE_3 = zeros(size(observations));
exponential_bias_3 = zeros(size(observations));
exponential_variance_3 = zeros(size(observations));

rayleigh_MSE_1 = zeros(size(observations));
rayleigh_bias_1 = zeros(size(observations));
rayleigh_variance_1 = zeros(size(observations));

rayleigh_MSE_2 = zeros(size(observations));
rayleigh_bias_2 = zeros(size(observations));
rayleigh_variance_2 = zeros(size(observations));

rayleigh_MSE_3 = zeros(size(observations));
rayleigh_bias_3 = zeros(size(observations));
rayleigh_variance_3 = zeros(size(observations));


index = 1;

for i = observations
    % populating from the exponential distribution
    exponential_1 = exprnd(1/lambda(1), [iterations i]);
    exponential_2 = exprnd(1/lambda(2), [iterations i]);
    exponential_3 = exprnd(1/lambda(3), [iterations i]);
    
    % First we get the estimators:
    exp_lambda_hat_1 = i./sum(exponential_1, 2);
    exp_lambda_hat_2 = i./sum(exponential_2, 2);
    exp_lambda_hat_3 = i./sum(exponential_3, 2);
    
    % populating from the rayleigh distribution
    rayleigh_1 = raylrnd(lambda(1), [iterations i]);
    rayleigh_2 = raylrnd(lambda(2), [iterations i]);
    rayleigh_3 = raylrnd(lambda(3), [iterations i]);
    
    ray_lambda_hat_1 = sqrt(0.5 * mean(rayleigh_1.^2, 2));
    ray_lambda_hat_2 = sqrt(0.5 * mean(rayleigh_2.^2, 2));
    ray_lambda_hat_3 = sqrt(0.5 * mean(rayleigh_3.^2, 2));
    
    % And now we get the MSE, bias, variance:
    exponential_MSE_1(index) = mean((lambda(1) - exp_lambda_hat_1).^2);
    exponential_bias_1(index) = mean(exp_lambda_hat_1) - lambda(1);
    exponential_variance_1(index) = var(exp_lambda_hat_1);
    
    exponential_MSE_2(index) = mean((lambda(2) - exp_lambda_hat_2).^2);
    exponential_bias_2(index) = mean(exp_lambda_hat_2) - lambda(2);
    exponential_variance_2(index) = var(exp_lambda_hat_2);
    
    exponential_MSE_3(index) = mean((lambda(3) - exp_lambda_hat_3).^2);
    exponential_bias_3(index) = mean(exp_lambda_hat_3) - lambda(3);
    exponential_variance_3(index) = var(exp_lambda_hat_3);
    
    rayleigh_MSE_1(index) = mean((lambda(1) - ray_lambda_hat_1).^2);
    rayleigh_bias_1(index) = mean(ray_lambda_hat_1) - lambda(1);
    rayleigh_variance_1(index) = var(ray_lambda_hat_1);
    
    rayleigh_MSE_2(index) = mean((lambda(2) - ray_lambda_hat_2).^2);
    rayleigh_bias_2(index) = mean(ray_lambda_hat_2) - lambda(2);
    rayleigh_variance_2(index) = var(ray_lambda_hat_2);
    
    rayleigh_MSE_3(index) = mean((lambda(3) - ray_lambda_hat_3).^2);
    rayleigh_bias_3(index) = mean(ray_lambda_hat_3) - lambda(3);
    rayleigh_variance_3(index) = var(ray_lambda_hat_3);

    index = index + 1;
end

figure;
plot(observations, exponential_MSE_1, observations, exponential_MSE_2, observations, exponential_MSE_3);
title("MSE of Exponential Distribution on Observations");
xlabel("Observations");
ylabel("MSE");
legend("lambda = " + lambda(1), "lambda = " + lambda(2), "lambda = " + lambda(3));
    
figure;
plot(observations, exponential_bias_1, observations, exponential_bias_2, observations, exponential_bias_3);
title("Bias of Exponential Distribution on Observations");
xlabel("Observations");
ylabel("Bias");
legend("lambda = " + lambda(1), "lambda = " + lambda(2), "lambda = " + lambda(3));
   
figure;
plot(observations, exponential_variance_1, observations, exponential_variance_2, observations, exponential_variance_3);
title("Variance of Exponential Distribution on Observations");
xlabel("Observations");
ylabel("Variance");
legend("lambda = " + lambda(1), "lambda = " + lambda(2), "lambda = " + lambda(3));
 
figure;
plot(observations, rayleigh_MSE_1, observations, rayleigh_MSE_2, observations, rayleigh_MSE_3);
title("MSE of Rayleigh Distribution on Observations");
xlabel("Observations");
ylabel("MSE");
legend("lambda = " + lambda(1), "lambda = " + lambda(2), "lambda = " + lambda(3));
   
    
figure;
plot(observations, rayleigh_bias_1, observations, rayleigh_bias_2, observations, rayleigh_bias_3);
title("Bias of Rayleigh Distribution on Observations");
xlabel("Observations");
ylabel("Bias");
legend("lambda = " + lambda(1), "lambda = " + lambda(2), "lambda = " + lambda(3));
   
figure;
plot(observations, rayleigh_variance_1, observations, rayleigh_variance_2, observations, rayleigh_variance_3);
title("Variance of Rayleigh Distribution on Observations");
xlabel("Observations");
ylabel("Variance");
legend("lambda = " + lambda(1), "lambda = " + lambda(2), "lambda = " + lambda(3));
 
% When looking over the graphs, for the Rayleigh distribution there seems
% to be a bias/variance tradeoff, similar to something we've seen in
% Machine Learning previously which is cool. Otherwise, MSE and variances decrease
% as the number of observations being used increases, which intuitively
% makes sense. 
%% Part 2
clear;
clc;
data = load('data.mat');
data = data.data;
% For exponential distribution estimator:
expL = size(data)/sum(data);

% Log of PDF of an exponential distribution, summed up:
exponential_Likelihood=sum(log(expL(2) * exp(-expL(2) * data)));

% For Rayleigh distribution estimator:
rayL = sqrt(sum(data.^2)./size(data));

% Log of Pdf of a Rayleigh distribution, summed up:
rayleigh_Likelihood = sum(((-1 * transpose(data)).^2)/(2*rayL(2)^2) .* log(transpose(data)./rayL(2)^2));
difference = rayleigh_Likelihood - exponential_Likelihood;

fprintf("More likely to be Rayleigh, higher likelihood for the parameter cumulatively. Higher by: %f\n", difference)