%% Danny Hong, Arthur Skok, Kenny Huang
% ECE 302 Project 2: MMSE Estimation

%% Scenario 1 
clc
clear
close all

%Declaring the number of iterations.
iterations = 10^7; 

Y = -1 + (2*rand(iterations, 1)); %Simulating 10^7 iterations of random draws of Y with type double values ranging from -1 to 1.
W = -2 + (4*rand(iterations, 1)); %Simulating 10^7 iterations of random draws of W with type double values ranging from -2 to 2.
X = Y + W; %Adding the Y and W values together in order to obtain the observations X.

%Declaring the theoretical Linear MMSE value that was given from the MIT notes.
Linear_MMSE_Empirical = 4/15; 

%Declaring the theoretical Linear MMSE value that was given from the MIT notes.
Linear_MMSE_Theoretical = 1/4;

%Obtaining the Linear y_hat value by implementing  the Linear MMSE Estimator Function from example 8.6 of the MIT notes.
Linear_y_hat = (1/5)*X; 

%Obtaining the Bayes y_hat value by implementing the Bayes MMSE Estimator Function from example 8.5 of the MIT notes. This function was simplified from the original for ease of coding.
Bayes_y_hat = ((X+1)/2).*and(X>=-3, X<-1) + 0.*and(X>=-1, X<1) + ((X-1)/2).*and(X>=1, X<=3); 

%Obtaining the corresponding Linear MMSE experimental value after finding the Linear y_hat value.
Linear_MMSE_Experimental = mean((Y - Linear_y_hat).^2);

%Obtaining the corresponding Bayes MMSE experimental value after finding the Bayes y_hat value.
Bayes_MMSE_Experimental = mean((Y - Bayes_y_hat).^2);

%Creating a table that lists and compares the theoretical Linear and Bayes MMSE values with our experimental Linear and Bayes MMSE values. 
table_of_values = table([Linear_MMSE_Empirical;
    Linear_MMSE_Theoretical], [Linear_MMSE_Experimental;
    Bayes_MMSE_Experimental], 'RowNames', {'Linear MMSE', 'Bayes MMSE'}, 'VariableNames', {'Theoretical Value'; 'Experimental Value'});
disp(table_of_values); %Displaying that table.

%Our experimental results for the Linear MMSE and Bayes MMSE values seemed
%to match as the amount of iterations was increased. For the most part, after using
%the estimator functions provided by the MIT notes, the MMSE values obtained seemed 
%to improve since they were slightly lower than the theoretical MMSE values given from 
%the MIT notes that were obtained from using the expected value of Y.


%% Scenario 2
% Implement the linear estimator for multiple noisy observations, similar to example 8.8 from the notes.
% Extend this example so that it works for an arbitrary number of observations.
% Use Gaussian random variables for Y and R. Set mean y = 1.
% Experiment with a few different variances for both Y and R.
% On one plot, show the mean squared error of your simulation compared to the theoretical values for at least 2 different pairs of variances.

clc;
clear all;
close all;

Y_mean = 1;
% First pair of variances:
Y_var1 = 1;
% R has zero mean
R_var1 = 1;
iterations = 1000;
N_observations = 100;
Y_standard_dev1 = sqrt(Y_var1);
N_observations_multiple = 2:N_observations;
% Making Y Gaussian:
Y = Y_standard_dev1 * randn(1,iterations) + Y_mean; 

% Empty array preallocation for plotting, Matlab was complaining about
% memory issues without it as the arrays were redefined
Linear_MMSE_Empirical = zeros(1, N_observations);
Linear_MMSE_Theoretical = zeros(1, N_observations);
observations_x_axis = zeros(1, N_observations);

for i = N_observations_multiple
    % Get X, an observation (Sum of R and Y, get R)
    R = R_var1.*randn(i,iterations);
    X = Y + R;
    % Equation 8.79 from MIT notes with "i" in denominator to scale
    Y_hat = (R_var1*Y_mean+Y_var1*sum(X))/((i*Y_var1)+R_var1);
    % Used to keep track of the x_axis and the estimates for that number of
    % observations
    observations_x_axis(i) = i;
    %Obtaining the corresponding Linear MMSE experimental value after finding the Linear y_hat value.
    Linear_MMSE_Empirical(i) = mean((Y-Y_hat).^2);
    %Obtaining the corresponding Linear MMSE theoretical value.
    Linear_MMSE_Theoretical(i) = (Y_var1*R_var1)/((i*Y_var1)+R_var1);
end
% Second pair of variances:
Y_var2 = 2;
R_var2 = 2;
Y_standard_dev2 = sqrt(Y_var2);
N_observations_multiple = 2:N_observations;
Y_2 = Y_standard_dev2 * randn(1,iterations) + Y_mean; % Making Y_2 Gaussian:
% R_2 has zero mean

% Empty array preallocation for plotting
Linear_MMSE_Empirical_2 = zeros(1, N_observations);
Linear_MMSE_Theoretical_2 = zeros(1, N_observations);
observations_x_axis_2 = zeros(1, N_observations);

for i = N_observations_multiple
    %Get X_2, an observation (Sum of R_2 and Y_2, get R_2)
    R_2 = R_var2.*randn(i,iterations);
    X_2 = Y_2 + R_2;
    Y_hat_2 = (R_var2*Y_mean+Y_var2*sum(X_2))/((i*Y_var2)+R_var2);
    observations_x_axis_2(i) = i;
    Linear_MMSE_Empirical_2(i) = mean((Y_2-Y_hat_2).^2);
    Linear_MMSE_Theoretical_2(i) = (Y_var2*R_var2)/((i*Y_var2)+ R_var2);
end
% Third pair of variances:
Y_var3 = 1/3;
R_var3 = 1/3;
Y_standard_dev2 = sqrt(Y_var3);
N_observations_multiple = 2:N_observations;
Y_3 = Y_standard_dev2 * randn(1,iterations) + Y_mean; % Making Y_3 Gaussian:
% R_3 has zero mean

% Empty array preallocation for plotting
Linear_MMSE_Empirical_3 = zeros(1, N_observations);
Linear_MMSE_Theoretical_3 = zeros(1, N_observations);
observations_x_axis_3 = zeros(1, N_observations);

for i = N_observations_multiple
    %Get X_3, an observation (Sum of R_3 and Y_3, get R_3)
    R_3 = R_var3.*randn(i,iterations);
    X_3 = Y_3 + R_3;
    Y_hat_3 = (R_var3*Y_mean+Y_var3*sum(X_3))/((i*Y_var3)+R_var3);
    observations_x_axis_3(i) = i;
    Linear_MMSE_Empirical_3(i) = mean((Y_3-Y_hat_3).^2);
    Linear_MMSE_Theoretical_3(i) = (Y_var3*R_var3)/((i*Y_var3)+R_var3);
end

% Plotting the three above results for each pair of variances:
figure('Name', 'Scenario 2: Example 8.8', 'NumberTitle', 'off');
hold on;
plot(observations_x_axis(2:end), Linear_MMSE_Theoretical(2:end))
plot(observations_x_axis(2:end), Linear_MMSE_Empirical(2:end))
plot(observations_x_axis_3(2:end), Linear_MMSE_Theoretical_2(2:end))
plot(observations_x_axis_2(2:end), Linear_MMSE_Empirical_2(2:end))
plot(observations_x_axis_3(2:end), Linear_MMSE_Theoretical_3(2:end))
plot(observations_x_axis_3(2:end), Linear_MMSE_Empirical_3(2:end))

legend('Theoretical:Yvar_1 = 1 Rvar_1 = 1', 'Empirical:Yvar_1 = 1 Rvar_1 = 1', 'Theoretical:Yvar_2 = 2 Rvar_2 = 2', 'Empirical:Yvar_2 = 2 Rvar_2 = 2', 'Theoretical:Yvar_3 = 1/2 Rvar_3 = 1/2', 'Empirical:Yvar_3 = 1/2 Rvar_3 = 1/2');
title('LMSE with Generalized # of Observations Made');
xlabel('Number of Observations');
ylabel('MMSE');
hold off;

% As the number of observations increased for any pair of variances, the
% MMSE decreased as expected. The theoretical and experimental results
% almost clearly lined up (which is great), and with higher variances, the
% MMSE increased as would make sense, as the data is less predictable.
% (Variance of the noise and signal).

% The graphing portion of the exercise for scenario 2 introduced a ton of
% redundancies, which could probably be fixed with another refined "for
% loop" for each of the pairs of variances...

