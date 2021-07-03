%% Danny Hong, Arthur Skok, Kenny Huang
% ECE 302 Project 5: MMSE estimation
clc;
clear;
close all;

N = [4, 6, 10];                     % Simulate the system for filters of length N = 4, 6 and 10
c = [1 0.2 0.4];                    % c[n] is an FIR filter with impulse response of [1 .2 .4]
std_noise = 0.5;                    % standard deviation of noise
MSE = zeros(1, length(N));          % array to store MSE of each N value
signal_length = 1000;               % signal length
s = randi(2, 1, signal_length);     % s[n] is an i.i.d processes which takes value +/-1 with equal probability for each sample
s(s == 2) = -1;
[Rss, lags] = xcorr(s);             % index 1000 is time delay = 0;
y = filter(c, 1, s);                % output of 1st filter c[n]
len_y = length(y)

% y + d[n] -> input of 2nd filter h[n]
r = y + normrnd(0, std_noise, 1, len_y);    

% For each N
for m = 1:length(N)
    Rsr = xcorr(s,r);                   % Rsr[n] is the cross-correlation of the observations R[n]
    Rrr = xcorr(r);                     % Rrr[n] is the auto-correlation of the observations
    Rsr_half = (length(Rsr) + 1) / 2;   % Getting the middle value of Rsr
    Rrr_half = (length(Rrr) + 1) / 2;   % Getting the middle value of Rrr
    left_side = zeros(N(m));
    right_side = transpose(Rsr(Rsr_half : Rsr_half + N(m)-1));
    
    for i = 1:N(m)
        left_side(i, :) = transpose(Rrr(Rrr_half - i + 1 : Rrr_half - i + N(m))); 
    end
    
    h = left_side \ right_side;                     % slove for h[n]
    s_hat = filter(transpose(h), 1, r);             % output of 2nd filter h[n]
    MSE(m) = sum((s_hat - s).^2) / signal_length;   % design a filter h[n] to estimate s[n] from r[n] such that s_hat[n] is an MMSE estimate
end

table(MSE(1), MSE(2), MSE(3), 'VariableNames', ["N = 4", "N = 6", "N = 10"], 'RowNames', "MSE")