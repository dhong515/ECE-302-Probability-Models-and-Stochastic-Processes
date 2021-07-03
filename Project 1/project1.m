%% Danny Hong, Arthur Skok, Kenny Huang
% ECE 302 Project 1: Dungeons and Dragons 

%% Question 1
clc
clear 
close all

iterations = 10^5; %declaring the amount of iterations

%% Question 1a

rolls1 = randi(6, 3, iterations); %rolling a set of 3 6-sided die 100000 times 
rolls_sums1 = sum(rolls1); %summing up each set of 3 roll numbers for all the rolls.
find_sum1 = find(rolls_sums1 == 18); %locates the indices for which the sum of a roll equals 18. 
count_sum1 = length(find_sum1); %counts the number of times for which the sum of a roll equals 18.
probability_a = count_sum1 / iterations; %approximates the probability of generating an ability score of 18.

%printing the probability for 1a.
fprintf("1a.) The probability of generating an ability score of 18 for any one role of 3 die is:");
disp(probability_a);

%% Question 1b

rolls2 = randi(6, 3, 3, iterations); %rolling 3 sets of 3 6-sided die 100000 times
rolls_sums2 = sum(max(rolls2)); %summing up the greatest roll numbers from each group of 3 sets.  
find_sum2 = find(rolls_sums2 == 18); %locates the indices for which the sum of a roll equals 18 after using the fun method. 
count_sum2 = length(find_sum2); %counts the number of times for which the sum of a roll equals 18 after using the fun method.
probability_b = count_sum2 / iterations; %approximates the probability of generating an ability score of 18 after using the fun method.

%printing the probability for 1b.
fprintf("1b.) The probability of generating an ability score of 18 after using the fun method of generating 3 scores and keeping the highest is:");
disp(probability_b);

%% Question 1c 

scores_array = zeros(1, iterations); %Initializing an array of zeros to store the ability scores after summing up each set of 6 ability scores

for j = 1:iterations 
    rolls_sums3 = zeros(1, 6); %initiales for each iteration a 1 by 6 array of zeros to store the sums of the roll numbers
    for i = 1:6 
        rolls3 = randi(6, 3, 3); %rolling 3 sets of 3 6-sided die for each one of the six iterations.
        rolls_sums3(i) = sum(max(rolls3));  %summing up the greatest roll numbers from each group of 3 sets and ultimately obtain 6 ability scores.
    end
    scores_array(1, j) = sum(rolls_sums3); %summing up the ability scores
end

find_sum3 = find(scores_array == 108); %locates the indices for which the sum of the ability scores equals 108 after using the fun method.
count_sum3 = length(find_sum3); %counts the number of times for which the sum of the ability scores equals 108 after using the fun method.
probability_c = count_sum3/iterations; %approximates the probability of generating a character with 18's in all ability scores using the fun method.

%printing the probability for 1c.
fprintf("1c.) The probability of generating a character with 18's in all ability scores using the fun method is:");
disp(probability_c);

%% Question 1d

count_sum4 = 0; %declaring the count for the amount of times in which the sum of one roll equals 9.

for j = 1:iterations
    
    rolls_sums4 = zeros(1, 6); %initiales for each iteration a 1 by 6 array of zeros to store the sums of the roll numbers
    for i = 1:6
        rolls4 = randi(6, 3, 1);  %rolling a set of 3 6-sided die once for each of the six iterations.
        rolls_sums4(i) = sum(rolls4); %summing up each set of 3 roll numbers and ultimately obtain six ability scores.
    end 
    find_sum4 = find(rolls_sums4 == 9); %locates the indices for which the sum of one roll (a given ability score) equals 9.
    
    %The number of times for which the sum of one roll equals 9 is found,
    %and if this number is equal to 6, this means that all 6 ability scores
    %are equal to 9 and that the sum of all the ability scores equals 54.
    %In that case, the count is incremented by 1. 
    if length(find_sum4) == 6 
        count_sum4 = count_sum4 + 1;
    end
    
end

probability_d = count_sum4/iterations; %approximation of the probability for part d.

%printing the probability for 1d.
fprintf("1d.) The probability of generating a character with 9's in all ability scores is:");
disp(probability_d);

%% Question 2

clc
clear 
close all

iterations = 10^5;  %declaring the amount of iterations

%% Question 2a

troll_rolls = randi(4, 1, iterations);   %Uniform random distribution for Troll Hit Points
mean_troll_hp = mean(troll_rolls);  %Finding the Average Troll Hit Points

%printing the Average Troll Hit Points for 2a
fprintf("2a.) The Average Troll Hit Points is :");
disp(mean_troll_hp);

fireball_damage = randi(2, 2, iterations); %Since fireballs are 2d2 you have 2 rows, one for each roll
sum_damage = sum(fireball_damage); %To find total damage per fireball.
mean_damage_F = mean(sum_damage); %Average damage per fireball

%printing the Average Damage per Fireball for 2a
fprintf("The Average Damage Per Fireball is :");
disp(mean_damage_F);

% Finding the bounding value on probability:
% This was modified as an example from the presentation on thursday for the
% ease of grabbing only a certain range of values
correct_bounds = sum_damage > 3; 
bounded_probability = sum(correct_bounds)/iterations;

%printing the Bounding Probability Value for 2a
fprintf("The Bounding Probability Value for which the fireball does greater than 3 points of damage is:");
disp(bounded_probability);

%% Question 2b

%Declaring count variables to keep track of the amount of times a fireball
%achieves a particular damage value when it hits. 
count_damage_2 = 0;
count_damage_3 = 0;
count_damage_4 = 0;

%Using a for loop to iterate through the sum damage array and adds one to 
%a particular count variable each time the loop encounters the associated 
%damage value while traversing the array. 
for j = 1:iterations
    if sum_damage(j) == 2
        count_damage_2 = count_damage_2 + 1;
    elseif sum_damage(j) == 3
        count_damage_3 = count_damage_3 + 1;
    else 
        count_damage_4 = count_damage_4 + 1;
    end
end

%Normalizing the count variables over the number of iterations.
fireball_damage_2 = count_damage_2/iterations;
fireball_damage_3 = count_damage_3/iterations;
fireball_damage_4 = count_damage_4/iterations;   

%Declaring count variables to keep track of the amount of times a troll
%achieves a particular hit point value.
count_point_1 = 0;
count_point_2 = 0;
count_point_3 = 0;
count_point_4 = 0;

%Using a for loop to iterate through the troll rolls array and adds one to 
%a particular count variable each time the loop encounters the associated 
%hit point value while traversing the array. 
for j = 1:iterations
    if troll_rolls(j) == 1
        count_point_1 = count_point_1 + 1;
    elseif troll_rolls(j) == 2
        count_point_2 = count_point_2 + 1;
    elseif troll_rolls(j) == 3
        count_point_3 = count_point_3 + 1;
    else
        count_point_4 = count_point_4 + 1;
    end
end

%Normalizing the count variables over the number of iterations.
troll_points_1 = count_point_1/iterations;
troll_points_2 = count_point_2/iterations;
troll_points_3 = count_point_3/iterations;
troll_points_4 = count_point_4/iterations;

%Concatenating results in order to organize them in PMF form
fireballs_pmf = cat(2, 0, fireball_damage_2, fireball_damage_3, fireball_damage_4);
trolls_pmf = cat(2, troll_points_1, troll_points_2, troll_points_3, troll_points_4);

%Displaying Fireball PMF results using stem plots.
figure
stem(fireballs_pmf);
title("2b.) Stem Plot displaying PMF of Fireball Damage");
xlim([0 5]);
xlabel("Fireball Damage Points");
ylabel("Probabilities")

%Displaying Troll PMF results using stem plots.
figure
stem(trolls_pmf);
title("2b.) Stem Plot displaying PMF of Troll Hit Points");
xlim([0 5]);
xlabel("Troll Hit Points");
ylabel("Probabilities")

%% Question 2c

trolls_health = randi(4, 6, iterations);  %6 trolls, 1d4 hit points each
max_trolls_health = zeros(1, iterations); %array of zeros from 1 to number of iterations (max health of trolls)
dead_trolls = zeros(1, iterations); %array of zeros from 1 to number of iterations (fireball damage vs max health of trolls)

%putting max health into array of zeros
for i = 1:iterations
    max_trolls_health(i) = max(trolls_health(i));
end

%checks when fireball kills trolls
for j = 1:iterations
    dead_trolls(j) = max_trolls_health(j) <= fireball_damage(j);
end

%probability of dead trolls
probability_dead_trolls = sum(dead_trolls)/iterations;

%printing the probability for 2c
fprintf("2c.) The probability Keene slays all the trolls with this spell is:");
disp(probability_dead_trolls);

%% Question 2d

% Array preallocation for troll health, arrays of trolls alive, and
% singular troll survivor arrays
troll_health_left = zeros(6, iterations);
survivors = zeros(6, iterations);
single_troll = zeros(1, iterations);

% First figure out what trolls "survived" the fireball by subtracting their
% hp by the damage, and then report the indices of each survivor by using the previous method
% of array conversion with the comparator
for trolls = 1:6
    for trial = 1:iterations
        troll_health_left(trolls, trial) = trolls_health(trolls, trial) - fireball_damage(trial);
        % same idea used here to simplify the array conversion
    end
end

for trolls = 1:6
    for trial = 1:iterations
        survivors(trolls, trial) = troll_health_left(trolls, trial) > 0;
    end
end

% Only choose trials where a single troll survived
for trial = 1:iterations
    % sum up the number of indices that have survivors and only set the 
    % singular troll survivor array to have a true value at the trial index
    % for singular survivor scenarios
    single_troll(:, trial) = sum(survivors(:, trial)) == 1;
end

% this part took some trial and error to get right, needed the troll health
% left to report the health per survivor for the average
% we had the indices of singular troll survivor trials and we want to have
% an array of the indices with solo survivors with their hit points to get
% the average health
survivor_health = single_troll.*max(troll_health_left);
% need the number of trials that had a single survivor to get average of
% their health values
viable_trials = sum(single_troll);
average_survivor_health_expected = sum(survivor_health)/viable_trials;

% Printing the expected hit points that the remaining troll has for 2d
fprintf("2d) The expected amount of hit points that the remaining troll has is  ");
disp(average_survivor_health_expected);

%% Question 2e

hit_chance = randi(20, 2, iterations);   %Hit chance rolling 2d20

sword_of_tuition_damage = randi(6, 2, iterations); %If first 2d20 rolled above 11, deals sword dmg w.r.t. 2d6
total_sword_of_tuition_damage = sum(sword_of_tuition_damage); %Sum of both rolls
hammer_of_tenure_denial_damage = randi(4, 1, iterations); %If second 2d20 rolled above 11, deals hammer dmg w.r.t. 1d4

hit_success = hit_chance >= 11; %Inputs 1 into hit_success array
hit_success(2, :) = hit_success(1,:).*hit_success(2, :);  %If first atk success second atk will succeed

shedjam_damage = cat(1, total_sword_of_tuition_damage, ...  %Combines both weapon damage array into one
                     hammer_of_tenure_denial_damage);                            
total_shedjam_damage = hit_success.*shedjam_damage; %If hit failed element wise multiplication zero out
average_shedjam_damage = sum(sum(total_shedjam_damage))/iterations; %Calculating the average damage for the amount of iterations

%Printing the expected amount of damage done to Keene from Shedjam per attack
fprintf("2e.) The expected amount of damage Shedjam would do to the Wizard Keene per attack is:");
disp(average_shedjam_damage);
