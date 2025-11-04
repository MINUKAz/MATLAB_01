%% Analyze Extracted Features
clear all; close all; clc;

if ~exist('extracted_features.mat', 'file')
    error('extracted_features.mat not found. Please run feature extraction first.');
end

load('extracted_features.mat', 'feature_table', 'label_table', 'user_ids');

fprintf('=== FEATURE ANALYSIS ===\n');

% Basic statistics
fprintf('Total feature vectors: %d\n', height(feature_table));
fprintf('Number of features per trial: %d\n', width(feature_table));
fprintf('Number of users: %d\n', length(unique(user_ids)));

% Display first few feature names
fprintf('\nFirst 10 feature names:\n');
for i = 1:min(10, width(feature_table))
    fprintf('  %d: %s\n', i, feature_table.Properties.VariableNames{i});
end

% Check feature distribution
figure('Position', [100, 100, 1200, 800]);

% Plot feature distributions for first 6 features
for i = 1:6
    subplot(2, 3, i);
    histogram(table2array(feature_table(:, i)), 50);
    title(sprintf('Feature %d: %s', i, feature_table.Properties.VariableNames{i}));
    xlabel('Feature Value');
    ylabel('Frequency');
    grid on;
end

% Visualize feature separability (first 2 features)
figure;
user_groups = unique(user_ids);
colors = hsv(length(user_groups));

for i = 1:length(user_groups)
    user_mask = user_ids == user_groups(i);
    scatter(table2array(feature_table(user_mask, 1)), ...
            table2array(feature_table(user_mask, 2)), ...
            30, colors(i,:), 'filled', 'DisplayName', sprintf('User %d', user_groups(i)));
    hold on;
end

xlabel(feature_table.Properties.VariableNames{1});
ylabel(feature_table.Properties.VariableNames{2});
title('Feature Space - First Two Features');
legend('show');
grid on;

fprintf('\nFeature analysis complete!\n');