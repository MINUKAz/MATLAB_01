%% Train Final Optimized Model - Enhanced Version
clear all; close all; clc;

fprintf('=== TRAINING FINAL OPTIMIZED MODEL ===\n');
fprintf('Enhanced version - supports all optimization approaches\n');

% Load features
load('extracted_features.mat', 'feature_table', 'label_table', 'user_ids');

% Convert data
X = table2array(feature_table);
y_original = table2array(label_table);
y_categorical = categorical(y_original);
y_onehot = dummyvar(y_categorical)';

%% Determine Best Configuration from Available Optimization Results
fprintf('\n--- Loading Optimization Results ---\n');

optimization_sources = {};
best_configs = [];

% Check for comprehensive optimization results
if exist('optimization_results.mat', 'file')
    load('optimization_results.mat', 'results_hidden', 'results_algorithms', 'results_splits');
    
    % Find best configuration from comprehensive optimization
    [best_hidden_acc, best_hidden_idx] = max([results_hidden.accuracy]);
    [best_algo_acc, best_algo_idx] = max([results_algorithms.accuracy]);
    [best_split_acc, best_split_idx] = max([results_splits.accuracy]);
    
    config.hidden_size = results_hidden(best_hidden_idx).hidden_size;
    config.algorithm = results_algorithms(best_algo_idx).algorithm;
    config.split_ratio = results_splits(best_split_idx).train_ratio;
    config.accuracy = (best_hidden_acc + best_algo_acc + best_split_acc) / 3; % Average
    config.source = 'Comprehensive Optimization';
    
    best_configs = [best_configs, config];
    optimization_sources{end+1} = 'Comprehensive (neural_network_optimization.m)';
    fprintf('✓ Loaded comprehensive optimization results\n');
end

% Check for fast optimization results
if exist('quick_optimization_results.mat', 'file')
    load('quick_optimization_results.mat', 'quick_results');
    
    % Find best from quick optimization
    quick_accuracies = [quick_results.accuracy];
    [best_quick_acc, best_quick_idx] = max(quick_accuracies);
    
    if isfield(quick_results(best_quick_idx), 'hidden_size')
        config.hidden_size = quick_results(best_quick_idx).hidden_size;
        config.algorithm = 'trainscg'; % Default for quick optimization
        config.split_ratio = 0.8; % Default for quick optimization
    else
        config.hidden_size = 64; % Default from quick optimization
        config.algorithm = quick_results(best_quick_idx).algorithm;
        config.split_ratio = 0.8;
    end
    
    config.accuracy = best_quick_acc;
    config.source = 'Fast Optimization';
    
    best_configs = [best_configs, config];
    optimization_sources{end+1} = 'Fast (fast_optimization.m)';
    fprintf('✓ Loaded fast optimization results\n');
end

% Check for targeted optimization results
if exist('optimization_report.mat', 'file')
    load('optimization_report.mat', 'optimization_report');
    
    config.hidden_size = optimization_report.tested_hidden_size;
    config.algorithm = 'trainscg'; % Default for targeted
    config.split_ratio = 0.8; % Default for targeted
    config.accuracy = optimization_report.new_accuracy;
    config.source = 'Targeted Optimization';
    
    best_configs = [best_configs, config];
    optimization_sources{end+1} = 'Targeted (targeted_optimization.m)';
    fprintf('✓ Loaded targeted optimization results\n');
end

% If no optimization results found, use default
if isempty(best_configs)
    fprintf('No optimization results found. Using default configuration.\n');
    best_config.hidden_size = 64;
    best_config.algorithm = 'trainscg';
    best_config.split_ratio = 0.8;
    best_config.source = 'Default';
    best_config.accuracy = NaN;
else
    % Choose the configuration with highest accuracy
    accuracies = [best_configs.accuracy];
    [~, best_idx] = max(accuracies);
    best_config = best_configs(best_idx);
end

%% Display Configuration Information
fprintf('\n=== SELECTED OPTIMIZED CONFIGURATION ===\n');
fprintf('Source: %s\n', best_config.source);
fprintf('Hidden Layer Size: %d neurons\n', best_config.hidden_size);
fprintf('Training Algorithm: %s\n', best_config.algorithm);
fprintf('Train/Test Split: %.0f/%.0f\n', best_config.split_ratio*100, (1-best_config.split_ratio)*100);

if ~isnan(best_config.accuracy)
    fprintf('Expected Accuracy: %.2f%%\n', best_config.accuracy);
end

fprintf('\nAvailable optimization sources:\n');
for i = 1:length(optimization_sources)
    fprintf('  %d. %s\n', i, optimization_sources{i});
end

%% Split Data with Selected Ratio
rng(42);
test_ratio = 1 - best_config.split_ratio;
cv = cvpartition(length(y_original), 'HoldOut', test_ratio);

X_train = X(cv.training,:)';
X_test = X(cv.test,:)';
y_train = y_onehot(:, cv.training);
y_test = y_onehot(:, cv.test);

fprintf('\nData split:\n');
fprintf('Training set: %d samples\n', size(X_train, 2));
fprintf('Testing set: %d samples\n', size(X_test, 2));

%% Create and Train Optimized Network
fprintf('\n--- Training Optimized Network ---\n');

net = patternnet(best_config.hidden_size);
net.divideParam.trainRatio = 0.85;
net.divideParam.valRatio = 0.15;
net.divideParam.testRatio = 0.0;

% Convert algorithm name to MATLAB function
algo_map = containers.Map(...
    {'Scaled CG', 'Levenberg-Marquardt', 'Bayesian Reg.', 'Resilient Backprop', ...
     'trainscg', 'trainlm', 'trainbr', 'trainrp'}, ...
    {'trainscg', 'trainlm', 'trainbr', 'trainrp', ...
     'trainscg', 'trainlm', 'trainbr', 'trainrp'});
 
if isKey(algo_map, best_config.algorithm)
    net.trainFcn = algo_map(best_config.algorithm);
else
    fprintf('Warning: Algorithm "%s" not recognized. Using trainscg.\n', best_config.algorithm);
    net.trainFcn = 'trainscg';
end

% Train the optimized network
[net_optimized, tr_optimized] = train(net, X_train, y_train);

%% Evaluate Optimized Model
fprintf('\n--- Evaluating Optimized Model ---\n');

% Predict on test set
y_pred = net_optimized(X_test);
[~, predicted_labels] = max(y_pred);
[~, true_labels] = max(y_test);

% Calculate accuracy
optimized_accuracy = sum(predicted_labels == true_labels) / length(true_labels) * 100;
fprintf('Optimized Model Accuracy: %.2f%%\n', optimized_accuracy);

% Compare with original if available
if exist('trained_neural_network.mat', 'file')
    load('trained_neural_network.mat', 'accuracy');
    fprintf('Original Model Accuracy: %.2f%%\n', accuracy);
    fprintf('Improvement: +%.2f%%\n', optimized_accuracy - accuracy);
else
    fprintf('Original model not found for comparison.\n');
end

%% Plot Comprehensive Results
figure('Position', [100, 100, 1200, 500]);

% Performance comparison
subplot(1,3,1);
if exist('trained_neural_network.mat', 'file')
    accuracies = [accuracy, optimized_accuracy];
    bar(accuracies, 'FaceColor', [0.2, 0.6, 0.8]);
    set(gca, 'XTickLabel', {'Original', 'Optimized'});
    ylabel('Accuracy (%)');
    title('Model Performance Comparison');
    grid on;
    for i = 1:length(accuracies)
        text(i, accuracies(i)+0.5, sprintf('%.2f%%', accuracies(i)), ...
             'HorizontalAlignment', 'center', 'FontWeight', 'bold');
    end
else
    bar(optimized_accuracy, 'FaceColor', [0.2, 0.6, 0.8]);
    ylabel('Accuracy (%)');
    title(sprintf('Optimized Model: %.2f%%', optimized_accuracy));
    grid on;
    text(1, optimized_accuracy+0.5, sprintf('%.2f%%', optimized_accuracy), ...
         'HorizontalAlignment', 'center', 'FontWeight', 'bold');
end

% Configuration summary
subplot(1,3,2);
config_text = sprintf(['Optimization Source: %s\n' ...
                      'Hidden Size: %d neurons\n' ...
                      'Algorithm: %s\n' ...
                      'Split: %.0f/%.0f\n' ...
                      'Final Accuracy: %.2f%%'], ...
                      best_config.source, best_config.hidden_size, ...
                      best_config.algorithm, best_config.split_ratio*100, ...
                      (1-best_config.split_ratio)*100, optimized_accuracy);
text(0.1, 0.7, config_text, 'FontSize', 12, 'VerticalAlignment', 'top');
axis off;
title('Optimized Configuration');

% Confusion matrix
subplot(1,3,3);
confusion_mat = confusionmat(true_labels, predicted_labels);
imagesc(confusion_mat);
colorbar;
xlabel('Predicted Label');
ylabel('True Label');
title('Confusion Matrix');
axis square;

%% Calculate Additional Metrics
fprintf('\n--- Additional Metrics ---\n');

% Per-class accuracy
num_classes = length(unique(true_labels));
class_accuracy = zeros(1, num_classes);
for i = 1:num_classes
    class_mask = (true_labels == i);
    if sum(class_mask) > 0
        class_accuracy(i) = sum(predicted_labels(class_mask) == i) / sum(class_mask) * 100;
    end
end

fprintf('Per-class accuracy:\n');
for i = 1:num_classes
    fprintf('  User %d: %.2f%%\n', i, class_accuracy(i));
end

fprintf('Overall accuracy std across users: %.2f%%\n', std(class_accuracy));

%% Save Optimized Model with Complete Information
fprintf('\n--- Saving Optimized Model ---\n');

optimization_info = struct();
optimization_info.source = best_config.source;
optimization_info.hidden_size = best_config.hidden_size;
optimization_info.algorithm = best_config.algorithm;
optimization_info.split_ratio = best_config.split_ratio;
optimization_info.expected_accuracy = best_config.accuracy;
optimization_info.actual_accuracy = optimized_accuracy;
optimization_info.training_time = tr_optimized.time(end);
optimization_info.available_sources = optimization_sources;

save('optimized_neural_network.mat', 'net_optimized', 'tr_optimized', ...
     'optimized_accuracy', 'optimization_info', 'class_accuracy');

fprintf('Optimized model saved to optimized_neural_network.mat\n');
fprintf('Optimization info structure includes all configuration details.\n');

fprintf('\n=== OPTIMIZATION COMPLETE ===\n');
fprintf('Final Optimized Accuracy: %.2f%%\n', optimized_accuracy);
fprintf('Using configuration from: %s\n', best_config.source);