%% Train Final Optimized Model
clear all; close all; clc;

fprintf('=== TRAINING FINAL OPTIMIZED MODEL ===\n');

% Load features and optimization results
load('extracted_features.mat', 'feature_table', 'label_table', 'user_ids');
load('optimization_results.mat', 'results_hidden', 'results_algorithms', 'results_splits');

% Find best configuration
[best_hidden_acc, best_hidden_idx] = max([results_hidden.accuracy]);
[best_algo_acc, best_algo_idx] = max([results_algorithms.accuracy]);
[best_split_acc, best_split_idx] = max([results_splits.accuracy]);

best_hidden_size = results_hidden(best_hidden_idx).hidden_size;
best_algorithm = results_algorithms(best_algo_idx).algorithm;
best_split = results_splits(best_split_idx).train_ratio;

fprintf('Using optimized configuration:\n');
fprintf('  - Hidden Layer Size: %d neurons\n', best_hidden_size);
fprintf('  - Training Algorithm: %s\n', best_algorithm);
fprintf('  - Train/Test Split: %.0f/%.0f\n', best_split*100, (1-best_split)*100);

% Convert data
X = table2array(feature_table);
y_original = table2array(label_table);
y_categorical = categorical(y_original);
y_onehot = dummyvar(y_categorical)';

% Split data with best ratio
rng(42);
test_ratio = 1 - best_split;
cv = cvpartition(length(y_original), 'HoldOut', test_ratio);

X_train = X(cv.training,:)';
X_test = X(cv.test,:)';
y_train = y_onehot(:, cv.training);
y_test = y_onehot(:, cv.test);

fprintf('Training set: %d samples\n', size(X_train, 2));
fprintf('Testing set: %d samples\n', size(X_test, 2));

%% Create and Train Optimized Network
fprintf('\n--- Training Optimized Network ---\n');

net = patternnet(best_hidden_size);
net.divideParam.trainRatio = 0.85;
net.divideParam.valRatio = 0.15;
net.divideParam.testRatio = 0.0;

% Convert algorithm name back to MATLAB function
algo_map = containers.Map(...
    {'Scaled CG', 'Levenberg-Marquardt', 'Bayesian Reg.', 'Resilient Backprop'}, ...
    {'trainscg', 'trainlm', 'trainbr', 'trainrp'});
net.trainFcn = algo_map(best_algorithm);

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

% Compare with original
load('trained_neural_network.mat', 'accuracy');
fprintf('Original Model Accuracy: %.2f%%\n', accuracy);
fprintf('Improvement: +%.2f%%\n', optimized_accuracy - accuracy);

%% Plot Comparison
figure('Position', [100, 100, 800, 400]);

subplot(1,2,1);
accuracies = [accuracy, optimized_accuracy];
bar(accuracies);
set(gca, 'XTickLabel', {'Original', 'Optimized'});
ylabel('Accuracy (%)');
title('Model Performance Comparison');
grid on;
for i = 1:length(accuracies)
    text(i, accuracies(i)+0.5, sprintf('%.2f%%', accuracies(i)), ...
         'HorizontalAlignment', 'center', 'FontWeight', 'bold');
end

subplot(1,2,2);
confusion_mat = confusionmat(true_labels, predicted_labels);
imagesc(confusion_mat);
colorbar;
xlabel('Predicted Label');
ylabel('True Label');
title('Optimized Model Confusion Matrix');

%% Save Optimized Model
fprintf('\n--- Saving Optimized Model ---\n');

save('optimized_neural_network.mat', 'net_optimized', 'tr_optimized', ...
     'optimized_accuracy', 'best_hidden_size', 'best_algorithm', 'best_split');

fprintf('Optimized model saved to optimized_neural_network.mat\n');
fprintf('\n=== OPTIMIZATION COMPLETE ===\n');
fprintf('Final Optimized Accuracy: %.2f%%\n', optimized_accuracy);