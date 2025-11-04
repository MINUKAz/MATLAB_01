%% Neural Network Optimization
clear all; close all; clc;

fprintf('=== NEURAL NETWORK OPTIMIZATION ===\n');

% Load features
load('extracted_features.mat', 'feature_table', 'label_table', 'user_ids');

% Convert to arrays
X = table2array(feature_table);
y_original = table2array(label_table);
y_categorical = categorical(y_original);
y_onehot = dummyvar(y_categorical)';

fprintf('Starting optimization experiments...\n');

%% Experiment 1: Different Hidden Layer Sizes
fprintf('\n--- Experiment 1: Hidden Layer Size ---\n');

hidden_sizes = [32, 64, 128, 256];
results_hidden = struct();

for i = 1:length(hidden_sizes)
    hidden_size = hidden_sizes(i);
    fprintf('Testing hidden size: %d neurons... ', hidden_size);
    
    % Split data
    rng(42);
    cv = cvpartition(length(y_original), 'HoldOut', 0.2);
    X_train = X(cv.training,:)';
    X_test = X(cv.test,:)';
    y_train = y_onehot(:, cv.training);
    y_test = y_onehot(:, cv.test);
    
    % Create and train network
    net = patternnet(hidden_size);
    net.divideParam.trainRatio = 0.85;
    net.divideParam.valRatio = 0.15;
    net.divideParam.testRatio = 0.0;
    net.trainFcn = 'trainscg';
    
    % Train
    [net, tr] = train(net, X_train, y_train);
    
    % Evaluate
    y_pred = net(X_test);
    [~, predicted_labels] = max(y_pred);
    [~, true_labels] = max(y_test);
    accuracy = sum(predicted_labels == true_labels) / length(true_labels) * 100;
    
    % Store results
    results_hidden(i).hidden_size = hidden_size;
    results_hidden(i).accuracy = accuracy;
    results_hidden(i).training_time = tr.time(end);
    
    fprintf('Accuracy: %.2f%%, Time: %.1fs\n', accuracy, tr.time(end));
end

%% Experiment 2: Different Training Algorithms
fprintf('\n--- Experiment 2: Training Algorithms ---\n');

training_algorithms = {'trainscg', 'trainlm', 'trainbr', 'trainrp'};
algorithm_names = {'Scaled CG', 'Levenberg-Marquardt', 'Bayesian Reg.', 'Resilient Backprop'};
results_algorithms = struct();

for i = 1:length(training_algorithms)
    train_algo = training_algorithms{i};
    fprintf('Testing algorithm: %s... ', algorithm_names{i});
    
    % Split data
    rng(42);
    cv = cvpartition(length(y_original), 'HoldOut', 0.2);
    X_train = X(cv.training,:)';
    X_test = X(cv.test,:)';
    y_train = y_onehot(:, cv.training);
    y_test = y_onehot(:, cv.test);
    
    % Create and train network
    net = patternnet(64); % Fixed hidden size
    net.divideParam.trainRatio = 0.85;
    net.divideParam.valRatio = 0.15;
    net.divideParam.testRatio = 0.0;
    net.trainFcn = train_algo;
    
    % Train
    [net, tr] = train(net, X_train, y_train);
    
    % Evaluate
    y_pred = net(X_test);
    [~, predicted_labels] = max(y_pred);
    [~, true_labels] = max(y_test);
    accuracy = sum(predicted_labels == true_labels) / length(true_labels) * 100;
    
    % Store results
    results_algorithms(i).algorithm = algorithm_names{i};
    results_algorithms(i).accuracy = accuracy;
    results_algorithms(i).training_time = tr.time(end);
    
    fprintf('Accuracy: %.2f%%, Time: %.1fs\n', accuracy, tr.time(end));
end

%% Experiment 3: Different Train/Test Splits
fprintf('\n--- Experiment 3: Train/Test Splits ---\n');

split_ratios = [0.7, 0.75, 0.8, 0.85];
results_splits = struct();

for i = 1:length(split_ratios)
    test_ratio = 1 - split_ratios(i);
    fprintf('Testing split: %.0f/%.0f... ', split_ratios(i)*100, test_ratio*100);
    
    % Split data
    rng(42);
    cv = cvpartition(length(y_original), 'HoldOut', test_ratio);
    X_train = X(cv.training,:)';
    X_test = X(cv.test,:)';
    y_train = y_onehot(:, cv.training);
    y_test = y_onehot(:, cv.test);
    
    % Create and train network
    net = patternnet(64);
    net.divideParam.trainRatio = 0.85;
    net.divideParam.valRatio = 0.15;
    net.divideParam.testRatio = 0.0;
    net.trainFcn = 'trainscg';
    
    % Train
    [net, tr] = train(net, X_train, y_train);
    
    % Evaluate
    y_pred = net(X_test);
    [~, predicted_labels] = max(y_pred);
    [~, true_labels] = max(y_test);
    accuracy = sum(predicted_labels == true_labels) / length(true_labels) * 100;
    
    % Store results
    results_splits(i).train_ratio = split_ratios(i);
    results_splits(i).test_ratio = test_ratio;
    results_splits(i).accuracy = accuracy;
    results_splits(i).training_samples = sum(cv.training);
    results_splits(i).testing_samples = sum(cv.test);
    
    fprintf('Accuracy: %.2f%%, Train samples: %d, Test samples: %d\n', ...
            accuracy, sum(cv.training), sum(cv.test));
end

%% Display Optimization Results
fprintf('\n=== OPTIMIZATION RESULTS SUMMARY ===\n');

% Plot comparison of all experiments
figure('Position', [100, 100, 1200, 800]);

% Hidden layer size results
subplot(2,2,1);
hidden_accuracies = [results_hidden.accuracy];
hidden_times = [results_hidden.training_time];
bar(hidden_accuracies);
set(gca, 'XTickLabel', {results_hidden.hidden_size});
xlabel('Hidden Layer Size');
ylabel('Accuracy (%)');
title('Accuracy vs Hidden Layer Size');
grid on;
for i = 1:length(hidden_accuracies)
    text(i, hidden_accuracies(i)+0.5, sprintf('%.2f%%', hidden_accuracies(i)), ...
         'HorizontalAlignment', 'center', 'FontWeight', 'bold');
end

% Training algorithm results
subplot(2,2,2);
algo_accuracies = [results_algorithms.accuracy];
algo_times = [results_algorithms.training_time];
bar(algo_accuracies);
set(gca, 'XTickLabel', {results_algorithms.algorithm});
xlabel('Training Algorithm');
ylabel('Accuracy (%)');
title('Accuracy vs Training Algorithm');
grid on;
for i = 1:length(algo_accuracies)
    text(i, algo_accuracies(i)+0.5, sprintf('%.2f%%', algo_accuracies(i)), ...
         'HorizontalAlignment', 'center', 'FontWeight', 'bold');
end

% Split ratio results
subplot(2,2,3);
split_accuracies = [results_splits.accuracy];
split_train_samples = [results_splits.training_samples];
bar(split_accuracies);
set(gca, 'XTickLabel', {'70/30', '75/25', '80/20', '85/15'});
xlabel('Train/Test Split Ratio');
ylabel('Accuracy (%)');
title('Accuracy vs Train/Test Split');
grid on;
for i = 1:length(split_accuracies)
    text(i, split_accuracies(i)+0.5, sprintf('%.2f%%', split_accuracies(i)), ...
         'HorizontalAlignment', 'center', 'FontWeight', 'bold');
end

% Training time comparison
subplot(2,2,4);
yyaxis left;
plot([results_hidden.hidden_size], hidden_accuracies, 'o-', 'LineWidth', 2, 'MarkerSize', 8);
ylabel('Accuracy (%)');
yyaxis right;
plot([results_hidden.hidden_size], hidden_times, 's-', 'LineWidth', 2, 'MarkerSize', 8);
ylabel('Training Time (s)');
xlabel('Hidden Layer Size');
title('Accuracy vs Time Trade-off');
legend('Accuracy', 'Training Time', 'Location', 'best');
grid on;

%% Find Best Configuration
fprintf('\n--- BEST CONFIGURATION ANALYSIS ---\n');

% Find best from each experiment
[best_hidden_acc, best_hidden_idx] = max([results_hidden.accuracy]);
[best_algo_acc, best_algo_idx] = max([results_algorithms.accuracy]);
[best_split_acc, best_split_idx] = max([results_splits.accuracy]);

fprintf('Best Hidden Layer Size: %d neurons (%.2f%% accuracy)\n', ...
        results_hidden(best_hidden_idx).hidden_size, best_hidden_acc);
fprintf('Best Training Algorithm: %s (%.2f%% accuracy)\n', ...
        results_algorithms(best_algo_idx).algorithm, best_algo_acc);
fprintf('Best Train/Test Split: %.0f/%.0f (%.2f%% accuracy)\n', ...
        results_splits(best_split_idx).train_ratio*100, ...
        results_splits(best_split_idx).test_ratio*100, best_split_acc);

% Save optimization results
save('optimization_results.mat', 'results_hidden', 'results_algorithms', 'results_splits');

fprintf('\nOptimization results saved to optimization_results.mat\n');
fprintf('Ready to train final optimized model!\n');