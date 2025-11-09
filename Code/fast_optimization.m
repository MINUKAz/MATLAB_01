%% Fast Neural Network Optimization
clear all; close all; clc;

fprintf('=== FAST NEURAL NETWORK OPTIMIZATION ===\n');
fprintf('Using reduced parameter set for speed...\n');

% Load features
load('extracted_features.mat', 'feature_table', 'label_table', 'user_ids');

% Convert to arrays
X = table2array(feature_table);
y_original = table2array(label_table);
y_categorical = categorical(y_original);
y_onehot = dummyvar(y_categorical)';

fprintf('Data loaded: %d samples, %d features, %d users\n', ...
        size(X, 1), size(X, 2), length(unique(y_original)));

%% Quick Experiment 1: Only 2 Hidden Layer Sizes
fprintf('\n--- Quick Test: Hidden Layer Sizes ---\n');

hidden_sizes = [32, 64]; % Only test 2 sizes instead of 4
quick_results = [];

for i = 1:length(hidden_sizes)
    hidden_size = hidden_sizes(i);
    fprintf('Testing %d neurons... ', hidden_size);
    
    % Use fixed 80/20 split and trainscg for consistency
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
    
    % Train with progress display
    net.trainParam.showWindow = false; % Hide training window for speed
    net.trainParam.showCommandLine = false;
    
    [net, tr] = train(net, X_train, y_train);
    
    % Quick evaluation
    y_pred = net(X_test);
    [~, predicted_labels] = max(y_pred);
    [~, true_labels] = max(y_test);
    accuracy = sum(predicted_labels == true_labels) / length(true_labels) * 100;
    
    quick_results(i).hidden_size = hidden_size;
    quick_results(i).accuracy = accuracy;
    quick_results(i).training_time = tr.time(end);
    
    fprintf('Accuracy: %.2f%%, Time: %.1fs\n', accuracy, tr.time(end));
end

%% Quick Experiment 2: Only Best Algorithm Comparison
fprintf('\n--- Quick Test: Training Algorithms (Best 2) ---\n');

% Only test the 2 most promising algorithms
training_algorithms = {'trainscg', 'trainlm'};
algorithm_names = {'Scaled CG', 'Levenberg-Marquardt'};

for i = 1:length(training_algorithms)
    train_algo = training_algorithms{i};
    fprintf('Testing %s... ', algorithm_names{i});
    
    % Use same split and hidden size (64)
    rng(42);
    cv = cvpartition(length(y_original), 'HoldOut', 0.2);
    X_train = X(cv.training,:)';
    X_test = X(cv.test,:)';
    y_train = y_onehot(:, cv.training);
    y_test = y_onehot(:, cv.test);
    
    net = patternnet(64);
    net.divideParam.trainRatio = 0.85;
    net.divideParam.valRatio = 0.15;
    net.divideParam.testRatio = 0.0;
    net.trainFcn = train_algo;
    net.trainParam.showWindow = false;
    net.trainParam.showCommandLine = false;
    
    [net, tr] = train(net, X_train, y_train);
    
    y_pred = net(X_test);
    [~, predicted_labels] = max(y_pred);
    [~, true_labels] = max(y_test);
    accuracy = sum(predicted_labels == true_labels) / length(true_labels) * 100;
    
    quick_results(length(hidden_sizes) + i).algorithm = algorithm_names{i};
    quick_results(length(hidden_sizes) + i).accuracy = accuracy;
    quick_results(length(hidden_sizes) + i).training_time = tr.time(end);
    
    fprintf('Accuracy: %.2f%%, Time: %.1fs\n', accuracy, tr.time(end));
end

%% Display Quick Results
fprintf('\n=== QUICK OPTIMIZATION RESULTS ===\n');

figure('Position', [100, 100, 800, 400]);

% Plot results
subplot(1,2,1);
accuracies = [quick_results.accuracy];
config_names = {'32 Neurons', '64 Neurons', 'Scaled CG', 'Levenberg-Marquardt'};

bar(accuracies);
set(gca, 'XTickLabel', config_names, 'XTickLabelRotation', 45);
ylabel('Accuracy (%)');
title('Quick Optimization Results');
grid on;

for i = 1:length(accuracies)
    text(i, accuracies(i)+0.5, sprintf('%.2f%%', accuracies(i)), ...
         'HorizontalAlignment', 'center', 'FontWeight', 'bold');
end

% Compare with original
subplot(1,2,2);
load('trained_neural_network.mat', 'accuracy');
comparison_acc = [accuracy, max(accuracies)];
bar(comparison_acc);
set(gca, 'XTickLabel', {'Original', 'Best Optimized'});
ylabel('Accuracy (%)');
title('Before vs After Optimization');
grid on;

for i = 1:length(comparison_acc)
    text(i, comparison_acc(i)+0.5, sprintf('%.2f%%', comparison_acc(i)), ...
         'HorizontalAlignment', 'center', 'FontWeight', 'bold');
end

% Save quick results
save('quick_optimization_results.mat', 'quick_results');

fprintf('\nBest configuration from quick test: %.2f%% accuracy\n', max(accuracies));
fprintf('Original model: %.2f%% accuracy\n', accuracy);
fprintf('Improvement: +%.2f%%\n', max(accuracies) - accuracy);

fprintf('\nQuick optimization complete! Results saved.\n');