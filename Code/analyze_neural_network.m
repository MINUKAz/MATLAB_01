%% Analyze Neural Network Results
clear all; close all; clc;

fprintf('=== NEURAL NETWORK ANALYSIS ===\n');

% Load trained model
if ~exist('trained_neural_network.mat', 'file')
    error('trained_neural_network.mat not found. Please run neural network training first.');
end

load('trained_neural_network.mat', 'net', 'tr', 'accuracy', 'authentication_results');

fprintf('Model loaded: %.2f%% accuracy\n', accuracy);
fprintf('Network architecture: %d inputs -> %d hidden -> %d outputs\n', ...
        net.inputs{1}.size, net.layers{1}.size, net.layers{2}.size);

% Plot final results
figure('Position', [100, 100, 1000, 800]);

% Training performance
subplot(2,2,1);
plotperform(tr);
title('Training Performance');

% Training performance over epochs (replacement for ploterrhist)
subplot(2,2,2);
if isfield(tr, 'perf')
    plot(tr.epoch, tr.perf, 'LineWidth', 2, 'Color', 'blue');
    hold on;
    if isfield(tr, 'vperf')
        plot(tr.epoch, tr.vperf, 'LineWidth', 2, 'Color', 'red');
        legend('Training', 'Validation', 'Location', 'best');
    else
        legend('Training', 'Location', 'best');
    end
    title('Performance Over Epochs');
    xlabel('Epoch');
    ylabel('Cross-Entropy Loss');
    grid on;
else
    text(0.5, 0.5, 'Performance data not available', ...
         'HorizontalAlignment', 'center', 'FontSize', 12);
    title('Performance Plot');
    axis off;
end

% Authentication scores distribution (for first user)
subplot(2,2,3);
if ~isempty(authentication_results) && isfield(authentication_results, 'genuine_scores')
    user1 = authentication_results(1);
    if ~isempty(user1.genuine_scores) && ~isempty(user1.impostor_scores)
        histogram(user1.genuine_scores, 'Normalization', 'probability', 'FaceColor', 'g', 'EdgeColor', 'none');
        hold on;
        histogram(user1.impostor_scores, 'Normalization', 'probability', 'FaceColor', 'r', 'EdgeColor', 'none');
        xlabel('Authentication Score');
        ylabel('Probability');
        title('User 1: Genuine vs Impostor Scores');
        legend('Genuine', 'Impostor');
        grid on;
    else
        text(0.5, 0.5, 'No authentication scores available', ...
             'HorizontalAlignment', 'center', 'FontSize', 12);
        title('Authentication Scores');
        axis off;
    end
else
    text(0.5, 0.5, 'Authentication results not available', ...
         'HorizontalAlignment', 'center', 'FontSize', 12);
    title('Authentication Analysis');
    axis off;
end

% Summary statistics
subplot(2,2,4);
if ~isempty(authentication_results) && isfield(authentication_results, 'num_genuine')
    avg_genuine = mean([authentication_results.num_genuine]);
    avg_impostor = mean([authentication_results.num_impostor]);
    
    bar([avg_genuine, avg_impostor]);
    set(gca, 'XTickLabel', {'Genuine', 'Impostor'});
    ylabel('Average Samples per User');
    title('Sample Distribution');
    grid on;
    
    % Add value labels on bars
    text(1, avg_genuine + 0.1, sprintf('%.1f', avg_genuine), ...
         'HorizontalAlignment', 'center', 'FontWeight', 'bold');
    text(2, avg_impostor + 0.1, sprintf('%.1f', avg_impostor), ...
         'HorizontalAlignment', 'center', 'FontWeight', 'bold');
else
    % Display training time if available
    if isfield(tr, 'time') && ~isempty(tr.time)
        training_time = tr.time(end);
        text(0.5, 0.7, sprintf('Training Time: %.1f seconds', training_time), ...
             'HorizontalAlignment', 'center', 'FontSize', 14, 'FontWeight', 'bold');
    end
    
    % Display final performance
    text(0.5, 0.5, sprintf('Final Accuracy: %.2f%%', accuracy), ...
         'HorizontalAlignment', 'center', 'FontSize', 16, 'FontWeight', 'bold', 'Color', 'blue');
    
    text(0.5, 0.3, 'Excellent Performance!', ...
         'HorizontalAlignment', 'center', 'FontSize', 14, 'FontWeight', 'bold', 'Color', 'green');
    
    title('Training Summary');
    axis off;
end

fprintf('\n=== ANALYSIS COMPLETE ===\n');
fprintf('Key metrics:\n');
fprintf('  - Test Accuracy: %.2f%%\n', accuracy);
if isfield(authentication_results, 'num_genuine')
    fprintf('  - Number of Users: %d\n', length(authentication_results));
end
fprintf('  - Hidden Layer Size: %d neurons\n', net.layers{1}.size);

% Display training time if available
if isfield(tr, 'time') && ~isempty(tr.time)
    fprintf('  - Training Time: %.1f seconds\n', tr.time(end));
end

% Display next steps
fprintf('\nNext steps for optimization:\n');
fprintf('1. Try different hidden layer sizes (32, 64, 128)\n');
fprintf('2. Experiment with multiple hidden layers\n');
fprintf('3. Adjust training/validation ratios\n');
fprintf('4. Try different training algorithms\n');