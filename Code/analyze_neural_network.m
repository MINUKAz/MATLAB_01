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

% Error histogram
subplot(2,2,2);
ploterrhist(tr);
title('Error Histogram');

% Authentication scores distribution (for first user)
subplot(2,2,3);
if ~isempty(authentication_results)
    user1 = authentication_results(1);
    histogram(user1.genuine_scores, 'Normalization', 'probability', 'FaceColor', 'g', 'EdgeColor', 'none');
    hold on;
    histogram(user1.impostor_scores, 'Normalization', 'probability', 'FaceColor', 'r', 'EdgeColor', 'none');
    xlabel('Authentication Score');
    ylabel('Probability');
    title('User 1: Genuine vs Impostor Scores');
    legend('Genuine', 'Impostor');
    grid on;
end

% Summary statistics
subplot(2,2,4);
if ~isempty(authentication_results)
    avg_genuine = mean([authentication_results.num_genuine]);
    avg_impostor = mean([authentication_results.num_impostor]);
    
    bar([avg_genuine, avg_impostor]);
    set(gca, 'XTickLabel', {'Genuine', 'Impostor'});
    ylabel('Average Samples per User');
    title('Sample Distribution');
    grid on;
end

fprintf('\n=== ANALYSIS COMPLETE ===\n');
fprintf('Key metrics:\n');
fprintf('  - Test Accuracy: %.2f%%\n', accuracy);
fprintf('  - Number of Users: %d\n', length(authentication_results));
fprintf('  - Hidden Layer Size: %d neurons\n', net.layers{1}.size);

% Display next steps
fprintf('\nNext steps for optimization:\n');
fprintf('1. Try different hidden layer sizes (32, 64, 128)\n');
fprintf('2. Experiment with multiple hidden layers\n');
fprintf('3. Adjust training/validation ratios\n');
fprintf('4. Try different training algorithms\n');