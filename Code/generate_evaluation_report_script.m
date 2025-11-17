%% SCRIPT VERSION: Generate Evaluation Report
clear all; close all; clc;

fprintf('Generating evaluation report...\n');

% Check if comprehensive metrics exist
if exist('comprehensive_metrics.mat', 'file')
    load('comprehensive_metrics.mat', 'metrics');
    generate_evaluation_report(metrics, 'final_evaluation_report.txt');
    fprintf('Comprehensive evaluation report generated!\n');
elseif exist('trained_neural_network.mat', 'file')
    fprintf('Comprehensive metrics not found. Using trained model accuracy...\n');
    
    load('trained_neural_network.mat', 'accuracy');
    
    % Create basic metrics structure
    basic_metrics.accuracy = accuracy;
    basic_metrics.macro_precision = NaN;
    basic_metrics.macro_recall = NaN;
    basic_metrics.macro_f1 = NaN;
    basic_metrics.macro_auc = NaN;
    
    generate_evaluation_report(basic_metrics, 'basic_evaluation_report.txt');
    fprintf('Basic evaluation report generated using model accuracy.\n');
else
    fprintf('Error: No model or metrics found. Please run neural network training first.\n');
end