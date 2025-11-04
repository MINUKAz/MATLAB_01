%% MAIN SCRIPT: Run Complete Feature Extraction
clear all; close all; clc;

fprintf('=== COMPLETE FEATURE EXTRACTION PIPELINE ===\n');

% Step 1: Load and verify data exists
if ~exist('processed_data.mat', 'file')
    error('processed_data.mat not found. Please run Step 1 data loading first.');
end

load('processed_data.mat', 'all_data');
fprintf('Loaded data: %d sessions\n', length(all_data));

% Step 2: Test feature extraction on a single trial first
fprintf('\n--- Testing feature extraction on first trial ---\n');
test_trial_data = all_data(1).segments{1};
fprintf('Test trial size: %dx%d\n', size(test_trial_data));

% Test the feature extraction function
test_features = extract_trial_features(test_trial_data);
fprintf('Features extracted: %d features\n', length(test_features));

% Verify it's exactly 121
if length(test_features) == 121
    fprintf('✓ Feature dimension correct! Proceeding with full extraction...\n');
    
    % Step 3: Run complete feature extraction
    fprintf('\n--- Running complete feature extraction ---\n');
    run_complete_feature_extraction(all_data);
    
    fprintf('\n=== FEATURE EXTRACTION COMPLETE ===\n');
else
    fprintf('✗ Feature dimension incorrect: expected 121, got %d\n', length(test_features));
    fprintf('Please check the feature extraction function.\n');
end