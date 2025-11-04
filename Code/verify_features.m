%% Test Simple Feature Extraction
clear all; close all; clc;

load('processed_data.mat', 'all_data');

% Test on first trial
test_trial = all_data(1).segments{1};
fprintf('Test trial size: %dx%d\n', size(test_trial));

% Test feature extraction
features = extract_trial_features(test_trial);
fprintf('Features extracted: %d\n', length(features));

if length(features) == 96
    fprintf('✓ SUCCESS: Time-domain feature extraction working!\n');
    fprintf('Proceeding with full extraction...\n\n');
    
    % Run full extraction
    run_complete_feature_extraction(all_data);
    
    % Load and verify results
    if exist('extracted_features.mat', 'file')
        load('extracted_features.mat');
        fprintf('\n=== VERIFICATION ===\n');
        fprintf('Feature table size: %dx%d\n', size(feature_table));
        fprintf('Unique users: %d\n', length(unique(user_ids)));
        fprintf('Ready for neural network training!\n');
    end
else
    fprintf('✗ FAILED: Expected 96 features, got %d\n', length(features));
end