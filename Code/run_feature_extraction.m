%% MAIN SCRIPT: Run Complete Feature Extraction - ENHANCED VERSION
clear all; close all; clc;

fprintf('=== COMPLETE FEATURE EXTRACTION PIPELINE - ENHANCED ===\n');

% Step 1: Load and verify ENHANCED data exists
if ~exist('processed_data_enhanced.mat', 'file')
    error('processed_data_enhanced.mat not found. Please run enhanced data loading first.');
end

load('processed_data_enhanced.mat', 'all_data');
fprintf('Loaded ENHANCED data: %d sessions\n', length(all_data));

% Step 2: Test feature extraction on a single trial first
fprintf('\n--- Testing feature extraction on first segment ---\n');

if isempty(all_data) || ~isfield(all_data(1), 'segments') || isempty(all_data(1).segments)
    error('No segments found in enhanced data. Check preprocessing.');
end

test_segment_data = all_data(1).segments{1};
fprintf('Test segment size: %dx%d (expected: 128x6)\n', size(test_segment_data));

% Verify segment size
expected_samples = 128; % 4 seconds × 32 Hz
if size(test_segment_data, 1) ~= expected_samples
    fprintf('⚠️  WARNING: Segment size is %d, expected %d\n', size(test_segment_data, 1), expected_samples);
end

% Test the feature extraction function
test_features = extract_trial_features(test_segment_data);
fprintf('Features extracted: %d features\n', length(test_features));

% Verify it's exactly 108 (our new feature count)
expected_features = 108;
if length(test_features) == expected_features
    fprintf('✓ Feature dimension correct! Proceeding with full extraction...\n');
    
    % Step 3: Run complete feature extraction
    fprintf('\n--- Running complete feature extraction ---\n');
    run_complete_feature_extraction(all_data);
    
    fprintf('\n=== FEATURE EXTRACTION COMPLETE ===\n');
else
    fprintf('✗ Feature dimension incorrect: expected %d, got %d\n', expected_features, length(test_features));
    fprintf('Please check the feature extraction function.\n');
end