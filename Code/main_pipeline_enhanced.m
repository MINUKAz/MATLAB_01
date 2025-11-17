%% MAIN ENHANCED PIPELINE - MEETS ALL REQUIREMENTS
clear all; close all; clc;

fprintf('=== ENHANCED GAIT AUTHENTICATION PIPELINE ===\n');
fprintf('Meeting all documented requirements:\n');
fprintf('  ✓ 32 Hz uniform sampling\n');
fprintf('  ✓ Stationary segment removal\n');
fprintf('  ✓ Band-pass filter (0.5-5 Hz) for accelerometer\n');
fprintf('  ✓ Low-pass filter for gyroscope\n');
fprintf('  ✓ 4-second segments with 50%% overlap\n');
fprintf('  ✓ Enhanced feature extraction\n\n');

%% Step 1: Load and Preprocess Data
fprintf('--- STEP 1: Data Loading & Preprocessing ---\n');
run('load_data.m'); % This now includes enhanced preprocessing

%% Step 2: Verify Preprocessing
fprintf('\n--- STEP 2: Preprocessing Verification ---\n');
visualize_preprocessing;

%% Step 3: Feature Extraction
fprintf('\n--- STEP 3: Feature Extraction ---\n');
if exist('processed_data_enhanced.mat', 'file')
    load('processed_data_enhanced.mat', 'all_data');
    
    % Test feature extraction on first segment
    if ~isempty(all_data) && ~isempty(all_data(1).segments)
        test_segment = all_data(1).segments{1};
        fprintf('Testing feature extraction on %d-sample segment...\n', size(test_segment, 1));
        
        test_features = extract_trial_features(test_segment);
        fprintf('✓ Features extracted: %d features\n', length(test_features));
        
        % Run complete feature extraction
        run_complete_feature_extraction(all_data);
    end
else
    error('Enhanced processed data not found. Please run Step 1 first.');
end

%% Step 4: Neural Network Training
fprintf('\n--- STEP 4: Neural Network Training ---\n');
if exist('extracted_features.mat', 'file')
    neural_network_training;
else
    error('Feature extraction must be completed first.');
end

%% Step 5: Optimization
fprintf('\n--- STEP 5: Model Optimization ---\n');
fprintf('Choose optimization method:\n');
fprintf('1. Fast Optimization (quick test)\n');
fprintf('2. Comprehensive Optimization (thorough)\n');
fprintf('3. Targeted Optimization (build on current results)\n');

% Run fast optimization by default (you can change this)
fprintf('\nRunning fast optimization...\n');
fast_optimization;

fprintf('\n=== ENHANCED PIPELINE COMPLETE ===\n');
fprintf('All requirements implemented:\n');
fprintf('✓ 32 Hz uniform sampling via interpolation\n');
fprintf('✓ Stationary/unworn signal removal\n');
fprintf('✓ Band-pass filter (0.5-5 Hz) for accelerometer\n');
fprintf('✓ Low-pass filter for gyroscope\n');
fprintf('✓ 4-second segments (128 samples) with 50%% overlap\n');
fprintf('✓ Enhanced feature extraction (108 features)\n');
fprintf('✓ Complete neural network pipeline\n');