function run_complete_feature_extraction(all_data)
    % Complete feature extraction for all data - TIME DOMAIN ONLY
    
    fprintf('Starting TIME-DOMAIN feature extraction...\n');
    
    % Pre-calculate total number of trials for pre-allocation
    total_trials = 0;
    for i = 1:length(all_data)
        total_trials = total_trials + all_data(i).num_trials;
    end
    
    fprintf('Total trials to process: %d\n', total_trials);
    
    % Use 96 features (time-domain only)
    expected_features = 96;
    all_features = zeros(total_trials, expected_features);
    all_labels = zeros(total_trials, 1);
    user_ids = zeros(total_trials, 1);
    session_types = cell(total_trials, 1);
    
    current_index = 1;
    
    % Loop through each session
    for session_idx = 1:length(all_data)
        user_id = all_data(session_idx).user_id;
        session = all_data(session_idx).session;
        segments = all_data(session_idx).segments;
        num_trials = all_data(session_idx).num_trials;
        
        fprintf('Processing User %d, Session %s: %d trials\n', ...
                user_id, session, num_trials);
        
        % Process each walking trial
        for trial_idx = 1:num_trials
            trial_data = segments{trial_idx};
            
            % Extract features from this trial
            features = extract_trial_features(trial_data);
            
            % Verify feature dimension
            if length(features) ~= expected_features
                fprintf('Warning: Trial %d has %d features (expected %d)\n', ...
                        trial_idx, length(features), expected_features);
                % Force correct size
                if length(features) > expected_features
                    features = features(1:expected_features);
                else
                    features = [features, zeros(1, expected_features - length(features))];
                end
            end
            
            % Store in pre-allocated arrays
            all_features(current_index, :) = features;
            all_labels(current_index) = user_id;
            user_ids(current_index) = user_id;
            session_types{current_index} = session;
            
            current_index = current_index + 1;
        end
    end
    
    % Convert to tables
    feature_table = array2table(all_features);
    label_table = array2table(all_labels, 'VariableNames', {'UserID'});
    
    % Generate feature names
    feature_names = generate_feature_names();
    if length(feature_names) == size(feature_table, 2)
        feature_table.Properties.VariableNames = feature_names;
    else
        fprintf('Warning: Feature names count (%d) doesnt match features (%d)\n', ...
                length(feature_names), size(feature_table, 2));
    end
    
    % Save the extracted features
    save('extracted_features.mat', 'feature_table', 'label_table', 'user_ids', 'session_types', 'feature_names');
    fprintf('\n=== FEATURE EXTRACTION COMPLETE ===\n');
    fprintf('Total feature vectors: %d\n', size(all_features, 1));
    fprintf('Features per trial: %d\n', size(all_features, 2));
    fprintf('Total users: %d\n', length(unique(user_ids)));
end