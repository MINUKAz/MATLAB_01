function run_complete_feature_extraction(all_data)
    % Complete feature extraction for ENHANCED data - TIME DOMAIN ONLY
    
    fprintf('Starting TIME-DOMAIN feature extraction for ENHANCED data...\n');
    
    % Pre-calculate total number of segments for pre-allocation
    total_segments = 0;
    for i = 1:length(all_data)
        if isfield(all_data(i), 'num_segments')
            total_segments = total_segments + all_data(i).num_segments;
        end
    end
    
    fprintf('Total segments to process: %d\n', total_segments);
    
    if total_segments == 0
        error('No segments found in the enhanced data. Please check preprocessing.');
    end
    
    % Use 108 features (enhanced time-domain)
    expected_features = 108;
    all_features = zeros(total_segments, expected_features);
    all_labels = zeros(total_segments, 1);
    user_ids = zeros(total_segments, 1);
    session_types = cell(total_segments, 1);
    
    current_index = 1;
    
    % Loop through each session
    for session_idx = 1:length(all_data)
        if ~isfield(all_data(session_idx), 'segments') || isempty(all_data(session_idx).segments)
            continue; % Skip sessions without segments
        end
        
        user_id = all_data(session_idx).user_id;
        session = all_data(session_idx).session;
        segments = all_data(session_idx).segments;
        num_segments = all_data(session_idx).num_segments;
        
        fprintf('Processing User %d, Session %s: %d segments\n', ...
                user_id, session, num_segments);
        
        % Process each walking segment
        for segment_idx = 1:num_segments
            segment_data = segments{segment_idx};
            
            % Verify segment size
            if size(segment_data, 1) < 100 % Warn if significantly less than 128
                fprintf('  Warning: Segment %d has only %d samples\n', segment_idx, size(segment_data, 1));
            end
            
            % Extract features from this segment
            features = extract_trial_features(segment_data);
            
            % Verify feature dimension
            if length(features) ~= expected_features
                fprintf('Warning: Segment %d has %d features (expected %d)\n', ...
                        segment_idx, length(features), expected_features);
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
    
    % Remove any unused pre-allocated space
    if current_index <= total_segments
        all_features = all_features(1:current_index-1, :);
        all_labels = all_labels(1:current_index-1);
        user_ids = user_ids(1:current_index-1);
        session_types = session_types(1:current_index-1);
    end
    
    % Convert to tables
    feature_table = array2table(all_features);
    label_table = array2table(all_labels, 'VariableNames', {'UserID'}); % FIXED: Removed extra parenthesis
    
    % Generate feature names
    feature_names = generate_feature_names();
    if length(feature_names) == size(feature_table, 2)
        feature_table.Properties.VariableNames = feature_names;
    else
        fprintf('Warning: Feature names count (%d) doesnt match features (%d)\n', ...
                length(feature_names), size(feature_table, 2));
        % Create generic names
        generic_names = arrayfun(@(x) sprintf('Feature_%d', x), 1:size(feature_table, 2), 'UniformOutput', false);
        feature_table.Properties.VariableNames = generic_names;
    end
    
    % Save the extracted features
    save('extracted_features.mat', 'feature_table', 'label_table', 'user_ids', 'session_types', 'feature_names', '-v7.3');
    fprintf('\n=== FEATURE EXTRACTION COMPLETE ===\n');
    fprintf('Total feature vectors: %d\n', size(all_features, 1));
    fprintf('Features per segment: %d\n', size(all_features, 2));
    fprintf('Total users: %d\n', length(unique(user_ids)));
end