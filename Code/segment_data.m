%% Data Segmentation into Walking Trials
fprintf('\n=== DATA SEGMENTATION ===\n');

for i = 1:length(all_data)
    data = all_data(i).raw_data;
    user_id = all_data(i).user_id;
    session = all_data(i).session;
    
    % Get unique trial IDs from column A
    trial_ids = unique(data(:,1));
    num_trials = length(trial_ids);
    
    fprintf('User %d, Session %s: %d trials found\n', user_id, session, num_trials);
    
    % Initialize cell array to store segments
    segments = cell(num_trials, 1);
    segment_lengths = zeros(num_trials, 1);
    
    % Extract each trial
    for trial_idx = 1:num_trials
        trial_id = trial_ids(trial_idx);
        trial_data = data(data(:,1) == trial_id, :);
        
        segments{trial_idx} = trial_data;
        segment_lengths(trial_idx) = size(trial_data, 1);
        
        % Verify segment length (should be around 30)
        if segment_lengths(trial_idx) ~= 30
            fprintf('  Warning: Trial %d has %d samples (expected 30)\n', ...
                    trial_id, segment_lengths(trial_idx));
        end
    end
    
    % Store segments in the structure
    all_data(i).trial_ids = trial_ids;
    all_data(i).segments = segments;
    all_data(i).segment_lengths = segment_lengths;
    all_data(i).num_trials = num_trials;
end

% Display segmentation summary
fprintf('\n=== SEGMENTATION SUMMARY ===\n');
for i = 1:length(all_data)
    fprintf('User %d, Session %s: %d trials, avg length: %.1f samples\n', ...
            all_data(i).user_id, all_data(i).session, ...
            all_data(i).num_trials, mean(all_data(i).segment_lengths));
end