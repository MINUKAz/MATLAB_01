%% Data Loading and Basic Exploration - ENHANCED VERSION
clear all; close all; clc;

% Define the path to your data folder
data_folder = 'Data/';

% List all CSV files in the folder
file_list = dir([data_folder '*.csv']);

% Initialize variables to store all data
all_data = struct();

% Sampling rate parameters
original_fs = 30;  % Original sampling rate (approx)
target_fs = 32;    % Target uniform sampling rate

fprintf('=== ENHANCED DATA LOADING WITH PREPROCESSING ===\n');

% Loop through each file
for i = 1:length(file_list)
    filename = file_list(i).name;
    fprintf('Loading and processing: %s\n', filename);
    
    % Read the CSV file (no headers)
    file_path = [data_folder filename];
    raw_data = readmatrix(file_path);
    
    % Extract user ID and session type from filename
    tokens = regexp(filename, 'U(\d+)NW_([FM]D)\.csv', 'tokens');
    if ~isempty(tokens)
        user_id = str2double(tokens{1}{1});
        session = tokens{1}{2};
        
        fprintf('  User %d, Session: %s - ', user_id, session);
        
        % APPLY PREPROCESSING PIPELINE
        [filtered_data, time_vector] = preprocessing_functions(raw_data, original_fs, target_fs);
        
        % Only proceed if we have enough data after preprocessing
        if size(filtered_data, 1) >= 128  % Need at least one 4-second window
            % SEGMENT INTO 4-SECOND WINDOWS WITH 50% OVERLAP
            segments = segment_4s_windows(filtered_data, target_fs, 4, 0.5);
            
            % Store processed data in structure
            all_data(i).user_id = user_id;
            all_data(i).session = session;
            all_data(i).filename = filename;
            all_data(i).raw_data = raw_data;
            all_data(i).filtered_data = filtered_data;
            all_data(i).time_vector = time_vector;
            all_data(i).segments = segments;
            all_data(i).num_segments = length(segments);
            all_data(i).original_fs = original_fs;
            all_data(i).target_fs = target_fs;
            
            fprintf('Created %d segments of %d samples\n', length(segments), size(segments{1}, 1));
        else
            fprintf('SKIPPED - Not enough data after preprocessing (%d samples)\n', size(filtered_data, 1));
        end
    else
        fprintf('  Warning: Could not parse filename %s\n', filename);
    end
end

% Remove empty entries
all_data = all_data(~cellfun('isempty', {all_data.user_id}));

fprintf('\n=== LOADING COMPLETE ===\n');
fprintf('Total files processed: %d\n', length(all_data));

% Calculate total segments
total_segments = 0;
for i = 1:length(all_data)
    if isfield(all_data(i), 'num_segments')
        total_segments = total_segments + all_data(i).num_segments;
    end
end
fprintf('Total 4-second segments: %d\n', total_segments);

% Save processed data
save('processed_data_enhanced.mat', 'all_data', 'target_fs', '-v7.3');
fprintf('Enhanced processed data saved to processed_data_enhanced.mat\n');

% Display sample verification
if ~isempty(all_data) && isfield(all_data(1), 'segments') && ~isempty(all_data(1).segments)
    sample_segment = all_data(1).segments{1};
    fprintf('Sample segment size: %dx%d (should be 128x6)\n', size(sample_segment));
end