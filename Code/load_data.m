%% Data Loading and Basic Exploration
clear all; close all; clc;

% Define the path to your data folder
data_folder = 'Data/';

% List all CSV files in the folder
file_list = dir([data_folder '*.csv']);

% Initialize variables to store all data
all_data = struct();
user_count = 0;

% Loop through each file
for i = 1:length(file_list)
    filename = file_list(i).name;
    fprintf('Loading file: %s\n', filename);
    
    % Read the CSV file (no headers)
    file_path = [data_folder filename];
    raw_data = readmatrix(file_path);
    
    % Extract user ID and session type from filename
    % Example: 'U1NW_FD.csv' -> user_id = 1, session = 'FD'
    tokens = regexp(filename, 'U(\d+)NW_([FM]D)\.csv', 'tokens');
    if ~isempty(tokens)
        user_id = str2double(tokens{1}{1});
        session = tokens{1}{2};
        
        % Store data in structure
        all_data(i).user_id = user_id;
        all_data(i).session = session;
        all_data(i).filename = filename;
        all_data(i).raw_data = raw_data;
        all_data(i).num_rows = size(raw_data, 1);
        all_data(i).num_cols = size(raw_data, 2);
        
        fprintf('  User %d, Session: %s, Size: %dx%d\n', ...
                user_id, session, size(raw_data, 1), size(raw_data, 2));
    else
        fprintf('  Warning: Could not parse filename %s\n', filename);
    end
end

fprintf('\nTotal files loaded: %d\n', length(all_data));