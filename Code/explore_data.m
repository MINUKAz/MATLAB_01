%% Data Exploration
fprintf('\n=== DATA EXPLORATION ===\n');

% Check data dimensions and basic statistics
for i = 1:length(all_data)
    data = all_data(i).raw_data;
    
    fprintf('\nFile: %s\n', all_data(i).filename);
    fprintf('Dimensions: %d rows x %d columns\n', size(data, 1), size(data, 2));
    fprintf('Column A unique values: %d\n', length(unique(data(:,1))));
    fprintf('Column A range: %d to %d\n', min(data(:,1)), max(data(:,1)));
    
    % Check for missing values
    missing_vals = sum(isnan(data), 'all');
    fprintf('Missing values: %d\n', missing_vals);
    
    % Basic statistics for sensor columns (B-G)
    sensor_data = data(:, 2:7); % Columns B-G
    fprintf('Sensor data stats (mean): [%.3f, %.3f, %.3f, %.3f, %.3f, %.3f]\n', mean(sensor_data));
    fprintf('Sensor data stats (std):  [%.3f, %.3f, %.3f, %.3f, %.3f, %.3f]\n', std(sensor_data));
end

% Plot sample data from first user, first session
if ~isempty(all_data)
    sample_data = all_data(1).raw_data;
    
    figure;
    subplot(2,1,1);
    plot(sample_data(1:500, 2:4)); % First 500 rows, columns B-D (likely accelerometer)
    title('Sample Accelerometer Data (First 500 samples)');
    xlabel('Sample Index');
    ylabel('Sensor Value');
    legend('B', 'C', 'D');
    grid on;
    
    subplot(2,1,2);
    plot(sample_data(1:500, 5:7)); % Columns E-G (likely gyroscope)
    title('Sample Gyroscope Data (First 500 samples)');
    xlabel('Sample Index');
    ylabel('Sensor Value');
    legend('E', 'F', 'G');
    grid on;
end