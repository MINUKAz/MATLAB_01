%% Visualize Preprocessing Results
clear all; close all; clc;

fprintf('=== PREPROCESSING VISUALIZATION ===\n');

load('processed_data_enhanced.mat', 'all_data');

% Plot before/after for first file
if ~isempty(all_data)
    sample_idx = 1;
    raw_data = all_data(sample_idx).raw_data;
    filtered_data = all_data(sample_idx).filtered_data;
    segments = all_data(sample_idx).segments;
    
    figure('Position', [100, 100, 1200, 800]);
    
    % Plot 1: Raw vs Filtered Accelerometer
    subplot(3,2,1);
    plot(raw_data(1:min(500,end), 2:4)); % First 3 columns (accelerometer)
    title('Raw Accelerometer Data (First 500 samples)');
    xlabel('Sample Index');
    ylabel('Acceleration');
    legend('X', 'Y', 'Z');
    grid on;
    
    subplot(3,2,2);
    plot(filtered_data(1:min(500,end), 1:3));
    title('Filtered Accelerometer Data (First 500 samples)');
    xlabel('Sample Index');
    ylabel('Acceleration (Filtered)');
    legend('X', 'Y', 'Z');
    grid on;
    
    % Plot 2: Raw vs Filtered Gyroscope
    subplot(3,2,3);
    plot(raw_data(1:min(500,end), 5:7)); % Last 3 columns (gyroscope)
    title('Raw Gyroscope Data (First 500 samples)');
    xlabel('Sample Index');
    ylabel('Angular Velocity');
    legend('X', 'Y', 'Z');
    grid on;
    
    subplot(3,2,4);
    plot(filtered_data(1:min(500,end), 4:6));
    title('Filtered Gyroscope Data (First 500 samples)');
    xlabel('Sample Index');
    ylabel('Angular Velocity (Filtered)');
    legend('X', 'Y', 'Z');
    grid on;
    
    % Plot 3: Sample segments
    subplot(3,2,5);
    if ~isempty(segments)
        first_segment = segments{1};
        plot(first_segment(:, 1:3));
        title('First 4-Second Segment (Accelerometer)');
        xlabel('Sample Index (0-127)');
        ylabel('Acceleration');
        legend('X', 'Y', 'Z');
        grid on;
    end
    
    subplot(3,2,6);
    if ~isempty(segments)
        plot(first_segment(:, 4:6));
        title('First 4-Second Segment (Gyroscope)');
        xlabel('Sample Index (0-127)');
        ylabel('Angular Velocity');
        legend('X', 'Y', 'Z');
        grid on;
    end
    
    % Display statistics
    fprintf('\nPreprocessing Statistics:\n');
    fprintf('Original samples: %d\n', size(raw_data, 1));
    fprintf('Filtered samples: %d\n', size(filtered_data, 1));
    fprintf('Number of 4-second segments: %d\n', length(segments));
    fprintf('Segment size: %d samples\n', size(segments{1}, 1));
end