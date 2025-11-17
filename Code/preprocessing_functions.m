function [filtered_data, time_vector] = preprocessing_functions(raw_data, original_fs, target_fs)
    % COMPREHENSIVE PREPROCESSING PIPELINE
    % Input: raw sensor data, original sampling rate, target sampling rate (32 Hz)
    % Output: filtered data and uniform time vector
    
    fprintf('Starting preprocessing pipeline...\n');
    
    % Extract sensor columns (assuming columns 2-7: B-G are sensor data)
    sensor_data = raw_data(:, 2:7);
    
    % Create original time vector
    original_time = (0:size(sensor_data,1)-1)' / original_fs;
    
    %% 1. TIME INTERPOLATION TO 32 Hz
    fprintf('1. Time interpolation to %d Hz... ', target_fs);
    
    % Create uniform time vector at target frequency
    max_time = original_time(end);
    time_vector = (0:1/target_fs:max_time)';
    
    % Linear interpolation for each sensor channel
    interpolated_data = zeros(length(time_vector), 6);
    for i = 1:6
        interpolated_data(:, i) = interp1(original_time, sensor_data(:, i), time_vector, 'linear');
    end
    fprintf('DONE\n');
    
    %% 2. REMOVE STATIONARY/UNWORN SEGMENTS
    fprintf('2. Removing stationary segments... ');
    
    % Calculate acceleration magnitude (first 3 columns are accelerometer)
    accel_magnitude = sqrt(sum(interpolated_data(:, 1:3).^2, 2));
    
    % Find segments where magnitude is near gravity (9.8 m/s²) ± threshold
    gravity_threshold = 0.5; % m/s² tolerance
    stationary_mask = abs(accel_magnitude - 9.8) < gravity_threshold;
    
    % Remove consecutive stationary segments longer than 2 seconds
    min_stationary_samples = 2 * target_fs; % 2 seconds at target frequency
    
    % Find runs of stationary samples
    [run_lengths, run_values] = RunLength(stationary_mask);
    long_stationary_runs = run_lengths >= min_stationary_samples & run_values == 1;
    
    % Create mask to keep only non-stationary segments
    keep_mask = true(size(stationary_mask));
    current_idx = 1;
    for i = 1:length(run_lengths)
        if long_stationary_runs(i)
            keep_mask(current_idx:current_idx+run_lengths(i)-1) = false;
        end
        current_idx = current_idx + run_lengths(i);
    end
    
    % Apply stationary removal
    non_stationary_data = interpolated_data(keep_mask, :);
    non_stationary_time = time_vector(keep_mask);
    
    fprintf('Removed %d stationary samples (%.1f%%)\n', ...
            sum(~keep_mask), sum(~keep_mask)/length(keep_mask)*100);
    
    %% 3. FILTERING
    fprintf('3. Applying filters... ');
    
    % 3A. Band-pass filter for Accelerometer (0.5-5 Hz)
    accel_data = non_stationary_data(:, 1:3);
    filtered_accel = bandpass_filter_accelerometer(accel_data, target_fs);
    
    % 3B. Low-pass filter for Gyroscope (cutoff 5 Hz)
    gyro_data = non_stationary_data(:, 4:6);
    filtered_gyro = lowpass_filter_gyroscope(gyro_data, target_fs);
    
    % Combine filtered data
    filtered_data = [filtered_accel, filtered_gyro];
    
    fprintf('DONE\n');
    
    fprintf('Preprocessing complete: %d → %d samples\n', ...
            size(raw_data, 1), size(filtered_data, 1));
end

function filtered_signal = bandpass_filter_accelerometer(signal, fs)
    % Band-pass filter: 0.5-5 Hz for accelerometer data
    % Removes gravity (DC) and high-frequency noise
    
    nyquist = fs / 2;
    low_cutoff = 0.5 / nyquist;   % 0.5 Hz
    high_cutoff = 5.0 / nyquist;   % 5 Hz
    
    % Design Butterworth band-pass filter
    [b, a] = butter(4, [low_cutoff, high_cutoff], 'bandpass');
    
    % Apply filter to each channel
    filtered_signal = zeros(size(signal));
    for i = 1:size(signal, 2)
        filtered_signal(:, i) = filtfilt(b, a, signal(:, i));
    end
end

function filtered_signal = lowpass_filter_gyroscope(signal, fs)
    % Low-pass filter: 5 Hz cutoff for gyroscope data
    % Removes high-frequency noise while preserving gait frequencies
    
    nyquist = fs / 2;
    cutoff = 5.0 / nyquist;  % 5 Hz cutoff
    
    % Design Butterworth low-pass filter
    [b, a] = butter(4, cutoff, 'low');
    
    % Apply filter to each channel
    filtered_signal = zeros(size(signal));
    for i = 1:size(signal, 2)
        filtered_signal(:, i) = filtfilt(b, a, signal(:, i));
    end
end

function [lengths, values] = RunLength(x)
    % RUNLENGTH - Run-length encoding
    if length(x) == 0
        lengths = [];
        values = [];
        return
    end
    
    indices = [1; find(diff(x) ~= 0) + 1; length(x) + 1];
    lengths = diff(indices);
    values = x(indices(1:end-1));
end