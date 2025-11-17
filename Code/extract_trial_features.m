function features = extract_trial_features(trial_data)
    % UPDATED FEATURE EXTRACTION FOR 128-SAMPLE SEGMENTS (4 seconds at 32 Hz)
    % trial_data: 128×6 matrix (columns: accelerometer XYZ, gyroscope XYZ)
    
    % Validate input
    if nargin < 1 || isempty(trial_data)
        error('Invalid input to extract_trial_features');
    end
    
    expected_samples = 128; % 4 seconds × 32 Hz
    if size(trial_data, 1) ~= expected_samples
        warning('Expected %d samples, got %d. Feature quality may be affected.', ...
                expected_samples, size(trial_data, 1));
    end
    
    % Extract sensor data (all 6 columns)
    sensor_data = trial_data;
    
    % Initialize feature vector
    features = [];
    
    % For each sensor channel (6 channels total)
    for channel = 1:6
        signal = sensor_data(:, channel);
        
        % === TIME-DOMAIN FEATURES (15 features per channel) ===
        
        % 1. Basic Statistical Features
        mean_val = mean(signal);
        std_val = std(signal);
        variance = var(signal);
        rms_val = rms(signal);
        
        % 2. Range and Peak Features
        min_val = min(signal);
        max_val = max(signal);
        peak_to_peak = max_val - min_val;
        
        % 3. Shape Features
        skewness_val = skewness(signal);
        kurtosis_val = kurtosis(signal);
        
        % 4. Energy Features
        signal_energy = sum(signal.^2);
        signal_power = signal_energy / length(signal);
        
        % 5. Distribution Features
        median_val = median(signal);
        mad_val = mad(signal); % Median Absolute Deviation
        iqr_val = iqr(signal); % Interquartile Range
        
        % 6. Zero-Crossing Rate
        zero_crossings = sum(diff(sign(signal)) ~= 0);
        zcr_normalized = zero_crossings / (length(signal) - 1);
        
        % 7. Additional gait-specific features
        % Signal Magnitude Area (SMA) - good for gait analysis
        sma_val = sum(abs(signal)) / length(signal);
        
        % Combine all 16 features for this channel
        channel_features = [mean_val, std_val, variance, rms_val, ...
                           min_val, max_val, peak_to_peak, ...
                           skewness_val, kurtosis_val, ...
                           signal_energy, signal_power, ...
                           median_val, mad_val, iqr_val, ...
                           zcr_normalized, sma_val];
        
        features = [features, channel_features];
    end
    
    % === CROSS-CHANNEL FEATURES (6 features) ===
    try
        % Accelerometer correlations
        accel_corr = corr(sensor_data(:, 1:3));
        cross_features = [accel_corr(1,2), accel_corr(1,3), accel_corr(2,3)];
        
        % Gyroscope correlations
        gyro_corr = corr(sensor_data(:, 4:6));
        cross_features = [cross_features, gyro_corr(1,2), gyro_corr(1,3), gyro_corr(2,3)];
    catch
        cross_features = zeros(1, 6);
    end
    
    % === OVERALL FEATURES (6 features) ===
    overall_mean = mean(sensor_data(:));
    overall_std = std(sensor_data(:));
    overall_energy = sum(sensor_data(:).^2);
    
    % Additional overall features
    overall_range = max(sensor_data(:)) - min(sensor_data(:));
    overall_abs_mean = mean(abs(sensor_data(:)));
    overall_rms = rms(sensor_data(:));
    
    overall_features = [overall_mean, overall_std, overall_energy, ...
                       overall_range, overall_abs_mean, overall_rms];
    
    % Final feature vector: 6×16 + 6 + 6 = 96 + 12 = 108 features
    features = [features, cross_features, overall_features];
    
    % Verify feature count
    expected_features = 108;
    if length(features) ~= expected_features
        fprintf('Adjusting features from %d to %d\n', length(features), expected_features);
        if length(features) > expected_features
            features = features(1:expected_features);
        else
            features = [features, zeros(1, expected_features - length(features))];
        end
    end
    
    fprintf('Extracted %d features from %d-sample segment\n', length(features), size(trial_data, 1));
end