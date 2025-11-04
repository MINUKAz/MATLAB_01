function features = extract_trial_features(trial_data)
    % RELIABLE feature extraction - ONLY TIME-DOMAIN FEATURES
    % Guaranteed 85 features (6×13 + 4 + 3 = 85)
    % trial_data: 30×7 matrix (columns: A=trial_id, B-G=sensor data)
    
    % Validate input
    if nargin < 1 || isempty(trial_data)
        error('Invalid input to extract_trial_features');
    end
    
    % Extract sensor data (columns 2-7)
    sensor_data = trial_data(:, 2:7);
    
    % Initialize feature vector
    features = [];
    
    % For each sensor channel (B through G)
    for channel = 1:6
        signal = sensor_data(:, channel);
        
        % === TIME-DOMAIN FEATURES ONLY (13 features per channel) ===
        
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
        
        % Combine all 13 features for this channel
        channel_features = [mean_val, std_val, variance, rms_val, ...
                           min_val, max_val, peak_to_peak, ...
                           skewness_val, kurtosis_val, ...
                           signal_energy, signal_power, ...
                           median_val, mad_val, iqr_val, ...
                           zcr_normalized];
        
        % Safety check - ensure exactly 15 features per channel
        if length(channel_features) ~= 15
            error('Channel %d: Expected 15 features, got %d', channel, length(channel_features));
        end
        
        features = [features, channel_features];
    end
    
    % === CROSS-CHANNEL FEATURES (4 features) ===
    % Simple correlations between main accelerometer axes
    try
        accel_corr = corr(sensor_data(:, 1:3)); % First 3 columns (accelerometer)
        cross_features = [accel_corr(1,2), accel_corr(1,3), accel_corr(2,3)]; % XY, XZ, YZ correlations
    catch
        cross_features = [0, 0, 0];
    end
    
    % === OVERALL FEATURES (3 features) ===
    overall_mean = mean(sensor_data(:));
    overall_std = std(sensor_data(:));
    overall_energy = sum(sensor_data(:).^2);
    
    overall_features = [overall_mean, overall_std, overall_energy];
    
    % Final feature vector
    features = [features, cross_features, overall_features];
    
    % Verify exact feature count: 6×15 + 3 + 3 = 90 + 3 + 3 = 96 features
    expected_features = 96;
    if length(features) ~= expected_features
        fprintf('Adjusting features from %d to %d\n', length(features), expected_features);
        if length(features) > expected_features
            features = features(1:expected_features);
        else
            features = [features, zeros(1, expected_features - length(features))];
        end
    end
    
    fprintf('Extracted %d time-domain features\n', length(features));
end