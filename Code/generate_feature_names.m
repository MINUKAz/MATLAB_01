function feature_names = generate_feature_names()
    % Generate descriptive names for 108 time-domain features
    
    sensors = {'AccelX', 'AccelY', 'AccelZ', 'GyroX', 'GyroY', 'GyroZ'};
    base_features = {'Mean', 'Std', 'Variance', 'RMS', ...
                    'Min', 'Max', 'PeakToPeak', ...
                    'Skewness', 'Kurtosis', ...
                    'Energy', 'Power', ...
                    'Median', 'MAD', 'IQR', 'ZCR', 'SMA'}; % Added SMA
    
    feature_names = {};
    
    % Per-sensor features: 6 sensors × 16 features = 96 features
    for sensor_idx = 1:6
        for feat_idx = 1:length(base_features)
            feature_names{end+1} = sprintf('%s_%s', sensors{sensor_idx}, base_features{feat_idx});
        end
    end
    
    % Cross-channel features (6 features)
    feature_names{end+1} = 'AccelXY_Corr';
    feature_names{end+1} = 'AccelXZ_Corr';
    feature_names{end+1} = 'AccelYZ_Corr';
    feature_names{end+1} = 'GyroXY_Corr';
    feature_names{end+1} = 'GyroXZ_Corr';
    feature_names{end+1} = 'GyroYZ_Corr';
    
    % Overall features (6 features)
    feature_names{end+1} = 'Overall_Mean';
    feature_names{end+1} = 'Overall_Std';
    feature_names{end+1} = 'Overall_Energy';
    feature_names{end+1} = 'Overall_Range';
    feature_names{end+1} = 'Overall_AbsMean';
    feature_names{end+1} = 'Overall_RMS';
    
    % Verify we have exactly 108 features
    if length(feature_names) ~= 108
        fprintf('Warning: Expected 108 feature names, generated %d\n', length(feature_names));
        % Trim or pad to exactly 108
        if length(feature_names) > 108
            feature_names = feature_names(1:108);
        else
            for i = length(feature_names)+1:108
                feature_names{i} = sprintf('Feature_%d', i);
            end
        end
    end
    
    fprintf('Generated %d feature names for enhanced time-domain features\n', length(feature_names));
end