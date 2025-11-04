function feature_names = generate_feature_names()
    % Generate descriptive names for 96 time-domain features
    
    sensors = {'AccelX', 'AccelY', 'AccelZ', 'GyroX', 'GyroY', 'GyroZ'};
    base_features = {'Mean', 'Std', 'Variance', 'RMS', ...
                    'Min', 'Max', 'PeakToPeak', ...
                    'Skewness', 'Kurtosis', ...
                    'Energy', 'Power', ...
                    'Median', 'MAD', 'IQR', 'ZCR'};
    
    feature_names = {};
    
    % Per-sensor features: 6 sensors × 15 features = 90 features
    for sensor_idx = 1:6
        for feat_idx = 1:length(base_features)
            feature_names{end+1} = sprintf('%s_%s', sensors{sensor_idx}, base_features{feat_idx});
        end
    end
    
    % Cross-channel features (3 features)
    feature_names{end+1} = 'AccelXY_Corr';
    feature_names{end+1} = 'AccelXZ_Corr';
    feature_names{end+1} = 'AccelYZ_Corr';
    
    % Overall features (3 features)
    feature_names{end+1} = 'Overall_Mean';
    feature_names{end+1} = 'Overall_Std';
    feature_names{end+1} = 'Overall_Energy';
    
    fprintf('Generated %d feature names for time-domain features\n', length(feature_names));
end