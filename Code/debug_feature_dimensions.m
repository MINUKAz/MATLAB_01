%% Debug Single Channel Feature Extraction
clear all; close all; clc;

load('processed_data.mat', 'all_data');

% Test on first trial, first channel only
test_trial = all_data(1).segments{1};
sensor_data = test_trial(:, 2:7);
signal = sensor_data(:, 1); % First channel only

fprintf('=== DEBUG SINGLE CHANNEL ===\n');
fprintf('Signal size: %dx%d\n', size(signal));

% Test time-domain features
fprintf('\n--- Time-Domain Features ---\n');
time_features = [mean(signal), std(signal), var(signal), rms(signal), ...
                max(signal)-min(signal), skewness(signal), kurtosis(signal), ...
                sum(signal.^2), sum(signal.^2)/length(signal), median(signal), ...
                mad(signal), iqr(signal), sum(diff(sign(signal)) ~= 0)/(length(signal)-1)];

fprintf('Time features count: %d\n', length(time_features));
fprintf('All time features are scalars: %d\n', all(arrayfun(@isscalar, time_features)));

% Test frequency-domain features
fprintf('\n--- Frequency-Domain Features ---\n');
N = length(signal);
fft_signal = fft(signal);

if mod(N, 2) == 0
    fft_magnitude = abs(fft_signal(1:N/2+1));
    frequencies = linspace(0, 0.5, N/2+1);
else
    fft_magnitude = abs(fft_signal(1:(N+1)/2));
    frequencies = linspace(0, 0.5, (N+1)/2);
end

fft_magnitude_no_dc = fft_magnitude(2:end);
frequencies_no_dc = frequencies(2:end);

fprintf('FFT magnitude size: %dx%d\n', size(fft_magnitude_no_dc));
fprintf('Frequencies size: %dx%d\n', size(frequencies_no_dc));

% Calculate each frequency feature separately
spectral_centroid = sum(frequencies_no_dc .* fft_magnitude_no_dc) / sum(fft_magnitude_no_dc);
spectral_bandwidth = sqrt(sum((frequencies_no_dc - spectral_centroid).^2 .* fft_magnitude_no_dc) / sum(fft_magnitude_no_dc));

cum_energy = cumsum(fft_magnitude_no_dc);
rolloff_idx = find(cum_energy >= 0.85 * sum(fft_magnitude_no_dc), 1);
spectral_rolloff = frequencies_no_dc(rolloff_idx);

[~, max_idx] = max(fft_magnitude_no_dc);
dominant_freq = frequencies_no_dc(max_idx);

fprintf('Spectral Centroid: %f (scalar: %d)\n', spectral_centroid, isscalar(spectral_centroid));
fprintf('Spectral Bandwidth: %f (scalar: %d)\n', spectral_bandwidth, isscalar(spectral_bandwidth));
fprintf('Spectral Rolloff: %f (scalar: %d)\n', spectral_rolloff, isscalar(spectral_rolloff));
fprintf('Dominant Freq: %f (scalar: %d)\n', dominant_freq, isscalar(dominant_freq));

% Combine and test
freq_features = [spectral_centroid, spectral_bandwidth, spectral_rolloff, dominant_freq];
fprintf('Frequency features count: %d\n', length(freq_features));
fprintf('All frequency features are scalars: %d\n', all(arrayfun(@isscalar, freq_features)));

% Test complete channel
channel_features = [time_features, freq_features];
fprintf('\nTotal channel features: %d\n', length(channel_features));