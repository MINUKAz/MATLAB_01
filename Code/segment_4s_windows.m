function segments = segment_4s_windows(filtered_data, fs, window_sec, overlap_ratio)
    % SEGMENT DATA INTO 4-SECOND WINDOWS WITH 50% OVERLAP
    % Input: filtered data, sampling rate, window duration, overlap ratio
    % Output: cell array of segments
    
    fprintf('Segmenting into %.1f-second windows with %.0f%% overlap...\n', ...
            window_sec, overlap_ratio*100);
    
    samples_per_window = window_sec * fs;  % 4 seconds × 32 Hz = 128 samples
    overlap_samples = round(samples_per_window * overlap_ratio); % 64 samples
    step_size = samples_per_window - overlap_samples; % 64 samples
    
    num_samples = size(filtered_data, 1);
    num_segments = floor((num_samples - samples_per_window) / step_size) + 1;
    
    segments = cell(num_segments, 1);
    
    for i = 1:num_segments
        start_idx = (i-1) * step_size + 1;
        end_idx = start_idx + samples_per_window - 1;
        
        if end_idx <= num_samples
            segments{i} = filtered_data(start_idx:end_idx, :);
        end
    end
    
    % Remove empty segments
    segments = segments(~cellfun('isempty', segments));
    
    fprintf('Created %d segments of %d samples each\n', ...
            length(segments), samples_per_window);
end