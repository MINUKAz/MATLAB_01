function results = calculate_authentication_metrics(net, X_test, y_test, true_labels)
    % Calculate FAR, FRR, and EER for authentication system
    
    fprintf('Calculating authentication metrics...\n');
    
    % Get network outputs (probabilities)
    y_scores = net(X_test);
    
    num_users = size(y_scores, 1);
    results = struct();
    
    % For each user, calculate genuine and impostor scores
    for user = 1:num_users
        % Genuine scores: when this user is the true label
        genuine_mask = (true_labels == user);
        genuine_scores = y_scores(user, genuine_mask);
        
        % Impostor scores: when this user is NOT the true label
        impostor_mask = (true_labels ~= user);
        impostor_scores = y_scores(user, impostor_mask);
        
        % Store results
        results(user).user_id = user;
        results(user).genuine_scores = genuine_scores;
        results(user).impostor_scores = impostor_scores;
        results(user).num_genuine = length(genuine_scores);
        results(user).num_impostor = length(impostor_scores);
        
        fprintf('User %d: %d genuine, %d impostor samples\n', ...
                user, length(genuine_scores), length(impostor_scores));
    end
    
    % Calculate overall FAR, FRR, EER (you can expand this)
    fprintf('\nAuthentication metrics calculation complete.\n');
    fprintf('Next step: Plot ROC curves and calculate EER for each user.\n');
end