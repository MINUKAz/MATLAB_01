%% Neural Network Training for Gait Authentication
clear all; close all; clc;

fprintf('=== NEURAL NETWORK TRAINING FOR GAIT AUTHENTICATION ===\n');

% Load extracted features
if ~exist('extracted_features.mat', 'file')
    error('extracted_features.mat not found. Please run feature extraction first.');
end

load('extracted_features.mat', 'feature_table', 'label_table', 'user_ids');

fprintf('Loaded features: %d samples, %d features\n', size(feature_table, 1), size(feature_table, 2));
fprintf('Number of users: %d\n', length(unique(user_ids)));

%% Prepare Data for Neural Network
fprintf('\n--- Preparing Data ---\n');

% Convert features and labels to arrays
X = table2array(feature_table);  % Features
y_original = table2array(label_table);  % Original user IDs (1-10)

% Convert to categorical labels for classification
y_categorical = categorical(y_original);

% One-hot encode the labels
y_onehot = dummyvar(y_categorical)';

fprintf('Feature matrix size: %dx%d\n', size(X));
fprintf('Labels size: %dx%d\n', size(y_onehot));

%% Split Data into Training and Testing
fprintf('\n--- Splitting Data ---\n');

% Use 80% for training, 20% for testing
rng(42); % Set seed for reproducibility
cv = cvpartition(length(y_original), 'HoldOut', 0.2);

X_train = X(cv.training,:)';
X_test = X(cv.test,:)';
y_train = y_onehot(:, cv.training);
y_test = y_onehot(:, cv.test);

fprintf('Training set: %d samples\n', size(X_train, 2));
fprintf('Testing set: %d samples\n', size(X_test, 2));

%% Create and Configure Neural Network
fprintf('\n--- Creating Neural Network ---\n');

% Network architecture
input_size = size(X_train, 1);  % 96 features
hidden_layer_size = 64;         % Neurons in hidden layer
num_classes = size(y_train, 1); % 10 users

fprintf('Network architecture: %d -> %d -> %d\n', input_size, hidden_layer_size, num_classes);

% Create pattern recognition network (feedforward)
net = patternnet(hidden_layer_size);

% Configure network parameters
net.divideParam.trainRatio = 0.85;  % 85% of training data for actual training
net.divideParam.valRatio = 0.15;    % 15% for validation during training
net.divideParam.testRatio = 0.0;    % 0% for testing (we have separate test set)

% Training parameters
net.trainFcn = 'trainscg';  % Scaled conjugate gradient
net.performFcn = 'crossentropy';  % Cross-entropy for classification
net.plotFcns = {'plotperform', 'plottrainstate', 'ploterrhist'};

fprintf('Neural network created successfully!\n');

%% Train the Neural Network
fprintf('\n--- Training Neural Network ---\n');
fprintf('This may take a few minutes...\n');

% Train the network
[net, tr] = train(net, X_train, y_train);

fprintf('Training complete!\n');

% Plot training performance
figure('Position', [100, 100, 1200, 400]);
subplot(1,2,1);
plotperform(tr);
title('Training Performance');

subplot(1,2,2);
plottrainstate(tr);
title('Training State');

%% Evaluate Network Performance
fprintf('\n--- Evaluating Network ---\n');

% Predict on test set
y_pred = net(X_test);

% Convert predictions to class labels
[~, predicted_labels] = max(y_pred);
[~, true_labels] = max(y_test);

% Calculate accuracy
accuracy = sum(predicted_labels == true_labels) / length(true_labels) * 100;
fprintf('Test Accuracy: %.2f%%\n', accuracy);

%% Enhanced Evaluation Metrics
fprintf('\n--- Enhanced Evaluation Metrics ---\n');

% Get prediction scores for ROC-AUC calculation
y_pred_scores = net(X_test);  % This gives probability scores

% Calculate comprehensive metrics
metrics = enhanced_evaluation_metrics(true_labels, predicted_labels, y_pred_scores, num_classes);

% Save enhanced metrics
save('comprehensive_metrics.mat', 'metrics', 'true_labels', 'predicted_labels', 'y_pred_scores');

% Confusion matrix
figure;
plotconfusion(y_test, y_pred);
title('Confusion Matrix - User Classification');

%% Calculate Authentication Metrics (FAR, FRR, EER)
fprintf('\n--- Calculating Authentication Metrics ---\n');

% For authentication, we need to compute scores and thresholds
% We'll treat this as a verification system: for each user, genuine vs impostor

authentication_results = calculate_authentication_metrics(net, X_test, y_test, true_labels);

fprintf('Authentication metrics calculated!\n');

%% Save the Trained Model
fprintf('\n--- Saving Model ---\n');

save('trained_neural_network.mat', 'net', 'tr', 'accuracy', 'authentication_results', 'hidden_layer_size');
fprintf('Trained model saved to trained_neural_network.mat\n');

fprintf('\n=== NEURAL NETWORK TRAINING COMPLETE ===\n');
fprintf('Final Test Accuracy: %.2f%%\n', accuracy);