function metrics = enhanced_evaluation_metrics(true_labels, predicted_labels, y_scores, num_classes)
    % COMPREHENSIVE EVALUATION METRICS FOR MULTI-CLASS CLASSIFICATION
    
    fprintf('\n=== COMPREHENSIVE MODEL EVALUATION ===\n');
    
    %% 1. Basic Accuracy
    accuracy = sum(true_labels == predicted_labels) / length(true_labels) * 100;
    fprintf('Accuracy: %.2f%%\n', accuracy);
    
    %% 2. Precision, Recall, F1-Score (Per-class and Macro)
    confusion_mat = confusionmat(true_labels, predicted_labels);
    
    precision_per_class = zeros(1, num_classes);
    recall_per_class = zeros(1, num_classes);
    f1_per_class = zeros(1, num_classes);
    
    for i = 1:num_classes
        TP = confusion_mat(i,i);
        FP = sum(confusion_mat(:,i)) - TP;
        FN = sum(confusion_mat(i,:)) - TP;
        
        precision_per_class(i) = TP / (TP + FP + eps);
        recall_per_class(i) = TP / (TP + FN + eps);
        f1_per_class(i) = 2 * (precision_per_class(i) * recall_per_class(i)) / ...
                          (precision_per_class(i) + recall_per_class(i) + eps);
    end
    
    % Macro averages
    macro_precision = mean(precision_per_class);
    macro_recall = mean(recall_per_class);
    macro_f1 = mean(f1_per_class);
    
    fprintf('\n--- Per-Class Metrics ---\n');
    for i = 1:num_classes
        fprintf('User %d: Precision=%.3f, Recall=%.3f, F1=%.3f\n', ...
                i, precision_per_class(i), recall_per_class(i), f1_per_class(i));
    end
    
    fprintf('\n--- Macro Averages ---\n');
    fprintf('Macro Precision: %.3f\n', macro_precision);
    fprintf('Macro Recall: %.3f\n', macro_recall);
    fprintf('Macro F1-Score: %.3f\n', macro_f1);
    
    %% 3. ROC-AUC for Multi-class (One-vs-Rest approach)
    fprintf('\n--- ROC-AUC Metrics (One-vs-Rest) ---\n');
    
    % Convert to one-hot encoding for ROC calculation
    true_labels_onehot = zeros(num_classes, length(true_labels));
    for i = 1:length(true_labels)
        true_labels_onehot(true_labels(i), i) = 1;
    end
    
    % Calculate AUC for each class
    auc_per_class = zeros(1, num_classes);
    figure('Position', [100, 100, 1200, 800]);
    
    for i = 1:num_classes
        [~, ~, ~, auc_per_class(i)] = perfcurve(true_labels_onehot(i,:), y_scores(i,:), 1);
        fprintf('User %d AUC: %.3f\n', i, auc_per_class(i));
    end
    
    macro_auc = mean(auc_per_class);
    fprintf('Macro Average AUC: %.3f\n', macro_auc);
    
    %% 4. Comprehensive Metrics Structure
    metrics = struct();
    metrics.accuracy = accuracy;
    metrics.precision_per_class = precision_per_class;
    metrics.recall_per_class = recall_per_class;
    metrics.f1_per_class = f1_per_class;
    metrics.macro_precision = macro_precision;
    metrics.macro_recall = macro_recall;
    metrics.macro_f1 = macro_f1;
    metrics.auc_per_class = auc_per_class;
    metrics.macro_auc = macro_auc;
    metrics.confusion_matrix = confusion_mat;
    
    %% 5. Visualization
    % ROC Curves
    subplot(2,3,1);
    colors = lines(num_classes);
    for i = 1:num_classes
        [X, Y, ~, AUC] = perfcurve(true_labels_onehot(i,:), y_scores(i,:), 1);
        plot(X, Y, 'Color', colors(i,:), 'LineWidth', 2);
        hold on;
    end
    plot([0 1], [0 1], 'k--', 'LineWidth', 1);
    xlabel('False Positive Rate');
    ylabel('True Positive Rate');
    title('ROC Curves (One-vs-Rest)');
    legend(arrayfun(@(x) sprintf('User %d (AUC=%.3f)', x, auc_per_class(x)), 1:num_classes, 'UniformOutput', false), ...
           'Location', 'southeast');
    grid on;
    
    % Precision-Recall per class
    subplot(2,3,2);
    for i = 1:num_classes
        plot(recall_per_class(i), precision_per_class(i), 'o', 'Color', colors(i,:), ...
             'MarkerSize', 10, 'LineWidth', 2);
        hold on;
        text(recall_per_class(i), precision_per_class(i), sprintf(' User %d', i));
    end
    xlabel('Recall');
    ylabel('Precision');
    title('Precision-Recall per User');
    grid on;
    xlim([0 1]);
    ylim([0 1]);
    
    % F1-Score Bar Chart
    subplot(2,3,3);
    bar(f1_per_class, 'FaceColor', [0.2 0.6 0.8]);
    set(gca, 'XTickLabel', arrayfun(@(x) sprintf('User %d', x), 1:num_classes, 'UniformOutput', false));
    ylabel('F1-Score');
    title('F1-Score per User');
    grid on;
    
    % AUC Bar Chart
    subplot(2,3,4);
    bar(auc_per_class, 'FaceColor', [0.8 0.4 0.2]);
    set(gca, 'XTickLabel', arrayfun(@(x) sprintf('User %d', x), 1:num_classes, 'UniformOutput', false));
    ylabel('AUC');
    title('AUC per User');
    grid on;
    
    % Metrics Summary
    subplot(2,3,5);
    summary_metrics = [macro_precision, macro_recall, macro_f1, macro_auc, accuracy/100];
    bar(summary_metrics, 'FaceColor', [0.3 0.7 0.3]);
    set(gca, 'XTickLabel', {'Precision', 'Recall', 'F1', 'AUC', 'Accuracy'});
    ylabel('Score');
    title('Overall Metrics Summary');
    ylim([0 1]);
    grid on;
    
    % Add value labels on bars
    for i = 1:length(summary_metrics)
        text(i, summary_metrics(i)+0.02, sprintf('%.3f', summary_metrics(i)), ...
             'HorizontalAlignment', 'center', 'FontWeight', 'bold');
    end
    
    % Confusion Matrix Heatmap
    subplot(2,3,6);
    imagesc(confusion_mat);
    colorbar;
    xlabel('Predicted Label');
    ylabel('True Label');
    title('Confusion Matrix');
    axis square;
    
    fprintf('\n=== EVALUATION COMPLETE ===\n');
end