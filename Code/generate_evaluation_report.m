function generate_evaluation_report(metrics, filename)
    % GENERATE TEXT REPORT OF ALL EVALUATION METRICS
    
    fprintf('Generating comprehensive evaluation report...\n');
    
    if nargin < 2
        filename = 'evaluation_report.txt';
    end
    
    % Open file for writing
    fileID = fopen(filename, 'w');
    
    fprintf(fileID, '=== GAIT AUTHENTICATION SYSTEM EVALUATION REPORT ===\n\n');
    
    % Basic Information
    fprintf(fileID, 'MODEL PERFORMANCE SUMMARY:\n');
    fprintf(fileID, '==========================\n');
    fprintf(fileID, 'Overall Accuracy: %.2f%%\n', metrics.accuracy);
    fprintf(fileID, 'Macro Precision: %.3f\n', metrics.macro_precision);
    fprintf(fileID, 'Macro Recall: %.3f\n', metrics.macro_recall);
    fprintf(fileID, 'Macro F1-Score: %.3f\n', metrics.macro_f1);
    fprintf(fileID, 'Macro AUC: %.3f\n\n', metrics.macro_auc);
    
    % Per-class metrics
    fprintf(fileID, 'PER-CLASS METRICS:\n');
    fprintf(fileID, '==================\n');
    fprintf(fileID, 'User\tPrecision\tRecall\t\tF1-Score\tAUC\n');
    fprintf(fileID, '----\t---------\t------\t\t--------\t---\n');
    
    for i = 1:length(metrics.precision_per_class)
        fprintf(fileID, '%d\t%.3f\t\t%.3f\t\t%.3f\t\t%.3f\n', ...
                i, metrics.precision_per_class(i), metrics.recall_per_class(i), ...
                metrics.f1_per_class(i), metrics.auc_per_class(i));
    end
    
    % Performance Interpretation
    fprintf(fileID, '\nPERFORMANCE INTERPRETATION:\n');
    fprintf(fileID, '===========================\n');
    
    if metrics.accuracy >= 95
        fprintf(fileID, '✓ EXCELLENT performance (Accuracy >= 95%%)\n');
    elseif metrics.accuracy >= 90
        fprintf(fileID, '✓ VERY GOOD performance (Accuracy >= 90%%)\n');
    elseif metrics.accuracy >= 85
        fprintf(fileID, '✓ GOOD performance (Accuracy >= 85%%)\n');
    else
        fprintf(fileID, '✓ MODERATE performance (Accuracy < 85%%)\n');
    end
    
    if metrics.macro_auc >= 0.9
        fprintf(fileID, '✓ EXCELLENT discrimination (AUC >= 0.9)\n');
    elseif metrics.macro_auc >= 0.8
        fprintf(fileID, '✓ GOOD discrimination (AUC >= 0.8)\n');
    else
        fprintf(fileID, '✓ MODERATE discrimination (AUC < 0.8)\n');
    end
    
    if metrics.macro_f1 >= 0.9
        fprintf(fileID, '✓ EXCELLENT balance between precision and recall (F1 >= 0.9)\n');
    elseif metrics.macro_f1 >= 0.8
        fprintf(fileID, '✓ GOOD balance between precision and recall (F1 >= 0.8)\n');
    else
        fprintf(fileID, '✓ MODERATE balance between precision and recall (F1 < 0.8)\n');
    end
    
    % Recommendations
    fprintf(fileID, '\nRECOMMENDATIONS:\n');
    fprintf(fileID, '================\n');
    
    low_perf_users = find(metrics.f1_per_class < 0.8);
    if ~isempty(low_perf_users)
        fprintf(fileID, 'Consider collecting more data for users: ');
        fprintf(fileID, '%d ', low_perf_users);
        fprintf(fileID, '\n');
    end
    
    if min(metrics.auc_per_class) < 0.7
        fprintf(fileID, 'Some users show poor discrimination - review feature extraction\n');
    end
    
    fclose(fileID);
    fprintf('Evaluation report saved to: %s\n', filename);
end