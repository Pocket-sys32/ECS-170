'''
Evaluation of all params (Accuracy, Precision, Recall, F1 Score)
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

import csv
import os
from local_code.base_class.evaluate import evaluate
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


class Evaluate_Accuracy(evaluate):
    data = None
    
    def evaluate(self):
        print('evaluating performance...')
        return accuracy_score(self.data['true_y'], self.data['pred_y'])

class Evaluate_Precision(evaluate):
    data = None

    def evaluate(self):
        print('evaluating performance...')
        return precision_score(self.data['true_y'], self.data['pred_y'], average='weighted')

class Evaluate_Recall(evaluate):
    data = None

    def evaluate(self):
        print('evaluating performance...')
        return recall_score(self.data['true_y'], self.data['pred_y'], average='weighted')

class Evaluate_F1(evaluate):
    data = None

    def evaluate(self):
        print('evaluating performance...')
        return f1_score(self.data['true_y'], self.data['pred_y'], average='weighted')


class Evaluate_Metrics(evaluate):
    data = None
    log_path = '../../result/stage_2_result/metrics_log.csv'
    run_label = ''

    def evaluate(self):
        print('=== Evaluating Performance ===')
        acc = accuracy_score(self.data['true_y'], self.data['pred_y'])
        prec = precision_score(self.data['true_y'], self.data['pred_y'], average='weighted')
        rec = recall_score(self.data['true_y'], self.data['pred_y'], average='weighted')
        f1 = f1_score(self.data['true_y'], self.data['pred_y'], average='weighted')

        print(f'Accuracy:  {acc:.4f}')
        print(f'Precision: {prec:.4f}')
        print(f'Recall:    {rec:.4f}')
        print(f'F1 Score:  {f1:.4f}')

        # save one row per run so results are recorded
        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
        new_file = not os.path.exists(self.log_path)
        with open(self.log_path, 'a', newline='') as f:
            writer = csv.writer(f)
            if new_file:
                writer.writerow(['run_label', 'accuracy', 'precision', 'recall', 'f1'])
            writer.writerow([self.run_label, acc, prec, rec, f1])

        return acc