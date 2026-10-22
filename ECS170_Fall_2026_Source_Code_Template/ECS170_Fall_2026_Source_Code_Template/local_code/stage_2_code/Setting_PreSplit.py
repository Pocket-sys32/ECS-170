'''
Setting for pre-split train/test data (no cross-validation needed)
'''

from local_code.base_class.setting import setting
import numpy as np

class Setting_PreSplit(setting):
    
    def prepare(self, dataset_train, dataset_test, method, result, evaluate):
        self.dataset_train = dataset_train
        self.dataset_test = dataset_test
        self.method = method
        self.result = result
        self.evaluate = evaluate
    
    def load_run_save_evaluate(self):
        print('Loading training data...')
        train_data = self.dataset_train.load()
        X_train = np.array(train_data['X'])
        y_train = np.array(train_data['y'])
        
        print('Loading test data...')
        test_data = self.dataset_test.load()
        X_test = np.array(test_data['X'])
        y_test = np.array(test_data['y'])
        
        print('============ Training on pre-split data ============')
        
        self.method.data = {'train': {'X': X_train, 'y': y_train}, 'test': {'X': X_test, 'y': y_test}}
        learned_result = self.method.run()
        
        self.result.data = learned_result
        self.result.save()
        
        self.evaluate.data = learned_result
        score = self.evaluate.evaluate()
        
        print('Overall Accuracy:', score)
        return score, 0
