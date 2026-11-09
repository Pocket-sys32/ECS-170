from local_code.base_class.setting import setting


class Setting_ORL(setting):

    def load_run_save_evaluate(self):
        print('Loading data...')
        loaded_data = self.dataset.load()
        X_train = loaded_data['train']['X']
        y_train = loaded_data['train']['y']
        X_test = loaded_data['test']['X']
        y_test = loaded_data['test']['y']

        print('============ Training on pre-split data ============')

        self.method.data = {'train': {'X': X_train, 'y': y_train}, 'test': {'X': X_test, 'y': y_test}}
        learned_result = self.method.run()

        self.result.data = learned_result
        self.result.save()

        self.evaluate.data = learned_result
        score = self.evaluate.evaluate()

        print('Overall Accuracy:', score)
        return score, 0