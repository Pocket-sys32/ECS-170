from local_code.base_class.setting import setting
# Format modified to mostly match the Setting_CNN code in other part of this stage

class Setting_CNN_CIFAR(setting):
    def __init__(self, sName=None, sDescription=None):
        super().__init__(sName, sDescription)

    def load_run_save_evaluate(self):
        # Load both training and testing
        loaded_data = self.dataset.load()
        # Run Model training and prediction
        self.method.data = {
            'train': {'X': loaded_data['X_train'], 'y': loaded_data['y_train']},
            'test': {'X': loaded_data['X_test'], 'y': loaded_data['y_test']},
        }

        learned_result = self.method.run()

        # Save predictions array
        self.result.data = learned_result['pred_y']
        self.result.save()

        # Pass dict to evaluate
        self.evaluate.data = learned_result
        eval_result = self.evaluate.evaluate()

        # Get accuracy float from dict before returning: ensures
        # that script_cnn_cifar gets single float
        acc = eval_result['accuracy'] if isinstance(eval_result, dict) else eval_result
        return acc, None