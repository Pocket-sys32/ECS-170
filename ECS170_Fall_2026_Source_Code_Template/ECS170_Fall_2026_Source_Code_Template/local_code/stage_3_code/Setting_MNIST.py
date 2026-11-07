'''
Setting for MNIST: the loader returns train and test together
'''

from local_code.base_class.setting import setting


class Setting_MNIST(setting):

    def load_run_save_evaluate(self):
        # load both splits at once
        loaded_data = self.dataset.load()

        # run the model
        self.method.data = loaded_data
        learned_result = self.method.run()

        # save raw predictions
        self.result.data = learned_result
        self.result.save()

        # evaluate
        self.evaluate.data = learned_result
        return self.evaluate.evaluate(), None