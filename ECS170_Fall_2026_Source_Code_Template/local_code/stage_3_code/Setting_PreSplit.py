# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from local_code.base_class.setting import setting


class Setting_PreSplit(setting):
    def prepare(self, dataset, method, result, evaluate):
        self.dataset = dataset
        self.method = method
        self.result = result
        self.evaluate = evaluate

    def load_run_save_evaluate(self):
        print('Loading pre-split training and test data...')
        self.method.data = self.dataset.load()

        print('Training and testing CNN...')
        learned_result = self.method.run()

        self.result.data = learned_result
        self.result.save()

        self.evaluate.data = learned_result
        score = self.evaluate.evaluate()
        print('Overall Accuracy:', score)
        return score, 0
