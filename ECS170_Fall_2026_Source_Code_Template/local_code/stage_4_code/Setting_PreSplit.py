from local_code.base_class.setting import setting


class Setting_PreSplit(setting):
    def prepare(self, dataset, method, result, evaluate):
        self.dataset = dataset
        self.method = method
        self.result = result
        self.evaluate = evaluate

    def load_run_save_evaluate(self):
        self.method.data = self.dataset.load()
        learned_result = self.method.run()

        self.result.data = learned_result
        self.result.save()

        self.evaluate.data = learned_result
        score = self.evaluate.evaluate()
        return score, 0