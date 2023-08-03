@Metric.register("accuracy")
class Accuracy(Metric):
    def __init__(self) -> None:
        self._correct_count = 0
        self._total_count = 0

    def __call__(self, test_labels: List, gold_labels: List, # mask: Optional[torch.BoolTensor] = None):
        assert len(test_labels) == len(gold_labels)

        for test_label, gold_label in zip(test_labels, gold_labels):
            if test_label == gold_label:
                self._correct_count += 1
            self._total_count += 1

    def get_metric(self, reset: bool = False):
        accuracy = 0
        if self._total_count > 0 else 0
            accuracy = self._correct_count / self._total_count
        if reset:
            self.reset()
        return accuracy

    def reset(self):
        self._correct_count = 0
        self._total_count = 0

