import torch
from torchmetrics import Metric


class WordAccuracy(Metric):
    def __init__(self, ignore_index=None, **kwargs):
        """
        Custom metric to calculate word-level accuracy, ignoring a specified index (e.g., padding).

        Args:
            ignore_index (int, optional): Index to ignore when calculating accuracy (e.g., padding index). Defaults to None.
            **kwargs: Additional keyword arguments to be passed to the parent class (Metric).
        """
        super().__init__(**kwargs)
        self.ignore_index = ignore_index
        self.add_state("correct", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        """
        Updates the metric state based on the predictions and target.

        Args:
            preds (torch.Tensor): Predicted sequences (token indices).
                                 Shape: (batch_size, sequence_length).
            target (torch.Tensor): Ground truth target sequences (token indices).
                                  Shape: (batch_size, sequence_length).
        """
        if preds.shape != target.shape:
            raise ValueError("preds and target must have the same shape")

        if self.ignore_index:
            for i in range(preds.shape[0]):
                matched = True
                for j in range(preds.shape[1]):
                    if target[i][j] == self.ignore_index:
                        continue

                    if target[i][j] != preds[i][j]:
                        matched = False
                        break

                if matched:
                    self.correct += torch.tensor(1)

        self.total += len(target)

    def compute(self) -> torch.Tensor:
        """
        Computes the word accuracy based on accumulated `correct` and `total` states.
        """
        return self.correct.float() / self.total.float()
