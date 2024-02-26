import torch
import torch.nn.functional as F

from ignite.metrics import Metric
# These decorators helps with distributed settings
from ignite.exceptions import NotComputableError
from ignite.metrics.metric import sync_all_reduce, reinit__is_reduced


class ECE(Metric):
    """calculates the expected calibration error
    """
    def __init__(
        self,
        output_transform = lambda x: x,
        batch_size = len,
        n_bins: int = 10,
        device = torch.device("cuda:0"),
    ):
        super(ECE, self).__init__(output_transform, device=device)
        self._batch_size = batch_size
        self._n_bins = n_bins

    def _bin_predictions(self, y_hat, y, n_bins: int = 10):
        """bins predictions based on predicted class probilities

        Args:
            y_hat (Prediction): predicted class probabilities
            y (Tensor): ground-truth labels
            n_bins (int, optional): number of bins used in the calculation. Defaults to 10.

        Returns:
            Tuple[Tensor, Tensor, Tensor]: tuple of binned accuracy values, confidence values and cardinalities of each bin
        """
        y_hat = F.softmax(y_hat, dim=-1)
        y_hat_label = y_hat.argmax(dim=1)
        y_hat = y_hat.max(-1)[0]
        corrects = (y_hat_label == y.squeeze())

        acc_binned = torch.zeros((n_bins, ), device=y_hat.device)
        conf_binned = torch.zeros((n_bins, ), device=y_hat.device)
        bin_cardinalities = torch.zeros((n_bins, ), device=y_hat.device)

        bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        lower_bin_boundary = bin_boundaries[:-1]
        upper_bin_boundary = bin_boundaries[1:]

        for b in range(n_bins):
            in_bin = (y_hat <= upper_bin_boundary[b]) & (y_hat > lower_bin_boundary[b])
            bin_cardinality = in_bin.sum()
            bin_cardinalities[b] = bin_cardinality

            if bin_cardinality > 0:
                acc_binned[b] = corrects[in_bin].float().mean()
                conf_binned[b] = y_hat[in_bin].mean()

        return acc_binned, conf_binned, bin_cardinalities

    @reinit__is_reduced
    def reset(self):
        self._sum = torch.tensor(0.0, device=self._device)
        self._num_examples = 0

    @reinit__is_reduced
    def update(self, output):
        y_pred, y = output[0].detach(), output[1].detach()

        n = self._batch_size(y)

        acc_binned, conf_binned, bin_cardinalities = self._bin_predictions(y_pred, y, self._n_bins)
        ece = torch.abs(acc_binned - conf_binned) * bin_cardinalities

        self._sum += ece.sum() * 1
        self._num_examples += n

    @sync_all_reduce("_num_examples", "_num_correct:SUM")
    def compute(self):
        if self._num_examples == 0:
            raise NotComputableError('CustomAccuracy must have at least one example before it can be computed.')
        return self._sum.item() / self._num_examples
