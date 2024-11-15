
from deepxde import callbacks
# from deepxde.backend import backend_name
# from deepxde import utils
import csv


class VariableSaver(callbacks.Callback):
    """Monitor and save the learning history of external trainable variables."""

    def __init__(self, var_dict, scale_dict, period=1, filename=None):
        super().__init__()
        self.var_dict = var_dict
        self.scale_dict = scale_dict
        self.period = period
        self.filename = filename

        self.value_dict = {}
        self.value_history = []
        self.epochs_since_last = 0

        csvfile = open(self.filename, "a", newline="")
        writer = csv.writer(csvfile)
        writer.writerow(["epoch"] + list(var_dict.keys()))
        csvfile.close()

    def on_train_begin(self):
        for k, v in self.var_dict.items():
            self.value_dict[k] = v.detach().item() / self.scale_dict[k]

        row = [self.model.train_state.epoch] + list(self.value_dict.values())
        self.value_history.append(row)

        csvfile = open(self.filename, "a", newline="")
        writer = csv.writer(csvfile)
        writer.writerow(row)
        csvfile.close()

    def on_epoch_end(self):
        self.epochs_since_last += 1
        if self.epochs_since_last >= self.period:
            self.epochs_since_last = 0
            self.on_train_begin()

    def on_train_end(self):
        if not self.epochs_since_last == 0:
            self.on_train_begin()
