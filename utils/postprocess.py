import deepxde as dde
import numpy as np
import torch
import os
from utils import metric_funcs

import matplotlib
matplotlib.use("Agg")  # do not show figures
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D

set_fs = 22
set_dpi = 200
plt.rcParams["font.sans-serif"] = "Arial"  # default font
# plt.rcParams["font.sans-serif"] = "Nimbus Sans"  # default font
plt.rcParams["font.size"] = set_fs  # default font size
# plt.rcParams["mathtext.fontset"] = "stix"  # default font of math text


class PostProcess:
    """Base class for post-processing.

    Args:
        args: args of the main function.
        case: case file variable.
        model: deepxde model.
        output_dir: output directory.
    """

    def __init__(self, args, case, model, output_dir):
        self.args = args
        self.case = case
        self.model = model
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        # The following 5 variables should be modified by inheritance, and they must have the same length.
        # Each item of preds and refes is ndarray, and their shape should match the problem dimension.
        # For example, (n_x, ) for 1D problems, (n_x, n_y) for 2D, (n_x, n_y, n_t) for 2Dt, etc.
        self.preds = []  # predicted field variables of interest
        self.refes = []  # corresponding reference field variables of interest
        self.mathnames = []  # the field variable names in math format, using for figure titles, for example
        self.textnames = []  # the field variable names in plain text format
        self.units = []  # the units of the field variables

        # for inferred parameters
        self.para_infes = []
        self.para_refes = []
        self.para_mathnames = []
        self.para_textnames = []
        self.para_units = []

    def _save_data(self, save_refe=True, suffix=""):
        """Save the predicted and reference fields."""
        print("Saving data...")
        output_dir = self.output_dir
        os.makedirs(output_dir + "data/", exist_ok=True)
        for i in range(len(self.preds)):
            np.save(output_dir + f"data/{self.textnames[i]}_{suffix}_predicted.npy", self.preds[i])
            if save_refe:
                np.save(output_dir + f"data/{self.textnames[i]}_{suffix}_reference.npy", self.refes[i])

    def _save_metrics(self, refes, preds, output_dir):
        """Save the evaluation metrics of given predicted results w.r.t. reference results."""
        refes_ = [field.ravel() for field in refes]
        preds_ = [field.ravel() for field in preds]

        formats = {"l2 relative error": "{:.4%}",
                   "l1 relative error": "{:.4%}",
                   "MSE": "{:.2e}",
                   "RMSE": "{:.2e}",
                   "MAE": "{:.2e}",
                   "MaxE": "{:.2e}",
                   "MAPE": "{:.4%}",
                   "R2": "{:.4f}",
                   "mean absolute of refe": "{:.2e}",
                   "mean absolute of pred": "{:.2e}",
                   "min absolute of refe": "{:.2e}",
                   "min absolute of pred": "{:.2e}",
                   "max absolute of refe": "{:.2e}",
                   "max absolute of pred": "{:.2e}",
                   }

        metrics = {}
        for key in formats.keys():
            metrics[key] = []
            # metric_func = dde.metrics.get(key)
            metric_func = metric_funcs.get(key)
            for i in range(len(self.preds)):
                metric = metric_func(refes_[i], preds_[i])
                metrics[key].append(formats[key].format(metric))

        file = open(output_dir + "metrics.txt", "a")
        # file = open(output_dir + "metrics.txt", "w")
        file.write("\n")
        file.write("field variables:   ")
        file.write(", ".join(self.textnames))
        file.write("\n")
        for key in metrics.keys():
            file.write(key + ":   ")
            file.write(", ".join(metrics[key]))
            file.write("\n")
        file.write("\n")
        file.close()

        return metrics

    def save_metrics(self):
        """Save the evaluation metrics of predicted results w.r.t. reference results."""
        print("Saving metrics...")
        self._save_metrics(self.refes, self.preds, self.output_dir)

    def save_para_metrics(self):
        """Save the evaluation metrics of inferred parameters (i.e. external trainable variables)."""
        print("Saving the metrics of inferred parameters...")
        n_para = len(self.para_refes)
        output_dir = self.output_dir
        file = open(output_dir + "metrics.txt", "a")
        file.write("\n")
        file.write("parameter:   ")
        file.write(", ".join(self.para_textnames))
        file.write("\n")

        file.write("reference:   ")
        file.write(", ".join(["{:.4e}".format(self.para_refes[i]) for i in range(n_para)]))
        file.write("\n")

        file.write("inferred:   ")
        file.write(", ".join(["{:.4e}".format(self.para_infes[i]) for i in range(n_para)]))
        file.write("\n")

        file.write("relative error:   ")
        file.write(", ".join(["{:.4%}".format(self.para_infes[i] / self.para_refes[i] - 1) for i in range(n_para)]))
        file.write("\n")
        file.close()

    @staticmethod
    def _plot_save_loss_history(model, names, output_dir, save_name):
        """Plot the loss history and save the history data."""
        print("Plotting and saving loss history...")
        os.makedirs(output_dir + "pics/", exist_ok=True)

        loss_history = model.losshistory
        loss_train = np.array(loss_history.loss_train)
        loss_names = names["equations"] + names["ICBCOCs"]

        s_de = slice(0, len(names["equations"]), 1)
        s_bc = slice(len(names["equations"]), len(loss_names), 1)
        ss = [s_de, s_bc]

        # plot in two figures
        fig, axes = plt.subplots(1, 2, sharey="all", figsize=(15, 6))
        for i in range(2):
            axes[i].set_xlabel("Epoch")
            axes[i].set_yscale("log")
            axes[i].tick_params(axis="y", labelleft=True)
            axes[i].plot(loss_history.steps, loss_train[:, ss[i]])
            axes[i].legend(loss_names[ss[i]], fontsize="small")
        plt.savefig(output_dir + f"pics/{save_name}_2figs.png", bbox_inches="tight", dpi=set_dpi)
        plt.close(fig)

        # plot in one figure
        plt.figure(figsize=(8, 6))
        plt.xlabel("Epoch")
        plt.yscale("log")
        plt.plot(loss_history.steps, loss_train, lw=2)
        plt.legend(loss_names, fontsize="small")
        plt.savefig(output_dir + f"pics/{save_name}_1fig.png", bbox_inches="tight", dpi=set_dpi)
        plt.close(fig)

        # save the loss history
        loss_save = np.hstack([
            np.array(loss_history.steps)[:, None],
            np.array(loss_history.loss_train),
        ])
        np.savetxt(output_dir + f"{save_name}.csv", loss_save, fmt="%.2e", delimiter=",",
                   header=",".join(["epoch"] + loss_names), comments="")

    def plot_save_loss_history(self):
        self._plot_save_loss_history(self.model, self.case.names, self.output_dir, "losses_history")

    def plot_para_history(self, var_saver):
        """Plot the learning history of inferred parameters (i.e. external trainable variables)."""
        print("Plotting the learning history of inferred parameters...")
        output_dir = self.output_dir
        os.makedirs(output_dir + "data/", exist_ok=True)
        os.makedirs(output_dir + "pics/", exist_ok=True)

        para_history = np.array(var_saver.value_history)
        epochs = para_history[:, 0]
        para_history = para_history[:, 1:]

        for i in range(para_history.shape[1]):
            plt.figure(figsize=(8, 6))
            plt.title(self.para_mathnames[i], fontsize="medium")
            plt.xlabel("Epoch")
            plt.ylabel(self.para_units[i])
            plt.plot(epochs, np.ones(len(epochs)) * self.para_refes[i], c="k", ls="--", lw=3, label="Reference")
            plt.plot(epochs, para_history[:, i], lw=2, label="Inferred")
            plt.legend(fontsize="small")
            plt.savefig(output_dir + f"pics/parameter{i + 1}_{self.para_textnames[i]}.png", bbox_inches="tight", dpi=set_dpi)
            plt.close()

    def delete_old_models(self):
        """Delete the old models produced during training"""
        print("Deleting old models...")
        output_dir = self.output_dir
        model_list = os.listdir(output_dir + "models/")
        model_list_better = [s for s in model_list if "better" in s]
        better_epochs = [int(s.split("-")[1][:-3]) for s in model_list_better]
        best_epoch_index = better_epochs.index(max(better_epochs))
        for filename in model_list_better:
            if filename != model_list_better[best_epoch_index]:
                os.remove(output_dir + "models/" + filename)

    def plot_lossgrads_fnn(self):
        """Plot the distribution of the losses gradients w.r.t. the NN parameters. Only valid for FNNs."""
        print("Plotting loss gradients...")
        # TODO


class PostProcess1D(PostProcess):
    """Post-processing for 1D problems (1D space or 1D time)."""

    def __init__(self, args, case, model, output_dir):
        super().__init__(args, case, model, output_dir)
        self.n_x = 5001
        self.x_l, self.x_r = case.x_l, case.x_r
        self.x = np.linspace(self.x_l, self.x_r, self.n_x)  # can be modified to t_l if the only dimension is t
        self.x_name = "$x$"  # can be modified by inheritance, to $t$" for example
        self.x_unit = "m"  # can be modified by inheritance, to "s" for example

    def save_data(self, save_refe=True):
        """Save the predicted and reference fields."""
        self._save_data(save_refe=save_refe, suffix="1D")

    def plot_1dfields(self, lw=2, format="png", extra_plot=None):
        """Plot the curves of predicted 1D fields."""
        print("Plotting 1D fields...")
        # TODO

    def plot_1dfields_comp(self, lws=(1, 2.5), label_refe="Reference", label_pred="PINN",
                           fsize_legend=set_fs - 6, format="png", extra_plot=None):
        """Plot the curves of predicted v.s. reference 1D fields."""
        print("Plotting 1D fields (predicted vs reference)...")
        # TODO


class PostProcess1Dt(PostProcess):
    """Post-processing for 1D unsteady problems, i.e. the independent variables are x and t."""

    def __init__(self, args, case, model, output_dir):
        super().__init__(args, case, model, output_dir)
        self.n_x, self.n_t = 501, 501
        self.x_l, self.x_r = case.x_l, case.x_r
        self.t_l, self.t_r = case.t_l, case.t_r
        self.x = np.linspace(self.x_l, self.x_r, self.n_x)
        self.t = np.linspace(self.t_l, self.t_r, self.n_t)
        self.xx, self.tt = np.meshgrid(self.x, self.t, indexing="ij")
        self.x_name, self.t_name = "$x$", "$t$"
        self.x_unit, self.t_unit = "m", "s"

    def save_data(self, save_refe=True):
        """Save the predicted and reference fields."""
        self._save_data(save_refe=save_refe, suffix="1Dt")

    def plot_sampling_points(self, figsize=(8, 6), format="png"):
        """Plot the sampling points, including PDE points and IC/BC/OC points."""
        print("Plotting sampling points...")
        # TODO

    def plot_2dfields(self, figsize=(8, 6), cmap="jet", format="png", extra_plot=None):
        """Plot the contours of predicted 2D fields, and streamline for flow problems."""
        print("Plotting 2D fields...")
        # TODO

    def plot_2dfields_comp(self, figsize=(13, 6), is_vertical=False, label_refe="Reference", label_pred="Predicted",
                           cmap="jet", adjust=(0.1, 0.8, None, None, None, None), format="png", extra_plot=None):
        """Plot the contours of predicted v.s. reference 2D fields."""
        print("Plotting 2D fields (predicted vs reference)...")
        # TODO

    def plot_1dcurves(self, select_t=(0., 1.,), lws=(1, 2.5),
                      label_refe="reference", label_pred="PINN",
                      n_col=2, fsize_legend=set_fs - 6, save_data=False, format="png"):
        """Plot the 1D curves at some moments of predicted v.s. reference 2D fields."""
        print("Plotting 1D curves...")
        # TODO


class PostProcess2D(PostProcess):
    """Post-processing for 2D steady problems, i.e. the independent variables are x and y."""

    def __init__(self, args, case, model, output_dir):
        super().__init__(args, case, model, output_dir)
        self.n_x, self.n_y = 501, 501
        self.x_l, self.x_r = case.x_l, case.x_r
        self.y_l, self.y_r = case.y_l, case.y_r
        self.x = np.linspace(self.x_l, self.x_r, self.n_x)
        self.y = np.linspace(self.y_l, self.y_r, self.n_y)
        self.xx, self.yy = np.meshgrid(self.x, self.y, indexing="ij")
        self.x_name, self.y_name = "$x$", "$y$"
        self.x_unit, self.y_unit = "m", "m"

    def save_data(self, save_refe=True):
        """Save the predicted and reference fields."""
        self._save_data(save_refe=save_refe, suffix="2D")

    def plot_sampling_points(self, figsize=(8, 6), format="png"):
        """Plot the sampling points, including PDE points and IC/BC/OC points."""
        print("Plotting sampling points...")
        model = self.model
        output_dir = self.output_dir
        os.makedirs(output_dir + "pics/", exist_ok=True)
        x_l, x_r, y_l, y_r = self.x_l, self.x_r, self.y_l, self.y_r

        plt.figure(figsize=figsize)
        plt.axis("scaled")
        plt.axis((x_l - 0.05 * (x_r - x_l), x_r + 0.05 * (x_r - x_l),
                  y_l - 0.05 * (y_r - y_l), y_r + 0.05 * (y_r - y_l)))
        plt.xlabel(f"{self.x_name}/{self.x_unit}")
        plt.ylabel(f"{self.y_name}/{self.y_unit}")
        plt.scatter(model.data.train_x_all[:, 0],
                    model.data.train_x_all[:, 1], s=0.2, lw=0.2)
        plt.scatter(model.data.train_x_bc[:, 0],
                    model.data.train_x_bc[:, 1], s=5, marker="x", lw=0.5)
        # plt.scatter(data.anchors[:, 0], data.anchors[:, 1], s=0.5)
        plt.savefig(output_dir + f"pics/sampling_points.{format}", bbox_inches="tight", dpi=500)  # .svg
        plt.close()

    def plot_2dfields(self, figsize=(8, 6), cmap="jet", format="png", extra_plot=None):
        """Plot the contours of predicted 2D fields, and streamline for flow problems."""
        print("Plotting 2D fields...")
        output_dir = self.output_dir
        os.makedirs(output_dir + "pics/", exist_ok=True)
        x_l, x_r, y_l, y_r = self.x_l, self.x_r, self.y_l, self.y_r
        xx, yy = self.xx, self.yy
        preds = self.preds
        mnames, tnames, units = self.mathnames, self.textnames, self.units

        n_level = 50
        for i in range(len(preds)):
            plt.figure(figsize=figsize)
            plt.title(mnames[i], fontsize="medium")
            plt.axis("scaled")
            plt.axis((x_l, x_r, y_l, y_r))
            plt.xlabel(f"{self.x_name}/{self.x_unit}")
            plt.ylabel(f"{self.y_name}/{self.y_unit}")
            if tnames[i] == "psi":
                plt.contour(xx, yy, preds[i], colors="k", levels=n_level, linewidths=0.8, linestyles="solid")
            else:
                plt.contourf(xx, yy, preds[i], cmap=cmap, levels=n_level)
                # cb = plt.colorbar(format="%.1e", pad=0.1)
                cb = plt.colorbar(format="%.1e")
                cb.ax.set_title(units[i], fontsize="medium")
            if extra_plot is not None:
                extra_plot()
            plt.savefig(output_dir + f"pics/contour{i + 1}_{tnames[i]}.{format}", bbox_inches="tight", dpi=set_dpi)
            plt.close()

    def plot_2dfields_comp(self, figsize=(13, 6), is_vertical=False, label_refe="Reference", label_pred="Predicted",
                           cmap="jet", adjust=(0.1, 0.8, None, None, None, None), format="png", extra_plot=None):
        """Plot the contours of predicted v.s. reference 2D fields."""
        print("Plotting 2D fields (predicted vs reference)...")
        output_dir = self.output_dir
        os.makedirs(output_dir + "pics/", exist_ok=True)
        x_l, x_r, y_l, y_r = self.x_l, self.x_r, self.y_l, self.y_r
        xx, yy = self.xx, self.yy
        preds = self.preds
        refes = self.refes
        mnames, tnames, units = self.mathnames, self.textnames, self.units

        n_row, n_col = (2, 1) if is_vertical else (1, 2)
        n_level = 50
        for i in range(len(preds)):
            fig, axes = plt.subplots(n_row, n_col, figsize=figsize)
            f_min = min(preds[i].min(), refes[i].min())
            f_max = max(preds[i].max(), refes[i].max())
            norm = matplotlib.colors.Normalize(vmin=f_min, vmax=f_max)
            subfigs = []
            for j in range(2):
                axes[j].set_title(f"{[label_refe, label_pred][j]} {mnames[i]}", fontsize="medium")
                axes[j].axis("scaled")
                axes[j].axis((x_l, x_r, y_l, y_r))
                if tnames[i] == "psi":
                    subfig = axes[j].contour(xx, yy, [refes, preds][j][i],
                                             colors="k", levels=n_level, linewidths=0.8, linestyles="solid")
                else:
                    subfig = axes[j].contourf(xx, yy, [refes, preds][j][i], cmap=cmap, levels=n_level, norm=norm)
                subfigs.append(subfig)
                if extra_plot is not None:
                    extra_plot(axes[j])
            if is_vertical:
                axes[0].set_xticklabels([])
                axes[1].set_xlabel(f"{self.x_name}/{self.x_unit}")
                axes[0].set_ylabel(f"{self.y_name}/{self.y_unit}")
                axes[1].set_ylabel(f"{self.y_name}/{self.y_unit}")
            else:
                axes[0].set_xlabel(f"{self.x_name}/{self.x_unit}")
                axes[1].set_xlabel(f"{self.x_name}/{self.x_unit}")
                axes[0].set_ylabel(f"{self.y_name}/{self.y_unit}")
                axes[1].set_yticklabels([])

            fig.subplots_adjust(left=adjust[0], right=adjust[1], bottom=adjust[2], top=adjust[3],
                                wspace=adjust[4], hspace=adjust[5])
            if tnames[i] != "psi":
                cb_ax = fig.add_axes([0.83, 0.15, 0.02, 0.7])  # l, b, w, h
                cb = fig.colorbar(subfigs[0], cax=cb_ax, format="%.1e")
                # cb.set_ticks(np.linspace(f_min, f_max, 8))
                cb.ax.set_title(units[i], fontsize="medium")
            plt.savefig(output_dir + f"pics/contourComp{i + 1}_{tnames[i]}.{format}", bbox_inches="tight", dpi=set_dpi)
            plt.close(fig)

    def plot_1dcurves(self, select_x=(0., 1.,), select_y=(0., 1.,), lws=(1, 2.5),
                      label_refe="reference", label_pred="PINN",
                      n_col=2, fsize_legend=set_fs - 6, save_data=False, format="png"):
        """Plot the 1D curves at some lines of predicted v.s. reference 2D fields."""
        print("Plotting 1D curves...")
        output_dir = self.output_dir
        os.makedirs(output_dir + "pics/", exist_ok=True)
        if save_data:
            os.makedirs(output_dir + "data/", exist_ok=True)
        x, y = self.x, self.y
        preds = self.preds
        refes = self.refes
        mnames, tnames, units = self.mathnames, self.textnames, self.units

        # field variables vs x at different y
        label_order = [2 * k for k in range(len(select_y))] + [2 * k + 1 for k in range(len(select_y))]
        for i in range(len(preds)):
            plt.figure(figsize=(8, 6))
            plt.title(mnames[i], fontsize="medium")
            plt.xlabel(f"{self.x_name}/{self.x_unit}")
            plt.ylabel(units[i])
            for j in range(len(select_y)):
                s = "{:.2f}".format(select_y[j])
                loc = np.sum(select_y[j] >= y) - 1
                loc = loc - 1 if loc == len(x) - 1 else loc
                d = (select_y[j] - y[loc]) / (y[loc + 1] - y[loc])
                f_refe = (1 - d) * refes[i][:, loc] + d * refes[i][:, loc + 1]
                f_pred = (1 - d) * preds[i][:, loc] + d * preds[i][:, loc + 1]
                plt.plot(x, f_refe, ls="-", c=f"C{j}", lw=lws[0],
                         label=f"{self.y_name} = {s}{self.y_unit}, {label_refe}")
                plt.plot(x, f_pred, ls="--", c=f"C{j}", lw=lws[1],
                         label=f"{self.y_name} = {s}{self.y_unit}, {label_pred}")
                if save_data:
                    np.save(output_dir + f"data/curve_{tnames[i]}-x_y{s}_reference.npy", f_refe)
                    np.save(output_dir + f"data/curve_{tnames[i]}-x_y{s}_predicted.npy", f_pred)
            handles, labels = plt.gca().get_legend_handles_labels()
            plt.legend([handles[idx] for idx in label_order], [labels[idx] for idx in label_order],
                       ncol=n_col, labelspacing=0.3, columnspacing=0.5, fontsize=fsize_legend)
            plt.savefig(output_dir + f"pics/curve{i + 1}_{tnames[i]}-x.{format}", bbox_inches="tight", dpi=set_dpi)
            plt.close()

        # field variables vs y at different x
        label_order = [2 * k for k in range(len(select_x))] + [2 * k + 1 for k in range(len(select_x))]
        for i in range(len(preds)):
            plt.figure(figsize=(8, 6))
            plt.title(mnames[i], fontsize="medium")
            plt.xlabel(f"{self.y_name}/{self.y_unit}")
            plt.ylabel(units[i])
            for j in range(len(select_x)):
                s = "{:.2f}".format(select_x[j])
                loc = np.sum(select_x[j] >= x) - 1
                loc = loc - 1 if loc == len(x) - 1 else loc
                d = (select_x[j] - x[loc]) / (x[loc + 1] - x[loc])
                f_refe = (1 - d) * refes[i][loc, :] + d * refes[i][loc + 1, :]
                f_pred = (1 - d) * preds[i][loc, :] + d * preds[i][loc + 1, :]
                plt.plot(y, f_refe, ls="-", c=f"C{j}", lw=lws[0],
                         label=f"{self.x_name} = {s}{self.x_unit}, {label_refe}")
                plt.plot(y, f_pred, ls="--", c=f"C{j}", lw=lws[1],
                         label=f"{self.x_name} = {s}{self.x_unit}, {label_pred}")
                if save_data:
                    np.save(output_dir + f"data/curve_{tnames[i]}-y_x{s}_reference.npy", f_refe)
                    np.save(output_dir + f"data/curve_{tnames[i]}-y_x{s}_predicted.npy", f_pred)
            handles, labels = plt.gca().get_legend_handles_labels()
            plt.legend([handles[idx] for idx in label_order], [labels[idx] for idx in label_order],
                       ncol=n_col, labelspacing=0.3, columnspacing=0.5, fontsize=fsize_legend)
            plt.savefig(output_dir + f"pics/curve{i + 1}_{tnames[i]}-y.{format}", bbox_inches="tight", dpi=set_dpi)
            plt.close()

        if save_data:
            np.save(output_dir + f"data/curve_x.npy", x)
            np.save(output_dir + f"data/curve_y.npy", y)

    @staticmethod
    def stream_function(uu, vv, dx, dy, psi0=0):
        psipsi = np.ones_like(uu) * psi0
        # u_x0_mid = 0.5 * (uu[0, 1:] + uu[0, :-1])
        # psipsi[0, 1:] = psi0 + np.cumsum(u_x0_mid * dy)
        # v_mid = 0.5 * (vv[1:, :] + vv[:-1, :])
        # psipsi[1:, :] = psipsi[0:1, :] + np.cumsum(-v_mid * dx, axis=0)
        v_y0_mid = 0.5 * (vv[1:, 0] + vv[:-1, 0])
        psipsi[1:, 0] = psi0 + np.cumsum(-v_y0_mid * dx, axis=0)
        u_mid = 0.5 * (uu[:, 1:] + uu[:, :-1])
        psipsi[:, 1:] = psipsi[:, 0:1] + np.cumsum(u_mid * dy, axis=1)
        return psipsi


class PostProcess2Dt(PostProcess):
    """Post-processing for 2D unsteady problems, i.e. the independent variables are x, y, and t."""

    def __init__(self, args, case, model, output_dir):
        super().__init__(args, case, model, output_dir)
        # self.n_x, self.n_y, self.n_t = 65, 65, 101
        self.n_x, self.n_y, self.n_t = 129, 129, 26
        # self.n_x, self.n_y, self.n_t = 257, 257, 11
        self.x_l, self.x_r = case.x_l, case.x_r
        self.y_l, self.y_r = case.y_l, case.y_r
        self.t_l, self.t_r = case.t_l, case.t_r
        self.x = np.linspace(self.x_l, self.x_r, self.n_x)
        self.y = np.linspace(self.y_l, self.y_r, self.n_y)
        self.t = np.linspace(self.t_l, self.t_r, self.n_t)
        self.xx, self.yy = np.meshgrid(self.x, self.y, indexing="ij")
        self.x_name, self.y_name = "$x$", "$y$"
        self.x_unit, self.y_unit = "m", "m"

    def save_data(self, save_refe=True):
        """Save the predicted and reference fields."""
        self._save_data(save_refe=save_refe, suffix="2Dt")

    def save_2dmetrics(self, n_moments=6):
        print("Saving 2D metrics...")
        output_dir = self.output_dir
        n_t = self.n_t
        for i_t in range(0, n_t, n_t // (n_moments - 1)):
            t_str = "{:.2f}".format(self.t[i_t])
            os.makedirs(output_dir + f"pics/{t_str}/", exist_ok=True)
            refes = [self.refes[i][:, :, i_t] for i in range(len(self.refes))]
            preds = [self.preds[i][:, :, i_t] for i in range(len(self.preds))]
            self._save_metrics(refes, preds, output_dir + f"pics/{t_str}/")

        os.makedirs(output_dir + f"2dmetrics/", exist_ok=True)
        l2res, l1res, maes = [], [], []
        for i_t in range(n_t):
            refes = [self.refes[i][:, :, i_t] for i in range(len(self.refes))]
            preds = [self.preds[i][:, :, i_t] for i in range(len(self.preds))]
            metrics = self._save_metrics(refes, preds, output_dir + f"2dmetrics/")
            l2re = [float(metrics["l2 relative error"][j][:-1]) for j in range(len(self.preds))]
            l1re = [float(metrics["l1 relative error"][j][:-1]) for j in range(len(self.preds))]
            mae = [float(metrics["MAE"][j]) for j in range(len(self.preds))]
            l2res.append(l2re)
            l1res.append(l1re)
            maes.append(mae)
        l2res, l1res, maes = np.array(l2res), np.array(l1res), np.array(maes)
        np.save(output_dir + f"2dmetrics/L2REs.npy", l2res)
        np.save(output_dir + f"2dmetrics/L1REs.npy", l1res)
        np.save(output_dir + f"2dmetrics/MAEs.npy", maes)

        for i in range(len(self.preds)):
            plt.figure(figsize=(8, 6))
            plt.title("$L_2$RE of " + self.mathnames[i], fontsize="medium")
            plt.xlabel("$t$/s")
            plt.ylabel("$L_2$RE/%")
            plt.plot(self.t, l2res[:, i], lw=2)
            plt.savefig(output_dir + f"2dmetrics/L2RE_{i + 1}_{self.textnames[i]}.png", bbox_inches="tight", dpi=set_dpi)
            plt.close()

        for i in range(len(self.preds)):
            plt.figure(figsize=(8, 6))
            plt.title("$L_1$RE of " + self.mathnames[i], fontsize="medium")
            plt.xlabel("$t$/s")
            plt.ylabel("$L_1$RE/%")
            plt.plot(self.t, l1res[:, i], lw=2)
            plt.savefig(output_dir + f"2dmetrics/L1RE_{i + 1}_{self.textnames[i]}.png", bbox_inches="tight", dpi=set_dpi)
            plt.close()

        for i in range(len(self.preds)):
            plt.figure(figsize=(8, 6))
            plt.title("MAE of " + self.mathnames[i], fontsize="medium")
            plt.xlabel("$t$/s")
            # plt.ylabel(f"MAE/({self.units[i]})")
            plt.ylabel("MAE")
            plt.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
            plt.plot(self.t, maes[:, i], lw=2)
            plt.savefig(output_dir + f"2dmetrics/MAE_{i + 1}_{self.textnames[i]}.png", bbox_inches="tight", dpi=set_dpi)
            plt.close()

    def plot_sampling_points(self, figsize=(8, 6)):
        """Plot the sampling points, including PDE points and IC/BC/OC points."""
        print("Plotting sampling points...")
        model = self.model
        output_dir = self.output_dir
        os.makedirs(output_dir + "pics/", exist_ok=True)
        # x_l, x_r, y_l, y_r, t_l, t_r = case.x_l, case.x_r, case.y_l, case.y_r, case.t_l, case.t_r

        fig = plt.figure(figsize=figsize)
        ax = Axes3D(fig)
        ax.view_init(30, 50)
        ax.xaxis.set_inverted(True)
        ax.set_xlabel("$t$/s", labelpad=10)
        ax.set_ylabel("$x$/m", labelpad=10)
        ax.set_zlabel("$y$/m", labelpad=10)
        ax.scatter(model.data.train_x_all[:, 2],
                   model.data.train_x_all[:, 0],
                   model.data.train_x_all[:, 1],
                   s=2, lw=0.2)
        ax.scatter(model.data.train_x_bc[:, 2],
                   model.data.train_x_bc[:, 0],
                   model.data.train_x_bc[:, 1],
                   s=5, marker="x", lw=0.5)
        plt.savefig(output_dir + "pics/sampling_points.png", bbox_inches="tight", dpi=400)
        plt.close()

    def plot_2dfields(self, n_moments=6, figsize=(8, 6), cmap="jet", format="png", extra_plot=None):
        """Plot the contours of predicted 2D fields at some moments."""
        output_dir = self.output_dir
        x_l, x_r, y_l, y_r = self.x_l, self.x_r, self.y_l, self.y_r
        xx, yy, t = self.xx, self.yy, self.t
        preds = self.preds
        mnames, tnames, units = self.mathnames, self.textnames, self.units

        n_level = 50
        n_t = self.n_t
        for i_t in range(0, n_t, n_t // (n_moments - 1)):
            t_str = "{:.2f}".format(t[i_t])
            os.makedirs(output_dir + f"pics/{t_str}/", exist_ok=True)
            print(f"Plotting 2D fields (t = {t_str} s)...")

            for i in range(len(preds)):
                plt.figure(figsize=figsize)
                plt.title(f"{mnames[i]}, $t$ = {t_str} s", fontsize="medium")
                plt.axis("scaled")
                plt.axis((x_l, x_r, y_l, y_r))
                plt.xlabel(f"{self.x_name}/{self.x_unit}")
                plt.ylabel(f"{self.y_name}/{self.y_unit}")
                if tnames[i] == "psi":
                    plt.contour(xx, yy, preds[i][:, :, i_t],
                                colors="k", levels=n_level, linewidths=0.8, linestyles="solid")
                else:
                    plt.contourf(xx, yy, preds[i][:, :, i_t], cmap=cmap, levels=n_level)
                    # cb = plt.colorbar(format="%.1e", pad=0.1)
                    cb = plt.colorbar(format="%.1e")
                    cb.ax.set_title(units[i], fontsize="medium")
                if extra_plot is not None:
                    extra_plot()
                plt.savefig(output_dir + f"pics/{t_str}/contour{i + 1}_{tnames[i]}.{format}",
                            bbox_inches="tight", dpi=set_dpi)
                plt.close()

    def plot_2dfields_comp(self, n_moments=6, figsize=(13, 6), is_vertical=False,
                           label_refe="Reference", label_pred="Predicted",
                           cmap="jet", adjust=(0.07, 0.84, 0.12, 0.92, 0.1, None), format="png",
                           extra_plot=None):
        """Plot the contours of predicted v.s. reference 2D fields at some moments."""
        output_dir = self.output_dir
        x_l, x_r, y_l, y_r = self.x_l, self.x_r, self.y_l, self.y_r
        xx, yy, t = self.xx, self.yy, self.t
        preds = self.preds
        refes = self.refes
        mnames, tnames, units = self.mathnames, self.textnames, self.units

        n_row, n_col = (2, 1) if is_vertical else (1, 2)
        n_level = 50
        n_t = self.n_t
        for i_t in range(0, n_t, n_t // (n_moments - 1)):
            t_str = "{:.2f}".format(t[i_t])
            os.makedirs(output_dir + f"pics/{t_str}/", exist_ok=True)
            print(f"Plotting 2D fields (t = {t_str} s) (predicted vs reference)...")

            for i in range(len(preds)):
                fig, axes = plt.subplots(n_row, n_col, figsize=figsize)
                f_min = min(preds[i][:, :, i_t].min(), refes[i][:, :, i_t].min())
                f_max = max(preds[i][:, :, i_t].max(), refes[i][:, :, i_t].max())
                norm = matplotlib.colors.Normalize(vmin=f_min, vmax=f_max)
                subfigs = []
                for j in range(2):
                    axes[j].set_title(f"{[label_refe, label_pred][j]} {mnames[i]}, $t$ = {t_str} s", fontsize="medium")
                    axes[j].axis("scaled")
                    axes[j].axis((x_l, x_r, y_l, y_r))
                    if tnames[i] == "psi":
                        subfig = axes[j].contour(xx, yy, [refes, preds][j][i][:, :, i_t],
                                                 colors="k", levels=n_level, linewidths=0.8, linestyles="solid")
                    else:
                        subfig = axes[j].contourf(xx, yy, [refes, preds][j][i][:, :, i_t],
                                                  cmap=cmap, levels=n_level, norm=norm)
                    subfigs.append(subfig)
                    if extra_plot is not None:
                        extra_plot(axes[j])
                if is_vertical:
                    axes[0].set_xticklabels([])
                    axes[1].set_xlabel(f"{self.x_name}/{self.x_unit}")
                    axes[0].set_ylabel(f"{self.y_name}/{self.y_unit}")
                    axes[1].set_ylabel(f"{self.y_name}/{self.y_unit}")
                else:
                    axes[0].set_xlabel(f"{self.x_name}/{self.x_unit}")
                    axes[1].set_xlabel(f"{self.x_name}/{self.x_unit}")
                    axes[0].set_ylabel(f"{self.y_name}/{self.y_unit}")
                    axes[1].set_yticklabels([])
                fig.subplots_adjust(left=adjust[0], right=adjust[1], bottom=adjust[2], top=adjust[3],
                                    wspace=adjust[4], hspace=adjust[5])
                if tnames[i] != "psi":
                    cb_ax = fig.add_axes([0.87, 0.15, 0.02, 0.7])  # l, b, w, h
                    cb = fig.colorbar(subfigs[0], cax=cb_ax, format="%.1e")
                    # cb.set_ticks(np.linspace(f_min, f_max, 8))
                    cb.ax.set_title(units[i], fontsize="medium")
                plt.savefig(output_dir + f"pics/{t_str}/contourComp{i + 1}_{tnames[i]}.{format}",
                            bbox_inches="tight", dpi=set_dpi)
                plt.close(fig)

    def plot_2danimations(self, figsize=(7.8, 6), ani_time=5, cmap="jet", adjust=(0.10, 0.92, 0.12, 0.92, None, None),
                          extra_plot=None):
        """Plot the contour animations of predicted 2D fields, and streamline animations for flow problems."""
        output_dir = self.output_dir
        os.makedirs(output_dir + "pics/", exist_ok=True)
        x_l, x_r, y_l, y_r = self.x_l, self.x_r, self.y_l, self.y_r
        xx, yy, t = self.xx, self.yy, self.t
        preds = self.preds
        mnames, tnames, units = self.mathnames, self.textnames, self.units
        # dt = self.t[1] - self.t[0]

        n_level = 50
        for i in range(len(preds)):
            print(f"Plotting 2D animations of {tnames[i]}...")
            fig = plt.figure(figsize=figsize)
            ax = fig.add_subplot(111)
            if tnames[i] != "psi":
                f_min, f_max = preds[i].min(), preds[i].max()
                norm = matplotlib.colors.Normalize(vmin=f_min, vmax=f_max)
                ff_init = np.linspace(f_min, f_max, self.n_x * self.n_y).reshape([self.n_x, self.n_y])
                contourf_init = ax.contourf(xx, yy, ff_init, cmap=cmap, levels=n_level)
                cb = fig.colorbar(contourf_init, format="%.1e")
                cb.set_ticks(np.linspace(f_min, f_max, 8))
                cb.ax.set_title(units[i], fontsize="medium")

            def update(i_t):
                t_str = "{:.2f}".format(t[i_t])
                ax.cla()  # can save plot time, and necessary for `contour`
                ax.set_title(mnames[i] + f", $t$ = {t_str} s", fontsize="medium")
                ax.axis("scaled")
                ax.axis((x_l, x_r, y_l, y_r))
                ax.set_xlabel(f"{self.x_name}/{self.x_unit}")
                ax.set_ylabel(f"{self.y_name}/{self.y_unit}")
                if tnames[i] == "psi":
                    ax.contour(xx, yy, preds[i][:, :, i_t],
                               colors="k", levels=n_level, linewidths=0.8, linestyles="solid")
                else:
                    ax.contourf(xx, yy, preds[i][:, :, i_t], cmap=cmap, levels=n_level, norm=norm)
                if extra_plot is not None:
                    extra_plot()
                fig.subplots_adjust(left=adjust[0], right=adjust[1], bottom=adjust[2], top=adjust[3],
                                    wspace=adjust[4], hspace=adjust[5])

            ani = animation.FuncAnimation(fig, update, frames=self.n_t, interval=ani_time * 1000 / self.n_t)
            ani.save(output_dir + f"pics/animation{i + 1}_{tnames[i]}.gif", writer="pillow", dpi=set_dpi)
            plt.close(fig)

    def plot_2danimations_comp(self, figsize=(13, 6), is_vertical=False, ani_time=5,
                               label_refe="Reference", label_pred="Predicted",
                               cmap="jet", adjust=(0.07, 0.84, 0.12, 0.92, 0.1, None), extra_plot=None):
        """Plot the contour animations of predicted v.s. reference 2D fields."""
        output_dir = self.output_dir
        os.makedirs(output_dir + "pics/", exist_ok=True)
        x_l, x_r, y_l, y_r = self.x_l, self.x_r, self.y_l, self.y_r
        xx, yy, t = self.xx, self.yy, self.t
        preds = self.preds
        refes = self.refes
        mnames, tnames, units = self.mathnames, self.textnames, self.units
        # dt = self.t[1] - self.t[0]

        n_row, n_col = (2, 1) if is_vertical else (1, 2)
        n_level = 50
        for i in range(len(preds)):
            print(f"Plotting 2D animations ({tnames[i]}) (predicted vs reference)...")
            fig, axes = plt.subplots(n_row, n_col, figsize=figsize)
            if tnames[i] != "psi":
                f_min = min(preds[i].min(), refes[i].min())
                f_max = max(preds[i].max(), refes[i].max())
                norm = matplotlib.colors.Normalize(vmin=f_min, vmax=f_max)
                ff_init = np.linspace(f_min, f_max, self.n_x * self.n_y).reshape([self.n_x, self.n_y])
                contourf_init = axes[0].contourf(xx, yy, ff_init, cmap=cmap, levels=n_level)
                cb_ax = fig.add_axes([0.87, 0.15, 0.02, 0.7])  # l, b, w, h
                cb = fig.colorbar(contourf_init, cax=cb_ax, format="%.1e")
                cb.set_ticks(np.linspace(f_min, f_max, 8))
                cb.ax.set_title(units[i], fontsize="medium")

            def update(i_t):
                t_str = "{:.2f}".format(t[i_t])
                for j in range(2):
                    axes[j].cla()  # can save plot time, and necessary for `contour`
                    axes[j].set_title(f"{[label_refe, label_pred][j]} {mnames[i]}, $t$ = {t_str} s", fontsize="medium")
                    axes[j].axis("scaled")
                    axes[j].axis((x_l, x_r, y_l, y_r))
                    if tnames[i] == "psi":
                        axes[j].contour(xx, yy, [refes, preds][j][i][:, :, i_t],
                                        colors="k", levels=n_level, linewidths=0.8, linestyles="solid")
                    else:
                        axes[j].contourf(xx, yy, [refes, preds][j][i][:, :, i_t],
                                         cmap=cmap, levels=n_level, norm=norm)
                    if extra_plot is not None:
                        extra_plot(axes[j])
                if is_vertical:
                    axes[0].set_xticklabels([])
                    axes[1].set_xlabel(f"{self.x_name}/{self.x_unit}")
                    axes[0].set_ylabel(f"{self.y_name}/{self.y_unit}")
                    axes[1].set_ylabel(f"{self.y_name}/{self.y_unit}")
                else:
                    axes[0].set_xlabel(f"{self.x_name}/{self.x_unit}")
                    axes[1].set_xlabel(f"{self.x_name}/{self.x_unit}")
                    axes[0].set_ylabel(f"{self.y_name}/{self.y_unit}")
                    axes[1].set_yticklabels([])
                fig.subplots_adjust(left=adjust[0], right=adjust[1], bottom=adjust[2], top=adjust[3],
                                    wspace=adjust[4], hspace=adjust[5])

            ani = animation.FuncAnimation(fig, update, frames=self.n_t, interval=ani_time * 1000 / self.n_t)
            ani.save(output_dir + f"pics/animationComp{i + 1}_{tnames[i]}.gif", writer="pillow", dpi=set_dpi)
            plt.close(fig)

    @staticmethod
    def stream_function(uuu, vvv, dx, dy, psi0=0):
        psipsipsi = np.ones_like(uuu) * psi0
        u_x0_mid = 0.5 * (uuu[0, 1:, :] + uuu[0, :-1, :])
        psipsipsi[0, 1:, :] = psi0 + np.cumsum(u_x0_mid * dy, axis=0)
        v_mid = 0.5 * (vvv[1:, :, :] + vvv[:-1, :, :])
        psipsipsi[1:, :, :] = psipsipsi[0:1, :, :] + np.cumsum(-v_mid * dx, axis=0)
        return psipsipsi

