import deepxde as dde
import numpy as np
# import torch
import os
import time
import argparse
from utils.dataset_modi import ScaledDataSet, ScaledComponentDataSet

# # relative import is not suggested
# # set projects_deepxde as root (by clicking), or use sys.path.insert like this (then ignore the red errors):
# import sys
# sys.path.insert(0, os.path.dirname("E:/Research_ASUS/1 PhD project/AI_PDE/projects_PINN/"))
from configs.case_KHI import Case
from configs.maps_KHI import Maps
from configs.post_KHI import PostProcessKHI
from utils.utils import efmt, cal_stat


def main(args):
    if args.variable_density:
        from configs.case_KHI_vd import Case
        case_name = "drho1.0_Re{:s}".format(args.Re_str)
    else:
        from configs.case_KHI import Case
        case_name = "drho0.0_Re{:s}".format(args.Re_str)
    case = Case(args)

    x_l, x_r, y_l, y_r, t_l, t_r = case.x_l, case.x_r, case.y_l, case.y_r, case.t_l, case.t_r

    # ----------------------------------------------------------------------
    # define observation points
    # nx_ob, ny_ob, nt_ob = 16, 32, 26
    nx_ob, ny_ob, nt_ob = 16, 32, 51
    # nx_ob, ny_ob, nt_ob = 16, 32, 101
    ob_x, dx_ob = np.linspace(x_l, x_r, nx_ob, endpoint=False, retstep=True)
    ob_y, dy_ob = np.linspace(y_l, y_r, ny_ob, endpoint=False, retstep=True)
    ob_t, dt_ob = np.linspace(t_l, t_r, nt_ob, endpoint=True, retstep=True)
    ob_xxx, ob_yyy, ob_ttt = np.meshgrid(ob_x, ob_y, ob_t, indexing="ij")
    ob_xyt = np.vstack([np.ravel(ob_xxx), np.ravel(ob_yyy), np.ravel(ob_ttt)]).T
    n_ob = nx_ob * ny_ob * nt_ob

    nx_refe, ny_refe = case.uuu_refe.shape[0], case.uuu_refe.shape[1]
    jump_x, jump_y, jump_t = nx_refe // nx_ob, ny_refe // ny_ob, (101 - 1) // (nt_ob - 1)
    ob_u = case.uuu_refe[::jump_x, ::jump_y, ::jump_t].ravel()[:, None]
    ob_v = case.vvv_refe[::jump_x, ::jump_y, ::jump_t].ravel()[:, None]
    ob_p = case.ppp_refe[::jump_x, ::jump_y, ::jump_t].ravel()[:, None]
    ob_c = case.ccc_refe[::jump_x, ::jump_y, ::jump_t].ravel()[:, None]
    if args.variable_density:
        ob_rho = case.rhorhorho_refe[::jump_x, ::jump_y, ::jump_t].ravel()[:, None]

    eps = 1e-5
    args.scales["u"] = 1 / (np.mean(np.abs(ob_u)) + eps)
    args.scales["v"] = 1 / (np.mean(np.abs(ob_v)) + eps)
    args.scales["p"] = 1 / (np.mean(np.abs(ob_p)) + eps)
    args.scales["c"] = 1 / (np.mean(np.abs(ob_c)) + eps)
    if args.variable_density:
        args.scales["rho"] = 1 / (np.mean(np.abs(ob_rho)) + eps)
    case.args = args

    normal_noise_u = np.random.randn(len(ob_u))[:, None]
    normal_noise_v = np.random.randn(len(ob_v))[:, None]
    normal_noise_p = np.random.randn(len(ob_p))[:, None]
    normal_noise_c = np.random.randn(len(ob_c))[:, None]
    ob_u += normal_noise_u * ob_u * args.noise_level
    ob_v += normal_noise_v * ob_v * args.noise_level
    ob_p += normal_noise_p * ob_p * args.noise_level
    ob_c += normal_noise_c * ob_c * args.noise_level
    if args.variable_density:
        normal_noise_rho = np.random.randn(len(ob_rho))[:, None]
        ob_rho += normal_noise_rho * ob_rho * args.noise_level

    if args.variable_density:
        data = ScaledDataSet(
            X_train=ob_xyt,
            y_train=np.hstack([ob_rho, ob_u, ob_v, ob_p, ob_c]),
            X_test=ob_xyt,
            y_test=np.hstack([ob_rho, ob_u, ob_v, ob_p, ob_c]),
            scales=(args.scales["rho"], args.scales["u"], args.scales["v"], args.scales["p"], args.scales["c"]),
            # standardize=True,
        )
    else:
        # data = dde.data.DataSet(
        data = ScaledDataSet(
            X_train=ob_xyt,
            y_train=np.hstack([ob_u, ob_v, ob_p, ob_c]),
            X_test=ob_xyt,
            y_test=np.hstack([ob_u, ob_v, ob_p, ob_c]),
            scales=(args.scales["u"], args.scales["v"], args.scales["p"], args.scales["c"]),
            # standardize=True,
        )

    # ----------------------------------------------------------------------
    # define maps (network and input/output transforms)
    maps = Maps(args=args, case=case)
    net = maps.net
    model = dde.Model(data, net)

    # ----------------------------------------------------------------------
    # define training and train
    output_dir = f"../results/{case_name}/fit/"
    # output_dir += f"ob{n_ob}-N{efmt(args.noise_level)}"
    output_dir += f"ob{nx_ob}x{ny_ob}x{nt_ob}-N{efmt(args.noise_level)}"

    scale_rho, scale_u, scale_v, scale_p, scale_c = (
        args.scales["rho"], args.scales["u"], args.scales["v"], args.scales["p"], args.scales["c"])
    if args.variable_density:
        output_dir += f"_rho{efmt(scale_rho)}-u{efmt(scale_u)}-v{efmt(scale_v)}-p{efmt(scale_p)}-c{efmt(scale_c)}"
    else:
        output_dir += f"_u{efmt(scale_u)}-v{efmt(scale_v)}-p{efmt(scale_p)}-c{efmt(scale_c)}"

    i_run = args.i_run
    while True:
        if not os.path.exists(output_dir + f"/{i_run}/"):
            output_dir += f"/{i_run}/"
            os.makedirs(output_dir)
            os.makedirs(output_dir + "models/")
            break
        else:
            i_run += 1
    # output_dir += "/1/"

    model_saver = dde.callbacks.ModelCheckpoint(
        output_dir + "models/model_better", save_better_only=True, period=100)
    callbacks = [model_saver, ]

    # external_trainable_variables = [maps.lnw0, maps.lnw1, maps.lnw2, maps.lnw3]
    # variable_saver = dde.callbacks.VariableValue(
    #     external_trainable_variables, period=100, filename=output_dir + "vars_history_scaled.txt")
    # callbacks.append(variable_saver)

    loss_weights = None
    # if args.variable_density:
    #     loss_weights = [1] * len(case.names["equations"]) + [100] * 5
    # else:
    #     loss_weights = [1] * len(case.names["equations"]) + [100] * 4
    decay_rate = 0.98 if args.variable_density else 0.97
    model.compile(optimizer="adam",  # "sgd", "rmsprop", "adam", "adamw"
                  lr=1e-3,
                  loss="MSE",
                  # loss="MAE",
                  decay=("step", 1000, decay_rate),
                  loss_weights=loss_weights,
                  # external_trainable_variables=external_trainable_variables,
                  )

    t0 = time.perf_counter()
    model.train(iterations=args.n_iter,
                display_every=100,
                disregard_previous_best=False,
                callbacks=callbacks,
                model_restore_path=None,
                model_save_path=output_dir + "models/model_last", )
    t_took = time.perf_counter() - t0
    np.savetxt(output_dir + f"training_time_is_{t_took:.2f}s.txt", np.array([t_took]), fmt="%.2f")

    # ----------------------------------------------------------------------
    # restore the best model (do not if using LBFGS)
    model_list = os.listdir(output_dir + "models/")
    model_list_better = [s for s in model_list if "better" in s]
    saved_epochs = [int(s.split("-")[1][:-3]) for s in model_list_better]
    best_epoch = max(saved_epochs)
    model.restore(output_dir + f"models/model_better-{best_epoch}.pt")

    # ----------------------------------------------------------------------
    # post-process
    import matplotlib.pyplot as plt

    def extra_plot1():
        plt.plot((case.x_r, case.x_r), (case.y_l, case.y_r), "k--", lw=1)  # add

    def extra_plot2(ax):
        ax.plot((case.x_r, case.x_r), (case.y_l, case.y_r), "k--", lw=1)  # add

    n_moments = 26
    pp2dt = PostProcessKHI(args=args, case=case, model=model, output_dir=output_dir)
    # pp2dt.save_data(save_refe=False)
    pp2dt.save_metrics()
    pp2dt.save_2dmetrics(n_moments=n_moments)
    # pp2dt.plot_save_loss_history()
    # if len(args.infer_paras) > 0:
    #     pp2dt.save_para_metrics()
    #     pp2dt.plot_para_history(var_saver)
    pp2dt.delete_old_models()
    # pp2dt.plot_sampling_points()
    # pp2dt.plot_2dfields(n_moments=n_moments, figsize=(8, 6), cmap="RdBu_r", extra_plot=extra_plot1)
    pp2dt.plot_2dfields_comp(n_moments=n_moments, figsize=(13, 6), cmap="RdBu_r", extra_plot=extra_plot2)
    # pp2dt.plot_2danimations(figsize=(7.8, 6), cmap="RdBu_r", extra_plot=extra_plot1)
    # pp2dt.plot_2danimations_comp(figsize=(13, 6), cmap="RdBu_r", extra_plot=extra_plot2)
    pp2dt.plot_tke_spectrums(n_moments=n_moments)
    pp2dt.plot_ke_t()
    pp2dt.plot_entropy_t()

    return output_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hyper-parameters")
    parser.add_argument("--case_id", type=int, default=1)
    parser.add_argument("--Re_str", type=str, default="5e3")

    # parser.add_argument("--problem_type", type=str, default="inverse", help="options: forward, inverse")
    parser.add_argument("--problem_type", type=str, default="fit")
    parser.add_argument("--ic_type", type=str, default="soft", help="options: none, soft, hard")
    parser.add_argument("--bc_type", type=str, default="hard", help="options: none, soft, hard")
    parser.add_argument("--oc_type", type=str, default="none", help="options: none, soft")

    parser.add_argument("--scales", type=dict,
                        default={"x": 1.0, "y": 1.0, "t": 1.0, "rho": 1.0, "u": 1.0, "v": 1.0, "p": 1.0, "c": 1.0},
                        help="(variables * scale) for NN I/O, PDE scaling, and parameter inference")
    parser.add_argument("--shifts", type=dict, default={"x": 0.0, "y": 0.0, "t": 0.0},
                        help="((independent variables + shift) * scale) for NN input and PDE scaling")

    parser.add_argument("--infer_paras", type=dict, default={},
                        help="initial values for unknown physical parameters to be inferred")
    parser.add_argument("--noise_level", type=float, default=0.00,
                        help="noise level of observed data for inverse problems, such as 0.02 (2%)")

    parser.add_argument("--n_iter", type=int, default=20000, help="number of training iterations")
    parser.add_argument("--i_run", type=int, default=1, help="index of the current run")

    # ----------------------------------------------------------------------
    # set arguments
    args = parser.parse_args()

    # args.scales["rho"], args.scales["u"], args.scales["v"], args.scales["p"], args.scales["c"] = 1, 1, 1, 1, 1
    args.scales["x"], args.shifts["x"] = 1, 0
    args.scales["y"], args.shifts["y"] = 1, 0
    # args.scales["t"], args.shifts["t"] = 1, 0
    args.scales["t"], args.shifts["t"] = 1.0, -2.5

    args.variable_density, args.n_iter = False, 50000
    # args.variable_density, args.n_iter = True, 80000
    args.Re_str = "5e3"

    # ----------------------------------------------------------------------
    # run
    n_run = 1
    for args.i_run in range(1, 1 + n_run):
        print(args)
        output_dir = main(args)

    # n_run = 1
    # for args.Re_str in ("1e3", "5e3", "1e4", "2e3", "8e3"):
    #     for args.i_run in range(1, 1 + n_run):
    #         print(args)
    #         output_dir = main(args)
