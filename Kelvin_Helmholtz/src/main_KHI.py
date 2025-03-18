import deepxde as dde
import numpy as np
# import torch
import os
import time
import argparse

# # relative import is not suggested
# # set projects_deepxde as root (by clicking), or use sys.path.insert like this (then ignore the red errors):
# import sys
# sys.path.insert(0, os.path.dirname("E:/Research_ASUS/1 PhD project/AI_PDE/projects_PINN/"))
from configs.maps_KHI import Maps
from configs.post_KHI import Postprocess
from utils.utils import efmt, cal_stat
from utils.callbacks_modi import VariableSaver


def main(args):
    if args.variable_density:
        from configs.case_KHI_vd import Case
        case_name = "drho1.0_Re{:s}".format(args.Re_str)
    else:
        from configs.case_KHI import Case
        case_name = "drho0.0_Re{:s}".format(args.Re_str)
    case = Case(args)

    # ----------------------------------------------------------------------
    # define sampling points
    n_bdr = (20 + 20) * 2 * 40 if args.bc_type == "soft" else 0
    # n_ini = 5000 if args.ic_type == "soft" else 0  # 20 * 40
    n_ini = 10000 if args.ic_type == "soft" else 0  # 20 * 40
    # n_dmn = 1000
    n_dmn = 2000
    # n_dmn = 20000
    # n_dmn = 150000

    data = dde.data.TimePDE(
        case.geom_time,
        case.pde,
        case.icbcocs,
        num_domain=n_dmn,
        num_boundary=n_bdr,
        num_initial=n_ini,
        train_distribution="pseudo",  # "Hammersley", "uniform", "pseudo"
        # train_distribution="Hammersley",  # "Hammersley", "uniform", "pseudo"
        # anchors=ob_xyt_s,
        # solution=func_sol,
        num_test=1000,
    )

    # ----------------------------------------------------------------------
    # define maps (network and input/output transforms)
    maps = Maps(args=args, case=case)
    net = maps.net
    model = dde.Model(data, net)

    # ----------------------------------------------------------------------
    # define training and train
    output_dir = f"../results/{case_name}/{args.problem_type}/"
    output_dir += f"dmn{n_dmn}"
    if args.bc_type == "soft":
        output_dir += f"_bdr{n_bdr}"
    # elif args.bc_type == "hard":
    #     output_dir += "_hardBC"

    if args.ic_type == "soft":
        output_dir += f"_ini{n_ini}"

    if args.oc_type == "soft":
        # output_dir += f"_ob{case.n_ob}-N{efmt(args.noise_level)}"
        output_dir += f"_ob{case.nx_ob}x{case.ny_ob}x{case.nt_ob}-N{efmt(args.noise_level)}"

    scale_rho, scale_u, scale_v, scale_p, scale_c = (
        args.scales["rho"], args.scales["u"], args.scales["v"], args.scales["p"], args.scales["c"])
    if args.variable_density:
        output_dir += f"_rho{efmt(scale_rho)}-u{efmt(scale_u)}-v{efmt(scale_v)}-p{efmt(scale_p)}-c{efmt(scale_c)}"
    else:
        output_dir += f"_u{efmt(scale_u)}-v{efmt(scale_v)}-p{efmt(scale_p)}-c{efmt(scale_c)}"

    if "nu" in args.infer_paras:
        output_dir += f"_nu{efmt(args.scales['nu'])}"
    if "D" in args.infer_paras:
        output_dir += f"_D{efmt(args.scales['D'])}"

    i_run = args.i_run
    while True:
        if not os.path.exists(output_dir + f"/{i_run}/"):
            output_dir += f"/{i_run}/"
            os.makedirs(output_dir)
            os.makedirs(output_dir + "models/")
            break
        else:
            i_run += 1

    model_saver = dde.callbacks.ModelCheckpoint(
        output_dir + "models/model_better", save_better_only=True, period=100)
    callbacks = [model_saver, ]
    resampler = dde.callbacks.PDEPointResampler(period=10, bc_points=True)
    callbacks += [resampler, ]

    external_trainable_variables, para_dict, var_saver = [], {}, None
    if "nu" in args.infer_paras:
        external_trainable_variables += [case.nu_infe_s, ]
        para_dict["nu"] = case.nu_infe_s
    if "D" in args.infer_paras:
        external_trainable_variables += [case.D_infe_s, ]
        para_dict["D"] = case.D_infe_s
    if len(para_dict) > 0:
        var_saver = VariableSaver(para_dict, args.scales, period=100, filename=output_dir + "parameters_history.csv")
        callbacks.append(var_saver)

    # loss_weights = None
    w_rpc = 200
    if args.variable_density:
        loss_weights = [1] * len(case.names["equations"]) + [w_rpc, 100, 100, w_rpc, w_rpc]
    else:
        loss_weights = [1] * len(case.names["equations"]) + [100, 100, w_rpc, w_rpc]
    decay_rate = 0.98 if args.variable_density else 0.97
    model.compile(optimizer="adam",  # "sgd", "rmsprop", "adam", "adamw"
                  lr=1e-3,
                  loss="MSE",
                  decay=("step", 1000, decay_rate),
                  loss_weights=loss_weights,
                  external_trainable_variables=external_trainable_variables,
                  )
    print("[" + ", ".join(case.names["equations"] + case.names["ICBCOCs"]) + "]" + "\n")

    t0 = time.perf_counter()
    model.train(iterations=args.n_iter,
                display_every=100,
                disregard_previous_best=False,
                callbacks=callbacks,
                model_restore_path=None,
                # model_restore_path=output_dir[:-2] + "2/models/model_last-30000.pt",
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
    pp2dt = Postprocess(args=args, case=case, model=model, output_dir=output_dir)
    # pp2dt.save_data(save_refe=False)
    pp2dt.save_metrics()
    pp2dt.save_2dmetrics(n_moments=n_moments)
    pp2dt.plot_save_loss_history()
    if len(args.infer_paras) > 0:
        pp2dt.save_para_metrics()
        pp2dt.plot_para_history(var_saver)
    pp2dt.delete_old_models()
    pp2dt.plot_sampling_points()
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
    parser.add_argument("--variable_density", type=bool, default=False)
    parser.add_argument("--Re_str", type=str, default="5e3")

    parser.add_argument("--problem_type", type=str, default="inverse", help="options: forward, inverse")
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

    # parser.add_argument("--nx_ob", type=int, default=16)
    # parser.add_argument("--ny_ob", type=int, default=32)
    # parser.add_argument("--nt_ob", type=int, default=51)

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

    # args.problem_type, args.ic_type, args.bc_type, args.oc_type = "forward", "hard", "hard", "none"
    args.problem_type, args.ic_type, args.bc_type, args.oc_type = "inverse", "none", "hard", "soft"
    # args.infer_paras["nu"], args.scales["nu"] = 5e-4, 1e3
    # args.infer_paras["D"], args.scales["D"] = 5e-4, 1e3

    # args.norm = True

    # args.n_iter = 500
    # args.n_iter = 20000

    # ----------------------------------------------------------------------
    # run

    n_run = 1
    for args.i_run in range(1, 1 + n_run):
        print(args)
        output_dir = main(args)

    # n_run = 1
    # for args.Re_str in ("1e4", "1e3", "2e3", ):
    #     for args.i_run in range(1, 1 + n_run):
    #         print(args)
    #         output_dir = main(args)
