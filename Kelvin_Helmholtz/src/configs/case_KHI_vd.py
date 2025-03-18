import math as mt
import numpy as np
import h5py
import torch
import deepxde as dde
import os
# from scipy import interpolate as interp
import copy
from utils.icbcs import ScaledPeriodicBC, ScaledPointSetBC, ScaledPointSetOperatorBC


class Case:
    def __init__(self, args):
        self.args = args
        # self.args = copy.copy(args)

        # ----------------------------------------------------------------------
        # define calculation domain
        self.x_l, self.x_r = 0.0, 0.5
        self.y_l, self.y_r = 0.0, 1.0
        # self.t_l, self.t_r = args.t_l, args.t_r
        self.t_l, self.t_r = 0.0, 5.0
        geom = dde.geometry.Rectangle(xmin=[self.x_l, self.y_l], xmax=[self.x_r, self.y_r])
        time_domain = dde.geometry.TimeDomain(self.t_l, self.t_r)
        self.geom_time = dde.geometry.GeometryXTime(geom, time_domain)

        # define the names of independents, dependents, and equations
        self.names = {
            "independents": ["x", "y", "t"],
            "dependents": ["rho", "u", "v", "p", "c"],
            "equations": ["incompress", "mass", "momentum_x", "momentum_y", "scalar"],
            "ICBCOCs": []}

        self.icbcocs = []  # initial, boundary, observation conditions

        self.nu_infe_s = dde.Variable(args.infer_paras["nu"] * args.scales["nu"]) if "nu" in args.infer_paras else None
        self.D_infe_s = dde.Variable(args.infer_paras["D"] * args.scales["D"]) if "D" in args.infer_paras else None

        # ----------------------------------------------------------------------
        # define parameters
        self.Re = float(args.Re_str)
        self.nu = 1 / self.Re
        self.D = self.nu
        self.u_flow = 1.0

        # IC parameters
        self.y1, self.y2 = 0.25, 0.75
        self.a = 0.025
        self.u_flow = 1.0
        self.A = 0.01
        self.p0 = 1.0
        self.rho0 = 1.0
        self.drho = 1.0

        # ----------------------------------------------------------------------
        # load data
        # data_path = f"../data/data_drho1.0_Re{args.Re_str}_512x1024x101.h5"
        data_path = f"../data/data_drho1.0_Re{args.Re_str}_256x512x101.h5"
        print("loading data...")
        data = h5py.File(data_path, "r")
        print("loading data: finished.")
        t = data["t"][:]
        x = data["x"][:]
        y = data["y"][:]
        rhorhorho = data["rhorhorho"][:].transpose(1, 2, 0)  # (nx, ny, nt)
        uuu = data["uuu"][:].transpose(1, 2, 0)  # (nx, ny, nt)
        vvv = data["vvv"][:].transpose(1, 2, 0)  # (nx, ny, nt)
        ppp = data["ppp"][:].transpose(1, 2, 0)  # (nx, ny, nt)
        ccc = data["ccc"][:].transpose(1, 2, 0)  # (nx, ny, nt)
        omomom = data["omegaomegaomega"][:].transpose(1, 2, 0)  # (nx, ny, nt)
        data.close()

        self.x, self.y, self.t = x, y, t
        self.rhorhorho_refe = rhorhorho
        self.uuu_refe = uuu
        self.vvv_refe = vvv
        self.ppp_refe = ppp
        self.ccc_refe = ccc
        self.omomom_refe = omomom

        # ----------------------------------------------------------------------
        # define ICs, BCs, OCs
        self.define_icbcocs()
        print(args)

    # ----------------------------------------------------------------------
    # IC functions
    def func_rho0(self, xyt):
        y1, y2, a, rho0, drho = self.y1, self.y2, self.a, self.rho0, self.drho
        y = xyt[:, 1:2]
        return 1.0 + drho / rho0 * 0.5 * (np.tanh((y - y1) / a) - np.tanh((y - y2) / a))

    def func_u0(self, xyt):
        y1, y2, a, u_flow = self.y1, self.y2, self.a, self.u_flow
        y = xyt[:, 1:2]
        return u_flow * (np.tanh((y - y1) / a) - np.tanh((y - y2) / a) - 1)

    def func_v0(self, xyt):
        y1, y2, A = self.y1, self.y2, self.A
        x, y = xyt[:, 0:1], xyt[:, 1:2]
        return A * np.sin(4 * mt.pi * x)

    def func_p0(self, xyt):
        x, y = xyt[:, 0:1], xyt[:, 1:2]
        # return np.zeros_like(x)
        return np.zeros_like(x) + self.p0

    def func_c0(self, xyt):
        y1, y2, a = self.y1, self.y2, self.a
        y = xyt[:, 1:2]
        return 0.5 * (np.tanh((y - y1) / a) - np.tanh((y - y2) / a))
    
    def func_rho0_tensor(self, x, y):
        y1, y2, a, rho0, drho = self.y1, self.y2, self.a, self.rho0, self.drho
        return 1.0 + drho / rho0 * 0.5 * (torch.tanh((y - y1) / a) - torch.tanh((y - y2) / a))

    def func_u0_tensor(self, x, y):
        y1, y2, a, u_flow = self.y1, self.y2, self.a, self.u_flow
        return u_flow * (torch.tanh((y - y1) / a) - torch.tanh((y - y2) / a) - 1)

    def func_v0_tensor(self, x, y):
        y1, y2, A = self.y1, self.y2, self.A
        return A * torch.sin(4 * mt.pi * x)

    def func_p0_tensor(self, x, y):
        # return torch.zeros_like(x)
        return torch.zeros_like(x) + self.p0

    def func_c0_tensor(self, x, y):
        y1, y2, a = self.y1, self.y2, self.a
        return 0.5 * (torch.tanh((y - y1) / a) - torch.tanh((y - y2) / a))

    # ----------------------------------------------------------------------
    # define pde
    def pde(self, xyt, ruvpc):
        args = self.args
        # x, y, t = xyt[:, 0:1], xyt[:, 1:2], xyt[:, 2:]
        rho, u, v, p, c = ruvpc[:, 0:1], ruvpc[:, 1:2], ruvpc[:, 2:3], ruvpc[:, 3:4], ruvpc[:, 4:]
        scale_rho, scale_u, scale_v, scale_p, scale_c = (
            args.scales["rho"], args.scales["u"], args.scales["v"], args.scales["p"], args.scales["c"])
        scale_x, scale_y, scale_t = args.scales["x"], args.scales["y"], args.scales["t"]
        # shift_x, shift_y, shift_t = args.shifts["x"], args.shifts["y"], args.shifts["t"]

        rho_x = dde.grad.jacobian(ruvpc, xyt, i=0, j=0)
        rho_y = dde.grad.jacobian(ruvpc, xyt, i=0, j=1)
        rho_t = dde.grad.jacobian(ruvpc, xyt, i=0, j=2)
        rho_xx = dde.grad.hessian(ruvpc, xyt, component=0, i=0, j=0)
        rho_yy = dde.grad.hessian(ruvpc, xyt, component=0, i=1, j=1)

        u_x = dde.grad.jacobian(ruvpc, xyt, i=1, j=0)
        u_y = dde.grad.jacobian(ruvpc, xyt, i=1, j=1)
        u_t = dde.grad.jacobian(ruvpc, xyt, i=1, j=2)
        u_xx = dde.grad.hessian(ruvpc, xyt, component=1, i=0, j=0)
        u_yy = dde.grad.hessian(ruvpc, xyt, component=1, i=1, j=1)

        v_x = dde.grad.jacobian(ruvpc, xyt, i=2, j=0)
        v_y = dde.grad.jacobian(ruvpc, xyt, i=2, j=1)
        v_t = dde.grad.jacobian(ruvpc, xyt, i=2, j=2)
        v_xx = dde.grad.hessian(ruvpc, xyt, component=2, i=0, j=0)
        v_yy = dde.grad.hessian(ruvpc, xyt, component=2, i=1, j=1)

        p_x = dde.grad.jacobian(ruvpc, xyt, i=3, j=0)
        p_y = dde.grad.jacobian(ruvpc, xyt, i=3, j=1)

        c_x = dde.grad.jacobian(ruvpc, xyt, i=4, j=0)
        c_y = dde.grad.jacobian(ruvpc, xyt, i=4, j=1)
        c_t = dde.grad.jacobian(ruvpc, xyt, i=4, j=2)
        c_xx = dde.grad.hessian(ruvpc, xyt, component=4, i=0, j=0)
        c_yy = dde.grad.hessian(ruvpc, xyt, component=4, i=1, j=1)

        nu = self.nu_infe_s / args.scales["nu"] if "nu" in args.infer_paras else self.nu
        D = self.D_infe_s / args.scales["D"] if "D" in args.infer_paras else self.D

        incompress = u_x + v_y
        mass = rho_t + u * rho_x + v * rho_y - D * (rho_xx + rho_yy)
        momentum_x = u_t + u * u_x + v * u_y + p_x / rho - nu * (u_xx + u_yy) - nu / rho * (2 * rho_x * u_x + rho_y * (u_y + v_x))
        momentum_y = v_t + u * v_x + v * v_y + p_y / rho - nu * (v_xx + v_yy) - nu / rho * (2 * rho_y * v_y + rho_x * (u_y + v_x))
        scalar = c_t + u * c_x + v * c_y - D * (c_xx + c_yy) - D / rho * (rho_x * c_x + rho_y * c_y)

        # coef = max(scale_u / scale_x, scale_v / scale_y)
        coef = 1.0
        incompress *= coef
        mass *= coef * scale_rho / scale_t
        momentum_x *= coef * scale_u / scale_t
        momentum_y *= coef * scale_v / scale_t
        scalar *= coef * scale_c / scale_t
        return [incompress, mass, momentum_x, momentum_y, scalar]

    def omega(self, xyt, ruvpc):
        u_y = dde.grad.jacobian(ruvpc, xyt, i=1, j=1)
        v_x = dde.grad.jacobian(ruvpc, xyt, i=2, j=0)
        return v_x - u_y

    # ----------------------------------------------------------------------
    # define ICs, BCs, OCs
    def define_icbcocs(self):
        args = self.args
        geom_time = self.geom_time
        scale_rho, scale_u, scale_v, scale_p, scale_c = (
            args.scales["rho"], args.scales["u"], args.scales["v"], args.scales["p"], args.scales["c"])

        # TODO: update scales
        bc_rho_lr = ScaledPeriodicBC(geom_time, 0, lambda _, on_bdr: on_bdr, component=0, scale=scale_rho)
        bc_u_lr = ScaledPeriodicBC(geom_time, 0, lambda _, on_bdr: on_bdr, component=1, scale=scale_u)
        bc_v_lr = ScaledPeriodicBC(geom_time, 0, lambda _, on_bdr: on_bdr, component=2, scale=scale_v)
        bc_p_lr = ScaledPeriodicBC(geom_time, 0, lambda _, on_bdr: on_bdr, component=3, scale=scale_p)
        bc_c_lr = ScaledPeriodicBC(geom_time, 0, lambda _, on_bdr: on_bdr, component=4, scale=scale_c)

        bc_rho_du = ScaledPeriodicBC(geom_time, 1, lambda _, on_bdr: on_bdr, component=0, scale=scale_rho)
        bc_u_du = ScaledPeriodicBC(geom_time, 1, lambda _, on_bdr: on_bdr, component=1, scale=scale_u)
        bc_v_du = ScaledPeriodicBC(geom_time, 1, lambda _, on_bdr: on_bdr, component=2, scale=scale_v)
        bc_p_du = ScaledPeriodicBC(geom_time, 1, lambda _, on_bdr: on_bdr, component=3, scale=scale_p)
        bc_c_du = ScaledPeriodicBC(geom_time, 1, lambda _, on_bdr: on_bdr, component=4, scale=scale_c)

        # ic_rho = dde.icbc.IC(geom_time, self.func_rho, lambda _, on_ini: on_ini, component=0)
        # ic_u = dde.icbc.IC(geom_time, self.func_u, lambda _, on_ini: on_ini, component=1)
        # ic_v = dde.icbc.IC(geom_time, self.func_v, lambda _, on_ini: on_ini, component=2)
        # ic_p = dde.icbc.IC(geom_time, self.func_p, lambda _, on_ini: on_ini, component=3)
        # ic_c = dde.icbc.IC(geom_time, self.func_c, lambda _, on_ini: on_ini, component=4)

        if args.bc_type == "soft":
            self.icbcocs += [bc_rho_lr, bc_u_lr, bc_v_lr, bc_p_lr, bc_c_lr,
                             bc_rho_du, bc_u_du, bc_v_du, bc_p_du, bc_c_du, ]
            self.names["ICBCOCs"] += ["BC_rho_lr", "BC_u_lr", "BC_v_lr", "BC_p_lr", "BC_c_lr",
                                      "BC_rho_du", "BC_u_du", "BC_v_du", "BC_p_du", "BC_c_du", ]
        else:  # "none"
            pass

        # if ic_type == "soft":
        #     self.icbcocs += [ic_rho, ic_u, ic_v, ic_p, ic_c]
        #     self.names["ICBCOCs"] += ["IC_rho", "IC_u", "IC_v", "IC_p", "IC_c"]
        # else:  # "none", "hard"
        #     pass

        if args.oc_type == "soft":
            # nx_ob, ny_ob, nt_ob = 16, 32, 26
            nx_ob, ny_ob, nt_ob = 16, 32, 51
            # nx_ob, ny_ob, nt_ob = 16, 32, 101
            ob_x, dx_ob = np.linspace(self.x_l, self.x_r, nx_ob, endpoint=False, retstep=True)
            ob_y, dy_ob = np.linspace(self.y_l, self.y_r, ny_ob, endpoint=False, retstep=True)
            ob_t, dt_ob = np.linspace(self.t_l, self.t_r, nt_ob, endpoint=True, retstep=True)
            ob_xxx, ob_yyy, ob_ttt = np.meshgrid(ob_x, ob_y, ob_t, indexing="ij")
            ob_xyt = np.vstack([np.ravel(ob_xxx), np.ravel(ob_yyy), np.ravel(ob_ttt)]).T
            n_ob = nx_ob * ny_ob * nt_ob

            nx_refe, ny_refe, nt_refe = self.uuu_refe.shape[0], self.uuu_refe.shape[1], self.uuu_refe.shape[2]
            jump_x, jump_y, jump_t = nx_refe // nx_ob, ny_refe // ny_ob, (nt_refe - 1) // (nt_ob - 1)
            ob_rho = self.rhorhorho_refe[::jump_x, ::jump_y, ::jump_t].ravel()[:, None]
            ob_u = self.uuu_refe[::jump_x, ::jump_y, ::jump_t].ravel()[:, None]
            ob_v = self.vvv_refe[::jump_x, ::jump_y, ::jump_t].ravel()[:, None]
            ob_p = self.ppp_refe[::jump_x, ::jump_y, ::jump_t].ravel()[:, None]
            ob_c = self.ccc_refe[::jump_x, ::jump_y, ::jump_t].ravel()[:, None]

            eps = 1e-5
            args.scales["rho"] = 1 / (np.mean(np.abs(ob_rho)) + eps)
            args.scales["u"] = 1 / (np.mean(np.abs(ob_u)) + eps)
            args.scales["v"] = 1 / (np.mean(np.abs(ob_v)) + eps)
            args.scales["p"] = 1 / (np.mean(np.abs(ob_p)) + eps)
            args.scales["c"] = 1 / (np.mean(np.abs(ob_c)) + eps)

            self.args = args

            normal_noise_rho = np.random.randn(len(ob_rho))[:, None]
            normal_noise_u = np.random.randn(len(ob_u))[:, None]
            normal_noise_v = np.random.randn(len(ob_v))[:, None]
            normal_noise_p = np.random.randn(len(ob_p))[:, None]
            normal_noise_c = np.random.randn(len(ob_c))[:, None]
            ob_rho += normal_noise_rho * ob_rho * args.noise_level
            ob_u += normal_noise_u * ob_u * args.noise_level
            ob_v += normal_noise_v * ob_v * args.noise_level
            ob_p += normal_noise_p * ob_p * args.noise_level
            ob_c += normal_noise_c * ob_c * args.noise_level

            # ob_u_trans = ob_u + np.tanh(18 * ob_u) / 2
            # ob_v_trans = ob_v + np.tanh(18 * ob_v) / 2
            ob_u_trans = ob_u + np.arctan(15 * ob_u) - np.arctan(6 * ob_u)  # f1
            ob_v_trans = ob_v + np.arctan(15 * ob_v) - np.arctan(6 * ob_v)  # f1
            # ob_u_trans = ob_u + 9 * ob_u / (1 + 100 * ob_u ** 2)  # f2
            # ob_v_trans = ob_v + 9 * ob_v / (1 + 100 * ob_v ** 2)  # f2
            # ob_u_trans = ob_u + np.tanh(13 * ob_u) - np.tanh(4 * ob_u)  # f3
            # ob_v_trans = ob_v + np.tanh(13 * ob_v) - np.tanh(4 * ob_v)  # f3
            # ob_u_trans = ob_u / (0.45 * np.tanh(10 * (np.abs(ob_u) - 0.25)) + 0.55)  # f4
            # ob_v_trans = ob_v / (0.45 * np.tanh(10 * (np.abs(ob_v) - 0.25)) + 0.55)  # f4

            def func_u_trans(xyt, ruvpc, _):
                rho, u, v, p, c = ruvpc[:, 0:1], ruvpc[:, 1:2], ruvpc[:, 2:3], ruvpc[:, 3:4], ruvpc[:, 4:]
                u_trans = u + torch.arctan(15 * u) - torch.arctan(6 * u)  # f1
                # u_trans = u + 9 * u / (1 + 100 * u ** 2)  # f2
                # u_trans = u + torch.tanh(13 * u) - torch.tanh(4 * u)  # f3
                # u_trans = u / (0.45 * torch.tanh(10 * (torch.abs(u) - 0.25)) + 0.55)  # f4
                return u_trans

            def func_v_trans(xyt, ruvpc, _):
                rho, u, v, p, c = ruvpc[:, 0:1], ruvpc[:, 1:2], ruvpc[:, 2:3], ruvpc[:, 3:4], ruvpc[:, 4:]
                v_trans = v + torch.arctan(15 * v) - torch.arctan(6 * v)  # f1
                # v_trans = v + 9 * v / (1 + 100 * v ** 2)  # f2
                # v_trans = v + torch.tanh(13 * v) - torch.tanh(4 * v)  # f3
                # v_trans = v / (0.45 * torch.tanh(10 * (torch.abs(v) - 0.25)) + 0.55)  # f4
                return v_trans

            # bachsize_oc = 1000
            # bachsize_oc = 2000
            bachsize_oc = n_ob // 4
            # bachsize_oc = n_ob // 16
            eps = None
            # eps = 1e-1
            oc_rho = ScaledPointSetBC(ob_xyt, ob_rho, component=0, batch_size=bachsize_oc, scale=args.scales["rho"])
            # oc_u = ScaledPointSetBC(ob_xyt, ob_u, component=1, batch_size=bachsize_oc, scale=args.scales["u"])
            # oc_v = ScaledPointSetBC(ob_xyt, ob_v, component=2, batch_size=bachsize_oc, scale=args.scales["v"])
            oc_u = ScaledPointSetOperatorBC(ob_xyt, ob_u_trans, func_u_trans, bachsize_oc, scale=args.scales["u"])
            oc_v = ScaledPointSetOperatorBC(ob_xyt, ob_v_trans, func_v_trans, bachsize_oc, scale=args.scales["v"])
            oc_p = ScaledPointSetBC(ob_xyt, ob_p, component=3, batch_size=bachsize_oc, scale=args.scales["p"])
            oc_c = ScaledPointSetBC(ob_xyt, ob_c, component=4, batch_size=bachsize_oc, scale=args.scales["c"])
            self.icbcocs += [oc_rho, oc_u, oc_v, oc_p, oc_c, ]
            self.names["ICBCOCs"] += ["OC_rho", "OC_u", "OC_v", "OC_p", "OC_c", ]
        else:  # "none"
            ob_xyt = np.empty([1, 3])
            n_ob = 0
            nx_ob, ny_ob, nt_ob = 0, 0, 0
        # n_ob = 0
        self.n_ob = n_ob
        self.nx_ob, self.ny_ob, self.nt_ob = nx_ob, ny_ob, nt_ob

