import numpy as np
from utils.postprocess import PostProcess2Dt
# import deepxde as dde
import torch
import os

import matplotlib
matplotlib.use("Agg")  # do not show figures
import matplotlib.pyplot as plt

set_fs = 22
set_dpi = 200
plt.rcParams["font.sans-serif"] = "Arial"  # default font
# plt.rcParams["font.sans-serif"] = "Nimbus Sans"  # default font
plt.rcParams["font.size"] = set_fs  # default font size
# plt.rcParams["mathtext.fontset"] = "stix"  # default font of math text


class PostProcessKHI(PostProcess2Dt):
    def __init__(self, args, case, model, output_dir):
        super().__init__(args, case, model, output_dir)
        jump_x, jump_y, jump_t = 1, 1, 1
        # jump_x, jump_y, jump_t = 2, 2, 1
        # jump_x, jump_y, jump_t = 4, 4, 4
        self.x, self.y, self.t = case.x, case.y, case.t
        self.x, self.x_r = np.hstack([self.x, self.x + 0.5]), case.x_r + 0.5
        self.x, self.y, self.t = self.x[::jump_x], self.y[::jump_y], self.t[::jump_t]  # downsample
        self.n_x, self.n_y, self.n_t = len(self.x), len(self.y), len(self.t)
        self.xx, self.yy = np.meshgrid(self.x, self.y, indexing="ij")
        dx, dy = self.x[1] - self.x[0], self.y[1] - self.y[0]

        # xxx, yyy, ttt = np.meshgrid(self.x, self.y, self.t, indexing="ij")
        # xyt = np.vstack([np.ravel(xxx), np.ravel(yyy), np.ravel(ttt)]).T

        # ----------------------------------------------------------------------
        # Get the predicted and reference fields
        rhorhorho_pred, uuu_pred, vvv_pred, ppp_pred, ccc_pred = [], [], [], [], []
        # omomom_pred = []
        x = self.x
        y = self.y
        for i_t in range(self.n_t):
            t = self.t[i_t]
            xxx, yyy, ttt = np.meshgrid(x, y, t, indexing="ij")
            xyt = np.vstack([np.ravel(xxx), np.ravel(yyy), np.ravel(ttt)]).T

            output = model.predict(xyt)
            if args.variable_density:
                rho_pred, u_pred, v_pred, p_pred, c_pred = output[:, 0], output[:, 1], output[:, 2], output[:, 3], output[:, 4]
                rhorhorho_pred.append(rho_pred.reshape([self.n_x, self.n_y, 1]))
            else:
                u_pred, v_pred, p_pred, c_pred = output[:, 0], output[:, 1], output[:, 2], output[:, 3]
            uuu_pred.append(u_pred.reshape([self.n_x, self.n_y, 1]))
            vvv_pred.append(v_pred.reshape([self.n_x, self.n_y, 1]))
            ppp_pred.append(p_pred.reshape([self.n_x, self.n_y, 1]))
            ccc_pred.append(c_pred.reshape([self.n_x, self.n_y, 1]))

            # # om_pred = model.predict(xyt, operator=case.omega)
            # # for case of OOM:
            # om_pred = []
            # for i in range(len(xyt) // self.n_x):
            #     om_pred.append(model.predict(xyt[i * self.n_x: (i + 1) * self.n_x, :], operator=case.omega).ravel())
            # om_pred = np.hstack(om_pred)
            # omomom_pred.append(om_pred.reshape([self.n_x, self.n_y, 1]))

        self.rhorhorho_pred = np.concatenate(rhorhorho_pred, axis=-1) if args.variable_density else None
        self.uuu_pred = np.concatenate(uuu_pred, axis=-1)
        self.vvv_pred = np.concatenate(vvv_pred, axis=-1)
        self.ppp_pred = np.concatenate(ppp_pred, axis=-1)
        self.ccc_pred = np.concatenate(ccc_pred, axis=-1)
        # self.omomom_pred = np.concatenate(omomom_pred, axis=-1)
        self.UUU_pred = np.sqrt(self.uuu_pred ** 2 + self.vvv_pred ** 2)
        self.psipsipsi_pred = self.stream_function(self.uuu_pred, self.vvv_pred, dx, dy)
        uuu_ypad = np.concatenate([self.uuu_pred[:, -1:, :], self.uuu_pred, self.uuu_pred[:, :1, :]], axis=1)
        vvv_xpad = np.concatenate([self.vvv_pred[-1:, :, :], self.vvv_pred, self.vvv_pred[:1, :, :]], axis=0)
        self.omomom_pred = (vvv_xpad[2:, :, :] - vvv_xpad[:-2, :, :]) / (2 * dx) - (uuu_ypad[:, 2:, :] - uuu_ypad[:, :-2, :]) / (2 * dy)

        # self.rhorhorho_refe = case.rhorhorho_refe.astype(np.float32) if args.variable_density else None
        # self.uuu_refe = case.uuu_refe.astype(np.float32)
        # self.vvv_refe = case.vvv_refe.astype(np.float32)
        # self.ppp_refe = case.ppp_refe.astype(np.float32)
        # self.ccc_refe = case.ccc_refe.astype(np.float32)
        # self.omomom_refe = case.omomom_refe.astype(np.float32)
        if args.variable_density:
            self.rhorhorho_refe = np.concatenate([case.rhorhorho_refe, case.rhorhorho_refe], axis=0, dtype=np.float32)
            self.rhorhorho_refe = self.rhorhorho_refe[::jump_x, ::jump_y, ::jump_t]
        else:
            self.rhorhorho_refe = None
        self.uuu_refe = np.concatenate([case.uuu_refe, case.uuu_refe], axis=0, dtype=np.float32)
        self.vvv_refe = np.concatenate([case.vvv_refe, case.vvv_refe], axis=0, dtype=np.float32)
        self.ppp_refe = np.concatenate([case.ppp_refe, case.ppp_refe], axis=0, dtype=np.float32)
        self.ccc_refe = np.concatenate([case.ccc_refe, case.ccc_refe], axis=0, dtype=np.float32)
        self.omomom_refe = np.concatenate([case.omomom_refe, case.omomom_refe], axis=0, dtype=np.float32)

        self.uuu_refe = self.uuu_refe[::jump_x, ::jump_y, ::jump_t]
        self.vvv_refe = self.vvv_refe[::jump_x, ::jump_y, ::jump_t]
        self.ppp_refe = self.ppp_refe[::jump_x, ::jump_y, ::jump_t]
        self.ccc_refe = self.ccc_refe[::jump_x, ::jump_y, ::jump_t]
        self.omomom_refe = self.omomom_refe[::jump_x, ::jump_y, ::jump_t]
        self.UUU_refe = np.sqrt(self.uuu_refe ** 2 + self.vvv_refe ** 2)
        self.psipsipsi_refe = self.stream_function(self.uuu_refe, self.vvv_refe, dx, dy)

        self.preds += [self.uuu_pred, self.vvv_pred, self.ppp_pred, self.ccc_pred,
                       self.omomom_pred, self.UUU_pred, self.psipsipsi_pred]
        self.refes += [self.uuu_refe, self.vvv_refe, self.ppp_refe, self.ccc_refe,
                       self.omomom_refe, self.UUU_refe, self.psipsipsi_refe]
        self.mathnames += ["$u$", "$v$", "$p$", "$c$", r"$\omega$", r"$|\mathbf{U}|$", r"$\psi$"]
        self.textnames += ["u", "v", "p", "c", "omega", "Um", "psi"]
        self.units += ["m/s", "m/s", "Pa", " ", "s$^{-1}$", "m/s", "m$^2$/s"]

        if args.variable_density:
            self.preds.insert(0, self.rhorhorho_pred)
            self.refes.insert(0, self.rhorhorho_refe)
            self.mathnames.insert(0, r"$\rho$")
            self.textnames.insert(0, "rho")
            self.units.insert(0, "kg/m$^3$")

        # # for linear input (no hard periodic BC) -- Do not use. Using half domain instead.
        # for i in range(len(self.preds)):
        #     self.preds[i][self.n_x//2:, ...] = self.preds[i][:self.n_x//2, ...]

        if "nu" in args.infer_paras:
            self.para_infes += [case.nu_infe_s / args.scales["nu"], ]
            self.para_refes += [case.nu, ]
            self.para_mathnames += [r"$\nu$", ]
            self.para_textnames += ["nu", ]
            self.para_units += ["m$^2$/s", ]

        if "D" in args.infer_paras:
            self.para_infes += [case.D_infe_s / args.scales["D"], ]
            self.para_refes += [case.D, ]
            self.para_mathnames += ["$D$", ]
            self.para_textnames += ["D", ]
            self.para_units += ["m$^2$/s", ]

    def _cal_tke_spectrums(self):
        nx, ny, nt = self.n_x, self.n_y, self.n_t
        nk = ny // 2 + 1
        kx, ky = np.fft.fftfreq(nx, d=1. / nx), np.fft.fftfreq(ny, d=1. / ny)
        k2 = kx[:, None] ** 2 + ky[:] ** 2  # (nx, 1) + (nk, ) -> (nx, nk)
        kmod = np.sqrt(k2)
        dk = 2  # integer
        # k = np.arange(0, (nk - 1) * 2 ** 0.5 + dk, dk, dtype=np.float64)[1:]
        k = np.arange(0, int(kmod.max()) + dk, dk, dtype=np.float64)[1:]
        self.k = k
        
        Es_refe = []
        for i_t in range(nt):
            uu, vv = self.uuu_refe[:, :, i_t], self.vvv_refe[:, :, i_t]
            uu, vv = uu - np.mean(uu), vv - np.mean(vv)
            uhuh, vhvh = np.fft.fft2(uu), np.fft.fft2(vv)
            tke_spec = 0.5 * (np.abs(uhuh) ** 2 + np.abs(vhvh) ** 2)
            tke_spec_avg = tke_spec / (nx * ny)  # angle averaged TKE spectrum
            
            E = np.zeros_like(k, dtype=np.float64)
            #  binning energies with wave number modulus in threshold
            for i in range(len(k)):
                E[i] = np.sum(tke_spec_avg[(kmod >= k[i] - dk / 2) & (kmod < k[i] + dk / 2)]) / dk
                E[i] = E[i] / (nx * ny)  # so that (area under E-k curve) == (space-averaged TKE)
            Es_refe.append(E)
        self.Es_refe = np.array(Es_refe)
        
        Es_pred = []
        for i_t in range(nt):
            uu, vv = self.uuu_pred[:, :, i_t], self.vvv_pred[:, :, i_t]
            uu, vv = uu - np.mean(uu), vv - np.mean(vv)
            uhuh, vhvh = np.fft.fft2(uu), np.fft.fft2(vv)
            tke_spec = 0.5 * (np.abs(uhuh) ** 2 + np.abs(vhvh) ** 2)
            tke_spec_avg = tke_spec / (nx * ny)  # angle averaged TKE spectrum
            
            E = np.zeros_like(k, dtype=np.float64)
            #  binning energies with wave number modulus in threshold
            for i in range(len(k)):
                E[i] = np.sum(tke_spec_avg[(kmod >= k[i] - dk / 2) & (kmod < k[i] + dk / 2)]) / dk
                E[i] = E[i] / (nx * ny)  # so that (area under E-k curve) == (space-averaged TKE)
            Es_pred.append(E)
        self.Es_pred = np.array(Es_pred)

        output_dir = self.output_dir
        os.makedirs(output_dir + "data/", exist_ok=True)
        np.save(output_dir + "data/energy_spectrums_k.npy", self.k)
        np.save(output_dir + "data/energy_spectrums_refe.npy", self.Es_refe)
        np.save(output_dir + "data/energy_spectrums_pred.npy", self.Es_pred)
        
    def plot_tke_spectrums(self, n_moments=6, lws=(1.5, 2)):
        """Plot the turbulent energy spectrums at some moments."""
        nt = self.n_t
        output_dir = self.output_dir
        self._cal_tke_spectrums()

        for i_t in range(0, nt, nt // (n_moments - 1)):
            t_str = "{:.2f}".format(self.t[i_t])
            os.makedirs(output_dir + f"pics/{t_str}/", exist_ok=True)
            print(f"Plotting energy spectrum (t = {t_str} s)...")
            plt.figure(figsize=(8, 6))
            plt.title(f"$t$ = {t_str} s", fontsize="medium")
            plt.ylim(1e-15, 2.0)
            plt.xlabel(r"$k / {\rm m}^{-1}$")
            plt.ylabel(r"$E(k) / ({\rm m}^3 \, {\rm s}^{-2})$")
            plt.loglog(self.k, self.Es_refe[i_t], '-b', lw=lws[0], label="Reference")
            plt.loglog(self.k, self.Es_pred[i_t], '--r', lw=lws[1], label="PINN")
            plt.legend(fontsize="small")
            plt.savefig(output_dir + f"pics/{t_str}/energy_spectrum.png", bbox_inches="tight", dpi=set_dpi)
            plt.close()

    def plot_ke_t(self, lws=(1.5, 2)):
        output_dir = self.output_dir
        os.makedirs(output_dir + "data/", exist_ok=True)
        os.makedirs(output_dir + "pics/", exist_ok=True)

        Ks_refe = []
        for i_t in range(self.n_t):
            uu, vv = self.uuu_refe[:, :, i_t], self.vvv_refe[:, :, i_t]
            # uu, vv = uu - np.mean(uu), vv - np.mean(vv)
            K_phys = 0.5 * (uu ** 2 + vv ** 2)
            if self.args.variable_density:
                K_phys *= self.rhorhorho_pred[:, :, i_t]
            K = K_phys.mean()
            Ks_refe.append(K)
        Ks_refe = np.array(Ks_refe)

        Ks_pred = []
        for i_t in range(self.n_t):
            uu, vv = self.uuu_pred[:, :, i_t], self.vvv_pred[:, :, i_t]
            # uu, vv = uu - np.mean(uu), vv - np.mean(vv)
            K_phys = 0.5 * (uu ** 2 + vv ** 2)
            if self.args.variable_density:
                K_phys *= self.rhorhorho_refe[:, :, i_t]
            K = K_phys.mean()
            Ks_pred.append(K)
        Ks_pred = np.array(Ks_pred)

        np.save(output_dir + "data/t.npy", self.t)
        np.save(output_dir + "data/K_t_refe.npy", Ks_refe)
        np.save(output_dir + "data/K_t_pred.npy", Ks_pred)

        print("Plotting kinetic energy-t curve....")
        plt.figure(figsize=(8, 6))
        plt.xlabel("$t$/s")
        # plt.ylabel(r"$k / ({\rm m}^2 \, {\rm s}^{-2})$")
        # plt.ylabel(r"${\rm K} / ({\rm m}^2 \cdot {\rm s}^{-2})$")
        plt.ylabel(r"$K / ({\rm J} \cdot {\rm m}^{-3})$")
        plt.plot(self.t, Ks_refe, "b-", lw=lws[0], label="Reference")
        plt.plot(self.t, Ks_pred, "r--", lw=lws[1], label="PINN")
        plt.legend(fontsize="small")
        plt.savefig(output_dir + f"pics/K-t.png", bbox_inches="tight", dpi=set_dpi)
        plt.close()

    def plot_entropy_t(self, lws=(1.5, 2)):
        output_dir = self.output_dir
        os.makedirs(output_dir + "data/", exist_ok=True)
        os.makedirs(output_dir + "pics/", exist_ok=True)
        dx, dy = self.x[1] - self.x[0], self.y[1] - self.y[0]

        rhorhorho_refe = self.rhorhorho_refe if self.args.variable_density else 1.0
        rhorhorho_pred = self.rhorhorho_pred if self.args.variable_density else 1.0

        ccc = self.ccc_refe
        sss = np.zeros_like(ccc)
        sss[ccc > 0.0] = -ccc[ccc > 0.0] * np.log(ccc[ccc > 0.0])  # kg^(-1)
        S_refe = np.sum(sss * rhorhorho_refe, axis=(0, 1)) * (dx * dy * 1)  # Nondimensional

        ccc = self.ccc_pred
        sss = np.zeros_like(ccc)
        sss[ccc > 0.0] = -ccc[ccc > 0.0] * np.log(ccc[ccc > 0.0])  # kg^(-1)
        S_pred = np.sum(sss * rhorhorho_pred, axis=(0, 1)) * (dx * dy * 1)  # Nondimensional

        np.save(output_dir + "data/t.npy", self.t)
        np.save(output_dir + "data/S_t_refe.npy", S_refe)
        np.save(output_dir + "data/S_t_pred.npy", S_pred)

        print("Plotting entropy-t curve....")
        plt.figure(figsize=(8, 6))
        plt.xlabel("$t$/s")
        plt.ylabel("$S$")
        plt.plot(self.t, S_refe, "b-", lw=lws[0], label="Reference")
        plt.plot(self.t, S_pred, "r--", lw=lws[1], label="PINN")
        plt.legend(fontsize="small")
        plt.savefig(output_dir + f"pics/S-t.png", bbox_inches="tight", dpi=set_dpi)
        plt.close()

