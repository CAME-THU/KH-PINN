import torch
import math as mt
from utils.networks import FNN, ModifiedMLP
# import deepxde as dde
# import torch.nn.functional as F


class Maps:
    def __init__(self, args, case):
        self.args = args
        self.case = case
        
        d_out = 5 if args.variable_density else 4
        
        self.net = ModifiedMLP(
        # self.net = FNN(
        #     layer_sizes=[3] + 6 * [64] + [d_out],
            layer_sizes=[3] + 5 * [128] + [d_out],
            activation="sin",  # "tanh", "sin"
            kernel_initializer="Glorot normal",
            weight_fact={"mean": 0.5, "std": 0.1},
            input_transform=self.input_transform,
        )

        if args.variable_density:
            self.net.apply_output_transform(self.output_denorm_transform_vd)
        else:
            self.net.apply_output_transform(self.output_denorm_transform)

    def input_transform(self, xyt):
        x, y, t = xyt[:, 0:1], xyt[:, 1:2], xyt[:, 2:3]
        # xs = (x + self.args.shifts["x"]) * self.args.scales["x"]
        # ys = (t + self.args.shifts["y"]) * self.args.scales["y"]
        ts = (t + self.args.shifts["t"]) * self.args.scales["t"]
        pi = mt.pi

        inputs = torch.cat([
            torch.sin(4 * pi * x * 1), torch.cos(4 * pi * x * 1),
            torch.sin(4 * pi * x * 2), torch.cos(4 * pi * x * 2),
            torch.sin(4 * pi * x * 3), torch.cos(4 * pi * x * 3),
            torch.sin(4 * pi * x * 4), torch.cos(4 * pi * x * 4),

            torch.sin(2 * pi * y * 1), torch.cos(2 * pi * y * 1),
            torch.sin(2 * pi * y * 2), torch.cos(2 * pi * y * 2),
            torch.sin(2 * pi * y * 3), torch.cos(2 * pi * y * 3),
            torch.sin(2 * pi * y * 4), torch.cos(2 * pi * y * 4),

            ts,
            # 2 * ts, 3 * ts, 4 * ts,
            torch.sin(1 * ts), torch.cos(1 * ts),
            torch.sin(2 * ts), torch.cos(2 * ts),
            torch.sin(3 * ts), torch.cos(3 * ts),
            torch.sin(4 * ts), torch.cos(4 * ts),
            # 5 * ts, 6 * ts, 7 * ts, 8 * ts,
            # 9 * ts, 10 * ts,
            # 100 * ts
            ], dim=1)

        return inputs

    def output_denorm_transform(self, xyt, uvpc_s):
        # x, y, t = xyt[:, 0:1], xyt[:, 1:2], xyt[:, 2:3]
        us, vs, ps, cs = uvpc_s[:, 0:1], uvpc_s[:, 1:2], uvpc_s[:, 2:3], uvpc_s[:, 3:4]
        u = us / self.args.scales["u"]
        v = vs / self.args.scales["v"]
        p = ps / self.args.scales["p"]
        # c = cs / self.args.scales["c"]
        c = torch.sigmoid(cs)
        return torch.cat([u, v, p, c], dim=1)

    def output_denorm_transform_vd(self, xyt, ruvpc_s):
        # x, y, t = xyt[:, 0:1], xyt[:, 1:2], xyt[:, 2:3]
        rhos, us, vs, ps, cs = ruvpc_s[:, 0:1], ruvpc_s[:, 1:2], ruvpc_s[:, 2:3], ruvpc_s[:, 3:4], ruvpc_s[:, 4:5]
        # rho = rhos / self.args.scales["rho"]
        rho = torch.sigmoid(rhos) + 1.0
        u = us / self.args.scales["u"]
        v = vs / self.args.scales["v"]
        p = ps / self.args.scales["p"]
        # c = cs / self.args.scales["c"]
        c = torch.sigmoid(cs)
        return torch.cat([rho, u, v, p, c], dim=1)
