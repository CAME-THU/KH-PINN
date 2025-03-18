# KH-PINN
Physics-informed neural networks for the inverse problems (field reconstruction and parameter inference) of Kelvin-Helmholtz instability (KHI) flows.

Codes for the paper: [*Physics-informed neural networks for Kelvin-Helmholtz instability with spatiotemporal and magnitude multiscale*](https://doi.org/10.1063/5.0251167).

Preprint version: [*https://arxiv.org/abs/2411.07524*](https://arxiv.org/abs/2411.07524).

Some DNS data can be downloaded from the supplementary materials of the paper.


## Framework

Variable density:

![framework](schematic_variable-density.png)

Constant density:

![framework](schematic_constant-density.png)


## Typical results

Variable density, *Re* = 10000
![results](/movie_results/drho1.0_Re1e4/animationComp1_rho.gif)

Variable density, *Re* = 10000
![results](/movie_results/drho1.0_Re1e4/animationComp6_omega.gif)

Constant density, *Re* = 10000
![results](/movie_results/drho0.0_Re1e4/animationComp4_c.gif)

Constant density, *Re* = 10000
![results](/movie_results/drho0.0_Re1e4/animationComp5_omega.gif)
