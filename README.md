# Differentiation tool to optimize  Lennard-Jones parameters from experimental isotherms.

Here we use automatic differentiation technique to optimize force fields for gas adsorption. In the paper, we mainly develop the refined force field for $CO_2$ adsorption within MOFs, which also works for $N_2$ and $CH_4$. More details can be found in the paper. 

## Two open-source packages are used in the tool:

[DMFF](https://github.com/deepmodeling/DMFF): It is an open-source automatic differentiable package for force field development.

[Aiida-LSMO](https://github.com/lsmo-epfl/aiida-lsmo): Aiida-LSMO workflow includes structure optimization, binding energy calculation, isotherm generation. And the tool mainly uses isotherm generation part to sample and the trajectories are used to develop FFs in DMFF. The intallation tutorial can be found in https://github.com/mpougin/aiida2.x-lsmo-setup.

## In-house modification:


[Modification in AiiDa-LSMO workflow](https://github.com/legend-L24/ff_optimizer/tree/main/sampler/calcfunctions)

I add "ff_optim" option to the Aiida-LSMO workflow, which allows self-defined FFs for simulation. In pratice, I add some lines in ff_builder_module.py, isotherm.py, simulation_annealing.py within calculation folder. The modified code can be found in calulation folder. 

[Execution scripts for AiiDa-LSMO](https://github.com/legend-L24/ff_optimizer/tree/main/sampler/applications)

[Execution scripts for DMFF](https://github.com/legend-L24/ff_optimizer/tree/main/optimizer/UFF_opt)

[Automated scripts in the tool](https://github.com/legend-L24/ff_optimizer/tree/main/sampler)

## Notes:
Before you use the tools to optimize force fields, you need to install DMFF and Aiida-LSMO firstly. If you meet errors in the installation of Aiida-LSMO, you can give up installing some packages as long as the isotherm generation workflows still work. The two packages can be install in two directories. The working directories of DMFF and Aiida-LSMO need to be defined as global variable for the tools.




