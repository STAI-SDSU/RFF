# RFF
This repository contains code for RF

Note:
There are 3 main files since each environment has different evaluation functions and supporting code:
- The ant-dir unlearning uses main_ant.py
- The cheetah-dir/cheetah-vel unlearning uses main_multi.py
- The single task agent unlearning uses main_single.py

Supporting unlearning loops/diffusion code can be found in ./agents/:
- ql_diffusion.py contains all the different types of unlearning loops.
- diffusion.py contains the added functions to calculate loss per timestep of the diffusion process.
