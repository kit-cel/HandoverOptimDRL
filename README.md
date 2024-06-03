# HandoverOptimDRL
Framework for learning handover algorithms using deep reinforcement learning. This repo provides the codebase and results for [1].

## Usage
- The routes used for the Vienna 5G Simulator can be found in the folder `routes` as .csv files. 
- The database used for the simulation is prestored in `data_from_vienna`. Use `karlsruheScenario.m` in the Vienna 5G Simulator to configure your own scenario.
- Take these files to start the Python script `main.py`. Do not forget to create an output folder and adjust the file paths .
- The folder structure is: `results/sweep_name/insl_num/run_name`, insl_num is the name of the simulation server, run_name is an arbitrary name.

---
[1] Peter J. Gu, Johannes Voigt and Peter M. Rost, "A Deep Reinforcement Learning-based Approach for Adaptive Handover Protocols in Mobile Networks"
