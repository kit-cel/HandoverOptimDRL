![Version](https://img.shields.io/github/v/tag/kit-cel/HandoverOptimDRL?label=version) ![pylint](https://img.shields.io/badge/PyLint-9.63-yellow?logo=python&logoColor=white)


# HandoverOptimDRL
A framework for developing and evaluating adaptive handover algorithms using deep reinforcement learning (PPO).

---

## Description
HandoverOptimDRL is a framework designed to facilitate developing and evaluating handover algorithms using deep reinforcement learning, i.e., proximal policy optimization (PPO).
It provides tools and environments to simulate the 3GPP handover protocol and to train and evaluate a PPO-based handover protocol.

This repository contains the source code, datasets, and trained PPO model for the paper:
**A Deep Reinforcement Learning-based Approach for Adaptive Handover Protocols**, see reference [1].

---

## Installation
To install the HandoverOptimDRL package, follow these steps:

1. Clone the repository:
    ```bash
    git clone https://github.com/kit-cel/HandoverOptimDRL
    ```

2. Navigate to the project directory:
    ```bash
    cd HandoverOptimDRL
    ```

3. Install the package:
    ```bash
    python -m pip install .
    ```
    i.e., to install it in editable mode/develop mode:
    ```bash
    python -m pip install -e .
    ```

You are now ready to use the HandoverOptimDRL framework for your projects.

---

## Getting Started
### **Generate New Datasets**
1. **Generate GPX Traces**:
    By using the open source software [Simulation of Urban MObility](https://eclipse.dev/sumo/), you can generate new user traces:
   ```bash
   sumo-gui -n map.net.xml -r routes.rou.xml --additional-files vehicles.rou.xml, map_buildings.poly.xml
   ```
   The GPX traces can be extracted using the included Python script:
   ```bash
   python -m gpx_extraction
   ```

2. **Generate RSRP and SINR Traces**:
    New RSRP and SINR traces can be generated using the open source [Vienna 5G System Level Simulator](https://www.tuwien.at/etit/tc/vienna-simulators/vienna-5g-simulators/) developed and provided by the TU Wien. The corresponding launcher and scenario files for our simulations can be found in the matlab folder of this repository.
    To run the simulation, first adapt the launcher files based on the SUMO GPX files, and then start the `runSimulation.m` script.


### Train a New PPO Agent
1. **Train a PPO agent**:
   ```bash
   python -m run train_ppo
   ```

### Run Protocol Validation
You can validate the PPO-based and 3GPP handover protocols using the `run.py` file:

1. **Validate the PPO-based protocol**:
   ```bash
   python -m run validate_ppo
   ```

2. **Validate 3GPP protocol**:
   ```bash
   python -m run validate_3gpp
   ```

## Citation [1]
If you use **HandoverOptimDRL** in your work, please cite our paper ([full-text on IEEE Xplore](https://ieeexplore.ieee.org/document/10949111)):
```
@INPROCEEDINGS{10949111,
  author={Voigt, Johannes and Gu, Peter J. and Rost, Peter M.},
  booktitle={2025 14th International ITG Conference on Systems, Communications and Coding (SCC)}, 
  title={A Deep Reinforcement Learning-Based Approach for Adaptive Handover Protocols}, 
  year={2025},
  pages={1-6},
  keywords={Cellular networks;Base stations;Protocols;5G mobile communication;Source coding;Handover;Radio links;Timing;Reliability;Research and development;Handover;Communication Protocols;Mobility Management;Deep Reinforcement Learning},
  doi={10.1109/IEEECONF62907.2025.10949111}
}
```

---

## **License**
This project is licensed under the MIT License. See the [LICENSE](./LICENSE) file for details.
