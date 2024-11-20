# HandoverOptimDRL
A framework for developing and evaluating adaptive handover algorithms using deep reinforcement learning.

> **Note**: *The code is currently being revised. A new version will be available soon.*

---

## Description
HandoverOptimDRL is a framework designed to facilitate developing and evaluating handover algorithms using deep reinforcement learning, i.e., proximal policy optimization (PPO).
It provides tools and environments to simulate the 3GPP handover protocol and train and evaluate a PPO-based handover protocol.

This repository contains the source code, data sets, and trained PPO model for the paper:
**A Deep Reinforcement Learning-based Approach for Adaptive Handover Protocols**, see reference below.

---

## **Installation**
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

## **Getting Started**
### **Run Protocol Validation**
You can validate the PPO-based and 3GPP handover protocols using the `run.py` file:

1. **Validate PPO-based Protocol**:
   ```bash
   python run.py validate_ppo
   ```

2. **Validate 3GPP Protocol**:
   ```bash
   python run.py validate_3gpp
   ```

## **Citation**
If you use **HandoverOptimDRL** in your research, please cite the accompanying paper:

```
@inproceedings{handoveroptimdrl,
  title={A Deep Reinforcement Learning-based Approach for Adaptive Handover Protocols},
  author={Johannes Voigt and Peter J. Gu and Peter M. Rost},
  year={2024},
  organization={KIT - Karlsruhe Institute of Technology},
}
```

---

## **License**
This project is licensed under the MIT License. See the [LICENSE](./LICENSE) file for details.
