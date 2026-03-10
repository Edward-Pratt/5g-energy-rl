# ECM3401 5G Energy Simulation

This repository is a monorepo for 5G network simulations focusing on energy efficiency and reinforcement learning. It integrates Simu5G with an OpenAI Gym-compatible interface for training and evaluating RL agents (PPO).

## Overview

The project aims to optimize energy consumption in 5G GNodeBs using Reinforcement Learning. It consists of a C++ simulation environment based on OMNeT++ and a Python-based RL training suite.

### Key Components

- **OMNeT++ 6.3.0**: The core discrete event simulation framework.
- **INET Framework v4.5.4**: Provides network protocols and models for OMNeT++.
- **Simu5G**: A 5G simulation library for OMNeT++.
- **Simu5G-Gym**: A C++ extension providing a ZMQ/Protobuf-based Gym interface for Simu5G.
- **Simu5G-python**: Python scripts for PPO training (using PyTorch), baseline evaluation, and results analysis.

## Project Structure

```text
.
├── omnetpp-6.3.0/          # OMNeT++ 6.3.0 installation
├── inet/                    # INET Framework v4.5.4
├── Simu5G/                  # Simu5G library
├── Simu5G-Gym/              # Simu5G-Gym extension (C++)
│   ├── src/                 # C++ source code for Gym connection
│   ├── ned/                 # NED files for Gym components
│   └── simulations/         # Simulation scenarios (StandaloneGym)
├── Simu5G-python/           # RL agents and analysis (Python)
│   ├── serverppo.py         # PPO training/evaluation server
│   ├── plot_rollouts.py     # Plotting and comparison scripts
│   ├── run_ppo_multi.sh     # Script to run PPO episodes
│   └── run_fixed_multi.sh   # Script to run fixed baseline episodes
└── CMakeLists.txt           # Main build configuration
```

## Requirements

- **Operating System**: Linux or WSL (Windows Subsystem for Linux).
- **C++**: GCC/Clang with C++17 support.
- **Python**: Python 3.10 or newer.
- **Package Managers**: 
  - `uv` (recommended) or `pip` for Python dependencies.
  - `cmake` and `make` for C++ components.
- **Other**: Protobuf compiler (`protoc`).

## Setup & Installation

### 1. OMNeT++ Environment
Ensure you have the OMNeT++ environment variables set up:
```bash
source omnetpp-6.3.0/setenv
```

### 2. Build C++ Components
Use CMake to build the entire suite (INET, Simu5G, Simu5G-Gym):
```bash
# Using the provided CMake profiles
cmake --build cmake-build-release --target all_release
```
Alternatively, build individual components:
```bash
cmake --build cmake-build-release --target simu5g_gym
```

### 3. Python Environment
Navigate to the `Simu5G-python` directory and install dependencies:
```bash
cd Simu5G-python
uv sync  # Recommended
# OR: pip install -r requirements.txt
```

## Running Simulations

All main execution scripts are located in `Simu5G-python/`.

### PPO Training and Evaluation
To run PPO evaluation for 10 episodes:
```bash
cd Simu5G-python
./run_ppo_multi.sh --episodes 10
```
To continue training for 100 episodes:
```bash
./run_ppo_multi.sh --mode train --episodes 100
```

### Fixed Baseline Evaluation
To run evaluation for fixed actions (0, 1, 2) for 5 episodes each:
```bash
cd Simu5G-python
./run_fixed_multi.sh --episodes 5
```

### Plotting Results
Compare different rollout results and generate plots:
```bash
cd Simu5G-python
python3 plot_rollouts.py --files eval_ppo.csv fixed_0.csv --labels "PPO" "Fixed 0"
```

## Environment Variables

- `OMNETPP_ROOT`: Path to OMNeT++ installation (default: `./omnetpp-6.3.0`).
- `INET_ROOT`: Path to INET framework (default: `./inet`).
- `SIMU5G_ROOT`: Path to Simu5G library (default: `./Simu5G`).
- `SIMU5G_GYM_ROOT`: Path to Simu5G-Gym extension (default: `./Simu5G-Gym`).
- `NEDPATH`: Search path for NED files (automatically configured in `run_*.sh` scripts).

## Tests

Run Simu5G internal tests:
```bash
cmake --build cmake-build-release --target simu5g_tests
```

## TODOs
- [ ] Add explicit license for the top-level repository.
- [ ] Document more complex multi-GNodeB scenarios if applicable.
- [ ] Add CI/CD pipeline for automated testing and simulation verification.

## License

- **Simu5G**: Licensed under [LGPL-3.0](Simu5G/LICENSE.md).
- **OMNeT++**: Subject to the [Academic Public License](https://omnetpp.org/intro/license).
- **INET**: Licensed under LGPL.
- **Overall Project**: TODO: Define repository-wide license.

---
Created for ECM3401 project.
