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

## Prerequisites

Before setting up the project, ensure your system meets the following requirements:

- **Operating System**: Linux (Ubuntu 20.04+, Fedora, etc.), macOS (10.15+), or Windows with WSL2 (Windows Subsystem for Linux).
- **Hardware**: Multi-core CPU (4+ cores recommended), at least 8GB RAM, sufficient disk space (10GB+ for OMNeT++ and frameworks).
- **Software Dependencies**:
  - GCC/Clang with C++17 support (e.g., `g++-9` or later).
  - Python 3.10 or newer.
  - CMake 3.16 or newer.
  - GNU Make.
  - Protobuf compiler (`protoc`).
  - Git for cloning repositories.
  - Package managers: `apt` (Ubuntu/Debian), `dnf` (Fedora), `brew` (macOS), or equivalents.
- **Optional but Recommended**: `uv` for Python dependency management, `ninja` for faster builds.

Install system dependencies (example for Ubuntu):
```bash
sudo apt update
sudo apt install build-essential cmake ninja-build python3 python3-pip git libprotobuf-dev protobuf-compiler libavcodec-dev libavformat-dev libswscale-dev
```

## Cloning the Repository

Clone the project repository and navigate to the directory:

```bash
git clone <repository-url>  # Replace <repository-url> with the actual Git repository URL
cd <repository-directory>    # Replace <repository-directory> with the cloned directory name
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

### Installing OMNeT++ 6.3.0

1. Download OMNeT++ 6.3.0 from the official website: https://omnetpp.org/download/
2. Extract the tarball into the project root directory (next to this README):
   ```bash
   tar -xzf omnetpp-6.3.0-src.tgz  # Adjust filename as needed
   cd omnetpp-6.3.0
   ```
3. Install OMNeT++:
   ```bash
   ./install.sh  # For Linux/macOS, or follow manual steps below
   ```
   Manual installation (if install.sh fails):
   ```bash
   source setenv
   ./configure
   make -j$(nproc)
   ```
4. Verify installation:
   ```bash
   source setenv
   omnetpp --version
   ```

### Installing INET Framework v4.5.4

1. Download INET v4.5.4 from https://inet.omnetpp.org/Download.html
2. Extract the tarball into the project root directory:
   ```bash
   tar -xzf inet-4.5.4-src.tgz  # Adjust filename
   cd inet
   ```
3. Set up environment and build:
   ```bash
   source ../omnetpp-6.3.0/setenv  # Ensure OMNeT++ is sourced
   source setenv
   pip install -r python/requirements.txt  # Install Python dependencies
   make makefiles
   make -j$(nproc)
   ```
4. Verify by running an example:
   ```bash
   cd examples/inet/routing
   ./run
   ```

### Installing Simu5G

1. Download Simu5G from http://simu5g.org/download/ (ensure version compatible with INET 4.5)
2. Extract the tarball into the project root directory:
   ```bash
   tar -xzf simu5g-x.y.z-src.tgz  # Adjust filename
   cd Simu5G
   ```
3. Set up environment and build:
   ```bash
   source ../omnetpp-6.3.0/setenv
   source ../inet/setenv
   source setenv
   make makefiles
   make -j$(nproc)
   ```
4. Run tests to verify:
   ```bash
   make tests
   ```

### Installing Simu5G-Gym

Simu5G-Gym is included in the repository under `Simu5G-Gym/`. It will be built with the CMake process below.

### Installing Dependencies

- **Python Dependencies**: Handled in the Python Environment section below.
- **System Libraries**: Already covered in Prerequisites. Additional libs may be needed for video processing if using certain Simu5G features.

### 1. OMNeT++ Environment
Ensure you have the OMNeT++ environment variables set up:
```bash
source omnetpp-6.3.0/setenv
```

### 2. Build C++ Components
Use CMake to build the entire suite (INET, Simu5G, Simu5G-Gym):
```bash
# Configure the build
cmake -S . -B cmake-build-release -DCMAKE_BUILD_TYPE=Release

# Build all components
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

## Verification Steps

After installation and build, verify the setup:

1. **OMNeT++**: Run a sample simulation.
2. **INET**: Run an INET example as above.
3. **Simu5G**: Run Simu5G tests or an example simulation.
4. **Full Project**: Run a simple PPO evaluation:
   ```bash
   cd Simu5G-python
   ./run_ppo_multi.sh --episodes 1
   ```
   Check for successful completion without errors.

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

To set up the environment for development and running simulations, source the following scripts in order:

```bash
source omnetpp-6.3.0/setenv
source inet/setenv
source Simu5G/setenv
```

This sets the necessary paths and environment variables. The CMake build system uses the following defaults:

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