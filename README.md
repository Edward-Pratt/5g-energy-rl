# ECM3401 5G Energy Simulation

This repository is a monorepo containing Simu5G extensions, gym bindings,
and analysis scripts.

Dependencies (not included in this repo):
- OMNeT++ 6.3.0 (install separately)
- INET v4.5.4 fork: https://github.com/Edward-Pratt/inet

Setup outline:
1. Install OMNeT++ 6.3.0 and set up its environment as usual.
2. Clone the INET fork into `inet/` at the repo root:
   `git clone https://github.com/Edward-Pratt/inet inet`
3. Build Simu5G/Simu5G-Gym as needed for your workflow.
