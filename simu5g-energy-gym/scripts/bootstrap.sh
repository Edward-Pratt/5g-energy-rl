#!/usr/bin/env bash
set -euo pipefail

# 1) clone deps (choose the exact versions/commits you want)
# git clone <SIMU5G_URL> deps/Simu5G
# git clone <INET_URL> deps/inet

# 2) apply your patch
# (cd deps/Simu5G && git apply ../../patches/simu5g.patch)

# 3) build steps (document your OMNeT++ env expectations here)
echo "Done. Now build Simu5G/INET, then Simu5G-Gym, then run python server + simulation."
