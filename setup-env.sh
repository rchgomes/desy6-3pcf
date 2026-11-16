#!/bin/bash
# Environment of the DES Y6 3PCF pipelines.  Usage:  source setup-env.sh
#
# The CosmoSIS configuration files refer to these two variables.

# this repository
export DESY6_3PCF_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# cosmosis-standard-library checkout (edit, or export before sourcing)
export COSMOSIS_STD_LIB="${COSMOSIS_STD_LIB:-/path/to/cosmosis-standard-library}"

if [ ! -d "$COSMOSIS_STD_LIB" ]; then
    echo "setup-env.sh: COSMOSIS_STD_LIB=$COSMOSIS_STD_LIB does not exist; set it to your cosmosis-standard-library" >&2
fi
