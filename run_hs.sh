#!/bin/bash
#set -e

readonly _DIR=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )
readonly PYTHON_DIR="${_DIR}/python"
readonly DL_DIR="${_DIR}/_downloads"
readonly RESULT_DIR="${_DIR}/_results"

# ==========
# Exp Results

function gen_requirements() {
        # write requirements in json
        (cd ${_DIR}
         python -m python.hs.main --run requirement
        )
}

function gen_templates() {
        # write templates in json
        (cd ${_DIR}
         CUDA_VISIBLE_DEVICES=5,6 python -m python.hs.main \
                             --run template --search_dataset sst \
                             --syntax_selection random > /dev/null 2>&1
        )
}


# ==========
# Main

function main() {
        gen_requirements # to generate test_type_hs.json and requirement_hs.json
        # gen_templates # to generate templates_sa/seeds_{cksum}.json, templates_sa/templates_seed_{cksum}.json and templates_sa/templates_exp_{cksum}.json and cfg_expanded_inputs_sa.json
}

# please make sure you actiavte nlptest conda environment
main
