#!/bin/bash

#set -e

readonly _DIR=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )
readonly PYTHON_DIR="${_DIR}/python"
readonly DL_DIR="${_DIR}/_downloads"
readonly RESULT_DIR="${_DIR}/_results"
readonly TASK=${1}; shift

function run_generator() {
        (cd ${PYTHON_DIR}
         python -m Generator
        )
}

function main() {

        # local task="${TASK}"
        local task="gen"
        
        if [ "${task}" == "gen" ]; then
                time run_generator
        fi
}

# please make sure you actiavte nlptest conda environment
main
