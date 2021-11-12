#!/bin/bash

#set -e

readonly _DIR=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )
readonly PYTHON_DIR="${_DIR}/python"
readonly DL_DIR="${_DIR}/_downloads"
readonly RESULT_DIR="${_DIR}/_results"
readonly TASK=${1}; shift

function gen_templates() {
        (cd ${PYTHON_DIR}
         python -m Template
        )
}

function gen_testsuite() {
        (cd ${PYTHON_DIR}
         python -m Testsuite
        )
}

function eval_models(){
        (cd ${PYTHON_DIR}
         python -m Testmodel
        )
}

function main() {
        # gen_templates
        gen_testsuite
}

# please make sure you actiavte nlptest conda environment
main
