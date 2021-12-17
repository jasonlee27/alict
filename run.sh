#!/bin/bash

#set -e

readonly _DIR=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )
readonly PYTHON_DIR="${_DIR}/python"
readonly DL_DIR="${_DIR}/_downloads"
readonly RESULT_DIR="${_DIR}/_results"


function gen_requirements() {
        # write requirements in json
        (cd ${_DIR}
         python -m python.main --run requirement
        )
}

function gen_templates() {
        # write templates in json
        (cd ${PYTHON_DIR}
         python -m Template
        )
}

function gen_testsuite() {
        # write test cases into Checklist Testsuite format
        (cd ${PYTHON_DIR}
         python -m python.main --run testsuite
        )
}

function eval_models(){
        # evaluate NLP models with generated testsuites
        (cd ${PYTHON_DIR}
         python -m python.main --run testmodel
        )
}

function retrain_models(){
        # evaluate NLP models with generated testsuites
        (cd ${_DIR}
         python -m python.main --run retrain
        )
}

function main() {
        # gen_requirements
        # gen_templates
        # gen_testsuite
        retrain_models
}

# please make sure you actiavte nlptest conda environment
main
