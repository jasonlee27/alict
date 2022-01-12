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
        (cd ${_DIR}
         CUDA_VISIBLE_DEVICES=1,2,3,4 python -m python.main --run template --search_dataset sst
        )
}

function gen_testsuite() {
        # write test cases into Checklist Testsuite format
        (cd ${_DIR}
         python -m python.main --run testsuite --search_dataset sst
        )
}

function eval_models(){
        # evaluate NLP models with generated testsuites
        (cd ${_DIR}
         # python -m python.main --run testmodel --test_baseline # evaluating models on checklist testcases
         python -m python.main --run testmodel # evaluating models on our generated testcases
        )
}

function retrain_models(){
        # evaluate NLP models with generated testsuites
        (cd ${_DIR}
         CUDA_VISIBLE_DEVICES=1,2,3,4 python -m python.main --run retrain --search_dataset checklist --model_name textattack/bert-base-uncased-SST-2
        )
}

function eval_retrained_models(){
        # evaluate NLP models with generated testsuites
        (cd ${_DIR}
         python -m python.main --run testmodel --local_model_name checklist-textattack-bert-base-uncased-SST-2
        )
}

function main() {
        # gen_requirements # to generate test_type_sa.json and requirement_sa.json
        # gen_templates # to generate templates_sa/seeds_{cksum}.json, templates_sa/templates_seed_{cksum}.json and templates_sa/templates_exp_{cksum}.json and cfg_expanded_inputs_sa.json
        gen_testsuite # to generate pkl checklist testsuite files in test_results directory
        # eval_models
        # retrain_models
        # eval_retrained_models
}

# please make sure you actiavte nlptest conda environment
main
