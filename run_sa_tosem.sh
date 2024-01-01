#!/bin/bash
#set -e

readonly _DIR=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )
readonly PYTHON_DIR="${_DIR}/python"
readonly DL_DIR="${_DIR}/_downloads"
readonly RESULT_DIR="${_DIR}/_results"


# ==========
# Human study

function func_humanstudy() {
        (cd ${_DIR}
         python -m python.sa.main \
                --run humanstudy_tosem \
                --search_dataset sst \
                --syntax_selection random
        )
}
# ==========


function func_gen_testsuite() {
        # write test cases into Checklist Testsuite format
        (cd ${_DIR}
         python -m python.sa.main \
                --run testsuite_tosem \
                --search_dataset sst \
                --syntax_selection random \
                --num_seeds -1 \
                --num_trials 1
        #  python -m python.sa.main \
        #         --run testmodel \
        #         --test_baseline
        )
}

function func_testmodel_chatgpt() {
        (cd ${_DIR}
        #  python -m python.sa.main \
        #         --run testmodel_tosem \
        #         --search_dataset sst \
        #         --syntax_selection random
         python -m python.sa.main \
                --run testmodel_tosem \
                --search_dataset sst \
                --syntax_selection random \
                --test_baseline
        )
}

function func_analyze_eval_models_chatgpt() {
        (cd ${_DIR}
         # evaluate NLP models with generated testsuites
        #  CUDA_VISIBLE_DEVICES=1,2,3 python -m python.sa.main \
        #         --run analyze_tosem \
        #         --search_dataset sst \
        #         --syntax_selection random \
        #         --num_seeds -1 \
        #         --num_trials 1
         CUDA_VISIBLE_DEVICES=1,2,3 python -m python.sa.main \
                --run analyze_tosem \
                --search_dataset sst \
                --syntax_selection random \
                --num_seeds -1 \
                --num_trials 1 \
                --test_baseline
        )
}

# ==========
# Linguistic capability of Fairness 
function func_fairness() {
        (cd ${_DIR}
         # evaluate NLP models with generated testsuites
        #  python -m python.sa.main \
        #         --run template_fairness \
        #         --search_dataset sst \
        #         --syntax_selection random \
        #         --num_seeds -1 \
        #         --num_trials 1
         python -m python.sa.main \
                --run testsuite_fairness \
                --search_dataset sst \
                --syntax_selection random \
                --num_seeds -1 \
                --num_trials 1
        )
}

# ==========
# Main

function main() {
        # func_humanstudy # sample sentences for manual study
        # func_gen_testsuite
        # func_testmodel_chatgpt # running chatgpt on the testcases
        # func_analyze_eval_models_chatgpt
        func_fairness
}




# please make sure you actiavte nlptest conda environment
main
