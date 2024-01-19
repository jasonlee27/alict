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

function func_humanstudy_result() {
        (cd ${_DIR}
         python -m python.sa.main \
                --run humanstudy_results_tosem \
                --search_dataset sst \
                --syntax_selection random
        )
}
# ==========


function func_gen_testsuite() {
        # write test cases into Checklist Testsuite format
        (cd ${_DIR}
         CUDA_VISIBLE_DEVICES=1,2,3,4,5,6 python -m python.sa.main \
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
function func_testsuite_fairness() {
        (cd ${_DIR}
         # evaluate NLP models with generated testsuites
        #  CUDA_VISIBLE_DEVICES=0,1,2,3 python -m python.sa.main \
        #         --run template_fairness \
        #         --search_dataset sst \
        #         --syntax_selection random \
        #         --num_seeds -1 \
        #         --num_trials 1 \
        #         --gpu_ids 0 1 2 3
         CUDA_VISIBLE_DEVICES=0,1,2,3 python -m python.sa.main \
                --run testsuite_fairness \
                --search_dataset sst \
                --syntax_selection random \
                --num_seeds -1 \
                --num_trials 1
        )
}

function func_eval_models_fairness {
        (cd ${_DIR}
        #  # evaluating models on checklist testcases
        #  python -m python.sa.main \
        #         --run testmodel_fairness \
        #         --search_dataset sst \
        #         --syntax_selection random \
        #         --num_seeds -1 \
        #         --num_trials 1 \
        #         --test_baseline

        # # evaluating models on our generated testcases
        #  CUDA_VISIBLE_DEVICES=5 python -m python.sa.main \
        #                      --run testmodel_fairness \
        #                      --search_dataset sst \
        #                      --syntax_selection random \
        #                      --num_seeds -1 \
        #                      --num_trials 1

        #  # analyze performance of NLP models
        #  python -m python.sa.main \
        #         --run analyze_fairness \
        #         --search_dataset sst \
        #         --syntax_selection random \
        #         --num_seeds -1 \
        #         --num_trials 1

        python -m python.sa.main \
                --run analyze_fairness \
                --search_dataset sst \
                --syntax_selection random \
                --test_baseline
        )
}

# ==========
# Main

function main() {
        # func_humanstudy # sample sentences for manual study
        func_humanstudy_result
        # func_gen_testsuite
        # func_testmodel_chatgpt # running chatgpt on the testcases
        # func_analyze_eval_moødels_chatgpt
        # func_testsuite_fairness
        # func_eval_models_fairness
}




# please make sure you actiavte nlptest conda environment
main
