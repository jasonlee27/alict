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
         CUDA_VISIBLE_DEVICES=1,2 python -m python.hs.main \
                             --run template \
                             --search_dataset hatexplain \
                             --syntax_selection random \
                             --num_seeds 100 \
                             --num_trials 1 # > /dev/null 2>&1
        #  CUDA_VISIBLE_DEVICES=1,2 python -m python.hs.main \
        #                      --run template \
        #                      --search_dataset hatecheck \
        #                      --syntax_selection random # > /dev/null 2>&1
        )
}

function gen_testsuite() {
        # write test cases into hatexplain Testsuite format
        (cd ${_DIR}
         python -m python.hs.main \
                --run testsuite \
                --search_dataset hatexplain \
                --syntax_selection random \
                --num_seeds 100 \
                --num_trials 1
         # python -m python.hs.main \
         #        --run testsuite \
         #        --search_dataset hatecheck \
         #        --syntax_selection random
        )
}

function eval_models() {
        # evaluate NLP models with generated testsuites
        (cd ${_DIR}
         # evaluating models on  our generated(hatexplain) testcases
         # CUDA_VISIBLE_DEVICES=2,3 python -m python.hs.main \
         #                     --run testmodel \
         #                     --search_dataset hatexplain \
         #                     --syntax_selection random \
         #                     --num_seeds 100 \
         #                     --num_trials 1
         # evaluating models on hatecheck testcases (not expanded) as baseline
         CUDA_VISIBLE_DEVICES=2,3 python -m python.hs.main \
                             --run testmodel \
                             --search_dataset hatexplain \
                             --syntax_selection random \
                             --num_seeds 100 \
                             --num_trials 1 \
                             --test_baseline
        )
}

function analyze_eval_models() {
        (cd ${_DIR}
         # evaluate NLP models with hatexplain expanded testsuites
         python -m python.hs.main \
                --run analyze \
                --search_dataset hatexplain \
                --syntax_selection random \
                --num_seeds 100 \
                --num_trials 1
         # python -m python.hs.main \
         #        --run analyze \
         #        --search_dataset hatexplain \
         #        --syntax_selection random \
         #        --test_baseline
        )
}


# ==========
# Main

function main() {
        # gen_requirements # to generate test_type_hs.json and requirement_hs.json
        # gen_templates # to generate templates_sa/seeds_{cksum}.json, templates_sa/templates_seed_{cksum}.json and templates_sa/templates_exp_{cksum}.json and cfg_expanded_inputs_sa.json
        # gen_testsuite
        eval_models
        # analyze_eval_models
}

# Please make sure you actiavte nlptest conda environment
main
