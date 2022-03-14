#!/bin/bash
#set -e

readonly _DIR=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )
readonly PYTHON_DIR="${_DIR}/python"
readonly DL_DIR="${_DIR}/_downloads"
readonly RESULT_DIR="${_DIR}/_results"


function gen_requirements() {
        # write requirements in json
        (cd ${_DIR}
         python -m python.sa.main --run requirement
        )
}

function gen_templates() {
        # write templates in json
        (cd ${_DIR}
         # CUDA_VISIBLE_DEVICES=5,6 python -m python.sa.main --run template --search_dataset sst --syntax_selection random > /dev/null 2>&1
         # CUDA_VISIBLE_DEVICES=6,7 python -m python.sa.main --run template --search_dataset sst --syntax_selection bertscore > /dev/null 2>&1
         CUDA_VISIBLE_DEVICES=6,7 python -m python.sa.main --run template --search_dataset sst --syntax_selection noselect > /dev/null 2>&1
        )
}

function gen_testsuite() {
        # write test cases into Checklist Testsuite format
        (cd ${_DIR}
         # python -m python.sa.main --run testsuite --search_dataset sst --syntax_selection random > /dev/null 2>&1
         # python -m python.sa.main --run testsuite --search_dataset sst --syntax_selection bertscore > /dev/null 2>&1
         python -m python.sa.main --run testsuite --search_dataset sst --syntax_selection noselect > /dev/null 2>&1
        )
}

function eval_models() {
        # evaluate NLP models with generated testsuites
        (cd ${_DIR}
         # evaluating models on checklist testcases
         # CUDA_VISIBLE_DEVICES=6,7 python -m python.sa.main --run testmodel --test_baseline > /dev/null 2>&1

         # evaluating models on our generated testcases
         # CUDA_VISIBLE_DEVICES=6,7 python -m python.sa.main --run testmodel --syntax_selection random > /dev/null 2>&1
         # CUDA_VISIBLE_DEVICES=6,7 python -m python.sa.main --run testmodel --syntax_selection bertscore > /dev/null 2>&1
         CUDA_VISIBLE_DEVICES=6,7 time python -m python.sa.main --run testmodel --syntax_selection noselect > /dev/null 2>&1
        )
}

function retrain_models() {
        # evaluate NLP models with generated testsuites
        (cd ${_DIR}
         echo "***** TRAIN: ours, EVAL: checklist *****"
         mkdir -p ./_results/retrain/models/sa/sa_sst_PROB_textattack-bert-base-uncased-SST-2
         CUDA_VISIBLE_DEVICES=6,7 python -m python.sa.main --run retrain --search_dataset sst --syntax_selection prob --model_name textattack/bert-base-uncased-SST-2 > ./_results/retrain/models/sa/sa_sst_PROB_textattack-bert-base-uncased-SST-2/eval_testsuite_results.txt

         CUDA_VISIBLE_DEVICES=6,7 python -m python.sa.main --run retrain --testing_on_trainset --search_dataset sst --syntax_selection prob --model_name textattack/bert-base-uncased-SST-2 > ./_results/retrain/models/sa/sa_sst_PROB_textattack-bert-base-uncased-SST-2/eval_testsuite_results_on_trainset.txt

         echo "***** TRAIN: checklist, EVAL: ours *****"
         mkdir -p ./_results/retrain/models/sa/sa_checklist_textattack-bert-base-uncased-SST-2
         CUDA_VISIBLE_DEVICES=6,7 python -m python.sa.main --run retrain --search_dataset checklist --model_name textattack/bert-base-uncased-SST-2 > ./_results/retrain/models/sa/sa_checklist_textattack-bert-base-uncased-SST-2/eval_testsuite_results.txt

         CUDA_VISIBLE_DEVICES=6,7 python -m python.sa.main --run retrain --testing_on_trainset --search_dataset checklist --model_name textattack/bert-base-uncased-SST-2 > ./_results/retrain/models/sa/sa_checklist_textattack-bert-base-uncased-SST-2/eval_testsuite_results_on_trainset.txt
        )
}

function eval_retrained_models() {
        # evaluate NLP models with generated testsuites
        (cd ${_DIR}
         python -m python.sa.main --run testmodel --local_model_name checklist-textattack-bert-base-uncased-SST-2
        )
}

function analyze_eval_models() {
        # evaluate NLP models with generated testsuites
        (cd ${_DIR}
         python -m python.sa.main --run analyze --search_dataset sst --syntax_selection random
         python -m python.sa.main --run analyze --search_dataset sst --syntax_selection bertscore
         # python -m python.sa.main --run analyze --search_dataset sst --syntax_selection noselect
        )
}

function analyze_retrained_models() {
        # evaluate NLP models with generated testsuites
        (cd ${_DIR}
         python -m python.sa.main --run retrain_analyze --search_dataset sst --syntax_selection random --model_name textattack/bert-base-uncased-SST-2
        )
}

function explain_nlp() {
        mkdir -p ./_results/retrain/explain_nlp
        CUDA_VISIBLE_DEVICES=6,7 python -m python.sa.main --run explain_nlp --search_dataset sst --syntax_selection random --model_name textattack/bert-base-uncased-SST-2 > ./_results/retrain/explain_nlp/explain_nlp_result_sst_PROB_textattack_bert-base-uncased-SST-2.txt
}

function explain_nlp() {
        (cd ${_DIR}
         python -m python.sa.main --run selfbleu --search_dataset sst --syntax_selection random
        )
}
}

function main() {
        # gen_requirements # to generate test_type_sa.json and requirement_sa.json
        gen_templates # to generate templates_sa/seeds_{cksum}.json, templates_sa/templates_seed_{cksum}.json and templates_sa/templates_exp_{cksum}.json and cfg_expanded_inputs_sa.json
        gen_testsuite # to generate pkl checklist testsuite files in test_results directory
        eval_models # run testsuite.run on our and checklist generated testsets
        analyze_eval_models # to generate test_results_analysis.json by reading test_results.txt and cfg_expanded_inputs_{nlp_task}_{search_dataset_name}_{selection_method}.json
        # retrain_models # to retrain models and test the retrained models on testsuite.run on our and checklist generated testsets
        # eval_retrained_models # to ...?
        # analyze_retrained_models # to generate debug_results.json and debug_comparision file
        # explain_nlp # to run the explainNLP
}

# please make sure you actiavte nlptest conda environment
main
