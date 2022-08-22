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
         python -m python.sa.main --run requirement
        )
}

function gen_templates() {
        # write templates in json
        (cd ${_DIR}
         # generate templates from sst dataset
         CUDA_VISIBLE_DEVICES=3,4,5 python -m python.sa.main \
                             --run template \
                             --search_dataset sst \
                             --syntax_selection random # > /dev/null 2>&1
         # CUDA_VISIBLE_DEVICES=6,7 python -m python.sa.main --run template --search_dataset sst --syntax_selection bertscore > /dev/null 2>&1
         # CUDA_VISIBLE_DEVICES=6,7 python -m python.sa.main --run template --search_dataset sst --syntax_selection noselect

         # generate tempaltes from checklist testcase
         # CUDA_VISIBLE_DEVICES=5,6 python -m python.sa.main \
         #                     --run template \
         #                     --search_dataset checklist \
         #                     --syntax_selection random # > /dev/null 2>&1
        )
}

function gen_testsuite() {
        # write test cases into Checklist Testsuite format
        (cd ${_DIR}
         python -m python.sa.main \
                --run testsuite \
                --search_dataset sst \
                --syntax_selection random
         # python -m python.sa.main --run testsuite --search_dataset sst --syntax_selection bertscore
         # python -m python.sa.main --run testsuite --search_dataset sst --syntax_selection noselect

         # write test cases into Checklist Testsuite format from checklist testcases
         # python -m python.sa.main \
         #        --run testsuite \
         #        --search_dataset checklist \
         #        --syntax_selection random
        )
}

function gen_seeds() {
        (cd ${_DIR}
         # write test cases into Checklist Testsuite format from checklist testcases
         python -m python.sa.main \
                --run seedgen \
                --search_dataset sst \
                --syntax_selection random \
                --num_seeds -1
        )
}

function eval_models() {
        # evaluate NLP models with generated testsuites
        (cd ${_DIR}
         # evaluating models on checklist testcases
         # CUDA_VISIBLE_DEVICES=1,3,5 python -m python.sa.main --run testmodel --test_baseline

         # evaluating models on our generated testcases
         # CUDA_VISIBLE_DEVICES=6,7 python -m python.sa.main --run testmodel --syntax_selection random
         # CUDA_VISIBLE_DEVICES=6,7 python -m python.sa.main --run testmodel --syntax_selection bertscore
         # CUDA_VISIBLE_DEVICES=6,7 time python -m python.sa.main --run testmodel --syntax_selection noselect

         # evaluating models on checklist expanded testcases
         # CUDA_VISIBLE_DEVICES=6,7 python -m python.sa.main \
         #                     --run testmodel \
         #                     --search_dataset checklist \
         #                     --syntax_selection random
         # CUDA_VISIBLE_DEVICES=6,7 python -m python.sa.main \
         #                     --run testmodel \
         #                     --search_dataset checklist \
         #                     --syntax_selection random \
         #                     --test_baseline
        )
}

function eval_models_seed() {
        # evaluate NLP models with generated testsuites
        (cd ${_DIR}
         # evaluating models on checklist testcases
         # CUDA_VISIBLE_DEVICES=6,7 python -m python.sa.main \
         #                     --run testmodel_seed \
         #                     --search_dataset sst \
         #                     --syntax_selection random \
         #                     --test_baseline

         # evaluating models on our generated testcases
         CUDA_VISIBLE_DEVICES=6,7 python -m python.sa.main \
                             --run testmodel_seed \
                             --search_dataset sst \
                             --syntax_selection random
        )
}

function retrain_models() {
        # evaluate NLP models with generated testsuites
        (cd ${_DIR}
         echo "***** TRAIN: ours, EVAL: checklist *****"
         CUDA_VISIBLE_DEVICES=0,1,2,3 python -m python.sa.main \
                             --run retrain \
                             --search_dataset sst \
                             --syntax_selection random \
                             --lcs \
                             --epochs 10 \
                             --testing_on_trainset \
                             --model_name textattack/bert-base-uncased-SST-2
         echo "***** TRAIN: checklist, EVAL: ours *****"
         CUDA_VISIBLE_DEVICES=0,1,2,3 python -m python.sa.main \
                             --run retrain \
                             --search_dataset checklist \
                             --lcs \
                             --epochs 10 \
                             --testing_on_trainset \
                             --model_name textattack/bert-base-uncased-SST-2
        )
}

function eval_retrained_models() {
        # evaluate NLP models with generated testsuites
        (cd ${_DIR}
         python -m python.sa.main \
                --run testmodel \
                --local_model_name checklist-textattack-bert-base-uncased-SST-2
        )
}

function analyze_eval_models() {
        (cd ${_DIR}
         # evaluate NLP models with generated testsuites
         # python -m python.sa.main \
         #        --run analyze \
         #        --search_dataset sst \
         #        --syntax_selection random
         # python -m python.sa.main \
         #        --run analyze \
         #        --search_dataset sst \
         #        --syntax_selection random \
         #        --test_baseline
         # python -m python.sa.main --run analyze --search_dataset sst --syntax_selection bertscore
         # python -m python.sa.main --run analyze --search_dataset sst --syntax_selection noselect

         # evaluate NLP models with checklist expanded testsuites
         python -m python.sa.main \
                --run analyze \
                --search_dataset checklist \
                --syntax_selection random
         python -m python.sa.main \
                --run analyze \
                --search_dataset checklist \
                --syntax_selection random \
                --test_baseline
        )
}

function analyze_eval_models_seed() {
        (cd ${_DIR}
         # evaluate NLP models with generated seed with baseline(checklist) testsuites
         python -m python.sa.main \
                --run analyze_seed \
                --search_dataset sst \
                --syntax_selection random
        )
}

function analyze_retrained_models() {
        # evaluate NLP models with generated testsuites
        (cd ${_DIR}
         python -m python.sa.main \
                --run retrain_analyze \
                --search_dataset sst \
                --syntax_selection random \
                --model_name textattack/bert-base-uncased-SST-2 \
                --epochs 10
        )
}

function explain_nlp() {
        mkdir -p ./_results/retrain/explain_nlp
        CUDA_VISIBLE_DEVICES=6,7 python -m python.sa.main \
                            --run explain_nlp \
                            --search_dataset sst \
                            --syntax_selection random \
                            --model_name textattack/bert-base-uncased-SST-2
}

function selfbleu() {
        (cd ${_DIR}
         python -m python.sa.main \
                --run selfbleu \
                --search_dataset sst \
                --syntax_selection random
        )
}

function pdrulecoverage() {
        (cd ${_DIR}
         python -m python.sa.main \
                --run pdrule_cov \
                --search_dataset sst \
                --syntax_selection random
        )
}

# ==========
# Human study

function humanstudy() {
        (cd ${_DIR}
         python -m python.sa.main \
                --run humanstudy \
                --search_dataset sst \
                --syntax_selection random
        )
}

function humanstudy_results() {
        (cd ${_DIR}
         python -m python.sa.main \
                --run humanstudy_results \
                --search_dataset sst \
                --syntax_selection random \
                --model_name textattack/bert-base-uncased-SST-2
        )
}

# ==========
# Coverage Exp

function gen_coverage_data() {
        (cd ${_DIR}
         python -m python.sa.main \
                --run coverage_data \
                --search_dataset sst \
                --syntax_selection random
        )
}

function compute_coverage() {
        COV_DIR=${PYTHON_DIR}/sa/coverage
        DATA_DIR=${COV_DIR}/data
        search_dataset="sst"
        syntax_selection="random"
        (cd ${PYTHON_DIR}/sa/coverage

         for file in $(find ${DATA_DIR} -name "seed_sa_${search_dataset}_*.txt" | sort); do
                 filename=$(basename "${file}")
                 filename="${filename%%.*}"
                 echo ${filename}
                 lc_cksum="${filename##*_}"
         
                 ours=${DATA_DIR}/"seed_sa_${search_dataset}_${lc_cksum}.txt"
                 bls=${DATA_DIR}/"checklist_sa_${search_dataset}_${lc_cksum}.txt"
                 CUDA_VISIBLE_DEVICES=6,7 python test_coverage.py \
                                     --our_sents ${ours} \
                                     --bl_sents ${bls} \
                                     --search_dataset ${search_dataset} \
                                     --syntax_selection ${syntax_selection} \
                                     --lc_cksum ${lc_cksum}
                 python post_coverage.py \
                        --search_dataset ${search_dataset} \
                        --syntax_selection ${syntax_selection} \
                        --lc_cksum ${lc_cksum}
         done
        )
}

# ==========
# Tables & Plots

function make_tables() {
        (cd ${_DIR}
         # Table 1
         # python -m python.sa.main --run tables --which lc-req
         # python -m python.sa.main --run tables --which selfbleu # deprecated
         # python -m python.sa.main \
         #        --run tables \
         #        --which retrain-debug \
         #        --search_dataset sst \
         #        --syntax_selection random \
         #        --epochs 5 \
         #        --model_name textattack/bert-base-uncased-SST-2 # deprecated
         python -m python.sa.main --run tables --which manual-study
        )
}

# ==========
# Main

function main_sst() {
        # gen_requirements # to generate test_type_sa.json and requirement_sa.json
        gen_templates # to generate templates_sa/seeds_{cksum}.json, templates_sa/templates_seed_{cksum}.json and templates_sa/templates_exp_{cksum}.json and cfg_expanded_inputs_sa.json
        # gen_testsuite # to generate pkl checklist testsuite files in test_results directory
        # gen_seeds
        # eval_models # run testsuite.run on our and checklist generated testsets
        # eval_models_seed
        # analyze_eval_models # to generate test_results_analysis.json and test_results_checklist_analysis.json by reading test_results.txt and cfg_expanded_inputs_{nlp_task}_{search_dataset_name}_{selection_method}.json
        # analyze_eval_models_seed
        # retrain_models # to retrain models and test the retrained models on testsuite.run on our and checklist generated testsets
        # analyze_retrained_models # to generate debug_results file
        # selfbleu # to compute the selfbleu
        # pdrulecoverage # to compute the diversity of grammatic structure of sentence
        # explain_nlp # to run the explainNLP
        # humanstudy # sample sentences for manual study
        # humanstudy_results # get results of manual study into human_study.json
        # gen_coverage_data # get sentences for coverage experiment
        # compute_coverage
        # make_tables


        # eval_retrained_models # to ...?
}


# please make sure you actiavte nlptest conda environment
# main_sst


function main_checklist() {
        # implement our expansion technique from checklist testcase
        # make sure that you change the name of dataset into checklist
        # gen_requirements # to generate test_type_sa.json and requirement_sa.json
        # gen_templates # to generate templates_sa_{dataset_name}/seeds_{cksum}.json, templates_sa/templates_seed_{cksum}.json and templates_sa_{dataset_name}/templates_exp_{cksum}.json and cfg_expanded_inputs_sa_{dataset_name}.json
        # gen_testsuite # to generate pkl checklist testsuite files in test_results directory
        # eval_models # run testsuite.run on checklist expanded testsets
        analyze_eval_models # to generate test_results_analysis.json and test_results_checklist_analysis.json by reading test_results.txt and cfg_expanded_inputs_{nlp_task}_{search_dataset_name}_{selection_method}.json 
}

main_sst
# main_checklist
