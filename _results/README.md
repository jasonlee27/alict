Results of ALiCT
=================

## Table of Contents
   * [Results of ALiCT](#results-of-alict)
      * [Table of Contents](#table-of-contents)
      * [Linguistic Capability Specifications](#linguistic-capability-specifications)
      * [Experiment Results](#experiment-results)
         * [RQ1: Diversity](#rq1-diversity)
         * [RQ2: Effectiveness](#rq2-effectiveness)
         * [RQ3: Consistency](#rq3-consistency)
<!-- 
You can find more results at the project site(https://sites.google.com/view/s2lct/home). -->


## Linguistic Capability Specifications
<div align="center">
    <span style="font-size:1.5em">
        <strong>Table 1: Search predicates for ten linguistic capabilities of sentiment analysis.</strong>
    </span>
</div>
<p align="center">
    <img src="./tables/lc-spec-table.png" alt="lc-spec-table" width=auto height=auto title="lc_spec_table">
</p>

<div align="center">
    <span style="font-size:1.5em">
        <strong>Table 2: Search predicates for ten linguistic capabilities of hate speech detection.</strong>
    </span>
</div>
<p align="center">
    <img src="./tables/hsd-lc-spec-table.png" alt="hsd-lc-spec-table" width=auto height=auto title="hsd_lc_spec_table">
</p>

## Baselines
### Capability Testing Baselines
ALiCT is evaluated by comparing with the state-of-the-art linguistic capability testing for sentiment analysis and hate speech detection as following:

> 1. CHECKLIST([paper](https://homes.cs.washington.edu/~marcotcr/acl20_checklist.pdf), [repo](https://github.com/marcotcr/checklist)) for sentiment analysis
> 2. Hatecheck([paper](https://aclanthology.org/2021.acl-long.4/), [repo](https://github.com/paul-rottger/hatecheck-data)) for hate speech detection

### Model Under Test
Given the generated test cases from the ALiCT and [capability testing baselines](#capability-testing), models in the table 3 are evaluated:
<div align="center">
    <span style="font-size:1.5em">
        <strong>Table 3: The NLP model used in our evaluation.</strong>
    </span>
</div>
<p align="center">
    <img src="./tables/model-under-test.png" alt="model-under-test" width=auto height=auto title="model_under_test">
</p>

### Evaluation of of expansion phase of ALiCT
the test case diversity provided by ALiCT expansion phase of ALiCT is also compared against that of one syntax-based (MT-NLP) and three adversarial (Alzantot-attack, BERT-Attack and SememePSO-attack) as follows:

- Syntax-based approach
> MT-NLP: [Metamorphic Testing and Certified Mitigation of Fairness Violations in NLP Models](https://www.ijcai.org/Proceedings/2020/64)

- Adversarial approaches
> Alzantot-attack: [Generating Natural Language Adversarial Examples](https://aclanthology.org/D18-1316/)   
> BERT-Attack: [BERT-ATTACK: Adversarial Attack Against BERT Using BERT](https://arxiv.org/abs/2004.09984)  
> SememePSO-attack: [Word-level textual adversarial attacking as combinatorial optimization](https://arxiv.org/abs/1910.12196)


## Experiment Results
### RQ1: Diversity
<div align="center">
    <span style="font-size:1.5em">
        <strong>Table 4: Comparison results against MT-NLP.</strong>
    </span>
</div>
<p align="center">
    <img src="./tables/exp-compare-mtnlp.png" alt="sa-test-results" width=auto height=auto title="sa_test_results">
</p>
<div align="center">
    <span style="font-size:1.5em">
        <strong>Table 5: Comparison results against adversarial attacks.</strong>
    </span>
</div>
<p align="center">
    <img src="./tables/exp-compare-advattacks.png" alt="sa-test-results" width=auto height=auto title="sa_test_results">
</p>

<div align="center">
    <span style="font-size:1.2em">
        <strong>Table 6: Examples for text generation compared with the syntax-based and adversarial generation baselines.</strong>
    </span>
</div>
<p align="center">
    <img src="./tables/text-generation-examples.png" alt="text-generation-examples" width=auto height=auto title="text_generation_examples">
</p>

### RQ2: Effectiveness
<div align="center">
    <span style="font-size:1.2em">
        <strong>Table 7: Results of BERT-base, RoBERTa-base and DistilBERT-base sentiment analysis models on ALiCT test cases using all seeds. BERT-base, RoBERTa-base and DistilBERT-base models are denoted as BERT, RoBERTa and dstBERT,respectively.</strong>
    </span>
</div>
<p align="center">
    <img src="./tables/sa-test-results.png" alt="sa-test-results" width=auto height=auto title="sa_test_results">
</p>

<div align="center">
    <span style="font-size:1.2em">
        <strong>Table 8: Results of dehate-BERT and twitter-RoBERTa hate speech detection models on ALiCT test cases using all seeds. dehate-BERT and twitter-RoBERTa models are denoted as BERT and RoBERTa respectively.</strong>
    </span>
</div>
<p align="center">
    <img src="./tables/hsd-test-results.png" alt="hsd-test-results" width=auto height=auto title="hsd_test_results">
</p>

### RQ3: Consistency
