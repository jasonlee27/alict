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
        <strong>Table 1: Structural predicates and generative rules for the linguistic capabilities of sentiment analysis.</strong>
    </span>
</div>
<p align="center">
    <img src="./tables/lc-spec-table.png" alt="lc-spec-table" width=auto height=auto title="lc_spec_table">
</p>

<div align="center">
    <span style="font-size:1.5em">
        <strong>Table 2: Structural predicates and generative rules for the linguistic capabilities of hate speech detection. 
          The <em>slur</em> and <em>profanity in LC1-LC4 are the collections of terms that express slur and profanity. 
          The <em>identity</em> in LC11-LC12 is a list of names that used to describe social groups. 
          In this work, we reuse these terms from Hatecheck.</strong>
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
<p align="center">
    <img src="./figures/alict-fig4.png" alt="alict-fig4" width="70%" height=auto title="alict_fig4">
</p>
<div align="center">
    <span style="font-size:1.5em">
        <strong>Figure 1: Results of Self-BLEU (left) and Syntactic diversity (right) of ALiCT and capability-based testing baselines for sentiment analysis and hate speech detection. 
          Use of only ALiCT seed sentences and all ALiCT sentences are denoted as ALiCT and ALiCT+EXP respectively.</strong>
    </span>
</div>

<p align="center">
    <img src="./figures/alict-fig5.png" alt="alict-fig5" width="70%" height=auto title="alict_fig5">
</p>
<div align="center">
    <span style="font-size:1.5em">
        <strong>Figure 2: Results of Self-BLEU (left) and Syntactic diversity (right) between original sentences of capability-based
          testing baselines and ALiCT generated sentences from the original sentences.</strong>
    </span>
</div>

<div align="center">
    <span style="font-size:1.5em">
        <strong>Table 4: Comparison results against MT-NLP.</strong>
    </span>
</div>
<p align="center">
    <img src="./tables/mtnlp-results.png" alt="mtnlp-results" width="50%" height=auto title="mtnlp-results">
</p>
<div align="center">
    <span style="font-size:1.5em">
        <strong>Table 5: Comparison results against adversarial attacks.</strong>
    </span>
</div>
<p align="center">
    <img src="./tables/adv-attack-results.png" alt="adv-attack-results" width="50%" height=auto title="adv-attack-results">
</p>
<p align="center">
    <img src="./tables/neuron-coverage.png" alt="neuron-coverage.png" width="70%" height=auto title="neuron-coverage.png">
</p>
<div align="center">
    <span style="font-size:1.2em">
        <strong>Figure 3: Neuron coverage results of ALiCT and CHECKLIST.</strong>
    </span>
</div>
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

<div align="center">
    <span style="font-size:1.2em">
        <strong>Table 9: Consistency Results.</strong>
    </span>
</div>
<p align="center">
    <img src="./tables/consistency-results.png" alt="consistency-results" width="50%" height=auto title="consistency-results">
</p>
