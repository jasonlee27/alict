# S<sup>2</sup>LCT

This repository contains code for testing NLP Models as described in the following paper:
>[S<sup>2</sup>LCT: Specification- and Syntax-based Automated Testing Linguistic Capabilities of NLP Models]
> Simin Chen, Cong Liu, Wei Yang, Shiyi Wei

S<sup>2</sup>LCT is an automated linguistic capability-based testing framework for NLP models. In this implementation, we generate testcases for sentiment analysis task. Seed test cases are generated from SST dataset(https://nlp.stanford.edu/sentiment/).
In the future, multiple tasks and multiple searching datasets are implemented for the S<sup>2</sup>LCT.

## Prerequisites
This application is written for ```Python>3.7.11```. All requirements are listed in ```requirements.txt```, and they are installed by pip with the following command.
```bash
pip install -r requirements.txt
```

## Usage


<!-- 
## Goal
The project is to generate comprehensive sets of test cases for evaluating NLP models on multiple linguistic capabilities of the NLP task.

## Problem
Prior works introduced multiple linguistic capabilities for a NLP task, and manually generated test templates for each linguistic capability. However, the generated test cases are highly restricted in elementary structures and vocabularies. The simplicity causes biases in the test set, thus it loses the comprehensive evaluation of the linguistic capabilities.

## Task
Given the limitations mentioned in the Problem section, we focus on improving comprehensivity of linguistic capability evaluation by generating more diverse realistic test cases.

## Idea 
1. For each linguistic capability, there are input/output properties that the input/output should meet for evaluating the linguistic capability.
2. With the help of a large amount of natural language dataset, for each linguistic capability we can increase the diversity of test cases by using the subset of inputs that meet the requirements from the dataset, and convert them into the test templates.
3. We obtain more diversity of test templates by appending structural components into structures of test templates by comparing context-free grammars (CFGs) between input and reference natural language datasets.
4. We fill the structures with the relevant vocabularies that do not affect the corresponding labels using language models.

## Steps:
1. Requirement extraction from linguistic capabilities.
2. Search/transform relevant inputs from the dataset for satisfying the requirements.
3. Generate test templates with the inputs from step 2.
4. Extract input-expansible structures in CFG from a reference rule set.
5. Expand input templates by adding the structures from step 4.
6. Fill the structures from step 5 with vocabularies suggested from a language model.

## Progress
1. I implemented the treebank dataset and its CFG production rule set as a reference rule set.
2. I implemented the Berkeley neural parser (https://github.com/nikitakit/self-attentive-parser) to parse each input sentence and construct its CFG.
3. I computed expandible rule sets of input sentences computed from the difference between reference and input CFG.
I manually extracted the requirements of two linguistic capabilities for semantic analysis reported in CHECKLIST paper.
4. For each requirement from step 4, I search relevant inputs that meet the requirement from widely used sentiment analysis dataset(Stanford Treebank dataset, link: https://nlp.stanford.edu/sentiment/index.html) and word sentiments dataset(SentiWordNe, link: https://github.com/aesuli/SentiWordNet)

 -->
