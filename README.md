# nlptest
NLPtest Project Description

Goal: The project is to generate comprehensive sets of test cases for evaluating NLP models on multiple linguistic capabilities of the NLP task.

Problem: Prior works introduced multiple linguistic capabilities for a NLP task, and manually generated test templates for each linguistic capability. However, the generated test cases are highly restricted in elementary structures and vocabularies. The simplicity causes biases in the test set, thus it loses the comprehensive evaluation of the linguistic capabilities.

Task: Given the limitations mentioned in the Problem section, we focus on improving comprehensivity of linguistic capability evaluation by generating more diverse realistic test cases.

Idea: 
For each linguistic capability, there are input/output properties that the input/output should meet for evaluating the linguistic capability.
With the help of a large amount of natural language dataset, for each linguistic capability we can increase the diversity of test cases by using the subset of inputs that meet the requirements from the dataset, and convert them into the test templates.
We obtain more diversity of test templates by appending structural components into structures of test templates by comparing context-free grammars (CFGs) between input and reference natural language datasets.
We fill the structures with the relevant vocabularies that do not affect the corresponding labels using language models.

Steps:
Requirement extraction from linguistic capabilities.
Search/transform relevant inputs from the dataset for satisfying the requirements.
Generate test templates with the inputs from step 2.
Extract input-expansible structures in CFG from a reference rule set.
Expand input templates by adding the structures from step 4.
Fill the structures from step 5 with vocabularies suggested from a language model.

Progress
I implemented the treebank dataset and its CFG production rule set as a reference rule set.
I implemented the Berkeley neural parser (https://github.com/nikitakit/self-attentive-parser) to parse each input sentence and construct its CFG.
I computed expandible rule sets of input sentences computed from the difference between reference and input CFG.
I manually extracted the requirements of two linguistic capabilities for semantic analysis reported in CHECKLIST paper.
For each requirement from step 4, I search relevant inputs that meet the requirement from widely used sentiment analysis dataset(Stanford Treebank dataset, link: https://nlp.stanford.edu/sentiment/index.html) and word sentiments dataset(SentiWordNe, link: https://github.com/aesuli/SentiWordNet)


