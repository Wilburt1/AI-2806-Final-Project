# Spring 2025 AIFirst Project Starter Code

## Project Introduction

This repository contains the implementation and analysis for the AI First (ECE2806) final project by Zewei Zhang at Georgia Institute of Technology. The project involves training and fine-tuning a custom Large Language Model (LLM) from scratch to serve as a study aid for the AI First course, answering Machine Learning-related questions. The focus is on dataset creation, hyperparameter optimization, and task-specific fine-tuning, with an emphasis on analyzing the impact of these processes on model performance. Creative modifications to the provided starter code are implemented to enhance the LLM's capabilities.

## Project Constraints
While pre-trained weights from libraries like Hugging Face could yield high-performing models, this project prioritizes the process of data preprocessing, training, hyperparameter tuning, and inference from scratch. The evaluation focuses on the analysis of decisions and outcomes, not the absolute performance of the model. Insightful analysis of challenges and failures is valued over using pre-trained models, and all training must be conducted independently.

## Directory Information

The repository follows these conventions:
Model Files: Saved in models/ (e.g., pretrain.pth for pre-trained model, finetuned_model.pth for fine-tuned model).
Model Implementation: Located in model.py, containing the GPT model architecture.
Data Directories:
train_data/: Contains training datasets (Dataset.txt, Dataset_improved.txt, QA.txt) in .txt format.
test_data/: Contains instructor-provided test files:
test_prompt_#.txt: Input prompts for the LLM.
test_answer_#.txt: Expected answers for prompts.
test_response_#.txt: Model-generated outputs.
Training and Testing Scripts:
training_base.py, testing_base.py: Base scripts for the GPT model.
training_tokenizer.py, testing_tokenizer.py: Scripts with tokenizer block for Part 1.
training_tokenize_Part2.py: Custom fine-tuning script for Part 2.
Additional Files:
tutorial_basic.ipynb: Jupyter notebook for code exploration.
utils.py: Helper functions for training and evaluation.
Customization: The code has been modified to support GPT2 tokenization, Adam optimizer, and QA-format fine-tuning, with new scripts added for Part 2.

## Code Usage

Part 1: Pre-Training


Place Dataset.txt (344KB, from AI 2806 slides) and Dataset_improved.txt (515KB, enhanced with Wikipedia content) in train_data/.

Modify training_tokenizer.py to set hyperparameters:
parser.add_argument('--max_iters', type=int, default=5000)
parser.add_argument('--tokenization', type=str, default='gpt2')
parser.add_argument('--optimizer', type=str, default='adam')
parser.add_argument('--dataset', type=str, default='train_data/Dataset_improved.txt')
Run training:
python3 training_tokenizer.py
Test the model:
python3 testing_tokenizer.py --model_path models/pretrain.pth
Evaluate results in test_data/test_response_#.txt using Levenshtein distance and ROUGE scores.

Part 2: Fine-Tuning
Place QA.txt (QA-formatted dataset) in train_data/.
Modify training_tokenize_Part2.py for fine-tuning:
parser.add_argument('--lr', type=float, default=1e-5)
parser.add_argument('--wd', type=float, default=0.25)
parser.add_argument('--model_path', type=str, default='models/pretrain.pth')
parser.add_argument('--dataset', type=str, default='train_data/QA.txt')
Run fine-tuning:
python3 training_tokenize_Part2.py
Test the fine-tuned model:
python3 testing_tokenizer.py --model_path models/finetuned_model.pth
Results are averaged over 10 runs to reduce randomness.
## Project Ideas
The project extends the starter code with the following implementations and analyses:





Custom Datasets:





Dataset.txt: Converted from 18 AI 2806 course slides (344KB).



Dataset_improved.txt: Augmented with Wikipedia content for key concepts (515KB).



QA.txt: Reformatted Dataset_improved.txt into QA format for fine-tuning.



Hyperparameter Analysis:





Part 1: Tested max_iters (1000, 5000, 10000, 15000), tokenization (BPE vs. GPT2), and optimizers (SGD vs. Adam). Optimal: max_iters=5000, GPT2, Adam.



Part 2: Tested five learning rate and weight decay combinations:





lr=3e-4, wd=0.1
lr=1e-4, wd=0.15
lr=5e-5, wd=0.2
lr=1e-5, wd=0.25 (optimal)
lr=5e-6, wd=0.3
Custom Tokenizer: Implemented GPT2 tokenization, outperforming BPE due to better semantic capture.
Performance Visualizations:
Line charts for Levenshtein distance and ROUGE scores over iterations.
Bar charts comparing pre-trained and fine-tuned models across metrics.
Evaluation Metrics:
Levenshtein distance for character-level alignment.
ROUGE-1, ROUGE-2, ROUGE-L F1 scores for text similarity.
Fine-Tuning Strategy: Developed training_tokenize_Part2.py to fine-tune on QA-formatted data, validating hypotheses on task adaptability and generalization.

## Acknowledgements/Resources

The codebase is inspired by Andrej Karpathy’s NLP courses (nn-zero-to-hero). All training and fine-tuning were performed from scratch, adhering to project constraints. Additional resources include the transformers library for GPT2 tokenization and rouge-score for evaluation.

