# Spring 2025 AIFirst Project Starter Code

## Project Introduction

In this project, we will be training and testing our own custom Large Language Model (LLM) from scratch. The main purpose
of this LLM is to act as a study aid for the course content in the AIFirst (ECE2806) course. Therefore, given any question on
Machine Learning topics, the LLM should provide a reasonable output that relates to different ML tasks. This repo contains the 
code for training and inference of your model. Note that you are free to imbue this project with your own creative solutions.

## Project Constraints
It is possible to generate a model within this github and then take pre-trained weights from a library like huggingface
to instantly get a model that will perform this task very well. Obviously, training
models from scratch will result in worse performing models compared to the open source weights that can be found online. However,
the major point of this project is go through the process of data pre-processing, training, hyperparameter tuning, and inference
on your own. The analysis of this process will compose most of your report. 

Note that we are not interested in the performance of your model. We are interested in how the analysis of your performance led to
different decisions and choices you make within this project. In other words, your grade depends on analysis, not on performance.
Interesting failures will make a better report than performing well on models that you got from the internet.

## Directory Information

Note the following features and conventions of this github repository:

* Model files should be saved in the models/ directory.
* The implementation of our GPT model is in the model.py file.
* Test and train data should be shaved in the test_data/ and train_data/ directories.
  * All data should be in the format of a single .txt file.
  * Test data will be provided in the form of test_answer_#.txt, test_prompt_#.txt, and test_response_#.txt.
    * All testing data will be provided by the instructors and will be composed of common machine learning questions that you learned in this course.
    * test_prompt_#.txt will be the input you give your LLM.
    * test_answer_#.txt will be the answer expected based on the corresponding prompt.
    * test_response_#.txt is the output file that your LLM should save everything to.
    * The inference code will then compute a metric to compare between the answer and response files to measure the quality of your LLM.
* There are two training files and two testing files.
  * Two of the files are based on the base GPT model itself.
  * Two of the files are copies using an additional tokenizer block.
* We provide a jupyter notebook called tutorial_basic.ipynb that places all the main points of the code together for your exploratory convenience. 
* utils.py contains helper functions that you may or may not use.
* You are free to add/modify any part of your code to suit the needs of your project.

## Code Usage

1. Go to the training_base.py file.
2. Modify the argparse options to suit the hyperparameters of your experiment.
3. Run the following command on the command line:
```
python3 training_base.py
```
4. Navigate to the corresponding testing file. In this case, that would be testing_base.py
5. Modify the argparse file in this file. Note that testing file must specify the path to the saved model checkpoint that you generated in Step 2.
```
python3 testing_base.py
```
6. Record the performance on your testing data.

## Project Ideas
The above Code Usage is simple. However, the complexity of the project has a lot to do with the analysis you can perform as well as the alterations you
can do on top of our basic setup.
Possible ideas include:
* Implement your own custom datasets.
  * For this project, you will almost certainly have to find a data source to input into your model.
  * Describing how you mined the data can be an interesting part of the report.
  * This could be as simple as inputting data into a large txt file.
* Perform analyses of performance on specific prompts as hyperparameter changes are made. This would be similar to your studios where you generate
plots of observed changes as different paramters are varied. These parameters could include:
  * Block Size
  * Batch Size
  * Learning Rate
  * Optimizer
  * Number of Heads or Layers in the GPT model.
  * etc.
* Incorporate your own custom tokenizer.
* Visualize the attention weights/Matrices for different input prompts.
* Generate curves of loss, gradient, training, and accuracy values over the course of training.
* Possible implementation of custom loss functions, optimizers, or training procedures.
* etc.

## Acknowledgements/Resources

This code base draws from the NLP courses taught by Andrej Karpathy. His github can be found at https://github.com/karpathy/nn-zero-to-hero. You are free to use
his resources and incorporate his ideas. However, as discussed earlier, all training must be done by you.

