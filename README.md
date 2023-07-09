[bitbucket repository]: https://bitbucket.org/adrian-secteam/ml-code-eval/src/main/
[data folder]: https://bitbucket.org/adrian-secteam/ml-code-eval/src/main/data
[model folder]: https://bitbucket.org/adrian-secteam/ml-code-eval/src/main/model
[keywords folder]: https://bitbucket.org/adrian-secteam/ml-code-eval/src/main/keywords
[Google Colab]: https://colab.research.google.com/
[preprocess/text_process.py]: https://bitbucket.org/adrian-secteam/ml-code-eval/src/main/preprocess/text_process.py
[model/keyword_model.py]: https://bitbucket.org/adrian-secteam/ml-code-eval/src/main/model/keyword_model.py
[model/lm_model.py]: https://bitbucket.org/adrian-secteam/ml-code-eval/src/main/model/lm_model.py

# ML Coding Evaluation #

## Goals ##

In this exercise the main goal is to build 2 Hatespeech classification models

1. A keyword based model classifier

2. A Language Model (LM) based model classifier

This exercise is designed to evaluate your abilities to

    - understand an ML problem, propose and implement a solution
    - preprocess text data
    - implement simple ML models
    - evaluate the performance of ML models
    - compare the performance of different ML models
    - use good code development practises
    - use the typical tools of an NLP researcher/engineer

You can go through as many steps of the exercise as you can in a reasonable amount of time.
All implementation should be done in Python. You can choose and use whichever Python packages you deem necessary.
Some of the steps can be implemented in different ways, you just need to choose one, it does not need to be the most complicated or detailed implementation for every step. It is better to implement a solution for each step and complete the exercise.

In the follow up interview we will discuss the different steps and focus on

    - the approach you chose for each step
    - the reasoning behind the approach
    - the output of each step
    - your analysis of the model results
    - your comparison of the model results
    - your coding style
    - what you could have done different

## Exercise ##

### Setup ###

1. Sign up for a free [BitBucket account](https://id.atlassian.com/signup?application=bitbucket) if you do not already have one.

2. When you have a BitBucket account send an email to adrian.flanagan@huawei.com from the email address you used to sign in to BitBucket requesting access to the repository. 
   
3. The git repository for this evaluation can be cloned from this [bitbucket repository]. (Before cloning you may need to create an [app password](https://bitbucket.org/account/settings/app-passwords/) if you do not add an ssh key)

4. The [data folder] folder contains all the data you need for this evaluation. There is also a README.md contained in this folder that explains the data structure.

5. The [model folder] contains a base Model class in model.py that should be inherited by each of your model class implementations.

6. The [keywords folder] contains a text file of Hate Speech keywords to use as the basis of your keyword model.

7. To train, evaluate and compare your models you can use [Google CoLab].
    1. Create a new notebook

    2. Import your code and data from Bitbucket

### Steps ###

1. Create a new branch in your local copy of the repository.

2. Implement a Python class in the file [preprocess/text_process.py] to preprocess the dataset into a suitable format for training the Hatespeech classification models.

3. Provide an analysis of the dataset.

4. Implement a keyword hatespeech classification model in the file [model/keyword_model.py].

5. Implement a language model based hatespeech classification model in the file [model/lm_model.py]. You can use a base BERT model.

6. For each model report the performance metrics you think are most relevant.

7. Compare the relative performance of the 2 models.

8. Create a pull request to incorporate all your code changes to the main branch of the repository assigning "Adrian Flanagan" as the reviewer.

9. Share with adrian.dreamlabs@gmail.com the CoLab notebook you created to analyse the data, train, evaluate and analyse the models.
