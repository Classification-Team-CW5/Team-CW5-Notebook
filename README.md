# Twitter-Sentiment-Analysis
It is a Natural Language Processing Problem where Sentiment Analysis is done by Classifying the tweets that show a belief in climate change  from tweets that do not by machine learning models for classification,  text mining, text analysis, data analysis and data visualization

## Introduction

Natural Language Processing (NLP) is a hotbed of research in data science these days and one of the most common applications of NLP is sentiment analysis. From opinion polls to creating entire marketing strategies, this domain has completely reshaped the way businesses work, which is why this is an area every data scientist must be familiar with.

Thousands of text documents can be processed for sentiment (and other features including named entities, topics, themes, etc.) in seconds, compared to the hours it would take a team of people to manually complete the same task. 

We will do so by following a sequence of steps needed to solve a general sentiment analysis problem. We will start with preprocessing and cleaning of the raw text of the tweets. Then we will explore the cleaned text and try to get some intuition about the context of the tweets. After that, we will extract numerical features from the data and finally use these feature sets to train models and identify the sentiments of the tweets.

This is one of the most interesting challenges in NLP so we are very excited to take this journey with you!

## Understand the Problem Statement

Let’s go through the problem statement once as it is very crucial to understand the objective before working on the dataset. The problem statement is as follows:

Many companies are built around lessening one’s environmental impact or carbon footprint. They offer products and services that are environmentally friendly and sustainable, in line with their values and ideals. They would like to determine how people perceive climate change and whether or not they believe it is a real threat. This would add to their market research efforts in gauging how their product/service may be received.

With this context, EDSA is challenging you during the Classification Sprint with the task of creating a Machine Learning model that is able to classify whether or not a person believes in climate change, based on their novel tweet data.

Providing an accurate and robust solution to this task gives companies access to a broad base of consumer sentiment, spanning multiple demographic and geographic categories - thus increasing their insights and informing future marketing strategies.

Note: The evaluation metric from this practice problem is F1-Score.

## Running the notebook

There are many different ways that the notebook can be run and in this section we will go through all those methods so that you can deploy the notebook and have the best experience in trying to replicate our results and making conclusions around our findings 

### Dependencies 

Please ensure that one or more of the following is installed on to whatever enviroment you want to run the notebook on first in order to not recieve any errors:

-
-
-

### Opening the notebook 

#### 1. Windows File Explorer + Command Prompt

>Please download the [notebook](https://github.com/Classification-Team-CW5/Team-CW5-Notebook/blob/main/Climate_Change_Beliefs_Analysis_(1).ipynb) to your local machine 

Whenever you open a Windows Explorer folder, you’ll see an address bar similar to that in a web browser. By default, it shows the path of the current folder. In this address bar, you can enter in text and navigate to other directories manually.

Once you’ve entered your specific folder with Windows Explorer, you can simply press `ALT` + `D`, type in `cmd` and press Enter. You can then type `jupyter notebook` to launch Jupyter Notebook within that specific folder.

<img src= "https://miro.medium.com/max/1400/0*5l8imTsZnIvqFDk4.gif" alt="Windows Explorer" title="Windows Explorer"/> 

NOTE: If you’re using Anaconda, you may have to type activate conda to switch to Anaconda Prompt within Command Prompt. Additionally, if you receive an error involving zqm.h, you’ll need to add the following folders to your PATH environment variable:

``` 
C:\Users\***\Anaconda3\Lib\site-packages\zmq
C:\Users\***\Anaconda3\Library\bin 
```



## Tweets Preprocessing and Cleaning

You are searching for a document in this office space. In which scenario are you more likely to find the document easily? Of course, in the less cluttered one because each item is kept in its proper place. The data cleaning exercise is quite similar. If the data is arranged in a structured format then it becomes easier to find the right information.

The preprocessing of the text data is an essential step as it makes the raw text ready for mining, i.e., it becomes easier to extract information from the text and apply machine learning algorithms to it. If we skip this step then there is a higher chance that you are working with noisy and inconsistent data. The objective of this step is to clean noise those are less relevant to find the sentiment of tweets such as punctuation, special characters, numbers, and terms which don’t carry much weightage in context to the text.

In one of the later stages, we will be extracting numeric features from our Twitter text data. This feature space is created using all the unique words present in the entire data. So, if we preprocess our data well, then we would be able to get a better quality feature space.

Let’s first read our data and load the necessary libraries.

## Story Generation and Visualization from Tweets

In this section, we will explore the cleaned tweets text. Exploring and visualizing data, no matter whether its text or any other data, is an essential step in gaining insights. Do not limit yourself to only these methods told in this tutorial, feel free to explore the data as much as possible.

Before we begin exploration, we must think and ask questions related to the data in hand. A few probable questions are as follows:

- What are the most common words in the entire dataset?
- What are the most common words in the dataset for negative and positive tweets, respectively?
- How many hashtags are there in a tweet?
- Which trends are associated with my dataset?
- Which trends are associated with either of the sentiments? Are they compatible with the sentiments?

## End Notes

We learned how to approach a sentiment analysis problem. We started with preprocessing and exploration of data. Then we extracted features from the cleaned text using Bag-of-Words and TF-IDF. Finally, we were able to build a couple of models using both the feature sets to classify the tweets.
