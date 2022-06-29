# Twitter-Sentiment-Analysis
It is a Natural Language Processing Problem where Sentiment Analysis is done by Classifying the tweets that show a belief in climate change from tweets that do not. We do this by using machine learning models for classification,  text mining, text analysis, data analysis and data visualization. These conseps are thouroughly explored and layed out in detail in the [notebook](https://github.com/Classification-Team-CW5/Team-CW5-Notebook/blob/main/Climate_Change_Beliefs_Analysis_(1).ipynb) above.

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

>Note: The evaluation metric for this practice problem is F1-Score. [More infromation about the f1 score here](https://www.educative.io/answers/what-is-the-f1-score)

## Running the notebook

There are many different ways that the notebook can be run and in this section we will go through all those methods so that you can deploy the notebook and have the best experience in trying to replicate our results and making conclusions around our findings 

### Dependencies 

Please ensure that the following is installed on to whatever enviroment you want to run the notebook on first in order to not recieve any errors:

- python 3 ([More information Available here](https://youtu.be/VWgs_iTojoA))
- NumPy ([More information Available here](https://numpy.org/install/))
- Pandas ([More information Available here](https://pandas.pydata.org/docs/getting_started/install.html))
- XGBoost ([More information Available here](https://xgboost.readthedocs.io/en/latest/install.html))
- Scikit-learn ([More information Available here](https://scikit-learn.org/stable/install.html))
- TensorFlow ([More information Available here](https://www.tensorflow.org/install))
- emoji ([More information Available here](https://pypi.org/project/emoji/))
- Nltk ([More information Available here](https://www.nltk.org/install.html))
- Internet connection with firewall permissions

It would be most advisable to use [VS Code](https://code.visualstudio.com/docs/datascience/jupyter-notebooks) to run the notebook.

### Opening the notebook 

#### 1. Windows File Explorer + Command Prompt

>Please download the [notebook](https://github.com/Classification-Team-CW5/Team-CW5-Notebook/blob/main/Climate_Change_Beliefs_Analysis_(1).ipynb) to your local machine 

Whenever you open a Windows Explorer folder, you’ll see an address bar similar to that in a web browser. By default, it shows the path of the current folder. In this address bar, you can enter in text and navigate to other directories manually.

Once you’ve entered your specific folder with Windows Explorer, you can simply press `ALT` + `D`, type in `cmd` and press `Enter`. You can then type `jupyter notebook` to launch Jupyter Notebook within that specific folder.

<img src= "https://miro.medium.com/max/1400/0*5l8imTsZnIvqFDk4.gif" alt="Windows Explorer" title="Windows Explorer"/> 

NOTE: If you’re using Anaconda, you may have to type activate conda to switch to Anaconda Prompt within Command Prompt. Additionally, if you receive an error involving zqm.h, you’ll need to add the following folders to your PATH environment variable:

``` 
C:\Users\***\Anaconda3\Lib\site-packages\zmq
C:\Users\***\Anaconda3\Library\bin 
```
#### 2. Google Colab

[Google Colaboratory](http://colab.research.google.com) is designed to integrate cleanly with GitHub, allowing both loading notebooks from github and saving notebooks to github.

**Loading Private Notebooks**

Loading a notebook from a private GitHub repository is possible, but requires an additional step to allow Colab to access your files.
Do the following:

1. Navigate to http://colab.research.google.com/github.
2. Click the "Include Private Repos" checkbox.
3. In the popup window, sign-in to your Github account and authorize Colab to read the private files.
4. Your private repositories and notebooks will now be available via the github navigation pane.

**Saving Notebooks To GitHub or Drive**

Any time you open a GitHub hosted notebook in Colab, it opens a new editable view of the notebook. You can run and modify the notebook without worrying about overwriting the source.

If you would like to save your changes from within Colab, you can use the File menu to save the modified notebook either to Google Drive or back to GitHub. Choose **File→Save a copy in Drive** or **File→Save a copy to GitHub** and follow the resulting prompts. To save a Colab notebook to GitHub requires giving Colab permission to push the commit to your repository

**Open In Colab Badge**

Anybody can open a copy of any github-hosted notebook within Colab. To make it easier to give people access to live views of GitHub-hosted notebooks,
colab provides a [shields.io](http://shields.io/)-style badge, which appears as follows:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/googlecolab/colabtools/blob/master/notebooks/colab-github-demo.ipynb)

#### 3. VS Code 

**Connect to a remote Jupyter server**

You can offload intensive computation in a Jupyter Notebook to other computers by connecting to a remote Jupyter server. Once connected, code cells run on the remote server rather than the local computer.

To connect to a remote Jupyter server:

1. Select the Jupyter Server: local button in the global Status bar or run the Jupyter: Specify local or remote Jupyter server for connections command from the Command Palette (Ctrl+Shift+P).

![specify-remote](https://user-images.githubusercontent.com/94076113/176407666-5d8032e5-df71-4b17-acb9-83a7d43f613e.png)


2. When prompted to Pick how to connect to Jupyter, select Existing: Specify the URI of an existing server.

![connect-to-existing](https://user-images.githubusercontent.com/94076113/176407976-4b8fecae-8715-4ec6-8703-cfcf14675e64.png)

3. When prompted to Enter the URI of a Jupyter server, provide the server's URI (hostname) with the authentication token included with a ?token= URL parameter. (If you start the server in the VS Code terminal with an authentication token enabled, the URL with the token typically appears in the terminal output from where you can copy it.) Alternatively, you can specify a username and password after providing the URI.

![enter-url-auth-token](https://user-images.githubusercontent.com/94076113/176408240-8bc573ed-d6c4-4099-870a-8f3664547c3f.png)

> Note: For added security, Microsoft recommends configuring your Jupyter server with security precautions such as SSL and token support. This helps ensure that requests sent to the Jupyter server are authenticated and connections to the remote server are encrypted. For guidance about securing a notebook server, refer to the Jupyter documentation.

**Setting up your environment**

To work with Python in Jupyter Notebooks, you must activate an Anaconda environment in VS Code, or another Python environment in which you've installed the Jupyter package. To select an environment, use the Python: Select Interpreter command from the Command Palette (Ctrl+Shift+P).

Once the appropriate environment is activated, you can create and open a Jupyter Notebook, connect to a remote Jupyter server for running code cells, and export a Jupyter Notebook as a Python file.

**Workspace Trust**

When getting started with Notebooks, you'll want to make sure that you are working in a trusted workspace. Harmful code can be embedded in notebooks and the Workspace Trust feature allows you to indicate which folders and their contents should allow or restrict automatic code execution.

If you attempt to open a notebook when VS Code is in an untrusted workspace running Restricted Mode, you will not be able to execute cells and rich outputs will be hidden.

**Create or open a Jupyter Notebook**

You can create a Jupyter Notebook by running the Jupyter: Create New Jupyter Notebook command from the Command Palette (Ctrl+Shift+P) or by creating a new .ipynb file in your workspace.

![native-code-cells-01](https://user-images.githubusercontent.com/94076113/176405804-a00e8a41-5a34-4ba7-b892-17a15274943c.png)

![native-kernel-picker](https://user-images.githubusercontent.com/94076113/176405888-388e13f4-c13d-496d-a7a0-7a37a08cf7de.png)

After selecting a kernel, the language picker located in the bottom right of each code cell will automatically update to the language supported by the kernel.

![native-language-picker-01](https://user-images.githubusercontent.com/94076113/176405978-2b1be00e-718f-4d37-96e7-56034cbd4534.png)

**Running cells**

Once you have a Notebook, you can run a code cell using the Run icon to the left of the cell and the output will appear directly below the code cell.

You can also use keyboard shortcuts to run code. When in command or edit mode, use Ctrl+Enter to run the current cell or Shift+Enter to run the current cell and advance to the next.

![native-code-cells-03](https://user-images.githubusercontent.com/94076113/176406131-6d6a7e05-6620-49c9-bfc7-73780f1c2526.png)

You can run multiple cells by selecting Run All, Run All Above, or Run All Below.

![native-code-runs](https://user-images.githubusercontent.com/94076113/176406290-d0af205b-766c-4947-9d4e-b6d0e798383b.png)


## Tweets Preprocessing and Cleaning

You are searching for a document in this office space. In which scenario are you more likely to find the document easily? Of course, in the less cluttered one because each item is kept in its proper place. The data cleaning exercise is quite similar. If the data is arranged in a structured format then it becomes easier to find the right information.

The preprocessing of the text data is an essential step as it makes the raw text ready for mining, i.e., it becomes easier to extract information from the text and apply machine learning algorithms to it. If we skip this step then there is a higher chance that you are working with noisy and inconsistent data. The objective of this step is to clean noises those that are less relevant to find the sentiment of tweets such as punctuation, special characters, numbers, and terms which don’t carry much weightage in context to the text.

In one of the later stages, we will be extracting numeric features from our Twitter text data. This feature space is created using all the unique words present in the entire data. So, if we preprocess our data well, then we would be able to get a better quality feature space.

>Please remember that in the notebook the data is accesed directly from Github and thus when running the notebook you need to ensure that your enviroment either has access to connect to github without firewall intrusion or the remote enviroment you are working on has the ability to connect to the remote files 

**The code that casues the above to be so is the following** 

```python:
# Load train and test datasets
train = pd.read_csv("https://raw.githubusercontent.com/Classification-Team-CW5/Classification-Data/main/train.csv")
test = pd.read_csv("https://raw.githubusercontent.com/Classification-Team-CW5/Classification-Data/main/test_with_no_labels.csv")

```
> Alternative if downloaded locally 
```python:
# Load train and test datasets
train = pd.read_csv("train.csv")
test = pd.read_csv("test_with_no_labels.csv")
```

## Story Generation and Visualization from Tweets

In this section, we will explore the cleaned tweets text. Exploring and visualizing data, no matter whether its text or any other data, is an essential step in gaining insights. Do not limit yourself to only these methods told in this tutorial, feel free to explore the data as much as possible.

Before we begin exploration, we must think and ask questions related to the data in hand. A few probable questions are as follows:

- What are the most common words in the entire dataset?
- What are the most common words in the dataset for negative and positive tweets, respectively?
- Which trends are associated with my dataset?
- Which trends are associated with either of the sentiments? Are they compatible with the sentiments?
- What is the distribution of the dataset?

## End Notes

We learned how to approach a sentiment analysis problem. We started with preprocessing and exploration of data. Then we extracted features from the cleaned text using Bag-of-Words and TF-IDF. Finally, we were able to build a couple of models using both the feature sets to classify the tweets.
