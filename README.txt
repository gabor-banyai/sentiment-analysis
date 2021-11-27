### Web Data Processing Systems 2, Vrije Universiteit Amsterdam, 19 Dec. 2020
Included in this directory is the report, code and video of the presentation
of the second assignment of Web Data Processing Systems of group 20.

# Requirements.txt
In this file the modules for the code can be installed using pip install

# reddit_scraper.py
Code for gathering the data

# urls.txt
All urls of reddit threads that were used

This program is made in Python and tested on Windows systems

## INSTRUCTIONS TO EXECUTE THE PROGRAM ##
execute the program with the following command in the terminal: 
	python Ensembling.py <n>
with:	n=1 VADER Sentiment Analysis
	n=2 TextBlob Sentiment Analysis
	n=3 AFINN Sentiment Analysis
	n=4 ensembling of sentiment analysis
	n=5 logistic regression
	n=6 SE weighted avg. logistic, blob, svm, nb
	n=7 SE max_voting logistic, blob, svm, nb
	n=8 AE logistic, blob+gradient boosting, svm
	n=9 AE logistic, blob+ada boosting, svm
	n=10 AE logistic, blob+bagging, svm

##OUTPUT
the accuracy of the model

##AUTHORS:
	P. Bijl, 
	M. Malkoc, 
	G. Banyai,
	A. Abdelghany