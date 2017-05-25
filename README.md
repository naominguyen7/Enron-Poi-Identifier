The project tries to predict the people involved in the Enron scheme, ‘persons of interest’. The model uses information about financial benefits and email communications, trying to figuring out patterns that distinguished the people of interests versus the rest. The dataset given includes individual’s information on financial benefits (payments and stock) involved with the company and email communications. It is obvious that the financial information will be useful for identifying people involved since money is the motive of the scheme. Individuals with especially high rate of communication the people of interests are likely to be one.

This is not a 50-50 bag of labels. There are 18 people of interests out of 144. There is a clear outlier with extreme financial data and this turns out to be the ‘total’ of all the data points, which is not a valid data point itself so I removed it. As for individuals that have extremely large financial values, I still keep the data points because they are people of interest. In terms of features, all features have missing data and some have a lot of missing data. Therefore, I edited the feature_format.py file to add an additional argument replace_median (replace NAs with median values).

*Note:*
Since this is a small dataset, I decided to run many grid-searched model. However, if you want to know the end of the movie, I find the model that applies principle component and logistic regression works best.
This is a project for my Data Science course at Udacity, and the code for feature_format.py(except for a few edits by me) and tester.py is written by Udacity



- poi_id.ipynb: Report with snippets of code. Can be used to run and generated the results.

- feature_format.py: Modified file from Udacity's. Add Option to fill NAs with medians.
    + converting data from dict form to numpy matrix for easy computation & manipulation
    + replace missing values with 0s or medians of the feature.
    + separating the data matrix to a vector of labels and a matrix of features
    
- my_classifier.pkl: Classifier with best performance

- my_dataset.pkl
- my_feature_list.pkl

- tester.py: include a function to test the classfier performance. 
    + employ stratified random shuffled splitting (since this is a small, skewed dataset)
    + give sensitivity, specificity, f1 score, accuracy
    


