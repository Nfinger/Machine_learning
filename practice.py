# dataframe
# import numpy
# import pandas
# myarray = numpy.array([[1, 2, 3], [4, 5, 6]])
# rownames = ['a', 'b']
# colnames = ['one', 'two', 'three']
# mydataframe = pandas.DataFrame(myarray, index=rownames, columns=colnames)
# #print(mydataframe)

# # Load CSV using Pandas from URL
# import pandas
# url = "https://goo.gl/vhm1eU"
# names = ["preg", "plas", "pres", "skin", "test", "mass", "pedi", "age", "class"]
# data = pandas.read_csv(url, names=names)
# description = data.describe()
# # print(description)

# # import matplotlib.pyplot as plt
# # import pandas
# # from pandas.tools.plotting import scatter_matrix
# # url = "https://goo.gl/vhm1eU"
# # names = ["preg", "plas", "pres", "skin", "test", "mass", "pedi", "age", "class"]
# # data = pandas.read_csv(url, names=names)
# # scatter_matrix(data)
# # plt.show()


# plt.scatter(test["value"], test["next_day"])
# plt.plot(test["value"], regressor.predict(test[["value"]]))
# plt.show()

# import pandas as pd
# from sklearn.cluster import KMeans
# import matplotlib.pyplot as plt

# nfl = pd.read_csv("2015_qb_rankings.csv")
# print(nfl.head())
# nfl = nfl[nfl["Cmp"] > 35]
# nfl = nfl.dropna()
# kmeans = KMeans(n_clusters=5)
# kmeans.fit(nfl[["Att","QBR"]])
# nfl["cluster"] = kmeans.labels_

# def visualize_clusters(df, num_clusters):
#     colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']

#     for n in range(num_clusters):
#         clustered_df = df[df['cluster'] == n]
#         plt.scatter(clustered_df['Att'], clustered_df['QBR'], c=colors[n-1])
#         plt.xlabel('Attempts', fontsize=13)
#         plt.ylabel('Quarterback Rating', fontsize=13)
#     plt.show()

# visualize_clusters(nfl, 5)

#Practice ML

##########################################################################################
####################################LINEAR REGRESSION#####################################
##########################################################################################


# import pandas

# #load data
# sp500 = pandas.read_csv("sp500.csv")
# sp500.head()
# #clean data
# # sp500 = sp500[sp500['Open'] != '.']
# # This prints the last 10 rows -- note where the dataset ends.
# print(sp500.tail(10))
# next_day = sp500["Open"].iloc[1:]
# #remove final day
# sp500 = sp500[sp500["Date"]!='2015-06-26']
# sp500["next_day"] = next_day.values
# print(sp500)
# # We can see the current types of the columns
# print(sp500.dtypes)
# #change them to floats
# sp500["Open"] = sp500["Open"].astype(float)
# sp500["next_day"] = sp500["next_day"].astype(float)
# # Import the linear regression class
# from sklearn.linear_model import LinearRegression

# # Initialize the linear regression class.
# regressor = LinearRegression()

# # We're using 'value' as a predictor, and making predictions for 'next_day'.
# # The predictors need to be in a dataframe.
# # We pass in a list when we select predictor columns from "sp500" to force pandas not to generate a series.
# predictors = sp500[["Open"]]
# to_predict = sp500["next_day"]

# # Train the linear regression model on our dataset.
# regressor.fit(predictors, to_predict)

# # Generate a list of predictions with our trained linear regression model
# next_day_predictions = regressor.predict(predictors)
# print(next_day_predictions)

# #The equation is MSE=1/n∑(Yi^−Yi)^2
# # The actual values are in to_predict, and the predictions are in next_day_predictions.
# _sum = 0
# sub = (next_day_predictions - to_predict) ** 2
# _sum = sub.sum()
# mse = _sum/len(next_day_predictions)
# print(mse.values)

# #Overfitting
# # Making predictions on data you've trained a model on is known as overfitting.
# # The best way to avoid overfitting is to make predictions on data that hasn't been
# # used to train the model. We randomly assign some data to a training set, to train the algorithm,
# # and some data to a test set, where we make predictions and evaluate error.
# import numpy as np
# import random

# # Set a random seed to make the shuffle deterministic.
# np.random.seed(1)
# random.seed(1)
# # Randomly shuffle the rows in our dataframe
# sp500 = sp500.loc[np.random.permutation(sp500.index)]

# # Select 70% of the dataset to be training data
# highest_train_row = int(sp500.shape[0] * .7)
# train = sp500.loc[:highest_train_row,:]

# # Select 30% of the dataset to be test data.
# test = sp500.loc[highest_train_row:,:]

# regressor = LinearRegression()
# regressor.fit(train[["value"]], train["next_day"])
# predictions = regressor.predict(test[["value"]])
# mse = sum((predictions - test["next_day"]) ** 2) / len(predictions)
# print(mse)

# #Error metrics can tell us a lot about how good a regression model is,
# #but sometimes being able to visualize what's going on is much more valuable.
# import matplotlib.pyplot as plt

# # Make a scatterplot with the actual values in the training set
# plt.scatter(train["value"], train["next_day"])
# plt.plot(train["value"], regressor.predict(train[["value"]]))
# plt.show()
# plt.scatter(test["value"], test["next_day"])
# plt.plot(test["value"], regressor.predict(test[["value"]]))
# plt.show()

# # Two other commonly used error metrics are root mean squared error,
# #  or RMSE, and mean absolute error, or MAE.

# # RMSE is just the square root of MSE.

# # RMSE=√1/n∑ni=1(Yi^−Yi)2

# # MAE is the mean of the absolute values of
# # all the differences between prediction and actual values.

# # MAE=1n∑ni=1∣∣Ŷ i−Yi∣∣MAE=1n∑i=1n|Y^i−Yi|
# # MSE and RMSE, because they square the errors,
# # penalize large errors way out of proportion to small errors. 
# # MAE, on the other hand, doesn't. MAE can be useful, because it is a 
# # more accurate look at the average error.

# # The test set predictions are in the predictions variable.
# import math
# rmse = math.sqrt(sum((test["next_day"]-predictions)**2) / len(predictions))
# mae = sum(abs(test["next_day"]-predictions)) / len(predictions)
# print(rmse)
# print(mae)


# ##########################################################################################
# ####################################Logistic Regression###################################
# ##########################################################################################

# # The fundamental goal of machine learning is to understand the relationship between the 
# # independent variable(s) and the dependent variable. Specifically, we're interested in the 
# # underlying mathematical function that uses the features to generate labels. In supervised 
# # machine learning, we use training data that contains a label for each row to approximate this function.


# # In classification, our target column has a finite set of possible values which represent different categories 
# # a row can belong to. We use integers to represent the different categories so we can continue to use 
# # mathematical functions to describe how the independent variables map to the dependent variable. 

# # When numbers are used to represent different options or categories, they are referred to
# # as categorical values. Classification focuses on estimating the relationship between the 
# # independent variables and the dependent, categorical variable.


# # Here's the mathematical representation of the logit function:

# # σ(t)=e^t/(1+e^t)
# # The logit function is broken up into 2 key parts:

# # The exponential transformation, transforming all values to be positive:
# # etet
# # The normalization transformation, transforming all values to range between 0 and 1:
# # # t1+t
# # import pandas as pd
# # import matplotlib.pyplot as plt
# # %matplotlib inline

# # admissions = pd.read_csv('admissions.csv')
# # plt.scatter(admissions["gpa"],admissions['admit'])
# # plt.show()


# # from sklearn.linear_model import LogisticRegression
# # logistic_model = LogisticRegression()
# # logistic_model.fit(admissions[["gpa"]], admissions["admit"])
# # #Plot the predicted probabilities
# # pred_probs = logistic_model.predict_proba(admissions[["gpa"]])
# # plt.scatter(admissions["gpa"], pred_probs[:,1])

# # # the scatter plot suggests a linear relationship between the gpa values 
# # # and the probability of being admitted. This if because logistic regression 
# # # is really just an adapted version of linear regression for classification problems. 
# # # Both logistic and linear regression are used to capture linear relationships between 
# # # the independent variables and the dependent variable.

# # # Let's now use the predict method to return the label predictions for each row 
# # # in our training dataset.

# # logistic_model = LogisticRegression()
# # logistic_model.fit(admissions[["gpa"]], admissions["admit"])
# # fitted_labels = logistic_model.predict(admissions[["gpa"]])
# # print(fitted_labels)

# # # The admissions Dataframe now contains the predicted value for that row, 
# # # in the predicted_label column, and the actual value for that row, in the 
# # # admit column. This format makes it easier for us to calculate how effective 
# # # the model was on the training data. The simplest way to determine the effectiveness 
# # # of a classification model is prediction accuracy. Accuracy helps us answer the question:

# # # What fraction of the predictions were correct (actual label matched predicted label)?
# # # Prediction accuracy boils down to the number of labels that were correctly predicted 
# # # divided by the total number of observations:

# # # Accuracy=# of Correctly Predicted# of ObservationsAccuracy=# of Correctly Predicted# of Observations
# # # In logistic regression, recall that the model's output is a probability 
# # # between 0 and 1. To decide who gets admitted, we set a threshold and accept 
# # # all of the students where their computed probability exceeds that threshold. This 
# # # threshold is called the discrimination threshold and scikit-learn sets it to 0.5 by 
# # # default when predicting labels. If the predicted probability is greater than 0.5, the 
# # # label for that observation is 1. If it is instead less than 0.5, the label for that observation is 0.

# # # An accuracy of 1.0 means that the model predicted 100% of admissions correctly 
# # # for the given discrimination threshold. An accuracy of 0.2 means that the model 
# # # predicted 20% of the admissions correctly. Let's calculate the accuracy for the predictions 
# # # the logistic regression model made.

# # labels = model.predict(admissions[["gpa"]])
# # admissions["predicted_label"] = labels
# # admissions["actual_label"] = admissions["admit"]
# # matches = admissions["predicted_label"] == admissions["actual_label"]
# # correct_predictions = admissions[matches]
# # print(correct_predictions.head())
# # accuracy = len(correct_predictions) / len(admissions)
# # print(accuracy)


























