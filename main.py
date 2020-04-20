import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn import linear_model as lm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree.export import export_text

from pca import pca_func


r = input('enter c for classification, r for regression\n')
if r == 'c':
  classify = True
else:
  classify = False

if classify == True:
  regression = False


def setplt(x = 13, y = 9, a = 1, b = 1):
    f, ax = plt.subplots(a,b,figsize = (x,y))
    sns.despine(f, left = True, bottom = True)
    return f, ax

# NEW FILTERED MOVIE NAMES - RE APPROACH other filters (years)
features = pd.read_csv('newDataSet.csv')
#﻿budget,company,country,director,genre,gross,name,rating,released,runtime,score,star,votes,writer,year,Column1,Name2,tweets analyzed,movieTermCount,total retweets,positive percentage,positive retweets,negative percentage,negative retweets,neutral tweet percentage,P_N,P_N_N,PoverNratio,PandNoverNratio
#data frame


features['country'] = features['country'].astype('category')
features['country_cat'] = features['country'].cat.codes
# UK = 35, USA = 36

features['rating'] = features['rating'].astype('category')
features['rating_cat'] = features['rating'].cat.codes

features['genre'] = features['genre'].astype('category')
features['genre_cat'] = features['genre'].cat.codes

features['score1'] = pd.qcut(features['score'], q=4)
features['score1'] = features['score1'].astype('category')
features['score_cat'] = features['score1'].cat.codes

features['P_Nratio1'] = pd.qcut(features['P_Nratio'], q=4)
features['P_Nratio1'] = features['P_Nratio1'].astype('category')
features['P_Nratio_cat'] = features['P_Nratio1'].cat.codes

features.head(10)

features = features.drop(columns=['company', 'score1', 'P_Nratio1', 'country', 'director', 'writer', 'Name2', 'Column1', 'rating', 'star', 'genre','name','votes', 'released'])


print(features.head(5))
#filter out low tweets, generic movie names, 0 budget
#drop.row
nonZBudg = features['budget'] > 0
features = features[nonZBudg]
manyTweets = features['tweets analyzed'] > 20
features = features[manyTweets]
USonly = features['country_cat'] == 36
features = features[USonly]
#manyVotes = features['votes'] > 80
#features = features[manyVotes]
#rescale "score"

#TRY CONTINUOUS VALS
features = features.drop(columns=['tweets analyzed', 'negative retweets', 'positive retweets', 'negative percentage', 'neutral tweet percentage', 'positive percentage', 'P_Nratio', 'PandN/Nratio', 'total retweets'])
#or convert cats to numerical
#Genre, Year/Season, 
# hypotheses: later years more accurate because twitter is more popular?


#PCA?
#features = pca_func(features)
print(features.head(5))
#cluster? - remove rows
#filtrations (interpret)
#PCA? - remove info overlap


#ML
#algorithms?? -- svm, random forest, knn
# logistic (classify)/ linear regression
# id3
#score?? + revenue --- contrinuos
x = input('Choose Label:\na)P_N\nb)P_N_N\nc)genre\nd)gross revenue(regression)\ne)score(regression)\nf)score_cat(class)\n')
if x == 'a':
  lab = 'P_N'
  labels = features.P_N
  features = features.drop(columns=['P_N_N'])
if x == 'b':
  lab = 'P_N_N'
  labels = features.P_N_N
  features = features.drop(columns=['P_N'])
if x == 'c':
  lab = 'genre_cat'
  labels = features.genre_cat
if x == 'd':
  lab = 'gross'
  labels = features.gross
if x == 'e':
  lab = 'score'
  labels = features.score
if x == 'f':
  lab = 'score_cat'
  labels = features.score_cat

print("\n\n\nLabel in use:" + str(lab))
# TRAIN TEST SPLIT
#try labels: budget, gross, score, sentiment
print(labels.head(2))
features = features.drop(columns=[lab])
X = features
y = labels
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)
print('Shape of training features : \t' + str(X_train.shape))
print('Shape of training labels : \t' + str(y_train.shape))
print('Shape of testing features : \t' + str(X_test.shape))
print('Shape of testing labels : \t' + str(y_test.shape))



#MODELS

#Regression::
if classify == False:
  #features = features.drop(columns=['country_cat'])
  print('\n\n\nRegression Coefficients:')
  reg = lm.LinearRegression()
  reg.fit(X_train, y_train)
  print(list(X_train))
  print(reg.coef_)
  print('R squared: \n' + str(reg.score(X_train, y_train)))
else:
    
  #Logistic Regression:: - classification
  #print('\n\n\nLogistic Regression:')


  # print("\n\n\nID3#2")

  # decision_tree = DecisionTreeClassifier(random_state=0, max_depth=4)
  # decision_tree = decision_tree.fit(X_train, y_train)
  # r = export_text(decision_tree, feature_names = list(X_train.columns.values))
  #print(r)


  print("\n\n\nID3#3 - depth = 4")

  decision_tree = DecisionTreeClassifier(random_state=0, max_depth=4)
  decision_tree = decision_tree.fit(X_train, y_train)
  #r = export_text(decision_tree, feature_names = list(X_train.columns.values))
  #print(r)
  tree.plot_tree(decision_tree.fit(X_test, y_test))
  plt.savefig('viz3.png')
  print("see viz3.png\nReference for X[i]")
  print(list(X_train))

  print("\n\n\nID3#3 - depth = 3")

  decision_tree = DecisionTreeClassifier(random_state=0, max_depth=3)
  decision_tree = decision_tree.fit(X_train, y_train)
  #r = export_text(decision_tree, feature_names = list(X_train.columns.values))
  #print(r)
  tree.plot_tree(decision_tree.fit(X_test, y_test))
  plt.savefig('viz4.png')
  print("see viz4.png\nReference for X[i]")
  print(list(X_train))



  #KNN::
  print('\n\n\nKNN:')
  acc_list = []
  for x in range(1,41):
      knn = KNeighborsClassifier(n_neighbors = x)
      knn.fit(X_train, y_train)
      acc_list.append(knn.score(X_test, y_test))

  setplt(8,5)
  sns.lineplot(x = range(1,41), y = acc_list)
  plt.savefig("viz.png")

  neighbors = acc_list.index(max(acc_list))
  print('Accuracy : ' + str(max(acc_list)) + ' at ' + str(neighbors) + ' nearest neighbors')
  print('see viz 1')



  #Random Forest:
  print('\n\n\nRandom Forest:')
  rf = RandomForestClassifier(n_estimators=10, criterion='gini') # Using Gini index to measure feature importance 
  for x in range(10,81,5):
      rf = RandomForestClassifier(n_estimators = x)
      rf.fit(X_train, y_train)
      acc_list.append(rf.score(X_test, y_test))

  setplt(8,5)
  sns.lineplot(x = range(len(acc_list)), y = acc_list)
  plt.savefig("viz2.png")

  trees = acc_list.index(max(acc_list))
  print('Accuracy : ' + str(max(acc_list)) + ' with ' + str(trees) + ' trees')
  print('see viz 2')


  #SVM::


  #tf.Keras / DEEP LEARN / Multi Layer Perceptron::
  #NaivesBayes??
  #conclusions? need dif data?
  #visualize ROC https://scikit-learn.org/stable/auto_examples/plot_roc_curve_visualization_api.html#sphx-glr-auto-examples-plot-roc-curve-visualization-api-py
  #own NLP