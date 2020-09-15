# info in https://colab.research.google.com/drive/1JW2I6cU_ypfRXfIfqMPQwMEA6LzGav-Y#forceEdit=true&sandboxMode=true&scrollTo=KY7W_YW5s5Jw
import pandas as pd
def preprocessing_samples(df):
    # preprocessing classification
    # convert tags into classes with numbers 
    df['category_id'] = df['tag'].factorize()[0]
    labels = df.category_id
    return df, labels

def feature_extraction(column):
    # feature extraction
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.feature_extraction import text
    # TF (term frequency) - IDF (inverse document frequency) --> 
    # frequency of the term multiplied with the frequency of the term in all the documents 
    # (to reduce the impact of stop words or context related less useful words)
    # sublinear_df = True to use a logarithmic form for frequency (Zipf's Law).
    # min_df is the minimum numbers of documents a word must be present in to be kept, to avoid rare words, which drastically increase the size of our features and might cause overfitting.
    # norm is l2, to ensure all our feature vectors have a euclidian norm of 1. 
    # encoding is latin-1 which is used by our input text.
    # ngram_range is (1, 2) to indicate that we want to consider both unigrams and bigrams
    # stop_words is "english" to remove stopwords and noise 
    my_stop_words = text.ENGLISH_STOP_WORDS.union(["coronavirus","covid19", "covid", "COVID19", "19"])
    tfidf = TfidfVectorizer(sublinear_tf=True, min_df=5, norm='l2', encoding='latin-1', 
                        ngram_range=(1, 2), stop_words=my_stop_words, use_idf = True)
    # columns are the different terms and the rows are the number of documents (here headlines)
    # apply function fit trasform to generate the features
    features = tfidf.fit_transform(column).toarray()
    
    return features, tfidf

def most_dominant_features(df, features, tfidf, labels):
    # Chi-square test to find the most dependant features that can predict the target variable
    from sklearn.feature_selection import chi2
    import numpy as np
    category_id_df = df[['tag', 'category_id']].drop_duplicates().sort_values('category_id')
    category_to_id = dict(category_id_df.values)
    
    N = 5
    for category, category_id in sorted(category_to_id.items()):
    # x- squared test is a feature selection technique
    # it measures independance between variables --> we aim to find dependant variables 
    # variables that predict the target variable --> 
    # the higher the score the higher the dependancy
  # x-squared features = words for one class --> the higher the number 
      features_chi2 = chi2(features, labels == category_id)
      indices = np.argsort(features_chi2[0])
      feature_names = np.array(tfidf.get_feature_names())[indices]
      unigrams = [v for v in feature_names if len(v.split(' ')) == 1]
      bigrams = [v for v in feature_names if len(v.split(' ')) == 2]
      print("# '{}':".format(category))
      print("  . Most correlated unigrams:\n       . {}".format('\n       . '.join(unigrams[-N:])))
      print("  . Most correlated bigrams:\n       . {}".format('\n       . '.join(bigrams[-N:])))
    return
 
#    # t-SNE clustering to find how much the documents are distinguishable   
#    from sklearn.manifold import TSNE
#    import matplotlib.pyplot as plt
#    # Sampling a subset of our dataset because t-SNE is computationally expensive
#    SAMPLE_SIZE = int(len(features))
#    np.random.seed(0)
#    indices = np.random.choice(range(len(features)), size=SAMPLE_SIZE, replace=False)
#    projected_features = TSNE(n_components=2, random_state=0).fit_transform(features[indices])
#    colors = ['pink', 'green', 'midnightblue', 'orange']
#    for category, category_id in sorted(category_to_id.items()):
#        points = projected_features[(labels[indices] == category_id).values]
#        plt.scatter(points[:, 0], points[:, 1], s=30, c=colors[category_id], label=category, alpha = 0.3)
#    plt.title("tf-idf feature vector for each article, projected on 2 dimensions.",
#              fontdict=dict(fontsize=15))
#    plt.legend()
  
    ## hyperparameter tuning for logistic regression(C regularization coefficient - l1,l2 regularization)
    #from sklearn.linear_model import LogisticRegression
    #from sklearn.model_selection import GridSearchCV
    ## Create regularization penalty space
    #penalty = ['l2']
    ## Create regularization hyperparameter space
    #C = [0.001,0.01,0.1,1,10,100]
    ## Create hyperparameter options
    #hyperparameters = dict(C=C, penalty=penalty)
    ## Create grid search using 5-fold cross validation
    #clf = GridSearchCV(LogisticRegression(verbose = 0, dual = False, max_iter  = 1000), hyperparameters, cv=10, 
    #                   scoring='accuracy')
    ## Fit grid search
    #best_model = clf.fit(features, labels)
    ## View best hyperparameters
    #print('Best Penalty:', best_model.best_estimator_.get_params()['penalty'])
    #print('Best C:', best_model.best_estimator_.get_params()['C'])

def cross_validation_logistic_regression(df):
    [df, labels] = preprocessing_samples(df)
    [features, tfidf] = feature_extraction(df.headline)
#     Validate different models
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score
    from sklearn.model_selection import cross_val_predict
    import numpy as np
    models = [
            LogisticRegression(C=5,solver='saga',penalty='elasticnet',l1_ratio=0.5)
            ]
    CV = 10
    cv_df = pd.DataFrame(index=range(CV * len(models)))
    entries = []
    for model in models:
        model_name = model.__class__.__name__
        print(model_name)
        accuracies = cross_val_score(model, features, labels, scoring='accuracy', cv=CV,  n_jobs = -1)
        predictions_labels = cross_val_predict(model, features, labels, cv=CV,  n_jobs = -1)
        predictions_proba = cross_val_predict(model, features, labels, cv=CV,  n_jobs = -1, method = 'predict_proba')
        predictions_proba = predictions_proba.max(axis = 1)
        for fold_idx, accuracy in enumerate(accuracies):
            entries.append((model_name, fold_idx, accuracy))
        cv_df = pd.DataFrame(entries, columns=['model_name', 'fold_idx', 'accuracy'])
    import seaborn as sns
    sns.boxplot(x='model_name', y='accuracy', data=cv_df)
    sns.stripplot(x='model_name', y='accuracy', data=cv_df, 
              size=8, jitter=True, edgecolor="gray", linewidth=2)
    predictions = pd.concat([pd.DataFrame(predictions_labels),pd.DataFrame(predictions_proba)], axis = 1)
    predictions.columns = ['label_id','probability']
    predictions['label'] = np.where(predictions['label_id'] == 0, 'economy' ,
                                       np.where(predictions['label_id'] == 1, 'healthcare', 
                                            np.where(predictions['label_id'] == 2,'science','travel' )))
    return predictions

def logistic_regression_classification(df, column_to_predict):
    import numpy as np
    [df, labels] = preprocessing_samples(df)
    # feature extraction
    [features, tfidf] = feature_extraction(df.headline)
    dutch_news_features = tfidf.transform(column_to_predict.astype('U')).toarray()
    
    most_dominant_features(df, features, tfidf, labels)
    # Select a model from the above and train it 
    from sklearn.linear_model import LogisticRegression
    model = LogisticRegression(C=5,solver='saga',penalty='elasticnet',l1_ratio=0.5)
    model.fit(features, labels)
    y_pred_proba = model.predict_proba(dutch_news_features)
    maximum_probabilities =  pd.DataFrame(y_pred_proba.max(axis = 1))
    y_pred = pd.DataFrame(model.predict(dutch_news_features))

    predictions = pd.DataFrame(pd.concat([y_pred, maximum_probabilities], axis = 1))
    predictions.columns = ['label_id','probability']
    predictions['label'] = np.where(predictions['label_id'] == 0, 'economy' ,
                                       np.where(predictions['label_id'] == 1, 'healthcare', 
                                            np.where(predictions['label_id'] == 2,'science','travel' )))
   
    
    return predictions


## model intrepretation 
#from sklearn.model_selection import train_test_split
#
#model = LogisticRegression(random_state=0)
#
#X_train, X_test, y_train, y_test, indices_train, indices_test = train_test_split(features, labels, df.index, test_size=0.33, random_state=0)
#model.fit(X_train, y_train)
#y_pred_proba = model.predict_proba(X_test)
#y_pred = model.predict(X_test)
#
#from sklearn.metrics import confusion_matrix
#import seaborn as sns
#
#conf_mat = confusion_matrix(y_test, y_pred)
#sns.heatmap(conf_mat, annot=True, fmt='d',
#            xticklabels=category_id_df.tag.values, yticklabels=category_id_df.tag.values)
#plt.ylabel('Actual')
#plt.xlabel('Predicted')
#
#
#from IPython.display import display
#id_to_category = dict(category_id_df[['category_id', 'tag']].values)
#for predicted in category_id_df.category_id:
#  for actual in category_id_df.category_id:
#    if predicted != actual and conf_mat[actual, predicted] >= 2:
#      print("'{}' predicted as '{}' : {} examples.".format(id_to_category[actual], id_to_category[predicted], conf_mat[actual, predicted]))
#      display(df.loc[indices_test[(y_test == actual) & (y_pred == predicted)]][['headline']])
#      print('')
      
#model.fit(features, labels)

#from sklearn.feature_selection import chi2
#
#N = 5
#for category, category_id in sorted(category_to_id.items()):
#  indices = np.argsort(model.coef_[category_id])
#  feature_names = np.array(tfidf.get_feature_names())[indices]
#  unigrams = [v for v in reversed(feature_names) if len(v.split(' ')) == 1][:N]
#  bigrams = [v for v in reversed(feature_names) if len(v.split(' ')) == 2][:N]
#  print("# '{}':".format(category))
#  print("  . Top unigrams:\n       . {}".format('\n       . '.join(unigrams)))
#  print("  . Top bigrams:\n       . {}".format('\n       . '.join(bigrams)))