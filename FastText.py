import fasttext

import pandas as pd
def fasttext_classification(usa, column_to_predict):   
    # validation: split the data into train and test set 
    import numpy as np
    import csv
#    from sklearn.model_selection import train_test_split
#    x_train, x_test, y_train, y_test = train_test_split(usa[['headline']], usa.tag, test_size=0.2)
#    usa_train = pd.concat([x_train,y_train], axis = 1)
#    usa_test = pd.concat([x_test,y_test], axis = 1)
#    # save the data
#    usa_train.to_csv(r'usa.train.txt', index=False, sep=' ', header=False, quoting=csv.QUOTE_NONE, quotechar="", escapechar=" ")
#    usa_test.to_csv(r'usa.test.txt', index=False, sep=' ', header=False, quoting=csv.QUOTE_NONE, quotechar="", escapechar=" ")
#
#    train_path = 'usa.train.txt'
#    model = fasttext.train_supervised(input = train_path)
#
#    results = model.test("usa.test.txt") # (number of samples, precision, recall)
#    print(results)

    # build the model
    usa[['headline', 'tag']].to_csv(r'usa.txt', index=False, sep=' ', header=False, quoting=csv.QUOTE_NONE, quotechar="", escapechar=" ")
    train_path = 'usa.txt'
    model = fasttext.train_supervised(input = train_path)
    
    # predict in unknown data
    labels = []
    for i in range(len(column_to_predict)):
        label = model.predict(column_to_predict[i])
        labels.append(label)

    predictions = pd.DataFrame(labels)
    predictions.columns = ['label','probability']
    predictions['label'] = predictions['label'].astype(str)
    predictions['label'] = [s.replace("(",'').replace(')','').replace('__label__','').replace(',','').replace("'",'')  for s in predictions['label']]
    predictions['probability'] = predictions['probability'].astype(str)
    predictions['probability'] = [s.replace("[",'').replace(']','')  for s in predictions['probability']]
    predictions['probability'] = predictions['probability'].astype(float)
    predictions['label_id'] = np.where(predictions['label'] == 'economy', 0 ,
                                       np.where(predictions['label'] == 'healthcare', 1, 
                                            np.where(predictions['label'] == 'science',2,3 )))
    return predictions

def cross_validation_fasttext(usa):
    from sklearn.model_selection import StratifiedKFold
    import numpy as np
    folds=StratifiedKFold(n_splits=10)
    train = usa[['headline', 'tag']]
    predictions = []
    accuracies = []
    entries = []
    for train_indices,val_indices in folds.split(train,usa.tag.values):
      x_train,x_val=train.iloc[train_indices],train.iloc[val_indices]
      x_train.to_csv(r'usa_train_cv.txt', index=False, sep=' ', header=False, quoting=csv.QUOTE_NONE, quotechar="", escapechar=" ")
      train_path = 'usa_train_cv.txt'
      
      model = fasttext.train_supervised(input = train_path)
      
      for i in range(len(x_val)):
        label_cv = model.predict(x_val.headline.iloc[i])
        predictions.append(label_cv)
      x_val.to_csv(r'usa_test_cv.txt', index=False, sep=' ', header=False, quoting=csv.QUOTE_NONE, quotechar="", escapechar=" ")
      result = model.test("usa_test_cv.txt")
      accuracies.append(result[1])
    
    predictions = pd.DataFrame(predictions)
    predictions.columns = ['label','probability']
    predictions['label'] = predictions['label'].astype(str)
    predictions['label'] = [s.replace("(",'').replace(')','').replace('__label__','').replace(',','').replace("'",'')  for s in predictions['label']]
    predictions['probability'] = predictions['probability'].astype(str)
    predictions['probability'] = [s.replace("[",'').replace(']','')  for s in predictions['probability']]
    predictions['probability'] = predictions['probability'].astype(float)
    predictions['label_id'] = np.where(predictions['label'] == 'economy', 0 ,
                                       np.where(predictions['label'] == 'healthcare', 1, 
                                            np.where(predictions['label'] == 'science',2,3 )))
   
    model_name = 'FastText' 
    for fold_idx, accuracy in enumerate(accuracies):
            entries.append((model_name, fold_idx, accuracy))
    cv_df = pd.DataFrame(entries, columns=['model_name', 'fold_idx', 'accuracy'])
    import seaborn as sns
    sns.boxplot(x='model_name', y='accuracy', data=cv_df)
    sns.stripplot(x='model_name', y='accuracy', data=cv_df, 
              size=8, jitter=True, edgecolor="gray", linewidth=2)
    return predictions