import preprocessing
import simple_text_classification
import FastText
import polarity_analysis
import pandas as pd
import numpy as np
### READ ###
# read the train data 
usa = preprocessing.train_data(20)
usa.to_csv("usa.csv", index=False)
# read the training data 
# from until 22 april - 2 may
dutch_news = preprocessing.read_dutchnews_translated_data()

### CLASSIFICATION: healthcare, science, economy, travel
## Logistic Regression 
dutch_news_topics = pd.DataFrame()
predictions_headlines = simple_text_classification.logistic_regression_classification(usa, dutch_news['headlines_en'])
predictions_headlines.columns = ['label_id_headline','probability_headline', 'label_headline']
predictions_content = simple_text_classification.logistic_regression_classification(usa, dutch_news['content_en'])
predictions_content.columns = ['label_id_content','probability_content', 'label_content']
dutch_news_topics = pd.DataFrame(pd.concat([dutch_news, predictions_headlines, predictions_content], axis = 1))

# FastText
predictions_headlines_fasttext = FastText.fasttext_classification(usa, dutch_news['headlines_en'])
predictions_headlines_fasttext.columns = ['f_label_headline','f_probability_headline', 'f_label_id_headline']
predictions_content_fasttext = FastText.fasttext_classification(usa, dutch_news['content_en'])
predictions_content_fasttext.columns = ['f_label_content','f_probability_content', 'f_label_id_content']
dutch_news_topics = pd.DataFrame(pd.concat([dutch_news_topics, predictions_headlines_fasttext, predictions_content_fasttext], axis = 1))

# define final tag
#
dutch_news_topics['final_tag'] = '0'

dutch_news_topics['maximum'] = dutch_news_topics[['probability_headline','probability_content', 'f_probability_headline', 'f_probability_content']].idxmax(axis=1)
dutch_news_topics['final_tag'] = np.where(dutch_news_topics['maximum']=='probability_headline',dutch_news_topics['label_headline'], dutch_news_topics['final_tag'])
dutch_news_topics['final_tag'] = np.where(dutch_news_topics['maximum']=='probability_content',dutch_news_topics['label_content'], dutch_news_topics['final_tag'])
dutch_news_topics['final_tag'] = np.where(dutch_news_topics['maximum']=='f_probability_headline',dutch_news_topics['f_label_headline'], dutch_news_topics['final_tag'])
dutch_news_topics['final_tag'] = np.where(dutch_news_topics['maximum']=='f_probability_content',dutch_news_topics['f_label_content'], dutch_news_topics['final_tag'])

dutch_news_topics_final = dutch_news_topics[['headlines', 'content', 'date', 'headlines_en', 'content_en','final_tag']]
dutch_news_topics_final.to_csv("topic_classification_predictions.csv", index = False)

### VISUALIZATIONS ### 
# preprocessing for visualization of categories
news_per_day_category = dutch_news_topics.groupby(['date', 'final_tag'], as_index=False).count()
news_per_day_category['date'] = pd.to_datetime(news_per_day_category['date'], format = '%d-%m-%y')
news_per_day_category = news_per_day_category.sort_values(by='date') 
economy_news = news_per_day_category[news_per_day_category['final_tag'] == 'economy']
healthcare_news = news_per_day_category[news_per_day_category['final_tag'] == 'healthcare']
science_news = news_per_day_category[news_per_day_category['final_tag'] == 'science']
travel_news = news_per_day_category[news_per_day_category['final_tag'] == 'travel']

import plotly.graph_objects as go
from plotly.offline import plot
fig = go.Figure(data=[
            go.Scatter(x=economy_news.date, y = economy_news.headlines, name = 'economy'), 
            go.Scatter(x=healthcare_news.date, y = healthcare_news.headlines, name = 'healthcare'), 
            go.Scatter(x=science_news.date, y = science_news.headlines, name = 'science'), 
            go.Scatter(x=travel_news.date, y = travel_news.headlines, name = 'travel')
            ])
fig.show()
fig.update_layout(
    title="Trends per topic through time",
    xaxis_title="Date",
    yaxis_title="number of newsitems",
    font=dict(
        size=18,
        color="black"
    ))
plot(fig)


### POLARITY ##
dutch_news_polarity = pd.DataFrame()
polarity_headlines = polarity_analysis.polarity(dutch_news['headlines_en']).reset_index(drop = True)
polarity_content = polarity_analysis.polarity(dutch_news['content_en']).reset_index(drop = True)
dutch_news_polarity = pd.DataFrame(pd.concat([dutch_news, polarity_headlines, polarity_content], axis = 1))
dutch_news_polarity.columns = ['headline', 'content','date', 'headline_en','content_en', 'hnegative', 'hneutral', 'hpositive', 'hcompound',
       'cnegative', 'cneutral', 'cpositive', 'ccompound']
dutch_news_polarity['average_polarity'] = (dutch_news_polarity.hcompound + dutch_news_polarity.ccompound)/2
polarity_news_final = dutch_news_polarity[['headline', 'content','date', 'average_polarity']]
polarity_news_final['tag'] = np.where(polarity_news_final['average_polarity']>0.5, "positive", 
                             np.where(polarity_news_final['average_polarity']<-0.3, "negative","neutral"))

### VISUALIZATIONS ### 
# preprocessing for visualization of categories
news_per_day_polarity = polarity_news_final.groupby(['date', 'tag'], as_index=False).count()
news_per_day_polarity['date'] = pd.to_datetime(news_per_day_polarity['date'], format = '%d-%m-%y')
news_per_day_polarity = news_per_day_polarity.sort_values(by='date') 
positive_news = news_per_day_polarity[news_per_day_polarity['tag'] == 'positive']
neutral_news = news_per_day_polarity[news_per_day_polarity['tag'] == 'neutral']
negative_news = news_per_day_polarity[news_per_day_polarity['tag'] == 'negative']

fig = go.Figure(data=[
            go.Scatter(x=positive_news.date, y = positive_news.headline, name = 'positive'), 
            go.Scatter(x=neutral_news.date, y = neutral_news.headline, name = 'neutral'), 
            go.Scatter(x=negative_news.date, y = negative_news.headline, name = 'negative'), 
            ])
fig.show()
fig.update_layout(
    title="Trends per polarity through time",
    xaxis_title="Date",
    yaxis_title="number of newsitems",
    font=dict(
        size=18,
        color="black"
    ))
plot(fig)




polarity_news_final.to_csv("polarity_predictions.csv", index = False)