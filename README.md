# Topic and Polarity Classification of news
Topic and Polarity Classification of Dutch news related to the Corona virus outbreak. 

<strong> Input Data: </strong> Google News from USA related to COVID-19 outbreak from the topics of Healthcare, Science, Economy, and Travel. Google News gives the possibility to filter the news based on the country, COVID-19, and topic.
<strong> Main goal: </strong> Classify the news based on the topic ( Healthcare, Science, Economy, and Travel) and on the polarity (positive, negative, neutral)
<strong> End result: </strong> Deploy a dashboard which shows for each newsitem the detected topic and polarity labels 
<strong>Steps </strong>

1) Scraping news from the Google News. 
We scraped the news from 18-04-2020 until 10-05-2020, the period of the coronavirus outbreak. 

2) Topic Classification
For having better results, we used an ensemble model, which combines the results of 2 Machine Learning algorithms. 
  a) Logistic Regression
  b) FastText
3) Polarity Clasification with VADER algorith

4) Dashboard, which displays the news with their labels
