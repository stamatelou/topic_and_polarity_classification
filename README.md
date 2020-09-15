# Topic and Polarity Classification of news
Topic and Polarity Classification of Dutch news related to the Corona virus outbreak. 

<strong> Input Data: </strong> Google News from USA related to COVID-19 outbreak from the topics of Healthcare, Science, Economy, and Travel. Google News gives the possibility to filter the news based on the country, COVID-19, and topic. </br>
<strong> Main goal: </strong> Classify the news based on the topic ( Healthcare, Science, Economy, and Travel) and on the polarity (positive, negative, neutral).</br>
<strong> End result: </strong> Deploy a dashboard which shows for each newsitem the detected topic and polarity labels.</br>
<strong>Steps </strong></br>

1) Scraping news from the Google News. </br>
We scraped the news from 18-04-2020 until 10-05-2020, the period of the coronavirus outbreak. </br>

2) Topic Classification</br>
For having better results, we used an ensemble model, which combines the results of 2 Machine Learning algorithms. </br>
  a) Logistic Regression</br>
  b) FastText</br>
3) Polarity Clasification with VADER algorith</br>

4) Dashboard, which displays the news with their labels</br>
