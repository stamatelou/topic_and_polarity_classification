# Topic and Polarity Classification of news
Topic and Polarity Classification of Dutch news related to the Corona virus outbreak. 

<strong> Input Data: </strong> Google News from USA related to COVID-19 outbreak from the topics of Healthcare, Science, Economy, and Travel. Google News gives the possibility to filter the news based on the country, COVID-19, and topic. </br>
<strong> Main goal: </strong> Classify the news based on the topic ( Healthcare, Science, Economy, and Travel) and on the polarity (positive, negative, neutral).</br>
<strong> End result: </strong> Deploy a dashboard which shows for each newsitem the detected topic and polarity labels.</br>

# <strong>Steps: </strong></br>

1) Scraping news from the Google News. </br>
We scraped the news from 18-04-2020 until 10-05-2020, the period of the coronavirus outbreak. </br>

2) Topic Classification</br>
For having better results, we used an ensemble model, which combines the results of 2 Machine Learning algorithms. </br>
  a) Logistic Regression</br>
  b) FastText</br>
3) Polarity Clasification with [VADER algorithm](https://github.com/cjhutto/vaderSentiment)</br>

4) Dashboard, which displays the news with their labels</br>

# How to start

Clone the repository and run main.py</br>

# Brief explanation of the files</br>
Input Data:
• Corona-ScrapedData: folder that contains the scraped data and the Python scripts used to scrape the data from Google News</br>

Main implementation:
• preprocessing.py: reads and preprocesses the scraped data</br>
• main.py : the main function which reads the training data, trains the models for topic and polarity classification, and predicts the labels for unknown newsitems</br>
• simple_text_classification.py: implements TFIDF and Logistic Regression training and prediction</br>
• FastText.py: implements the FastText algorith for Topic Classification. </br>
     Training: It uses the Google News with the topic categories (Healthcare, Science, Economy, and Travel) </br>
     Prediction: Given a newsitem it predicts its label (Healthcare, Science, Economy, and Travel)</br>
• polarity_analysis.py: implements the polarity classification algorithm (VADER). It uses a rule-based technique </br>
     Prediction: Given a newsitem it predicts its label (positive, negative, neutral)</br>

Results: 
• topic_classification_predictions.csv: prediction results of topic classification</br>
• polarity_predictions.csv : prediction results of polarity classification</br>



