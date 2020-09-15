import pandas as pd
import datetime
from datetime import timedelta

def read(file_path):
    df = pd.read_csv(file_path)
    df = pd.DataFrame(df.iloc[:,1]).astype(str)
    df.columns = ['headline']
    return df

def clean(df, column):
    # clean 
    df[column]=[s.replace("'",'').replace("[]",str(0)).replace('[','').replace(']','').replace("\\","'").replace("&amp", "and")  for s in df[column]]
    df[column] = df[column].str.lower()
    return df
    # add labels
def label(df, tag_name):
    df['tag'] = ['__label__' + tag_name for s in df['headline']]
    return df

def read_economy(file_path):
    tag_name = 'economy'
    economy_usa = read(file_path)
    economy_usa = clean(economy_usa, 'headline')
    economy_usa = label(economy_usa,tag_name)
    return economy_usa

def read_healthcare(file_path):
    tag_name = 'healthcare'
    healthcare_usa = read(file_path)
    healthcare_usa = clean(healthcare_usa, 'headline')
    healthcare_usa =label(healthcare_usa,tag_name)
    return healthcare_usa

def read_science(file_path):
    tag_name = 'science'
    science_usa = read(file_path)
    science_usa = clean(science_usa, 'headline')
    science_usa = label(science_usa,tag_name)
    return science_usa

def read_travel(file_path):
    tag_name = 'travel'
    travel_usa = read(file_path)
    travel_usa = clean(travel_usa, 'headline')
    travel_usa = label(travel_usa,tag_name)
    return travel_usa

def train_data(number_of_dates_usa):
  ### create train data ###
#  number_of_dates_usa = 11
  start_date = datetime.date(2020, 4, 18)
  
  date = start_date
  economy_usa = pd.DataFrame()
  for i in range(number_of_dates_usa):
      file_path = 'Corona-ScrapedData/EcoUsaNews_'+ str(date) + '.csv'
      economy_usa_new = read_economy(file_path)
      economy_usa_new['date'] = date
      economy_usa = pd.concat([economy_usa, economy_usa_new])
      date = date + timedelta(days=1) 

  date = start_date
  healthcare_usa = pd.DataFrame()
  for i in range(number_of_dates_usa):
      file_path = 'Corona-ScrapedData/HlthUsaNews_'+ str(date) + '.csv'
      healthcare_usa_new = read_healthcare(file_path)
      healthcare_usa_new['date'] = date
      healthcare_usa = pd.concat([healthcare_usa, healthcare_usa_new])
      date = date + timedelta(days=1) 
      
  date = start_date
  science_usa = pd.DataFrame()
  for i in range(number_of_dates_usa):
      file_path = 'Corona-ScrapedData/SciUsaNews_'+ str(date) + '.csv'
      science_usa_new = read_science(file_path)
      science_usa_new['date'] = date
      science_usa = pd.concat([science_usa, science_usa_new])
      date = date + timedelta(days=1)
      
  date = start_date
  travel_usa = pd.DataFrame()
  for i in range(number_of_dates_usa):
      file_path = 'Corona-ScrapedData/TrvUsaNews_'+ str(date) + '.csv'
      travel_usa_new = read_travel(file_path)
      travel_usa_new['date'] = date
      travel_usa = pd.concat([travel_usa, travel_usa_new])
      date = date + timedelta(days=1)

  # save train data
  usa = pd.concat([economy_usa,healthcare_usa,science_usa, travel_usa])
  usa = usa[usa['headline'] != '0']
  return usa

# ### test data ###
def separate(df):
    new =  df['headline'].str.split("]", n = 1, expand = True)   
    df['headline']= new[0] 
    df['content']= new[1] 
    return df
def read_dutch_news(file_path):
    df = read(file_path)
    df = separate(df)
    df = clean(df, 'headline')
    df = clean(df, 'content')
    return df


def translate(dutch_news_new, date):
    ## Google translate
    from googletrans import Translator
    translated = pd.DataFrame()
    translated_all = pd.DataFrame()
    for i in range(len(dutch_news_new['headline'])):
            translator = Translator()
            translated['headlines_en']= pd.Series(translator.translate(dutch_news_new['headline'][i]).text)
#            translator = Translator()
            translated['content_en']= pd.Series(translator.translate(dutch_news_new['content'][i]).text)
            translated_all = pd.concat([translated_all,translated])

    translated_all = translated_all.reset_index(drop = True)
    dutch_news_new = pd.concat([dutch_news_new,translated_all],axis= 1, ignore_index=True)
    dutch_news_new.columns = ['headlines','content','date','headlines_en', 'content_en']
    return dutch_news_new

def new_test_data(date):
    # 22/4 until 4/5 , start with 5
    number_of_dates_dutch = 1 
    start_date = datetime.date(2020, 5, 5)
    
    date = start_date
    dutch_news = pd.DataFrame()
    for i in range(number_of_dates_dutch):
      print("start with the date", date)
      file_path = 'Corona-ScrapedData/GoogleNews_'+ str(date) + '.csv'
      dutch_news_new = read_dutch_news(file_path)
      dutch_news_new['date'] = date
      dutch_news_new = translate(dutch_news_new, date)
      dutch_news = pd.concat([dutch_news, dutch_news_new])
      print("complete with the date", date)
      date = date + timedelta(days=1) 
      
      
    dutch_news = dutch_news.reset_index(drop=True)
    file_path = 'Corona-ScrapedData/Dutch_news_translated_1'+ '.csv'
    dutch_news.to_csv(file_path, index=False)
    return dutch_news

def read_dutchnews_translated_data():
    dutch_news_translated = pd.read_csv("Corona-ScrapedData/Dutch_news_translated_.csv")
    return dutch_news_translated



