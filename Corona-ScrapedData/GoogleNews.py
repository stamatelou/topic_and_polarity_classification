import requests
from requests import get
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import re
from datetime import date


headers = {"Accept-Language": "en-US, en;q=0.5"}
##only for last 24 hours
url = "https://news.google.com/search?q=Corona%20when%3A1d&hl=nl&gl=NL&ceid=NL%3Anl"
results = requests.get(url, headers=headers)
soup = BeautifulSoup(results.text, "html.parser")

headlines=[]
a_part = soup.find_all('div', jslog='93789')
for container in a_part:
    b_part=container.find_all('article',jslog='85008')
    headlines.append(b_part) 

titles=[]
for wrd in headlines:
    e_part=re.findall('<a class="DY5T1d" .+?</a>',str(wrd))
    f_part=re.sub(r'</a>', '', str(e_part), flags=re.MULTILINE)
    c_part=re.findall('<span class="xBbh9">.+?</span>',str(wrd))
    d_part=re.sub(r'<span class="xBbh9">', '', str(c_part), flags=re.MULTILINE)
    titles.append(re.sub(r'<a class="DY5T1d" .+?>', '', str(f_part), flags=re.MULTILINE)+
    re.sub(r'</span>', '', str(d_part), flags=re.MULTILINE))

pd.DataFrame(titles).to_csv('GoogleNews_'+str(date.today())+'.csv')