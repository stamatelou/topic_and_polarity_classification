import requests
from requests import get
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import re
from datetime import date


headers = {"Accept-Language": "en-US, en;q=0.5"}
url = "https://webcache.googleusercontent.com/search?q=cache:pWsqbcGanDoJ:https://www.dutchnews.nl/news/category/corona/+&cd=1&hl=en&ct=clnk&gl=nl"
results = requests.get(url, headers=headers)
soup = BeautifulSoup(results.text, "html.parser")

headlines=[]
first_part = soup.find_all('h2', class_='d-none d-md-block mb-3')
second_part = soup.find_all('h2', class_='h3 d-none d-md-block mb-3')
third_part = soup.find_all('ul', class_='blocklist')

for container in first_part:
    headlines.append(container.a.text.strip())
    
for container in second_part:
    headlines.append(container.a.text.strip())

thrid=[]
third_par=re.sub(r'"blocklist"', 'blocklist', str(third_part), flags=re.MULTILINE)
third_pa=third_par[third_par.find("<ul class=blocklist>")+20:third_par.find("</ul>")]
for txt in third_pa.split("<li>"):
    #thrid.append(re.sub(r'<.+?>', '', txt))
    p1=re.sub(r'<a.+?>', '', txt, flags=re.MULTILINE)
    p2=re.sub(r'</a>', '', p1, flags=re.MULTILINE)
    p3=re.sub(r'</li>', '', p2, flags=re.MULTILINE)
    headlines.append(p3.strip())

print(date.today())
pd.DataFrame(headlines).to_csv('DutchNews_'+str(date.today())+'.csv')
