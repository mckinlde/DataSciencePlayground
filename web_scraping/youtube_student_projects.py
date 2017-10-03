import requests
from time import sleep
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import re
from collections import Counter

#we want to scrape all the Youtube links from past student capstone projects

#step 1 let's put our base url somewhere convinient
base_url = "http://eecs.oregonstate.edu/industry-relations/capstone-and-senior-design-projects"

#okay now time to grab a webpage
def retrieve(url: str):
    print("*", url)
    sleep(1) #let's play nice with OSU's bandwidth
    r = requests.get(url, verify=False) #get the HTML, don't worry about SSL
    soup = BeautifulSoup(r.text, "lxml") #parse the HTML
    return soup

print(retrieve(base_url)) #does retrieve work?