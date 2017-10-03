import requests
from time import sleep
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import string
import re
from collections import Counter

#we want to scrape all the Youtube links from past student capstone projects

#step 1 let's put our base url somewhere convinient
BASE_URL = "http://eecs.oregonstate.edu/industry-relations/capstone-and-senior-design-projects"

#okay now time to grab a webpage
def retrieve(url: str):
    print("*", url)
    sleep(1) #let's play nice with OSU's bandwidth
    r = requests.get(url, verify=False) #get the HTML, don't worry about SSL
    soup = BeautifulSoup(r.text, "lxml") #parse the HTML
    return soup

#nice retrieve works let's get all the youtube links
def get_links():
    #we'll store the links here
    result_list = {}

    def load_video_links_from(url):
        soup = retrieve(url)
        #I'm a caveman so let's just grab every stinkin' link
        links = soup.select("a ")

        #but I only want youtube links
        for link in links:
            video_url = link.get('href')
            if 'www.youtube.com' in video_url:
                video_title = link.get('title')
                print(video_title)
                print(video_url)
                result_list[video_title] = video_url
                
    load_video_links_from(BASE_URL)
    return result_list

#Did our link retrival work?
links = get_links()
print(links)