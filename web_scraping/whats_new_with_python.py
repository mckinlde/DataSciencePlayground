# What's New With Python?
# Let's scrape Wikipedia and print out all the unique links
# that were modified in the last month.

import requests
from time import sleep
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import string
import re
from collections import Counter

#step 1 let's put our base url somewhere convinient
BASE_URL = "https://en.wikipedia.org/wiki/Python_(programming_language)"

#okay now time to grab a webpage
def retrieve(url: str):
    print("*", url)
    sleep(1) #let's play nice with bandwidth
    r = requests.get(url, verify=False) #get the HTML, don't worry about SSL
    soup = BeautifulSoup(r.text, "lxml") #parse the HTML
    return soup

#nice retrieve works let's get all the links
def get_links():
    #we'll store the links here
    result_list = []

    def load_links_from(url):
        soup = retrieve(url)
        #I'm a caveman so let's just grab every stinkin' link
        links = soup.select("a ")

        for link in links:
            result_list.append(link.get('href'))

    load_links_from(BASE_URL)
    return result_list

#iterate over all urls
links = get_links()

for link in links:
    absolute = urljoin(BASE_URL, links.pop())#convert to absolut
    page = requests.get(absolute) #get the HTML
    soup = BeautifulSoup(page.text, "lxml") #parse the HTML
    links_in_page = soup.find_all('li') #find all the links in the page
    for item in links_in_page:
        if(item.find('en.wikipedia.org') != -1): #only want english links
            if(item.get('id') == 'footer-info-lastmod'): #only want last modified text
                last_modified = item.string
                print(last_modified)


