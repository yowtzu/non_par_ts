
# coding: utf-8

# In[ ]:

##################################### Method 1
import mechanize
import cookielib
from BeautifulSoup import BeautifulSoup
import html2text

# Browser
br = mechanize.Browser()

# Cookie Jar
cj = cookielib.LWPCookieJar()
br.set_cookiejar(cj)

# Browser options
br.set_handle_equiv(True)
br.set_handle_gzip(True)
br.set_handle_redirect(True)
br.set_handle_referer(True)
br.set_handle_robots(False)
br.set_handle_refresh(mechanize._http.HTTPRefreshProcessor(), max_time=1)

br.addheaders = [('User-agent', 'Chrome')]

# The site we will navigate into, handling it's session
br.open('https://github.com/login')

# View available forms
for f in br.forms():
    print f

# Select the second (index one) form (the first form is a search query box)
br.select_form(nr=1)

# User credentials
br.form['login'] = 'mylogin'
br.form['password'] = 'mypass'

# Login
br.submit()

print(br.open('https://github.com/settings/emails').read())


# In[27]:

from selenium import webdriver
driver = webdriver.Safari()
driver.get('http://pems.dot.ca.gov/?redirect=&username=ebusseti%40stanford.edu&password=%3AUujazzy1%40&login=Login')

print(driver.page_source)


# In[24]:

import requests

LOGINURL = 'http://pems.dot.ca.gov/'
DATAURL = 'http://pems.dot.ca.gov/?dnode=Clearinghouse&type=station_5min&district_id=4&submit=Submit'

session = requests.session()

# req_headers = {
#     'Content-Type': 'application/x-www-form-urlencoded'
# }

formdata = {
    'redirect': None,
    'username': 'ebusseti@stanford.edu',
    'password': ':Uujazzy1@',
    'login':'Login'
}

# formdata2 = {
#     'dnode':'Clearinghouse',
#     'type':'station_5min',
#     'district_id':4,
#     'submit':'Submit'
# }

# Authenticate
r = session.get(LOGINURL, data=formdata)#, headers=req_headers, allow_redirects=False)
print (r.headers)
print (r.status_code)
print (r.text)

# # Read data
# r2 = session.get(DATAURL, data=formdata2, headers=req_headers)
# print( "___________DATA____________")
# print (r2.headers)
# print (r2.status_code)
# print (r2.text)


# In[4]:

r2.status_code


# In[5]:

r2.text


# In[ ]:



