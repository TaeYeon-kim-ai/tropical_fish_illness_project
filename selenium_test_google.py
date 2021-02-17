from selenium import webdriver
from selenium.webdriver.common.keys import Keys
import time
from bs4 import BeautifulSoup
import urllib.request
path = "C:/tropical_fish_illness_project/data/fish/chromedriver.exe"
driver = webdriver.Chrome(path)
driver.get("https://www.google.co.kr/imghp?hl=ko&ogbl")
search = driver.find_element_by_name("q")
search.send_keys("guppy") #이미지 열심히 찾자
search.send_keys(Keys.RETURN)
driver.find_elements_by_css_selector(".rg_i.Q4LuWd")[0].click() #class 선택, 첫번째요소 끄집어내기
time.sleep(2)
imgurl = driver.find_element_by_css_selector(".n3VNCb").get_attribute("src")
urllib.request.urlretrieve(imgurl,"test.jpg")