from selenium import webdriver
from selenium.webdriver.common.keys import Keys
import time
import urllib.request
#selenium, chromedriver

driver=webdriver.Chrome("C:/tropical_fish_illness_project/chromedriver.exe")
driver.get("https://www.google.co.kr/imghp?hl=ko&ogbl")
search = driver.find_element_by_name("q")
search.send_keys("구피") #이미지 열심히 찾자
search.send_keys(Keys.RETURN)

#스크롤 내려서 미리 펼쳐두자
SCROLL_PAUSE_TIME = 1

# Get scroll height
last_height = driver.execute_script("return document.body.scrollHeight")

while True:
    # Scroll down to bottom
    driver.execute_script("window.scrollTo(0, document.body.scrollHeight);") #브라우저 끝까지 스크롤 내리기

    # Wait to load page
    time.sleep(SCROLL_PAUSE_TIME)

    # Calculate new scroll height and compare with last scroll height
    new_height = driver.execute_script("return document.body.scrollHeight") #브라우저 높이 구하기
    if new_height == last_height: # 만약 스크롤 끝까지 내려가면
        try : #코드 실행했을때 오류생기면 나가라
            driver.find_element_by_css_selector(".mye4qd").click()        
        except :
            break

    last_height = new_height


#다운로드
images = driver.find_elements_by_css_selector(".rg_i.Q4LuWd") #class 선택, 첫번째요소 끄집어내기
count = 1
for image in images :
    try :
        image.click()
        time.sleep(0.8)
        imgurl = driver.find_element_by_class_name('n3VNCb').get_attribute("src")
        opener=urllib.request.build_opener()
        opener.addheaders=[('User-Agent','Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/36.0.1941.0 Safari/537.36')]
        urllib.request.install_opener(opener)
        urllib.request.urlretrieve(imgurl, str(count) + ".jpg")

        count = count + 1
    except :
        pass

driver.close()
