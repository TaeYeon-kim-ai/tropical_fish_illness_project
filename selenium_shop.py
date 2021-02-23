from bs4 import BeautifulSoup as bs
from urllib.request import urlopen
from urllib.parse import quote_plus
 
baseUrl = 'http://www.greenfish.co.kr/shop/shopbrand.html?xcode=011&mcode=001&type=X'
html = urlopen(baseUrl)
soup = bs(html, "html.parser")
img = soup.find_all(class_='MS_prod_img_s')
 
n = 1
for i in img:
    print(n)
    imgUrl = i['data-source']
    with urlopen(imgUrl) as f:
        with open('C:/tropical_fish_illness_project/data/fish_guppy_naver' + str(n)+'.jpg','wb') as h: # w - write b - binary
            img = f.read()
            h.write(img)
    n += 1
    if n > crawl_num:
        break
