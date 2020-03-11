import re
import requests
from bs4 import BeautifulSoup
from PIL import Image
from PIL import ImageEnhance
from pytesseract import *
import os.path
from lxml import html
import csv


def webspider(weblink):
    page = requests.get(weblink, timeout=600)
    weblist = []

    if page.status_code == requests.codes.ok:

        tree = html.fromstring(page.content)
        elements = tree.xpath('//ul[@class="listing-results clearfix js-gtm-list"]/li')

        for node in elements:
            id = node.xpath('./@data-listing-id')
            if id and id[0] != []:
                weblist.append(''.join(id))

        if weblist is None:
            print("Start Web Link Invalid")


    return weblist


def gethousesize_byfulldes(soup):
    size_ft_byDes = 0
    houseDescriptionhtml = soup.find("div", class_="dp-description__text")
    houseDescription = houseDescriptionhtml.get_text("\n", strip=True)

    # Get house size through square footage
    square_footage_text = re.findall(r"square footage: [0-9]+[.]?[0-9]*", houseDescription, flags=re.IGNORECASE)

    if square_footage_text:
        square_footage_text = re.sub("square footage: ", "", square_footage_text[0], flags=re.IGNORECASE)
        square_footage_text = square_footage_text.replace(",", "").replace(" ", "")
        square_footage = int(square_footage_text)
        return square_footage

    # Get house size through key features
    key_feature = soup.find("ul", class_="dp-features-list dp-features-list--bullets ui-list-bullets")

    if key_feature:
        key_feature = key_feature.get_text("\n", strip=True)
        key_feature = key_feature.split("\n")
        for i in key_feature:
            i = i.lower().replace(" ", "")
            if i.startswith("total"):
                size_byKeyFeature = re.findall(r"[0-9]+[.][0-9]+sq.m", i)
                if size_byKeyFeature:
                    size_ft_byDes = float(size_byKeyFeature[0].replace("sq.m", "")) * 10.7639
                    return size_ft_byDes
            if re.findall(r"internalarea", i):
                size_byKeyFeature = re.findall(r'[0-9]+[.]?[0-9]sq[.]?ft', i)
                if size_byKeyFeature:
                    size_byKeyFeature = re.findall(r"[0-9]+[.]?[0-9]", size_byKeyFeature[0])
                    size_ft_byDes = float(size_byKeyFeature[0])
                    return size_ft_byDes

    houseDescription = houseDescription.replace('”', '"').replace('"', '$').replace("’", "'")
    house_text = houseDescription.split("\n")

    sq_inch = []
    for i in house_text:

        sizeft_inch = re.findall(r"[0-9]+[ ]?['][ ]?[0-9]+", i)
        sizeft = re.findall(r"[0-9]+[']", i)
        sizem = re.findall(r"[0-9]+[.][0-9]*[m]", i)

        if len(sizem) == 1:
            x = re.findall(r"[0-9]+[m ]", i)
            if not x:
                sizem = []
            else:
                sizem.append(x[0])

        if sizeft_inch or sizem:
            if len(sizeft_inch) < len(sizem):
                if len(sizem) % 2 == 0:
                    for k in range(0, len(sizem), 2):
                        x = sizem[k].replace("m", "")
                        y = sizem[k + 1].replace("m", "")
                        sq_inch.append(float(x) * float(y) * 10.7639)
                        # print(str(x) + " " + str(y))
            else:
                if len(sizeft_inch) == 2:
                    x = (sizeft_inch[0].split("'"))
                    y = (sizeft_inch[1].split("'"))

                    mul = (int(x[0].replace(" ", "")) * 12 + int(x[1].replace(" ", ""))) * (
                            int(y[0].replace(" ", "")) * 12 + int(y[1].replace(" ", ""))) / 144
                    sq_inch.append(mul)
                else:
                    if len(sizeft_inch) == 0:
                        x = (sizeft[0].replace("'", "")).replace(" ", "")
                        y = (sizeft[1].replace("'", "")).replace(" ", "")
                        sq_inch.append(int(x) * int(y))

    for k in sq_inch:
        size_ft_byDes = size_ft_byDes + k

    return size_ft_byDes


def gethousesize_byfloorplan(soup):
    floorplan = soup.find("div", class_="dp-floorplan-assets dp-tabpanel__subgroup")
    if floorplan:
        sizebyPhoto = []

        floorplan = floorplan.find("img")
        if floorplan:
            # download the photo into folder
            floorplan_url = floorplan.get('data-src').replace("/u/480/360", "").replace("lid","lc")

            floorPlanfolder_path = "./floorplan/"
            if not os.path.exists(floorPlanfolder_path):
                os.makedirs(floorPlanfolder_path)

            floorPlan_link = requests.get(floorplan_url)
            img_name = floorPlanfolder_path + "floorplan.jpg"
            with open(img_name, "wb") as file:
                file.write(floorPlan_link.content)
                file.flush()
            file.close()

            img = Image.open(img_name)

            img = img.resize((1300, 1300))
            img = img.convert("RGB")
            img = ImageEnhance.Color(img).enhance(2)
            img = ImageEnhance.Contrast(img).enhance(2)
            img = ImageEnhance.Sharpness(img).enhance(1)

            img = img.convert("L")

            text = pytesseract.image_to_string(img)
            text = text.split("\n")

            for i in range(0, len(text)):
                if text[i].lower().startswith("total"):
                    sizebyPhoto = re.findall(r"[0-9]+[.]?[0-9]*", text[i])
                    if not sizebyPhoto:
                        sizebyPhoto = re.findall(r"[0-9]+", text[i])
                    break

            for i in range(0, len(sizebyPhoto)):
                sizebyPhoto[i] = float(sizebyPhoto[i])
    else:
        sizebyPhoto = []

    return sizebyPhoto


def webscraping(url):
    page = requests.get(url, timeout=600)
    print("Start: " + url)
    house_info = []

    if page.status_code == requests.codes.ok:
        html = page.text
        html = (html.replace("<br>", " ")).replace("<br/>", " ")
        soup = BeautifulSoup(html, "lxml")

        # house price
        house_price = soup.find("div", class_="ui-pricing").get_text(strip=True)
        house_price = re.findall(r'£[0-9,]+', house_price)
        if house_price:
            house_price = house_price[0]
            house_price = (house_price.replace("£", "")).replace(",", "")
            house_price = int(house_price)
        else:
            house_price = 0

        # bedroom number
        room_features = soup.find("ul", class_="dp-features-list dp-features-list--counts ui-list-icons")
        bedroom_num = 0
        bathroom_num = 0
        reception_num = 0
        if room_features:
            room_features = room_features.get_text(strip=False)
            bedroom = re.findall(r"[0-9]*[ ]*bedroom", room_features)
            if bedroom:
                bedroom = bedroom[0].replace("bedroom", "").replace(" ", "")
                bedroom_num = int(bedroom)

            bathroom = re.findall(r"[0-9]*[ ]*bathroom", room_features)
            if bathroom:
                bathroom = bathroom[0].replace("bathroom", "").replace(" ", "")
                bathroom_num = int(bathroom)

            reception = re.findall(r"[0-9]*[ ]*reception", room_features)
            if reception:
                reception = reception[0].replace("reception", "").replace(" ", "")
                reception_num = int(reception)

        # house address
        house_address = soup.find("h2", class_="ui-property-summary__address").get_text(strip=False)

        ########################################################################################
        # house size in Full description
        size_ft_bydes = gethousesize_byfulldes(soup)

        ########################################################################################
        # house size in Floor Plan
        # create the photo folder if not exist
        sizebyPhoto = gethousesize_byfloorplan(soup)
        size_photo = 0
        if sizebyPhoto:
            if len(sizebyPhoto) >= 2:
                size_photo = max(sizebyPhoto)

            if len(sizebyPhoto) == 1:
                size_photo = float(sizebyPhoto[0])

            size = size_photo

            if size_photo * 1.2 < size_ft_bydes:
                size = size_photo * 10.7639

        else:
            size = size_ft_bydes

        house_info = [house_price, house_address, size, bedroom_num, bathroom_num, reception_num]

        print("Finish：" + url)
    else:
        print("Failed：" + url)

    return house_info


if __name__ == '__main__':
    # nottinghamshire
    # derbyshire
    # lincolnshire

    with open("data_uk_derbyshire.csv", "w") as csvfile:

        writer = csv.writer(csvfile)
        writer.writerow(["url", "address", "price", "size", "average", "bedroom-number", "bathroom-number", "reception-number"])

        for i in range(1, 183):
            url = 'https://www.zoopla.co.uk/for-sale/property/derbyshire/?identifier=derbyshire&pagesize=25&q=DERBYSHIRE&search_source=refine&radius=0&pn=' + str(i)
            weblist = webspider(url)

            if not weblist:
                break
            else:
                for j in weblist:
                    houseurl = 'https://www.zoopla.co.uk/for-sale/details/' + j
                    house_info = webscraping(houseurl)
                    if house_info[2] == 0:
                        print("Invalid sample - house size")
                    elif house_info[0] == 0:
                        print("Invalid sample - house price")
                    else:
                        writer.writerow([j, house_info[1], house_info[0], house_info[2], int(house_info[0]) / house_info[2],
                                         house_info[3], house_info[4], house_info[5]])
