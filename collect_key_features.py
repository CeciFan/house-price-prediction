import requests
from bs4 import BeautifulSoup
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


if __name__ == '__main__':

    folder_path = "./key_features/"

    # nottinghamshire
    # derbyshire
    # lincolnshire
    # south-yorkshire

    with open("key_features_south-yorkshire.csv", "w") as csvfile:

        writer = csv.writer(csvfile)
        writer.writerow(["id"])

        for i in range(1, 191):
            url = 'https://www.zoopla.co.uk/for-sale/property/south-yorkshire/?identifier=south-yorkshire&pagesize=25&q=SOUTH-YORKSHIRE&search_source=refine&radius=0&pn='+str(i)
            weblist = webspider(url)

            if not weblist:
                break
            else:
                for j in weblist:
                    houseurl = 'https://www.zoopla.co.uk/for-sale/details/' + j
                    page = requests.get(houseurl, timeout=600)

                    if page.status_code == requests.codes.ok:
                        pagehtml = page.text
                        pagehtml = (pagehtml.replace("<br>", " ")).replace("<br/>", " ")
                        soup = BeautifulSoup(pagehtml, "lxml")
                        key_features = soup.find("ul", class_="dp-features-list dp-features-list--bullets ui-list-bullets")
                        if key_features:
                            file_name = folder_path + j + ".txt"
                            f = open(file_name, "w+")
                            key_features = key_features.get_text("\n", strip=True)
                            key_features = key_features.split("\n")
                            for i in key_features:
                                f.write(i + "\n")
                            f.close()
                            writer.writerow([j])
                            print("Succeed：" + houseurl)
                        else:
                            print("Failed: " + houseurl)

                    else:
                        print("Failed：" + houseurl)

