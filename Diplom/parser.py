#Loading libraries
import os
import pandas as pd
import requests
import time
from fake_headers import Headers
from random import randint
from bs4 import BeautifulSoup

# Generating headers
HEADERS = Headers(browser="chrome",
                  os="win",
                  headers=True).generate()

# Dictionary with links to parse:
url_dict = {'proc' : 'https://www.e-katalog.ru/list/186/',
            'mb' : 'https://www.e-katalog.ru/list/187/',
            'video': 'https://www.e-katalog.ru/list/189/',
            'ram' : 'https://www.e-katalog.ru/list/188/',
            'ssd':'https://www.e-katalog.ru/list/61/',
            'hdd':'https://www.e-katalog.ru/list/190/',
            'case': 'https://www.e-katalog.ru/list/193/',
            'power':'https://www.e-katalog.ru/list/351/',
            'cool':'https://www.e-katalog.ru/list/303/'}

# Notebooks link, it needs another parse method:
notes_dict = {'notebooks':'https://www.e-katalog.ru/list/298/'}

# Some parts of links, used to create links
DOMAIN = 'https://www.e-katalog.ru'
DOMAIN_PHP = 'https://www.e-katalog.ru/ek-item.php?resolved_name_='
review_php = '&view_=review'
table_php = '&view_=tbl'

''' This function is trying to get response 
    and returns it if everything is ok:'''

def test_request(url, retry=5):
    try:
        response = requests.get(url=url, headers=HEADERS)
        if response.status_code != 200:
            raise
    except:
        time.sleep(randint(30,50))
        if retry:# retries several times (default - 5)
            print(F'Request retries left: {retry}')
            return test_request(url, retry=(retry - 1))
        else:
            raise
    else:
        return response

'''In the begining we need to count the number of page to parse
and make the links set with the next function:'''

def pagination(page_url, retry=5):
    
    try:
        response = test_request(page_url)# getting page
        soup = BeautifulSoup(response.content, "lxml")
        # looking for the last page number:  
        pages_count = int(soup.find('div',class_='ib page-num').find_all('a')[-1].text)
        print(f"{soup.title.text}")
        print(f"Pagination:{pages_count}")
    except:
        time.sleep(randint(5,10))
        if retry:# retries several times (default - 5)
            print(F'Pagination retries left: {retry}')
            pagination(page_url,retry = retry - 1)
        else:
            print(f"Can't achieve url {page_url}\n{'-' * 20}" )
            return 'error'
    else:
        pages_set = set()
        for n in range(0,pages_count):
            full_url = f"{page_url}{n}/"
            pages_set.add(full_url)
    
        return pages_set

'''This function collect all of the links in selected category(url)
and saving them to list'''

def get_links(url):
    links = pagination(url)
    if type(links) != set:
        return 'error'
    else:
        urls = set()
        while len(links) != 0:
            current_url = links.pop()
            try:
                response = test_request(current_url)
            except:
                
                print(f"error with current url: {current_url}\n{'-' * 20}" )
                time.sleep(randint(5,10))# waiting to not get banned 
                continue   
            else:
                soup = BeautifulSoup(response.text, 'lxml')
                page = soup.findAll('td',class_='model-short-info')
            
                for p in page:
                    for a in p.findAll('a'):
                        if a.get('href') != None:
                            urls.add(a.get('href'))
                
                time.sleep(randint(5,10))# waiting to not get banned 
                
        
        # Filter links for wrong ones and save: 
        id_list = [u.replace('/','').replace('.htm','') for u in urls if (u != '#' and 'list' not in u)]
        print('Number of links received:',len(id_list),f"\n{'-' * 20}")
        return id_list 
        
'''Parse all products links from dict and save them
to  "links" directory'''
def links_from_dict(dict):
    
    for i in dict:
        links_list = get_links(dict[i])
        if links_list == 'error':
            print(f'error with {i}')
        else:
            df = pd.DataFrame({i:links_list})
            df.to_csv(f'data/links/{i}.csv',index=False)
            del df

'''Notebooks needs special parser:'''    
def note_links(url):
    links = pagination(url)
    if type(links) != set:
        return 'error'
    else:
        urls = set()
        while len(links) != 0:
            current_url = links.pop()
            try:
                response = test_request(current_url)
            except:
                print(f"error with current url: {current_url}\n{'-' * 20}" )
                time.sleep(randint(5,10))# waiting to not get banned 
                continue   
            else:
                soup = BeautifulSoup(response.text, 'lxml')
                page = soup.findAll('td',class_='model-conf-title')
            
                for p in page:
                    for a in p.findAll('a'):
                        if a.get('href') != None:
                            urls.add(a.get('href'))
                
                time.sleep(randint(5,10))# waiting to not get banned 
                
        
        # Filter links for wrong ones and save: 
        id_list = [u for u in urls if '#' not in u]
        
        note_urls = set()
        for i in id_list:
            try:
                response = test_request(DOMAIN + i) 
            except:
                print(f"error with current url: {i}\n{'-' * 20}" )
                time.sleep(randint(5,10))# waiting to not get banned 
                continue
            else:        
                soup = BeautifulSoup(response.text, 'lxml')
                page = soup.findAll('tr',class_='conf-tr')
                
                for p in page:
                    for a in p.findAll('a'):
                        ref = a.get('href')
                        if (ref != None) and ('#' not in ref):
                            note_urls.add(ref)
        notes_list = [u for u in note_urls if 'php' not in u]
                       
        print('Number of links received:',len(notes_list),f"\n{'-' * 20}")
        notes = pd.DataFrame(notes_list,columns=['notebooks'])    
        return notes 

'''This function extract product's characteristics 
for every product in series and returns dataframe'''

def get_data(series):
    items = [] # Here product specifications will be stored
    oos = [] # out of sale products
  
    for s in series:
        try:
            if series.name == 'notebooks':# separate link for notebooks
                response = test_request(DOMAIN + s)
                product_link = DOMAIN + s
            else:
                product_link = DOMAIN + '/' + s + '.htm'
                response = test_request(DOMAIN_PHP + s + table_php)
            data = pd.read_html(response.content, attrs = {'cellpadding':'3'})
            soup = BeautifulSoup(response.text, 'lxml')
            time.sleep(randint(1,2))
            if ('Ожидается в продаже'in soup.text) or ('устарел' in soup.text):
                oos.append(s)
            else:
                labels = []  
                vals = [] 
                price = int(soup.find('span',
                            itemprop = 'lowPrice').get_text().replace(u"\xa0",''))
                spec = {'id' : s,
                        'name' : soup.title.get_text().split(' – ')[0],
                        'price': price,
                        'link': product_link}        
            
                if 'Свободный множитель' in soup.text:
                    spec.update({'overclock': 1})
                
                for d in data:
                    d.dropna(inplace=True)
                    for f in range(len(d)):
                        if d.iloc[f,0] not in labels:
                            labels.append(d.iloc[f,0])
                        else:   
                            labels.append(d.iloc[f,0]+'*') 
                    vals.extend(list(d.iloc[:,1]))
                
                spec.update(dict(zip(labels,vals)))                    
                items.append(spec)
                
        except:
            print(f'Can not get data from: {s}')
            time.sleep(randint(5,10))
            continue
       
    return pd.DataFrame(items),oos

'''Saving all data for products in directory'''

def data_from_dir(dir):
    
    out_of_sale = []
    for file in os.listdir(dir):
        print(f"Loading data from {file}\n{'-' * 20}" )
        df = pd.read_csv(f'{dir}/{file}')
        data_df,oos= get_data(df.iloc[:,0])
        data_df.to_csv('data/raw_data/' + file,index=False)
        out_of_sale.extend(oos)
    pd.DataFrame(out_of_sale).to_csv('data/raw_data/out_of_sale.csv',index=False)

'''Like the get_data function gets product's reviews'''

def get_reviews(series):
    reviews = []
    for s in series:
        try:
            response = test_request(DOMAIN_PHP + s + review_php)
            data = pd.read_html(response.content,
                                attrs = {'class':'review-table'})
            
            soup = BeautifulSoup(response.content,'lxml')
            page = soup.findAll('div',class_='review-title') 
            for i,d in enumerate(data):
                rev = { 'id' : s,
                        'date':d[1][0].split('\xa0')[1],
                        'text': d[1][0].split('\xa0')[0]+d[1][0].split('\xa0')[2],
                        'likes': d[0][0].split(',')[3],
                        'dislikes' : d[0][0].split(',')[4],
                        'rating' : page[i].find('img').get('src').split('smile-')[1].replace('.svg','')
                        }
                reviews.append(rev)

            time.sleep(randint(1,2))    
        except:
            print(f'Can not get reviews from: {s}')
            time.sleep(randint(1,3))
            continue
    return pd.DataFrame(reviews)

'''Get reviews for links in directory and save them'''

def reviews_from_dir(dir):
    
    for file in os.listdir(dir):
        df = pd.read_csv(f'{dir}/{file}')
        rev_df= get_reviews(df.iloc[:,0])
        rev_df['product'] = file.split('.')[0]
        rev_df.to_csv('data/reviews/' + file,index=False)
        
'''Run needed operations'''

def main():
    
    #links_from_dict(url_dict)
    #note_links(notes_dict['notebooks']).to_csv('data/links/notebooks.csv',index = False)
    data_from_dir('data/links/')
    import  preprocess # Preprocess raw data to datasets
    #reviews_from_dir('links')
    
if __name__ == "__main__":
    main()
