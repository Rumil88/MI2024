import requests
from bs4 import BeautifulSoup
import csv

# URL сторінки
url = 'https://www.zsu.gov.ua/category/news/page/15/'

# Імітація Google Chrome
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 '
                  '(KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36'
}

# Запит до сайту
response = requests.get(url, headers=headers)

if response.status_code == 200:
    soup = BeautifulSoup(response.text, 'html.parser')

    # Отримання даних з <head>
    page_title = soup.title.string if soup.title else 'No title found'
    meta_og_title = soup.find('meta', property='og:title')
    og_title = meta_og_title['content'] if meta_og_title else 'No og:title found'

    # Запис даних з head у CSV
    with open('zsu_page_head_info.csv', 'w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['Title', 'OG Title'])
        writer.writerow([page_title, og_title])

    print("✅ Дані з head збережено у 'zsu_page_head_info.csv'")

    # Отримання заголовків новин
    try:
        # Знаходимо всі div-елементи з класом post__title
        post_titles = soup.find_all('div', class_='post__title')
        
        # Створюємо CSV файл для збереження заголовків новин
        with open('zsu_div_info.csv', 'w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow(['Title', 'Link'])
            
            for title_div in post_titles:
                # Отримуємо текст заголовка
                title_text = title_div.get_text(strip=True)
                
                # Отримуємо посилання, якщо воно є
                link = title_div.find('a')
                link_href = link['href'] if link else 'No link'
                
                # Записуємо інформацію
                writer.writerow([title_text, link_href])
            
            print(f"✅ Знайдено та збережено {len(post_titles)} заголовків новин у 'zsu_div_info.csv'")
        
    except Exception as e:
        print(f"❌ Помилка при обробці даних: {str(e)}")
else:
    print(f"❌ Помилка запиту: {response.status_code}")
