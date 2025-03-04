import requests
from bs4 import BeautifulSoup
import re
import json

def get_article_links(base_url, pages=1):
    """从列表页抓取所有文章链接"""
    articles = []
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
        for page in range(1, pages + 1):
            url = f"{base_url}?page={page}"
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')
            
            for a_tag in soup.select('div.m-statement__quote a'):
                link = a_tag.get('href')
                if link and link.startswith('/factchecks/'):
                    full_url = "https://www.politifact.com" + link
                    articles.append(full_url)

    except Exception as e:
        print(f"Error fetching article links: {str(e)}")
    
    return articles

def extract_content(url):
    """从单篇文章抓取数据"""
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')

        # 1. 精确提取标题
        title_quote = None
        for quote in soup.select('div.m-statement__quote'):
            text = clean_text(quote.get_text())
            if text:
                title_quote = text
                break

        # 2. 提取 summary
        summary = None
        summary_tag = soup.select_one('h1.c-title')
        if summary_tag:
            summary = clean_text(summary_tag.get_text())

        # 3. 提取声明内容（排除所有 quote）
        claims = []
        main_content = soup.select_one('article.m-textblock')
        if main_content:
            for p in main_content.select('p:not(.m-statement__quote)'):
                text = clean_text(p.get_text())
                if text and text != title_quote:
                    claims.append(text)

        # 显示抓取成功的URL和标题
        print(f"✅ {url} - {title_quote if title_quote else 'No Title'}")

        return {'url': url, 'title': title_quote, 'summary': summary, 'claims': claims}

    except Exception as e:
        print(f"❌ Error extracting content from {url}: {str(e)}")
        return {'url': url, 'title': None, 'summary': None, 'claims': []}

def clean_text(text):
    """强化文本清洗"""
    text = re.sub(r'\s+', ' ', text)  # 合并空白字符
    text = re.sub(r'[\xa0\u200b]', '', text)  # 移除特殊字符
    return text.strip()

def save_results(data_list, filename='results2.json'):
    """保存结果为 JSON"""
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(data_list, f, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    base_url = 'https://www.politifact.com/factchecks/list/'
    article_links = get_article_links(base_url, pages=2)  # 抓取前2页的文章链接

    extracted_data = [extract_content(url) for url in article_links]

    if extracted_data:
        save_results(extracted_data)
        print("\n🎉 数据提取完成，已保存为 results2.json")
    else:
        print("❌ 未找到有效内容")
