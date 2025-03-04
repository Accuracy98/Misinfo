import requests
from bs4 import BeautifulSoup
import re

def extract_content(url):
    """精准提取目标内容"""
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, 'html.parser')
        
        # 精确提取标题
        title_quote = None
        for quote in soup.select('div.m-statement__quote'):
            text = clean_text(quote.get_text())
            if "Jonathan Simpson, the pilot of" in text:
                title_quote = text
                break
        
        # 提取 summary
        summary = None
        summary_tag = soup.select_one('h1.c-title')
        if summary_tag:
            summary = clean_text(summary_tag.get_text())

        # 提取声明内容（排除所有quote）
        claims = []
        main_content = soup.select_one('article.m-textblock')
        if main_content:
            for p in main_content.select('p:not(.m-statement__quote)'):
                text = clean_text(p.get_text())
                if text and text != title_quote:
                    claims.append(text)

        return {'title': title_quote, 'summary': summary, 'claims': claims}

    except Exception as e:
        print(f"Error: {str(e)}")
        return {'title': None, 'summary': None, 'claims': []}

def clean_text(text):
    """强化文本清洗"""
    text = re.sub(r'\s+', ' ', text)  # 合并空白字符
    text = re.sub(r'[\xa0\u200b]', '', text)  # 移除特殊字符
    return text.strip()

def save_results(data, filename='results.txt'):
    """结构化保存结果"""
    with open(filename, 'w', encoding='utf-8') as f:
        if data['title']:
            f.write(f"Title:\n{data['title']}\n\n")
        
        if data['summary']:
            f.write(f"Summary:\n{data['summary']}\n\n")
        
        if data['claims']:
            f.write("Claims:\n")
            unique_claims = list(dict.fromkeys(data['claims']))  # 去重保留顺序
            for claim in unique_claims:
                f.write(f"{claim}\n")

if __name__ == "__main__":
    target_url = 'https://www.politifact.com/factchecks/2025/feb/20/threads-posts/is-this-the-pilot-from-the-delta-air-lines-crash-n/'  # 替换实际URL
    data = extract_content(target_url)
    
    if data['title'] or data['summary'] or data['claims']:
        save_results(data)
        print("数据提取成功")
    else:
        print("未找到有效内容")