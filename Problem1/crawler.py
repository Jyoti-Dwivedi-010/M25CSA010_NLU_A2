import requests
from bs4 import BeautifulSoup
import re
import os
import json
from urllib.parse import urljoin, urlparse
from collections import deque
import PyPDF2
from io import BytesIO

# Crawler configuration
MAX_DOCS = 200

#  seeds strictly matching documents in the assignment :
# (Departments, Academic Programs, Research Pages, Announcements, and Academic Regulations)
SEED_URLS = [
    "https://iitj.ac.in/academics/index.php?id=regulations", # Regulations (Must)
    "https://iitj.ac.in/department/index.php?id=cse", # Departments
    "https://iitj.ac.in/department/index.php?id=ee", # departments
    "https://iitj.ac.in/department/index.php?id=me", # departments
    "https://iitj.ac.in/academics/index.php?id=programs", # Academic Programs
    "https://iitj.ac.in/research/index.php", # Research Pages
    "https://iitj.ac.in/institute/index.php?id=announcements_list", # Announcements
    "https://iitj.ac.in/" # Main site for circulars/newsletters
]
# check if the text is predominantly English and does not contain Hindi characters
def is_english_only(text):
    
    # hindi language block filter
    if re.search(r'[\u0900-\u097F]', text):
        return False
    
    # Must be > 80% standard English/ASCII characters to avoid random non-text encoded garbage
    ascii_count = sum(1 for c in text if ord(c) < 128)
    if len(text) > 0 and (ascii_count / len(text)) < 0.8:
        return False
        
    return True

def clean_sentence(text):
    # Remove weird portal tokens
    text = re.sub(r'###\d+\$\$\$_.*?_%%%\d+!!!', '', text)
    if "Page/File Not Found" in text or "RedirectToLoginPage" in text:
        return ""
    
    text = text.strip()
    words = text.split()
    if len(words) < 5:
        return ""
        
    # Apply English Constraint
    if not is_english_only(text):
        return ""
        
    return text

# function to extract text from PDFs, specifically for regulations which are often in PDF format, and clean it similarly to HTML text
def parse_pdf(url, headers):
    try:
        response = requests.get(url, headers=headers, timeout=10)
        pdf_file = BytesIO(response.content)
        reader = PyPDF2.PdfReader(pdf_file)
        
        pdf_texts = []
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                cleaned = re.sub(r'\s+', ' ', page_text)
                for sentence in cleaned.split('. '):
                    sent = clean_sentence(sentence)
                    if sent:
                        pdf_texts.append(sent)
        return " ".join(pdf_texts)
    except Exception as e:
        return ""

# function to check if a URL is valid for crawling based on domain and file type constraints
def is_valid_link(url):
    parsed = urlparse(url)
    if "iitj.ac.in" not in parsed.netloc:
        return False
    # Only allow HTMLs and PDFs (Since policies/regulations are PDFs)
    bad_exts = ['.jpg', '.png', '.jpeg', '.gif', '.zip', '.rar', '.doc', '.xlsx']
    if any(url.lower().endswith(ext) for ext in bad_exts):
        return False
    return True

# main crawling function that uses BFS to traverse the website starting from the seed URLs, extracts text content, and stores it in a structured format while respecting the constraints of English-only content and prioritizing PDF regulations.
def crawl_and_extract_documents():
    visited_urls = set()
    queue = deque(SEED_URLS)
    
    documents = [] # Will store {"url": url, "text": content}
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
    
    while queue and len(documents) < MAX_DOCS:
        url = queue.popleft()
        url = url.split('#')[0]
        
        if url in visited_urls: continue
        visited_urls.add(url)
        
        print(f"[{len(documents)}/{MAX_DOCS}] Crawling: {url}")
        
        try:
            # Handle PDF for Regulations specifically
            if url.lower().endswith('.pdf'):
                pdf_text = parse_pdf(url, headers)
                if len(pdf_text.split()) > 20: 
                    documents.append({"url": url, "text": pdf_text, "type": "pdf"})
                continue
                
            response = requests.get(url, headers=headers, timeout=8)
            if 'text/html' not in response.headers.get('Content-Type', ''):
                continue
                
            soup = BeautifulSoup(response.text, 'html.parser')
            
            doc_texts = []
            for tag in soup.find_all(['p', 'article', 'section', 'li']):
                if tag.find_parent(['nav', 'header', 'footer', 'aside']):
                    continue
                text = tag.get_text(separator=' ')
                cleaned_text = re.sub(r'\s+', ' ', text)
                sentence = clean_sentence(cleaned_text)
                if sentence:
                    doc_texts.append(sentence)
            
            if doc_texts:
                full_doc_text = " ".join(doc_texts)
                documents.append({"url": url, "text": full_doc_text, "type": "html"})
                
            # Queue children links
            for a_tag in soup.find_all('a', href=True):
                next_url = urljoin(url, a_tag['href']).split('#')[0]
                if is_valid_link(next_url) and next_url not in visited_urls:
                    # Give Priority queue to PDFs containing regulations/rules
                    lower_url = next_url.lower()
                    if '.pdf' in lower_url and ('regul' in lower_url or 'rule' in lower_url or 'ordinance' in lower_url):
                        queue.appendleft(next_url)
                    else:
                        queue.append(next_url)
                        
        except Exception as e:
            pass
            
    return documents

def main():
    corpus_dir = "data"
    os.makedirs(corpus_dir, exist_ok=True)
    
    corpus_path = os.path.join(corpus_dir, "iitj_raw_documents.jsonl")
    
    print("Starting Focused Document Crawler for IIT Jodhpur Dataset...")
    print("Enforcing strictly: English Only, PDF Academic Regulations, and Required Domains.")
    
    docs = crawl_and_extract_documents()
    
    print("\nWriting JSON Lines (Document Level) to disk...")
    with open(corpus_path, "w", encoding="utf-8") as f:
        for doc in docs:
            f.write(json.dumps(doc) + "\n")
            
    print(f"Raw documents successfully saved to {corpus_path}")

if __name__ == "__main__":
    main()