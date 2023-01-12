import re
import unicodedata
import os
from collections import Counter
from bs4 import BeautifulSoup


# ------------------------------------------------------------
#                       text cleaning
#                       preprocessing 
# ------------------------------------------------------------
# remove some html tags frequently shown in the downloaded articles 


def clean_text(text):
    replaced_text = re.sub(r'[【】]', ' ', text)       # 【】の除去
    replaced_text = re.sub(r'[（）()]', ' ', replaced_text)     # （）の除去
    replaced_text = re.sub(r'[［］\[\]]', ' ', replaced_text)   # ［］の除去
    replaced_text = re.sub(r'[@＠]\w+', '', replaced_text)  # メンションの除去
    replaced_text = re.sub(r'https?:\/\/.*?[\r\n ]', '', replaced_text)  # URLの除去
    replaced_text = re.sub(r'　', ' ', replaced_text)  # 全角空白の除去
    replaced_text = is_japanese(replaced_text)
    # replaced_text = clean_html_tags(replaced_text)
    replaced_text = clean_html_and_js_tags(replaced_text)
    replaced_text = clean_url(replaced_text)
    replaced_text = clean_alt_tags(replaced_text)
    # replaced_text = clean_code(replaced_text)
    return replaced_text

    
def is_japanese(string):
    for ch in string:
        name = unicodedata.name(ch) 
        if "CJK UNIFIED" in name \
        or "HIRAGANA" in name \
        or "KATAKANA" in name:
            return string
    return ''

def clean_html_tags(html_text):
    soup = BeautifulSoup(html_text, 'html.parser')
    cleaned_text = soup.get_text()
    cleaned_text = ''.join(cleaned_text.splitlines())
    return cleaned_text


def clean_html_and_js_tags(html_text):
    soup = BeautifulSoup(html_text, 'html.parser')
    [x.extract() for x in soup.findAll(['script', 'style'])]
    cleaned_text = soup.get_text()
    cleaned_text = ''.join(cleaned_text.splitlines())
    return cleaned_text


def clean_alt_tags(alt_text): 
    clean_text = re.sub(r'"alt ="', '', alt_text)
    clean_text = re.sub(r'"/>', '', clean_text)

    return clean_text


def clean_url(html_text):
    """
    \S+ matches all non-whitespace characters (the end of the url)
    :param html_text:
    :return:
    """
    clean_text = re.sub(r'http\S+', '', html_text)
    return clean_text


def clean_code(html_text):
    """Qiitaのコードを取り除きます
    :param html_text:
    :return:
    """
    soup = BeautifulSoup(html_text, 'html.parser')
    [x.extract() for x in soup.findAll(class_="code-frame")]
    cleaned_text = soup.get_text()
    cleaned_text = ''.join(cleaned_text.splitlines())
    return cleaned_text

# ------------------------------------------------------------
#                       text normalization
# ------------------------------------------------------------
# 1. normalize data via unicodedata 
# 2. replace the numbers into 0
# 3. lower the case 
def normalize(text):
    normalized_text = normalize_unicode(text)
    normalized_text = normalize_number(normalized_text)
    normalized_text = lower_text(normalized_text)
    return normalized_text



def normalize_number(text):
    """
    pattern = r'\d+'
    replacer = re.compile(pattern)
    result = replacer.sub('0', text)
    """
    replaced_text = re.sub(r'\d+', '0', text)
    return replaced_text

def lower_text(text):
    return text.lower()

def normalize_unicode(text, form='NFKC'):
    normalized_text = unicodedata.normalize(form, text)
    return normalized_text




# ------------------------------------------------------------
#                    remove stop words
# ------------------------------------------------------------
# remove the stop words based on the word counts

def remove_stopwords(words, docs):
    stopwords = get_stop_words(docs)
    words = [word for word in words if word not in stopwords]
    return words

def most_common(docs, n=100):
    fdist = Counter()
    for doc in docs:
        for word in doc:
            fdist[word] += 1
    common_words = {word for word, freq in fdist.most_common(n)}
    print('{}/{}'.format(n, len(fdist)))
    return common_words

def get_stop_words(docs, n=100, min_freq=1):
    fdist = Counter()
    for doc in docs:
        for word in doc:
            fdist[word] += 1
    common_words = {word for word, freq in fdist.most_common(n)}
    rare_words = {word for word, freq in fdist.items() if freq <= min_freq}
    stopwords = common_words.union(rare_words)
    print('{}/{}'.format(len(stopwords), len(fdist)))
    return stopwords

