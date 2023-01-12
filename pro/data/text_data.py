import os 
from pydantic import BaseModel 
from pathlib  import Path 
from typing import List, Dict 
import pandas as pd
from sklearn.model_selection import KFold 
import re
import unicodedata
import os
from collections import Counter
from bs4 import BeautifulSoup
from torch.utils.data import Dataset
import torch

DIR_PATH = Path(os.path.abspath(__file__)).parent


# ============================================================
#                 TextData Class defination
# ============================================================
class TextData(BaseModel): 
    """declare a TextData object

    Args:
        BaseModel (_type_): _description_
    """
    url: str 
    title: str 
    content: str
    label: int

class TextDatas(BaseModel): 
    textdata_list: List[TextData] 
    textdata_dict: Dict[str, str] = {}
    textdata_dict_new: Dict[str, str] = {}
    filename_list: List[str] = []

    @classmethod
    def from_txt(
            cls, 
            filepath="txtdata", 
            output=False,  ):

        
        txt_file_dir_1 = DIR_PATH / filepath / "label_1"
        txt_file_dir_0 = DIR_PATH / filepath / "label_0"

        txt_data_path_1 = sorted(txt_file_dir_1.glob('*.txt'))
        txt_data_path_0 = sorted(txt_file_dir_0.glob('*.txt')) 
    
        textdata_list = []
        textdata_list = read_data(textdata_list, txt_data_path_1, 1, output=output, )
        textdata_list = read_data(textdata_list, txt_data_path_0, 0, output=output, )

        cls.txt_data_path_1 = txt_data_path_1
        cls.txt_data_path_0 = txt_data_path_0

        return cls(textdata_list=textdata_list)

    def get_filename_list(self) -> List : 
        txt_data_path_1 = self.txt_data_path_1
        txt_data_path_0 = self.txt_data_path_0
        filename_list = [None] * (len(list(txt_data_path_1))+len(list(txt_data_path_0)))
        for i in range(len(list(txt_data_path_1))): 
            filename_list[i] = txt_data_path_1[i].name
        for i in range(len(list(txt_data_path_0))): 
            filename_list[i+  len(list(txt_data_path_1)) ] = txt_data_path_0[i].name
        return filename_list


    def set_filename_list(self): 
        if len(self.filename_list) == 0: 
            filename_list = self.get_filename_list()
            self.filename_list = filename_list


    def set_data_dict(self): 
        data_num = len(self.textdata_list)
        url_list = [None]*data_num
        title_list = [None]*data_num
        content_list = [None]*data_num
        label_list = [None]*data_num
        if self.textdata_dict == {}:
            for i, textdata_list_item in enumerate(self.textdata_list): 
                url_list[i]     = textdata_list_item.url
                title_list[i]   = textdata_list_item.title
                content_list[i] = textdata_list_item.content
                label_list[i]   = textdata_list_item.label

        self.textdata_dict['url'] = url_list
        self.textdata_dict['title'] = title_list
        self.textdata_dict['content'] = content_list
        self.textdata_dict['label'] = label_list


    def get_select_data_list(self, column: str) -> (List, List): 
        if self.textdata_dict == {}:
            self.set_data_dict()
        if column not in ['url', 'title', 'content']:
            raise TypeError(" only the columns 'url','title', 'content' are allowed")
        
        return self.textdata_dict[column], self.textdata_dict['label']
    


    # ------------------------------------------------------------
    #         method: data frame dataset, k-fold data split
    # ------------------------------------------------------------

    def set_select_data_df(self, column: str):  
        if self.textdata_dict == {}:
            self.set_data_dict()

        if self.textdata_dict_new == {}: 
            if column not in ['url', 'title', 'content']:
                raise TypeError(" onlt the columns 'url','title', 'content' are allowed")
            textdata_dict_new = {}
            textdata_dict_new['text'] = self.textdata_dict[column]
            textdata_dict_new['label'] = self.textdata_dict['label']
            self.textdata_dict_new = textdata_dict_new 
    
    def get_splited_data_df(self, column, split_n=4) -> (List, List):
        if self.textdata_dict_new == {}: 
            self.set_select_data_df(column)

        textdata_dict_new_df = pd.DataFrame(data=self.textdata_dict_new)        
        # Define the split - into 4 folds
        kf = KFold(n_splits=split_n, random_state=42, shuffle=True)  
        train_data_df_list = [None]*split_n
        eval_data_df_list  = [None]*split_n
        for i, (train_index, eval_index) in enumerate(kf.split(textdata_dict_new_df)): 
            train_df = textdata_dict_new_df.loc[train_index]
            eval_df  = textdata_dict_new_df.loc[eval_index]
            train_data_df_list[i] = train_df
            eval_data_df_list[i]  = eval_df
    
        return train_data_df_list, eval_data_df_list

    def get_splited_filename_list(self, split_n=4) -> (List, List): 
        if len(self.filename_list) == 0: 
            self.set_filename_list() 
        
        filename_list = self.filename_list
        kf = KFold(n_splits=split_n, random_state=42, shuffle=True)  
        train_filename_list = [None]*split_n
        eval_filename_list  = [None]*split_n
        for i, (train_index, eval_index) in enumerate(kf.split(filename_list)): 
            train_filename_list[i] = [filename_list[x] for x in train_index]
            eval_filename_list[i]  = [filename_list[x] for x in eval_index]

        
        return train_filename_list, eval_filename_list 



def read_data(
        textdata_list, 
        txt_data_path, 
        label, 
        header=False, 
        output=False,) -> List:


    for txt_file in txt_data_path: 
        with open(txt_file, 'r') as f:    
            # title, url part
            if header: 
                url = f.readline().splitlines()[0]
                title = f.readline().splitlines()[0]
            else:
                url = f.readline().splitlines()[0].split('；')[1]
                title = f.readline().splitlines()[0].split('；')[1]

            # content part
            content0 = f.read().splitlines()
            content = ''

            # clean the secntence 
            if output: 
                with open(DIR_PATH / 'preproceed_txt' / txt_file.name, 'w') as f: 
                    for content_item in content0: 
                        content_item = clean_text(content_item)
                        content_item = normalize(content_item)
                        if content_item != '': 
                            content = content+ ' ' + content_item
                            f.write(content_item+'\n')
            else: 
                for content_item in content0: 
                    content_item = clean_text(content_item)
                    content_item = normalize(content_item)
                    if content_item != '':  
                        content = content+ ' ' + content_item
                        
            # split into words



            # fill the text data set
            textdata_list.append(
                    TextData(
                        url=url, 
                        title=title, 
                        content=content, 
                        label=label
                    )) 


    return textdata_list



# ------------------------------------------------------------
#                       text cleaning
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

def normalize(text):
    normalized_text = normalize_unicode(text)
    normalized_text = normalize_number(normalized_text)
    normalized_text = lower_text(normalized_text)
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


# --------------------
#  pytorch dataset form 
# --------------------

class LinDataset(Dataset):
    def __init__(self, data):
        if isinstance(data, pd.DataFrame):
            df = data
            self.features = [
                {
                    'text': row.text, 
                    'label': row.label, 
                } for row in df.itertuples() ]
        else: 
            self.features = [ { 'text': data['text'][i], 'label': data['label'][i], } for i in range()  ]
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx]



# --------------------
#  collate_fn function
# --------------------

class LinCollator():
    def __init__(self, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.encoding = []
        self.examples = []
    
    def __call__(self, examples):
        examples = {
            'text': list(map(lambda x: x['text'], examples)),
            'label': list(map(lambda x: x['label'], examples))
        }
        self.examples = examples
        # encodings = self.tokenizer(examples['text'],
        #                            padding=True,
        #                            truncation=True,
        #                            max_length=self.max_length,
        #                            return_tensors='pt')

        encodings = self.cut_head_and_tail(examples['text'], 
                                            self.tokenizer, 
                                            max_length=self.max_length)
        
        encodings['label'] = torch.tensor(examples['label'])
        self.encoding = encodings
        
        return encodings

    def cut_head_and_tail(self, text_list, tokenizer, max_length=512):
        input_ids_list = []
        attention_mask_list = []
        token_type_ids_list = []

        # if len(text_list) 1: 

        for text in text_list: 
            # まずは限界を設定せずにトークナイズする
            input_ids = tokenizer.encode(text)
     
            n_token = len(input_ids)

            # トークン数が最大数と同じ場合
            if n_token == max_length:
                input_ids = input_ids
                attention_mask = [1 for _ in range(max_length)]
                token_type_ids = [1 for _ in range(max_length)]
            # トークン数が最大数より少ない場合
            elif n_token < max_length:
                pad = [1 for _ in range(max_length-n_token)]
                input_ids = input_ids + pad
                attention_mask = [1 if n_token > i else 0 for i in range(max_length)]
                token_type_ids = [1 if n_token > i else 0 for i in range(max_length)]
            # トークン数が最大数より多い場合
            else:
                harf_len = (max_length-2)//2
                _input_ids = input_ids[1:-1]
                input_ids = [0]+ _input_ids[:harf_len] + _input_ids[-harf_len:] + [2]
                attention_mask = [1 for _ in range(max_length)]
                token_type_ids = [1 for _ in range(max_length)]

            input_ids_list.append(input_ids)
            attention_mask_list.append(attention_mask)
            token_type_ids_list.append(token_type_ids)
            
        d = {
            "input_ids": torch.tensor( input_ids_list, dtype=torch.long),
            "attention_mask": torch.tensor( attention_mask_list, dtype=torch.long),
            "token_type_ids": torch.tensor(token_type_ids_list, dtype=torch.long),
        }
                
        return d
        

