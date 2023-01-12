from torch.utils.data import Dataset
from tqdm import tqdm
import torch
import pandas as pd
from transformers import AutoTokenizer


# --------------------
#  pytorch dataset form 
# --------------------

class LinDataset(Dataset):
    def __init__(self, data) -> None : 
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

    def cut_head_and_tail(self, text_list, tokenizer, max_length=512) -> Dict :
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
        

