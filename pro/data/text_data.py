import os 
from pydantic import BaseModel 
from pathlib  import Path 
from typing import List, Dict 
import pandas as pd
from sklearn.model_selection import KFold 
from pro.data.text_preprosseing import clean_text, normalize, remove_stopwords


DIR_PATH = Path(os.path.abspath(__file__)).parent

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
    def from_txt(cls, filepath: str ="txtdata", output: bool = False,):

        txt_file_dir_1 = DIR_PATH / filepath / "label_1"
        txt_file_dir_0 = DIR_PATH / filepath / "label_0"

        txt_data_path_1 = sorted(txt_file_dir_1.glob('*.txt'))
        txt_data_path_0 = sorted(txt_file_dir_0.glob('*.txt')) 
    
        textdata_list = []

        textdata_list.append(cls.read_data(txt_data_path_1, 1, output=output,))
        textdata_list.append(cls.read_data(txt_data_path_0, 0, output=output,))

        cls.txt_data_path_1 = txt_data_path_1
        cls.txt_data_path_0 = txt_data_path_0

        return cls(textdata_list=textdata_list)

    def read_data( self, 
            txt_data_path, 
            label: int, 
            header: bool = False, 
            output: bool = False,) -> List:
        """read the data (.txt format) from the given file list and store the data (TextData object) in a list
            (the order of the resulting data list is the same as the given file list)

        Args:
            txt_data_path (List or Path object, any iterable list): file path list that would be read 
            label (int): the labels of the data being read
            header (bool, optional): specifying if the header is needed or not (the word before ';'). Defaults to False.
            output (bool, optional): specifying if the output file is generated or not. Defaults to False.

        Returns:
            List: textdata_list, 
        """

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

                # clean the sentences  
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
                         
                # fill the text data set
                textdata_list = []
                textdata_list.append(
                        TextData(
                            url=url, 
                            title=title, 
                            content=content, 
                            label=label)) 
        return textdata_list


    def get_filename_list(self) -> List : 
        """get the filename list

        Returns:
            List: _description_
        """
        txt_data_path_1 = self.txt_data_path_1
        txt_data_path_0 = self.txt_data_path_0
        filename_list = [None] * (len(list(txt_data_path_1))+len(list(txt_data_path_0)))
        for i in range(len(list(txt_data_path_1))): 
            filename_list[i] = txt_data_path_1[i].name
        for i in range(len(list(txt_data_path_0))): 
            filename_list[i+  len(list(txt_data_path_1)) ] = txt_data_path_0[i].name
        return filename_list


    def set_data_dict(self) -> None: 
        """_summary_
        """
        data_num = len(self.textdata_list)
        url_list = [None]*data_num
        title_list = [None]*data_num
        content_list = [None]*data_num
        label_list = [None]*data_num
        if self.textdata_dict == {}:
            for i, textdata_list_item in enumerate(self.textdata_list): 
                url_list[i] = textdata_list_item.url
                title_list[i] = textdata_list_item.title
                content_list[i] = textdata_list_item.content
                label_list[i] = textdata_list_item.label

        self.textdata_dict['url'] = url_list
        self.textdata_dict['title'] = title_list
        self.textdata_dict['content'] = content_list
        self.textdata_dict['label'] = label_list


    def get_select_data_list(self, column: str) -> tuple[List, List]: 
        """_summary_

        Args:
            column (str): _description_

        Raises:
            TypeError: _description_

        Returns:
            tuple[List, List]: _description_
        """
        if self.textdata_dict == {}:
            self.set_data_dict()
        if column not in ['url', 'title', 'content']:
            raise TypeError(" only the columns 'url','title', 'content' are allowed")
        
        return self.textdata_dict[column], self.textdata_dict['label']
    

    def set_select_data_df(self, column: str) -> None:
        """

        Args:
            column (str): _description_

        Raises:
            TypeError: _description_
        """
        if self.textdata_dict == {}:
            self.set_data_dict()

        if self.textdata_dict_new == {}: 
            if column not in ['url', 'title', 'content']:
                raise TypeError(" only the columns 'url','title', 'content' are allowed")
            textdata_dict_new = {}
            textdata_dict_new['text'] = self.textdata_dict[column]
            textdata_dict_new['label'] = self.textdata_dict['label']
            self.textdata_dict_new = textdata_dict_new 
    
    def get_splited_data_df(self, column: str, split_n: int=4) -> tuple[List, List]:
        """_summary_

        Args:
            column (str): _description_
            split_n (int, optional): _description_. Defaults to 4.

        Returns:
            tuple[List, List]: _description_
        """


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

    def get_splited_filename_list(self, split_n=4) -> tuple[List, List]: 
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

    def set_filename_list(self): 
        """ setup the filename list variable, which is used in showing up the source file of each rows
        """
        if len(self.filename_list) == 0: 
            filename_list = self.get_filename_list()
            self.filename_list = filename_list


