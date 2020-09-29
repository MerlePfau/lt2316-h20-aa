
#basics
import random
import pandas as pd
import torch
from pathlib import Path
from glob import glob
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
from collections import Counter
import re


class DataLoaderBase:

    #### DO NOT CHANGE ANYTHING IN THIS CLASS ### !!!!

    def __init__(self, data_dir:str, device=None):
        self._parse_data(data_dir)
        assert list(self.data_df.columns) == [
                                                "sentence_id",
                                                "token_id",
                                                "char_start_id",
                                                "char_end_id",
                                                "split"
                                                ]

        assert list(self.ner_df.columns) == [
                                                "sentence_id",
                                                "ner_id",
                                                "char_start_id",
                                                "char_end_id",
                                                ]
        self.device = device
        

    def get_random_sample(self):
        # DO NOT TOUCH THIS
        # simply picks a random sample from the dataset, labels and formats it.
        # Meant to be used as a naive check to see if the data looks ok
        sentence_id = random.choice(list(self.data_df["sentence_id"].unique()))
        sample_ners = self.ner_df[self.ner_df["sentence_id"]==sentence_id]
        sample_tokens = self.data_df[self.data_df["sentence_id"]==sentence_id]

        decode_word = lambda x: self.id2word[x]
        sample_tokens["token"] = sample_tokens.loc[:,"token_id"].apply(decode_word)

        sample = ""
        for i,t_row in sample_tokens.iterrows():

            is_ner = False
            for i, l_row in sample_ners.iterrows():
                 if t_row["char_start_id"] >= l_row["char_start_id"] and t_row["char_start_id"] <= l_row["char_end_id"]:
                    sample += f'{self.id2ner[l_row["ner_id"]].upper()}:{t_row["token"]} '
                    is_ner = True
            
            if not is_ner:
                sample += t_row["token"] + " "

        return sample.rstrip()



class DataLoader(DataLoaderBase):


    def __init__(self, data_dir:str, device=None):
        super().__init__(data_dir=data_dir, device=device)
        
    
    def fillout_frames(self, filename_list):
        print('reading in data...')
        #reads in all the xml files and fills the two dataframes with the corresponding values
        i = 1
        for filename in filename_list:
            #get split
            if 'Test' in str(filename):
                split = 'test'
            else:
                split = 'train'
            tree = ET.parse(filename)
            root = tree.getroot()
            for elem in root:
                #get sent_id
                sent_id = elem.get("id")
                for subelem in elem:
                    if subelem.tag == "entity":
                        #get entity
                        entity = subelem.get("text")
                        #get ner
                        ner = subelem.get("type")
                        #get char_start_id and char_end_id
                        if ";" not in subelem.get("charOffset"):
                            char_start, char_end = subelem.get("charOffset").split("-")
                            char_start, char_end = int(char_start), int(char_end)
                            self.ner_df.loc[i] = [sent_id, ner, char_start, char_end]
                            self.data_df.loc[i] = [sent_id, entity, char_start, char_end, split]
                            i += 1
                        #if more than one mention of a token, split into several lines
                        else:
                            occurences = subelem.get("charOffset").split(";")
                            for occurence in occurences:
                                char_start, char_end = occurence.split("-")
                                char_start, char_end = int(char_start), int(char_end)
                                self.ner_df.loc[i] = [sent_id, ner, char_start, char_end]
                                self.data_df.loc[i] = [sent_id, entity, char_start, char_end, split]
                                i += 1
                                
           # if i > 5000:
           #     break
        print('data read')                                                    
        pass
        
    
    def word_map_df(self):
        print('creating word map...')
        #maps vocabulary of entities to integers in a dictionary
        self.id2word = {}
        word_id = 0
        tokens = self.data_df['token_id']
        for token in tokens:
            if token not in self.id2word.values():
                self.id2word[word_id] = token
                word_id += 1
        #create vocabulary of all tokes
        self.vocab = list(self.id2word.values())
        #replace tokens with token_ids in dataframe
        for word_id, token in self.id2word.items():
            self.data_df.loc[self.data_df['token_id'] == token, 'token_id'] = word_id
        print('word map done')
        pass
          
      
    def ner_map_df(self):
        print('creating ner map...')
        #maps labels to integers in a dictionary
        self.id2ner = {}
        ner_id = 1
        tokens = self.ner_df['ner_id']
        for token in tokens:
            if token not in self.id2ner.values():
                self.id2ner[ner_id] = token
                ner_id += 1
        for ner_id, token in self.id2ner.items():
            self.ner_df.loc[self.ner_df['ner_id'] == token, 'ner_id'] = ner_id
        print('ner map done')
        pass
    
    
    def create_val_set(self):
        print('creating validation set...')
        #takes 25% of the training set as validation set
        train_len = len(self.data_df.loc[self.data_df['split'] == 'train'])
        val_len = int(train_len*0.25)
        i = 0
        while i < val_len:
            rand = random.choice(self.data_df.index)
            if self.data_df.loc[rand, 'split'] == 'train':
                self.data_df.at[rand, 'split'] = 'val'
                i += 1
        print('validation set ready')
        pass
    
    
    def _parse_data(self,data_dir):
        print('processing data')
        # Should parse data in the data_dir, create two dataframes with the format specified in
        # __init__(), and set all the variables so that run.ipynb run as it is.

        #initiate the dataframes with columns
        self.data_df = pd.DataFrame(columns = ["sentence_id", "token_id", "char_start_id", "char_end_id", "split"])
        self.ner_df = pd.DataFrame(columns = ["sentence_id", "ner_id", "char_start_id", "char_end_id"])
        
        #read in data
        all_files = Path(data_dir)
        filename_list = [f for f in all_files.glob('**/*.xml')]
        self.fillout_frames(filename_list)
        
        print(self.data_df)
        print(self.ner_df)
        
        #create vocabularies
        self.word_map_df()
        self.ner_map_df()
        
        #set maximum sample length
        self.max_sample_length = 50
        
        #25% of train test as val set
        self.create_val_set()
        
        self.data_df = self.data_df.sample(frac=1).reset_index(drop=True)
        self.ner_df = self.ner_df.sample(frac=1).reset_index(drop=True)
        
        #prepare data for plotting and labeling
        #split the data_df into three sub df: train, val and test
        val_df = self.data_df.loc[self.data_df['split'] == 'val']
        train_df = self.data_df.loc[self.data_df['split'] == 'train']
        test_df = self.data_df.loc[self.data_df['split'] == 'test']
        
        #get labels for each of the split dfs and shape into the correct dimensions
        self.train_list, train_tensor_len = self.get_labels_from_ner_df(train_df)
        train_tensor = torch.LongTensor(self.train_list)
        self.train_tensor = train_tensor.reshape([(train_tensor_len//self.max_sample_length),self.max_sample_length])
        
        self.val_list, val_tensor_len = self.get_labels_from_ner_df(val_df)
        val_tensor = torch.LongTensor(self.val_list)
        self.val_tensor = val_tensor.reshape([(val_tensor_len//self.max_sample_length),self.max_sample_length])
        
        self.test_list, test_tensor_len = self.get_labels_from_ner_df(test_df)
        test_tensor = torch.LongTensor(self.test_list)
        self.test_tensor = test_tensor.reshape([(test_tensor_len//self.max_sample_length),self.max_sample_length])
        
        print('data processed')
        pass


    def get_labels_from_ner_df(self, df):
        #takes a dataframe and returns a list of all ner labels (devidable by the max_sample_length)
        lst = []
        print(len(df))
        for i in range(1,len(df)):
            instance = df.iloc[i]
            instance_ner = self.ner_df.loc[(self.ner_df['sentence_id'] == instance['sentence_id']) & (self.ner_df['char_start_id'] == instance['char_start_id']) & (self.ner_df['char_end_id'] == instance['char_end_id'])]
            label = instance_ner.iloc[0]['ner_id']
            lst.append(label)
        lst = lst[:(len(lst)-len(lst)%self.max_sample_length)]
        return lst, len(lst)
    
    
    def get_y(self):
        # Should return a tensor containing the ner labels for all samples in each split.
        # the tensors should have the following following dimensions:
        # (NUMBER_SAMPLES, MAX_SAMPLE_LENGTH)
        # NOTE! the labels for each split should be on the GPU
       
        return self.train_tensor, self.val_tensor, self.test_tensor


    def plot_split_ner_distribution(self):
        # should plot a histogram displaying ner label counts for each split
        train_c = Counter(self.train_list)
        val_c = Counter(self.val_list)
        test_c = Counter(self.test_list)
        data = [train_c, val_c, test_c]
        print(data)
        to_plot= pd.DataFrame(data,index=['train', 'val', 'test'])
        to_plot.plot(kind='bar')
        print(self.id2ner)
        plt.show()
        pass


    def plot_sample_length_distribution(self):
        # FOR BONUS PART!!
        # Should plot a histogram displaying the distribution of sample lengths in number tokens
        pass


    def plot_ner_per_sample_distribution(self):        
        # FOR BONUS PART!!
        # Should plot a histogram displaying the distribution of number of NERs in sentences
        # e.g. how many sentences has 1 ner, 2 ner and so on
        pass


    def plot_ner_cooccurence_venndiagram(self):
        # FOR BONUS PART!!
        # Should plot a ven-diagram displaying how the ner labels co-occur
        pass



