
#basics
import random
import pandas as pd
import torch
from pathlib import Path
from glob import glob
import xml.etree.ElementTree as ET


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
        #reads in all the xml files and fills the two dataframes with the corresponding values
        i = 1
        for filename in filename_list:
            if 'Test' in str(filename):
                split = 'test'
            else:
                split = 'train'
            tree = ET.parse(filename)
            root = tree.getroot()
            for elem in root:
                #print(elem)
                sent_id = elem.get("id")
                for subelem in elem:
                    if subelem.tag == "entity":
                        entity = subelem.get("text")
                        ner = subelem.get("type")
                        if ";" not in subelem.get("charOffset"):
                            char_start, char_end = subelem.get("charOffset").split("-")
                            #print(entity)
                            self.data_df.loc[i] = [sent_id, entity, char_start, char_end, split]
                            self.ner_df.loc[i] = [sent_id, ner, char_start, char_end]
                            i += 1
                        else:
                            occurences = subelem.get("charOffset").split(";")
                            for occurence in occurences:
                                char_start, char_end = occurence.split("-")
                                self.data_df.loc[i] = [sent_id, entity, char_start, char_end, split]
                                self.ner_df.loc[i] = [sent_id, ner, char_start, char_end]
                                i += 1
                                
            if i > 10:
                break
                                                
        pass
        
    
    def word_map_df(self):
        #maps vocabulary of entities to integers in a dictionary
        self.id2word = {}
        word_id = 0
        tokens = self.data_df['token_id']
        for token in tokens:
            if token not in self.id2word.values():
                self.id2word[word_id] = token
                word_id += 1
        for word_id, token in self.id2word.items():
            self.data_df = self.data_df.replace({token: word_id}, regex=True)
        pass
          
      
    def ner_map_df(self):
        #maps labels to integers in a dictionary
        self.id2ner = {}
        ner_id = 0
        tokens = self.ner_df['ner_id']
        for token in tokens:
            if token not in self.id2ner.values():
                self.id2ner[ner_id] = token
                ner_id += 1
        for ner_id, token in self.id2ner.items():
            self.ner_df = self.ner_df.replace({token: ner_id}, regex=True)
        pass
    
    
    def _parse_data(self,data_dir):
        # Should parse data in the data_dir, create two dataframes with the format specified in
        # __init__(), and set all the variables so that run.ipynb run as it is.

        #initiate the dataframes with columns
        self.data_df = pd.DataFrame(columns = ["sentence_id", "token_id", "char_start_id", "char_end_id", "split"])
        self.ner_df = pd.DataFrame(columns = ["sentence_id", "ner_id", "char_start_id", "char_end_id"])
        
        #read in data
        all_files = Path(data_dir)
        filename_list = [f for f in all_files.glob('**/*.xml')]
        self.fillout_frames(filename_list)
        
        #create vocabularies
        self.word_map_df()
        self.ner_map_df()
        
        #set maximum sample length
        self.max_sample_length = 50
           
        pass


    def get_y(self):
        # Should return a tensor containing the ner labels for all samples in each split.
        # the tensors should have the following following dimensions:
        # (NUMBER_SAMPLES, MAX_SAMPLE_LENGTH)
        # NOTE! the labels for each split should be on the GPU
        
        pass


    def plot_split_ner_distribution(self):
        # should plot a histogram displaying ner label counts for each split
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



