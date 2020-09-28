
#basics
import random
import pandas as pd
import torch
from pathlib import Path
from glob import glob
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
from collections import Counter
from matplotlib_venn import venn2, venn2_circles, venn2_unweighted
from matplotlib_venn import venn3, venn3_circles



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
        #also creates mapping from tokens and ners to ids
        
        #initiate lists
        data_list = [["sentence_id", "token_id", "char_start_id", "char_end_id", "split"]]
        ner_list = [["sentence_id", "ner_id", "char_start_id", "char_end_id"]]
        
        #initiate word and ner mapping dictionaries
        self.id2word = {}
        self.id2ner = {}
        self.id2ner[0] = 'None'
        punct = ['.',',',':',';','!','?']
        ner_id = 1
        word_id = 1
        #start reading in the files
        for filename in filename_list:
            #get split from pathname and create validation set
            if 'Test' in str(filename):
                split = 'test'
            else:
                split = random.choices(["train", "val"], weights = (75, 25), k = 1)[0]  # split train into train 
            #access xml data
            tree = ET.parse(filename)
            root = tree.getroot()
            for elem in root:
                #get sent_id
                sent_id = elem.get("id")
                #get tokens from sentence
                tokens = elem.get("text")
                tokens = tokens.replace(";"," ")
                tokens_list = tokens.split(" ")
                k = 0
                for token in tokens_list:
                    #get char start and end by looping htrough sentence
                    char_start = k
                    char_end = k + len(token)-1
                    #clean tokens from punctuation and update char end
                    if token:
                        if token[-1] in punct:
                            token = token[:-1]
                            char_end -= 1
                    k += len(token)+1
                    #update word id dict
                    if token not in self.id2word.values():
                        self.id2word[word_id] = token
                        word_id += 1
                    token_id = self.get_id(token, self.id2word)
                    #add list to data_list for current token
                    data_list.append([sent_id, token_id, char_start, char_end, split])
                for subelem in elem:
                    if subelem.tag == "entity":
                        #get ner
                        ner = subelem.get("type")
                        #update ner id dict
                        if ner not in self.id2ner.values():
                            self.id2ner[ner_id] = ner
                            ner_id += 1
                        label = self.get_id(ner, self.id2ner)
                        #get char_start_id and char_end_id
                        if ";" not in subelem.get("charOffset"):
                            char_start, char_end = subelem.get("charOffset").split("-")
                            char_start, char_end = int(char_start), int(char_end)
                            #add row in data_df for current entity
                            ner_list.append([sent_id, label, char_start, char_end])
                        #if more than one mention of an entity, split into several lines
                        else:
                            occurences = subelem.get("charOffset").split(";")
                            for occurence in occurences:
                                char_start, char_end = occurence.split("-")
                                char_start, char_end = int(char_start), int(char_end)
                                #add row in data_df for current entity
                                ner_list.append([sent_id, label, char_start, char_end])
                                
        self.data_df = pd.DataFrame(data_list[1:], columns=data_list[0])
        self.ner_df = pd.DataFrame(ner_list[1:], columns=ner_list[0])           
        
        pass
        
        
    def get_id(self, token, dct):
        #takes a token and a dictionary and returns the id
        for ids, words in dct.items():
            if words == token:
                return ids

    
    def get_max_length(self):
        #gets the length of the longest sentence 
        sentences = list(self.data_df["sentence_id"])
        word_count = Counter(sentences)
        max_l = max(list(word_count.values()))
                
        return max_l
    
    
    def _parse_data(self,data_dir):
        # Should parse data in the data_dir, create two dataframes with the format specified in
        # __init__(), and set all the variables so that run.ipynb run as it is.
        
        #read in data
        all_files = Path(data_dir)
        filename_list = [f for f in all_files.glob('**/*.xml')]
        self.fillout_frames(filename_list)
        self.vocab = list(self.id2word.values())
        
        #set maximum sample length
        self.max_sample_length = self.get_max_length()

        pass

    
    def padding(self, lst):
        sample_len = len(lst)
        diff = self.max_sample_length - sample_len
        padding = diff * [-1]
        lst.extend(padding)

        return lst[:self.max_sample_length]

    
    def get_labels_from_ner_df(self, df): 
        #takes a dataframe and returns a list of all ner labels (devidable by the max_sample_length)
    
        label_list = []
        all_labels = []
        
        sent_ids = [s for s in df["sentence_id"]]
        start_ids = [s for s in df["char_start_id"]]
        end_ids = [s for s in df["char_end_id"]]
        id_tuples = list(zip(sent_ids, start_ids, end_ids))
        
        label_sent_ids = [s for s in self.ner_df["sentence_id"]]
        label_start_ids = [s for s in self.ner_df["char_start_id"]]
        label_end_ids = [s for s in self.ner_df["char_end_id"]]
        labels = [s for s in self.ner_df["ner_id"]]
        label_tuples = list(zip(label_sent_ids, label_start_ids, label_end_ids))
        
        sentence = 'x'
        sent_labels = []
        for t in id_tuples:
            label = 0
            if t in label_tuples:
                label = labels[label_tuples.index(t)]
            if t[0] == sentence:
                sent_labels.append(label)
            else: 
                padded_labels = self.padding(sent_labels)
                label_list.append(padded_labels)
                sent_labels = [label]
                sentence = t[0]
            if label != 0:
                all_labels.append(label) 
               
        return label_list, all_labels
    
    
    def get_y(self):
        # Should return a tensor containing the ner labels for all samples in each split.
        # the tensors should have the following following dimensions:
        # (NUMBER_SAMPLES, MAX_SAMPLE_LENGTH)
        # NOTE! the labels for each split should be on the GPU
        
        device = torch.device('cuda:1')
        
        #prepare data for plotting and labeling
        #split the data_df into three sub df: train, val and test
        val_df = self.data_df.loc[self.data_df['split'] == 'val']
        train_df = self.data_df.loc[self.data_df['split'] == 'train']
        test_df = self.data_df.loc[self.data_df['split'] == 'test']
        
        #get labels for each of the split dfs and shape into the correct dimensions
        self.train_list, self.all_labels_train = self.get_labels_from_ner_df(train_df)
        self.train_tensor_y = torch.LongTensor(self.train_list)
        self.train_tensor_y = self.train_tensor_y.to(device)
        
        self.val_list, self.all_labels_val = self.get_labels_from_ner_df(val_df)
        self.val_tensor_y = torch.LongTensor(self.val_list)
        self.val_tensor_y = self.val_tensor_y.to(device)
        
        self.test_list, self.all_labels_test = self.get_labels_from_ner_df(test_df)
        self.test_tensor_y = torch.LongTensor(self.test_list)
        self.test_tensor_y = self.test_tensor_y.to(device)
        
        return self.train_tensor_y, self.val_tensor_y, self.test_tensor_y


    def plot_split_ner_distribution(self):
        # should plot a histogram displaying ner label counts for each split
        self.get_y()
        
        train_c = Counter(self.all_labels_train)
        val_c = Counter(self.all_labels_val)
        test_c = Counter(self.all_labels_test)
        data = [train_c, val_c, test_c]
        print(data)
        to_plot= pd.DataFrame(data,index=['train', 'val', 'test'])
        to_plot.plot.bar(figsize=(5,10))
        print(self.id2ner)
        plt.show()

        pass


    def plot_sample_length_distribution(self):
        # FOR BONUS PART!!
        # Should plot a histogram displaying the distribution of sample lengths in number tokens
        counter_dict= {}
        sentence_ids = list(self.data_df["sentence_id"].unique())
        for sentence in sentence_ids:
            sub_df = self.data_df.loc[self.data_df['sentence_id'] == sentence]
            count = len(sub_df.index)
            if count not in counter_dict.keys():
                counter_dict[count] = 1
            else:
                counter_dict[count] += 1      
        keys = list(counter_dict.keys())
        data = counter_dict.values()
        keys.sort()
        to_plot= pd.DataFrame(data,index=keys)
        to_plot.plot.bar(figsize=(20,5))
        plt.show() 
        
        pass


    def plot_ner_per_sample_distribution(self):        
        # FOR BONUS PART!!
        # Should plot a histogram displaying the distribution of number of NERs in sentences
        # e.g. how many sentences has 1 ner, 2 ner and so on
        counter_dict= {}
        sentence_ids = list(self.data_df["sentence_id"].unique())
        for sentence_id in sentence_ids:
            sub_ner_df = self.ner_df.loc[self.ner_df['sentence_id'] == sentence_id]
            count = len(sub_ner_df.index)
            if count not in counter_dict.keys():
                counter_dict[count] = [sentence_id]
            else:
                if sentence_id not in counter_dict[count]:
                    counter_dict[count].append(sentence_id)
        keys = list(counter_dict.keys())
        data = [len(sentences) for sentences in counter_dict.values()]
        keys.sort()
        to_plot= pd.DataFrame(data,index=keys)
        to_plot.plot.bar(figsize=(10,5))
        plt.show()
                                         
        pass


    def plot_ner_cooccurence_venndiagram(self):
        # FOR BONUS PART!!
        # Should plot a ven-diagram displaying how the ner labels co-occur
        
        counter_dict= {}
        sentence_ids = list(self.data_df["sentence_id"].unique())
        for sentence_id in sentence_ids:
            if sentence_id not in counter_dict.keys():
                counter_dict[sentence_id] = {}
                counter_dict[sentence_id][1] = 0
                counter_dict[sentence_id][2] = 0
                counter_dict[sentence_id][3] = 0
                counter_dict[sentence_id][4] = 0
                sub_ner_df = self.ner_df.loc[self.ner_df['sentence_id'] == sentence_id]
                if not sub_ner_df.empty:
                    for j in range(len(sub_ner_df.index)):
                        ner = sub_ner_df.iloc[j]['ner_id']
                        counter_dict[sentence_id][ner] += 1
        self.counter_dict = counter_dict
        list_dict = {}
        for label in [1,2,3,4]:
            for sentence, id_dict in self.counter_dict.items():
                for label, count in id_dict.items():
                    if self.id2ner[label] not in list_dict.keys():
                        list_dict[self.id2ner[label]] = []
                    if count >= 1:
                        list_dict[self.id2ner[label]].append(sentence)
        list_1 = list_dict['drug'] 
        list_1.extend(list_dict['drug_n'])
        list_2 = list_dict['group']
        list_3 = list_dict['brand']
        venn3([set(list_1), set(list_2), set(list_3)], set_labels = ('Drug', 'Group', 'Brand'))                    
                        
        pass
    
    