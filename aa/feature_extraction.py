
#basics
import pandas as pd
import torch

def get_features(data, max_sample_length, id2word):
    
    sent_ids = [s for s in data["sentence_id"]]
    token_ids = [s for s in data["token_id"]]
    start_ids = [s for s in data["char_start_id"]]
    end_ids = [s for s in data["char_end_id"]]
    split = [s for s in data["split"]]
    all_rows = list(zip(sent_ids, token_ids, start_ids, end_ids, split)) 
    
    all_list = []
    sent_list = []
    #turning the df into a feature df
    for i in range(len(all_rows)):
        
        data_tuple = all_rows[i]
        #first features: left and right neighbour in the sentence
        sent_id = data_tuple[0]
        token_id = data_tuple[1]
        start_id = data_tuple[2]
        end_id = data_tuple[3]
        split = data_tuple[4]
        
        n_l = 0
        n_r = 0
        if i != 0:
            if all_rows[i-1][0] == sent_id:
                n_l = all_rows[i-1][1]
        if i < len(all_rows)-1:
            if all_rows[i+1][0] == sent_id:
                n_r = all_rows[i+1][1]
        #second and third feature: first letter / whole word is capitalized 
        #fourth feature: word is alphabetical
        token = id2word[token_id]
        if token:
            capital = 0
            all_caps = 0
            not_alpha = 0
            if token[0].isupper():
                capital = 1
            if token.isupper():
                all_caps = 1
            if token.isalpha():
                not_alpha = 1

        #fifth feature: word length
        word_len = end_id - start_id

        word_list = [n_l, n_r, capital, all_caps, not_alpha]
        
        if i < len(all_rows)-1:
            if sent_id == all_rows[i+1][0]:
                sent_list.append(word_list)
            else:
                len_sent = len(sent_list)
                diff = max_sample_length - len_sent
                padding = diff * [[-1] * len(word_list)]
                sent_list.extend(padding)
                all_list.append(sent_list)
                sent_list = []
                sent_list.append(word_list)
        else:
            len_sent = len(sent_list)
            diff = max_sample_length - len_sent
            padding = diff * [[-1] * len(word_list)]
            sent_list.extend(padding)
            all_list.append(sent_list)
            sent_list = []
            sent_list.append(word_list)
            
    return torch.tensor(all_list)
       
    

def extract_features(data:pd.DataFrame, max_sample_length:int, id2word, device):
    # this function should extract features for all samples and 
    # return a features for each split. The dimensions for each split
    # should be (NUMBER_SAMPLES, MAX_SAMPLE_LENGTH, FEATURE_DIM)
    # NOTE! Tensors returned should be on GPU
    #
    # NOTE! Feel free to add any additional arguments to this function. If so
    # document these well and make sure you dont forget to add them in run.ipynb
   
    train_df = data.loc[data['split'] == 'train']
    train_tensor = get_features(train_df, max_sample_length, id2word)
    train_tensor = train_tensor.to(device)
    
    val_df = data.loc[data['split'] == 'val']
    val_tensor = get_features(val_df, max_sample_length, id2word)
    val_tensor = val_tensor.to(device)
    
    test_df = data.loc[data['split'] == 'test']
    test_tensor = get_features(test_df, max_sample_length, id2word)
    test_tensor = test_tensor.to(device)  

    return train_tensor, val_tensor, test_tensor
