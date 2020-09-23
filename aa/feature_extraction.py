
#basics
import pandas as pd
import torch


def df_to_tensor(df):
    return tens


def extract_features(data:pd.DataFrame, max_sample_length:int, id2word):
    # this function should extract features for all samples and 
    # return a features for each split. The dimensions for each split
    # should be (NUMBER_SAMPLES, MAX_SAMPLE_LENGTH, FEATURE_DIM)
    # NOTE! Tensors returned should be on GPU
    #
    # NOTE! Feel free to add any additional arguments to this function. If so
    # document these well and make sure you dont forget to add them in run.ipynb
    
    data["word_len"] = data["char_end_id"] - data["char_start_id"]
   
    for i in range(1, len(data)+1):
        token_id = data.loc[i, "token_id"]
        
        if i > 1:
            if data.loc[(i), "sentence_id"] == data.loc[(i-1), "sentence_id"]:
                neighbour_l = data.loc[(i-1), "token_id"]
                data.at[i, 'neighbour_l'] = neighbour_l
            else:
                data.at[i, 'neighbour_l'] = 0
        else:
            data.at[i, 'neighbour_l'] = 0
            
        if i < len(data):
            if (data.loc[(i), "sentence_id"] == data.loc[(i+1), "sentence_id"]):
                neighbour_r = data.loc[(i+1), "token_id"]
                data.at[i, 'neighbour_r'] = neighbour_r
            else:
                data.at[i, 'neighbour_r'] = 0
        else:
            data.at[i, 'neighbour_r'] = 0
            
        token = id2word[token_id]
        if token:
            #if " " in token:
            #    data.at[i, "several_words"] = 1
            #else:
            #    data.at[i, "several_words"] = 0
            if token[0].isupper():
                data.at[i, "capital"] = 1
            else:
                data.at[i, "capital"] = 0
    
    
    data = data.drop(["token_id", "sentence_id", "char_end_id", "char_start_id"], axis=1)
    
    
    train_df = data.loc[data['split'] == 'train']
    train_tensor = df_to_tensor(train_df)
    val_df = data.loc[data['split'] == 'val']
    val_tensor = df_to_tensor(val_df)
    test_df = data.loc[data['split'] == 'test']
    test_tensor = df_to_tensor(test_df)
    
    print(val_df)
    
    return train_tensor, val_tensor, test_tensor
