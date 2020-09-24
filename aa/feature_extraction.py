
#basics
import pandas as pd
import torch


def df_to_tensor(df):
    return torch.from_numpy(df.values)

def shape_data(df, device, max_sample_length):
    df = df.drop(['split'], axis=1)
    n = len(df.index)%max_sample_length
    df.drop(df.tail(n).index,inplace=True)
    tensor = df_to_tensor(df)
    tensor_len = tensor.shape[0] - (tensor.shape[0]//max_sample_length)
    tensor = tensor.reshape([(tensor.shape[0]//max_sample_length),max_sample_length,4])
    return tensor


def extract_features(data:pd.DataFrame, max_sample_length:int, id2word, device):
    # this function should extract features for all samples and 
    # return a features for each split. The dimensions for each split
    # should be (NUMBER_SAMPLES, MAX_SAMPLE_LENGTH, FEATURE_DIM)
    # NOTE! Tensors returned should be on GPU
    #
    # NOTE! Feel free to add any additional arguments to this function. If so
    # document these well and make sure you dont forget to add them in run.ipynb
   

    #turning the df into a feature df
    for i in range(1, len(data)+1):
        #first features: left and right neighbour in the sentence
        token_id = data.loc[i, "token_id"]      
        if i > 1:
            if data.loc[i, "sentence_id"] == data.loc[(i-1), "sentence_id"]:
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
        
        #second feature: first letter is capitalized
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
                
        #third feature: word length
        data.at[i, "word_len"] = data.loc[i, "char_end_id"] - data.loc[i, "char_start_id"]
    
    #remove obsolete columns
    data = data.drop(["token_id", "sentence_id", "char_end_id", "char_start_id"], axis=1)
    
    #create tensors for each of the splits
    train_df = data.loc[data['split'] == 'train']
    train_tensor = shape_data(train_df, device, max_sample_length)
    train_tensor.cuda(0)
    
    val_df = data.loc[data['split'] == 'val']
    val_tensor = shape_data(val_df, device, max_sample_length)
    val_tensor.cuda(0)
    
    test_df = data.loc[data['split'] == 'test']
    test_tensor = shape_data(test_df, device, max_sample_length)
    test_tensor.cuda(0)
    
    print('train:', train_tensor.shape)
    print('val:', val_tensor.shape)
    print('test:', test_tensor.shape)

       
    return train_tensor, val_tensor, test_tensor
