
#basics
import pandas as pd


def extract_features(data:pd.DataFrame, max_sample_length:int, id2word):
    # this function should extract features for all samples and 
    # return a features for each split. The dimensions for each split
    # should be (NUMBER_SAMPLES, MAX_SAMPLE_LENGTH, FEATURE_DIM)
    # NOTE! Tensors returned should be on GPU
    #
    # NOTE! Feel free to add any additional arguments to this function. If so
    # document these well and make sure you dont forget to add them in run.ipynb
    
    data["word_len"] = data["char_end_id"] - data["char_start_id"]
        
    for i in range(len(data)):
        token_id = data.loc[i, "token_id"]
        token = id2word[token_id]
        if " " in token:
            data.at[i, "several_words"] = 1
        else:
            data.at[i, "several_words"] = 0
        if token[0].isupper():
            data.at[i, "capital"] = 1
        else:
            data.at[i, "capital"] = 0
    
    
    data = data.drop(["token_id", "sentence_id", "char_end_id", "char_start_id"], axis=1)
    
    
    train_df = data.loc[data['split'] == 'train']
    val_df = data.loc[data['split'] == 'val']
    test_df = data.loc[data['split'] == 'test']
    
    print(val_df)
    
    #features: Word length, Capitalization, several words
    
    pass
