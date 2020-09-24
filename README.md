# LT2316 H20 Assignment A1

Name: Merle Pfau

## Notes on Part 1.

In part 1 I filled out the given functions to read  in and process the data in the file data_loading.py.

I decided to split the first function _parse_data into several smaller functions to keep my code more organized. 

The biggest chunk of reading in the data and creating the mapping between the tokens and their ids, as well as the NER labels and their Ids is done in the function fillout_frames. It takes the list of filenames and creates one DataFrame for the tokens and one for the NER labels, as well as the mapping dictionaries and a vocabulary with all the tokens. 
The first column I filled out was the split, which I gathered from the pathname of the file. 
Next, i accessed the xml data through xml.etree.ElementTree. In the first layer I read in the sentence ID and saved it in a variable while I moved on to the subelements. While reading in the tokens I looped through all words of a sentence end determined the starting and ending points of the tokens through adding their lengths up. Afterwards I adjusted the ending point, if the last character of a token was a punctuation mark and removed that. That was the only "tokenization" I decided to do since some of the drug names contained other special characters and are not words a tokenizer might recocgnize. I used the function get id to get the key to the current token from the id2word dictionary that I constantly updated and filled the data_df with the gathered information on split, sentence id, token id and char start and end.
While having a document open I also filled out the ner_df with the same information, just replacing the token id with the ner id that i got the same way. Here the char start and end were taken directly from the xml file. Some entities had more than one occurence in a sentence, for those I added several lines with the coresponding start and end points to the df.

Next, I set the max_sample_length = 50.

In the function create_val_set() I took a random 25% of the train split and turned it into the validation split.

The second function we were given was get_y() which returns a tensor containing the ner labels for all samples in each split with the dimensions (NUMBER_SAMPLES, MAX_SAMPLE_LENGTH). I split up the data_df into the three splits and used the function get_labels_from_ner_df(df) to get a list of all corresponding NER labels. To achieve this I looped through the rows of each df and checked whether there was an instance in the ner_df whith the same sentence id, where the char start was bigger or identical to the char start of the token and the end smaller or identical to the token. I didn't just check whether they were identical because there were some entities that contained several words, and i wanted all of them to receive the NER label. If there was a match, the label was set to the NER id, otherwise it stayed 0 (no NER). The function returns the list of labels shortened to a length that is devidable by the max_sample_length, to prepare it for being reshaped into the given dimensions. I then turned the returned list into a tensor, gave it the correct dimensions, saved it to the gpu and returned all three tensors.

The last part of part 1 was to plot a histogram displaying ner label counts for each split. For this I took the lists that get_labels_from_ner_df(df) returns and ran a Counter on them. I plotted those counts using matplotlib plt. I chose to also print out the Counter and id2ner dicts to make the data more interpretable.


## Notes on Part 2.

In part 2 we were asked to extract some features from our collected data. 
I decided to use 5 features, with 2 of them being the left and right neighbour of the word, if there was one in the same sentence. The context of a word tells a lot about the meaning, so I thought that these features would be useful in performing the NER task later on.
I also used the word length and capitalization of the token as features. I noticed the drug names were rather long compared to the other words of the corpus, and quite a lot of them, especially the brand names had a capitalized first letter or were all caps. Therefore I think these features contain valuable information for the task.
After saving all feature values in the df, i dropped all now obsolete columns and devided the df into the splits.
I turned all three dfs into tensors and again, made sure the length was devidable by the max_sample_length to match my earlier decision, before reshaping it into the dimensions (NUMBER_SAMPLES, MAX_SAMPLE_LENGTH, FEATURE_DIM).
I saved the tensors to the gpu and returned them. 


## Notes on Part Bonus.

*fill in notes and documentation for the bonus as mentioned in the assignment description, if you choose to do the bonus*
