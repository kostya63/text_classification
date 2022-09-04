import spacy
from spacy import displacy
from collections import Counter
import pandas as pd
pd.options.display.max_rows = 600
pd.options.display.max_colwidth = 400

nlp = spacy.load('ru_core_news_md')

data_file = '/home/mkr/tc1/data/test_data.csv'
target_file = '/home/mkr/tc1/data/ners.csv'

source_data = pd.read_csv(data_file, sep = ';', header = 0)
target_data = source_data
target_data.insert(loc=5, column='named_entities', value=None)

for index, row in source_data.iterrows():
    text = source_data.iloc[index, 3].lower()
    document = nlp(text)
    ner = ''
    for named_entity in document.ents: ner = str(named_entity) + " - " + str(named_entity.label_)
    target_data.at[index, 'named_entities'] = ner

target_data.to_csv(target_file, index = False)   
