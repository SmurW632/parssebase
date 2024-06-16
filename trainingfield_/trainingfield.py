#python -m spacy download ru_core_news_sm

import spacy
from spacy.training.example import Example
from spacy.util import minibatch, compounding
from trainingdata import TRAIN_DATA
from trainingdatasent import TRAIN_SET
import random
from spacy.symbols import ORTH

nlp = spacy.load("ru_core_news_sm")

for i in range(len(TRAIN_DATA)):
    TRAIN_DATA[i] = (TRAIN_DATA[i][0].lower(), TRAIN_DATA[i][1])
for i in range(len(TRAIN_SET)):
    TRAIN_SET[i] = (TRAIN_SET[i][0].lower(), TRAIN_SET[i][1])
    
unique_words = set()

for sentence, _ in TRAIN_DATA:
    for word in sentence.split():
        unique_words.add(word)

for word in unique_words:
    special_case = [{ORTH: word}]
    nlp.tokenizer.add_special_case(word, special_case)

for _, annotations in TRAIN_DATA:
    for ent in annotations.get("entities"):
        nlp.get_pipe("ner").add_label(ent[2])
for _, annotations in TRAIN_SET:
    for ent in annotations.get("entities"):
        nlp.get_pipe("ner").add_label(ent[2])

print("Training started!")
#запускаем обучение
optimizer = nlp.initialize()

for i in range(100):  # настройка цикла обучения.
    random.shuffle(TRAIN_DATA)
    losses = {}

    for batch in minibatch(TRAIN_DATA, size=compounding(1.0, 4.0, 1.001)):
        examples = [Example.from_dict(nlp.make_doc(text), annots) for text, annots in batch]
        nlp.update(examples,  drop=0.1, losses=losses)
    for batch in minibatch(TRAIN_SET, size=compounding(1.0, 4.0, 1.001)):
        examples = [Example.from_dict(nlp.make_doc(text), annots) for text, annots in batch]
        nlp.update(examples,  drop=0.1, losses=losses) #drop 0.2, iter = 3500, size=compounding(4.0, 32.0, 1.001)) в update sgd=optimizer:

print("Training complete!")
nlp.to_disk('Model')
