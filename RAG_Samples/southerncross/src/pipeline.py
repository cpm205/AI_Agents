#1. Document/Text Processing and Embedding Creation
#Steps:
    #1.Import PDF document.
    #2.Process text for embedding (e.g. split into chunks of sentences).
    #3.Embed text chunks with embedding model.
    #4.Save embeddings to file for later use (embeddings will store on file for many years or until you lose your hard drive).

import os
import fitz 
from tqdm.auto import tqdm # for progress bars, requires !pip install tqdm 
import pandas as pd
from spacy.lang.en import English
import random
import spacy

# Load the English language model
nlp = spacy.load('en_core_web_sm')

# Pre-processing the text
def text_formatter(text: str) -> str:
    """Performs minor formatting on text."""
    """Strip the white spaces"""
    cleaned_text = text.replace("\n", " ").strip() # note: this might be different for each doc (best to experiment)

    # Other potential text formatting functions can go here
    return cleaned_text

# Open PDF and get lines/pages
def open_and_read_pdf(pdf_path: str) -> list[dict]:
    """
    Opens a PDF file, reads its text content page by page, and collects statistics.

    Parameters:
        pdf_path (str): The file path to the PDF document to be opened and read.

    Returns:
        list[dict]: A list of dictionaries, each containing the page number
        (adjusted), character count, word count, sentence count, token count, and the extracted text
        for each page.
    """
    doc = fitz.open(pdf_path)  # open a document
    pages_and_texts = []
    for page_number, page in tqdm(enumerate(doc)):  # iterate the document pages
        text = page.get_text()  # get plain text encoded as UTF-8
        text = text_formatter(text)
        pages_and_texts.append({"page_number": page_number,
                                "page_char_count": len(text),
                                "page_word_count": len(text.split(" ")),
                                "page_sentence_count_raw": len(text.split(". ")),
                                "page_token_count": len(text) / 4,  # 1 token = ~4 chars, see: https://help.openai.com/en/articles/4936856-what-are-tokens-and-how-to-count-them
                                "text": text})
    return pages_and_texts

#PDF file Path
pdf_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../doc/Combined_RegularCare_KiwiCare.pdf"))
pages_and_texts = open_and_read_pdf(pdf_path=pdf_path)
print("First two pages of processed text:")
print(pages_and_texts[:2])

# Perform the exploratory data analysis (EDA) to get an idea of the size of the texts (e.g. character counts, word counts etc)
# The different sizes of texts will be a good indicator into how we should split our texts.
# Many embedding models have limits on the size of texts they can ingest, 
# for example, the sentence-transformers model all-mpnet-base-v2 has an input size of 384 tokens.
# This means that the model has been trained in ingest and turn into embeddings texts with 384 tokens (1 token ~= 4 characters ~= 0.75 words).
# Texts over 384 tokens which are encoded by this model will be auotmatically reduced to 384 tokens in length, potentially losing some information.
df = pd.DataFrame(pages_and_texts)
print(df.head())
print(df.describe().round(2))

# Because the average token count per page is 287, it means we could embed an average whole page with the all-mpnet-base-v2 model 
# as this model has an input capacity of 384.

# Text Processing - splitting pages into sentences
# The ideal way of processing text before embedding it is still an active area of research.
# A simple method I've found helpful is to break the text into chunks of sentences.
# As in, chunk a page of text into groups of 5, 7, 10 or more sentences (these values are not set in stone and can be explored).
# But we want to follow the workflow of: Ingest text -> split it into groups/chunks -> embed the groups/chunks -> use the embeddings

# Some options for splitting text into sentences:
# 1. Split into sentences with simple rules (e.g. split on ". " with text = text.split(". "), like we did above).
# 2. Split into sentences with a natural language processing (NLP) library such as spaCy or nltk.

# Why split into sentences? 
# 1. Easier to handle than larger pages of text (especially if pages are densely filled with text).
# 2. Can get specific and find out which group of sentences were used to help within a RAG pipeline.

# Let's use spaCy to break our text into sentences.
# spaCy is an open-source library designed to break the text into sentences for NLP tasks.
nlp = English()
# Add a sentencizer pipeline, see https://spacy.io/api/sentencizer/ 
nlp.add_pipe("sentencizer")
for item in tqdm(pages_and_texts):
    item["sentences"] = list(nlp(item["text"]).sents)
    # Make sure all sentences are strings
    item["sentences"] = [str(sentence) for sentence in item["sentences"]]
    # Count the sentences 
    item["page_sentence_count_spacy"] = len(item["sentences"])

# Inspect an example
print(random.sample(pages_and_texts, k=1))

# Let's turn out list of dictionaries into a DataFrame and get some stats.
# The output shows our raw sentence count (e.g. splitting on ". ") is quite close to what spaCy came up with.
df = pd.DataFrame(pages_and_texts)
print(df.describe().round(2))