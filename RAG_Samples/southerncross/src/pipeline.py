"""
1. Load a PDF document
2. Text splitting/chunking - Format the text of the PDF textbook ready for an embedding model.
3. Embed all of the chunks of text in the textbook and turn them into numerical representation which we can store for later.
4. Build a retrieval system that uses vector search to find relevant chunks of text based on a query.
5. Create a prompt that incorporates the retrieved pieces of text.
6. Generate an answer to a query based on passages from the textbook.
"""

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

#1. Load a PDF document
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
# We need to pay attention to the token count of per page, because some embedding models have limits on the size of texts they can ingest.
df = pd.DataFrame(pages_and_texts)
print("\n")
print(df.head())
print("\n")
print(df.describe().round(2))

# 2. Text splitting/chunking
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

print("\n")
print("Splitting the text into sentences...\n")
# spaCy is an open-source library designed to break the text into sentences for NLP tasks.
nlp = English()
# Add a sentencizer pipeline. Sentencizer is a pipeline component that turn text into sentences.
nlp.add_pipe("sentencizer")
for item in tqdm(pages_and_texts):
    item["sentences"] = list(nlp(item["text"]).sents)
    # Make sure all sentences are strings
    item["sentences"] = [str(sentence) for sentence in item["sentences"]]
    # Count the sentences 
    item["page_sentence_count_spacy"] = len(item["sentences"])

# Inspect an example
print(random.sample(pages_and_texts, k=1))
# The output shows our raw sentence count (e.g. splitting on ". ") is quite close to what spaCy came up with.
df = pd.DataFrame(pages_and_texts)
print("\n")
print("Statistics after sentence splitting...")
print(df.describe().round(2))

# Chunking - Break down our list of sentences/text into smaller chunks.
# Why do we do this?
# 1. Easier to filter for RAG queries.
# 2. The text chunks can fit into the context window of the embedding model.
# 3. Framework such as Langchain which can help us with chunking as well.

# Define split size to turn a group of sentences into a chunk of text.
num_sentences_per_chunk = 10 # 10 is arbitrary number, we can change it to 5, 7, 8.
# Create a function to chunk the text
def split_list(input_list: list[str], slice_size: int) -> list[str]:
    return [input_list[i: i + slice_size] for i in range(0, len(input_list), slice_size)]

# Loop through pages and texts and split sentences into chunks
print("\n")
print("Chuncking the text...\n")
for item in tqdm(pages_and_texts):
    item["sentence_chunks"] = split_list(input_list = item["sentences"], slice_size = num_sentences_per_chunk)
    item["num_chunks"] = len(item["sentence_chunks"])

print(random.sample(pages_and_texts, k=1))

print("\n")
print("Statistics after chunking...")
df = pd.DataFrame(pages_and_texts)
print(df.describe().round(2))
 
# 3. Embedding the chunks of text
# How to choose the right embedding model?
# Consider the size of the text you want to embed. Because both embedding models and LLM cannot deal with infinite tokens.
# Some embedding models may have been trained to embedd a sequence of 384 tokens into numeric space, 
# if we pass anything more than 384 tokens, it will be truncated, which means we will lose some information.
# Please use this link https://huggingface.co/spaces/mteb/leaderboard to find the right embedding model for your use case.
# Because the average token count per page is 287, it means we could embed an average whole page with the all-mpnet-base-v2 model 
# as this model has an input capacity of 384.
