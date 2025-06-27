"""
1. Load a PDF document
2. Text splitting/chunking - Format the text of the PDF textbook ready for an embedding model.
3. Embed all of the chunks of text and turn them into the numerical representation - Vectors.
4. Build a retrieval system that uses vector search to find relevant chunks of text based on a query.
5. Create a prompt that incorporates the retrieved pieces of text.
6. Generate an answer to a query based on passages from the textbook.
"""

import os
import fitz 
from tqdm.auto import tqdm # for progress bars, requires !pip install tqdm 
import pandas as pd
import random
import re
from sentence_transformers import SentenceTransformer
import spacy
from spacy.lang.en import English

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
print("1. Loading the PDF file...\n")
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
print("2. Splitting the text into sentences...\n")
# spaCy is an open-source library designed to break the text into sentences for NLP tasks.
# Load the English language model
nlp = spacy.load('en_core_web_sm')
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
# The output shows our raw sentence count (e.g. splitting on ". ? ! ") is quite close to what spaCy came up with.
df = pd.DataFrame(pages_and_texts)
print("\n")
print("Statistics after sentence splitting...")
print(df.describe().round(2))

# Chunking - Break down our list of sentences into smaller chunks.
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

# Split each chunk into a dictionary
pages_and_chunks = []
for item in tqdm(pages_and_texts):
    for sentence_chunk in item["sentence_chunks"]:
        chunk_dict = {}
        chunk_dict["page_number"] = item["page_number"]
        
        # Join the sentences together into a paragraph-like structure, aka a chunk (so they are a single string)
        joined_sentence_chunk = "".join(sentence_chunk).replace("  ", " ").strip()
        #joined_sentence_chunk = re.sub(r'\.([A-Z])', r'. \1', joined_sentence_chunk) # ".A" -> ". A" for any full-stop/capital letter combo 
        chunk_dict["sentence_chunk"] = joined_sentence_chunk

        # Get stats about the chunk
        chunk_dict["chunk_char_count"] = len(joined_sentence_chunk)
        chunk_dict["chunk_word_count"] = len([word for word in joined_sentence_chunk.split(" ")])
        chunk_dict["chunk_token_count"] = len(joined_sentence_chunk) / 4 # 1 token = ~4 characters
        
        pages_and_chunks.append(chunk_dict)

# Inspect an example
print(random.sample(pages_and_chunks, k=1))
print("\n")
print("Statistics after joining sentences...")
df = pd.DataFrame(pages_and_chunks)
print(df.describe().round(2))

# 3. Embedding the chunks of text
# What is embedding?
# Embedding is a way to turn text into a numerical representation - vector that can be used by LLM models.
# Embeddings of text will mean that similar meaning texts have similar numerical representation.
# Please note once our text samples are in embedding vectors, humans will no longer be able to understand them.

# How to choose the right embedding model?
# Consider the size of the text you want to embed. Because both embedding models and LLM cannot deal with infinite tokens.
# Some embedding models may have been trained to embedd a sequence of 384 tokens into numeric space, 
# if we pass anything more than 384 tokens, it will be truncated, which means we will lose some information.
# Please use this link https://huggingface.co/spaces/mteb/leaderboard to find the right embedding model for your use case.

# Where should I store my embeddings?
# The best place to store your embeddings is in a vector database.
# But for now, we will store them in a CSV file.

print("\n")
print("3. Embedding the chunks of text...\n")
# If you want to use a different embedding model, please change the model_name_or_path parameter.
# If your laptop has a GPU, you can use the GPU to speed up the embedding process.
embedding_model= SentenceTransformer(model_name_or_path="all-mpnet-base-v2",  device="cpu")

# Create a list of sentences to turn into numbers
sentences = [
    "The Sentences Transformers library provides an easy and open-source way to create embeddings.",
    "Sentences can be embedded one by one or as a list of strings.",
    "I like to eat pizza."
]

# Sentences are encoded/embedded by calling model.encode()
embeddings = embedding_model.encode(sentences)
embeddings_dict = dict(zip(sentences, embeddings))

# This model has 768 dimensions, so each embedding is a vector of 768 numbers.
for sentence, embedding in embeddings_dict.items():
    print("Sentence:", sentence)
    print("Embedding:", embedding)
    print("Embedding length:", len(embedding))
    print("")

# Let's embed our pages_and_chunks
print("\n")
print("Embedding each chunk of text in pages_and_chunks...\n")
for item in tqdm(pages_and_chunks):
    item["embedding"] = embedding_model.encode(item["sentence_chunk"])


print("\n")
print("Or Embedding the chunks in batches...\n")
# Turn text chunks into a single list
text_chunks = [item["sentence_chunk"] for item in pages_and_chunks]
# Embed all texts in batches
text_chunk_embeddings = embedding_model.encode(text_chunks,
                                               batch_size=32, # you can use different batch sizes here for speed/performance, I found 32 works well for this use case
                                               convert_to_tensor=True) # optional to return embeddings as tensor instead of array
print("Done")
print(text_chunk_embeddings)

# Save embeddings to a file
print("\n")
print("Saving embeddings to a file...\n")
text_chunks_and_embeddings_df = pd.DataFrame(pages_and_chunks)

# Create embeddings directory if it doesn't exist
embeddings_dir = os.path.join(os.path.dirname(__file__), "../embeddings")
os.makedirs(embeddings_dir, exist_ok=True)
embeddings_df_save_path = os.path.join(embeddings_dir, "kiwicare_embeddings.csv")
text_chunks_and_embeddings_df.to_csv(embeddings_df_save_path, index=False)
print(f"Embeddings saved to: {embeddings_df_save_path}")

# Import saved file and view
text_chunks_and_embedding_df_load = pd.read_csv(embeddings_df_save_path)
text_chunks_and_embedding_df_load.head()