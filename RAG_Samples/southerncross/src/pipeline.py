#1. Document/Text Processing and Embedding Creation
#Steps:
    #1.Import PDF document.
    #2.Process text for embedding (e.g. split into chunks of sentences).
    #3.Embed text chunks with embedding model.
    #4.Save embeddings to file for later use (embeddings will store on file for many years or until you lose your hard drive).

import os
import fitz 
from tqdm.auto import tqdm # for progress bars, requires !pip install tqdm 

def text_formatter(text: str) -> str:
    """Performs minor formatting on text."""
    cleaned_text = text.replace("\n", " ").strip() # note: this might be different for each doc (best to experiment)

    # Other potential text formatting functions can go here
    return cleaned_text

# Open PDF and get lines/pages
# Note: this only focuses on text, rather than images/figures etc
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