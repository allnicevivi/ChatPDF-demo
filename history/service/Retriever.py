import os
from pypdf import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter, SentenceTransformersTokenTextSplitter
from transformers import BertTokenizer

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

embedding_function = SentenceTransformerEmbeddingFunction()

class getRetriver:

    def __init__(self, uploaded_file):
        
        self.file = uploaded_file

    def load_pdf_file(self, filename):

        print('Upload New Files')
        # pdf_texts = ''
        # for filename in filename_ls:
        reader = PdfReader(filename)
        texts = [p.extract_text().strip() for p in reader.pages]
        # Filter the empty strings
        texts = [text for text in texts if text]
        pdf_texts = '\n\n'.join(texts)

        return pdf_texts

    def split_chunks(self, pdf_texts):

        ### textSplitter
        tokenizer = BertTokenizer.from_pretrained("hfl/chinese-bert-wwm-ext")
        # tokenizer = BertTokenizer.from_pretrained("shibing624/text2vec-base-chinese")

        text_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
            tokenizer, chunk_size=400, chunk_overlap=100,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        character_split_texts = text_splitter.split_text(pdf_texts)

        return character_split_texts
    
    def load_chroma(self, token_split_texts):

        collection_name = 'pdf_retriever'
        # Create the collection
        chroma_client = chromadb.Client()
        try:
            chroma_client.delete_collection(name=collection_name)
        except:
            pass
    
        chroma_collection = chroma_client.get_or_create_collection(name=collection_name,
                                                                embedding_function=embedding_function,
                                                                metadata={"hnsw:space": "cosine"})

        ids = [str(i) for i in range(len(token_split_texts))]

        chroma_collection.add(ids=ids, documents=token_split_texts)
        return chroma_collection


    def main(self):

        pdf_texts = self.load_pdf_file(self.file)

        chunks = self.split_chunks(pdf_texts)

        chroma_collection = self.load_chroma(token_split_texts=chunks)

        return chroma_collection
