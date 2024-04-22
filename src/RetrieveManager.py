import pickle
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core import VectorStoreIndex, ServiceContext, Document, SimpleDirectoryReader, StorageContext
from llama_index.core.text_splitter import TokenTextSplitter
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
# from llama_index.core.indices.service_context import ServiceContext
from llama_index.core.extractors import (
    SummaryExtractor,
    QuestionsAnsweredExtractor,
    TitleExtractor,
    KeywordExtractor,
)
import chromadb

# define embedding function
embedding_function = HuggingFaceEmbedding(model_name="infgrad/stella-base-zh-v2") # "infgrad/stella-large-zh-v2"

#construct text splitter to split texts into chunks for processing
text_splitter = TokenTextSplitter(separator=" ", chunk_size=1024, chunk_overlap=128)


def get_retriever(uploaded_file, llm):

    ### load file
    # uploaded_file = '新向系統'
    # docs_path = f'/content/data/{uploaded_file}_atl_docs_llamaindex.pkl'
    # docs_path = f'/content/drive/MyDrive/Vivi/LLM/data/員工工作手冊/{uploaded_file}_atl_docs_llamaindex.pkl'
    docs_path = f'D:/05_Study/ChatPDF-demo/docs/data/{uploaded_file}_atl_docs_llamaindex.pkl'

    with open(docs_path, 'rb') as file:
        documents = pickle.load(file)

    #create metadata extractor
    extractors = [
        # TitleExtractor(nodes=1, llm=llm), #title is located on the first page, so pass 1 to nodes param
        # QuestionsAnsweredExtractor(questions=3, llm=llm), #let's extract 3 questions for each node, you can customize this.
        # SummaryExtractor(summaries=["prev", "self"], llm=llm), #let's extract the summary for both previous node and current node.
        KeywordExtractor(keywords=5, llm=llm), #let's extract 10 keywords for each node.
        # embedding_function
    ]

    ### create pipeline
    pipeline = IngestionPipeline(transformations=[text_splitter, *extractors])

    # Ingest directly into a vector db
    nodes = pipeline.run(documents=documents)

    ### create vector store
    # EphemeralClient operates purely in-memory, PersistentClient will also save to disk
    chroma_client = chromadb.EphemeralClient()
    try:
        chroma_client.delete_collection(name="quickstart")
    except:
        pass
    chroma_collection = chroma_client.create_collection("quickstart")

    # set up ChromaVectorStore and load in data
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    index = VectorStoreIndex.from_documents(
        documents, storage_context=storage_context, #service_context=service_context,
        embed_model=embedding_function, show_progress=True
    )
    index.insert_nodes(nodes)

    return nodes, index