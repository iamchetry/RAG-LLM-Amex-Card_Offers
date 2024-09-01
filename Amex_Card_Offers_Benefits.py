# Importation
import os
from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext, load_index_from_storage
from llama_index.core.extractors import (
    QuestionsAnsweredExtractor,
    KeywordExtractor,
    TitleExtractor,
)

from llama_index.core.ingestion import IngestionPipeline
from llama_index.llms.openai import OpenAI
from llama_index.core.node_parser import TokenTextSplitter
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.indices.postprocessor import SimilarityPostprocessor

from llama_index.core.response.pprint_utils import pprint_response

# Import environment variables
dotenv_path = ('env')
load_dotenv(dotenv_path=dotenv_path)
os.environ['OPENAI_API_KEY']=os.getenv("OPENAI_API_KEY")
llm = OpenAI(temperature=0.1, model="gpt-3.5-turbo", max_tokens=512)

# Index Directory
index_dir = 'index'

if not os.path.exists(index_dir):
    print('--------------------- Index Data NOT Found ---------------------')
    # Vector Embedding of local text data
    documents=SimpleDirectoryReader(input_files=[
        "data/Amex_Offers_FAQs.pdf", "data/Gold_Card_Benefits.pdf", "data/Green_Card_Benefits.pdf",
        "data/Platinum_Card_Benefits.pdf"]).load_data()

    # Node Parser & Indexing
    text_splitter = TokenTextSplitter(
        separator=" ", chunk_size=512, chunk_overlap=128
    )

    extractors = [
        TitleExtractor(nodes=5, llm=llm),
        QuestionsAnsweredExtractor(questions=3, llm=llm),
        KeywordExtractor(keywords=10, llm=llm)
    ]

    transformations = [text_splitter] + extractors

    pipeline = IngestionPipeline(transformations=transformations)
    nodes = pipeline.run(documents=documents)

    index = VectorStoreIndex(nodes=nodes)

    # Save the index to disk
    index.storage_context.persist(persist_dir=index_dir)

else:
    # load the existing index
    print('--------------------- Index Data Found ---------------------')
    storage_context = StorageContext.from_defaults(persist_dir=index_dir)
    index = load_index_from_storage(storage_context)

# Create Query Engine
retriever=VectorIndexRetriever(index=index, similarity_top_k=4)
postprocessor=SimilarityPostprocessor(similarity_cutoff=0.8)
query_engine=RetrieverQueryEngine(retriever=retriever, node_postprocessors=[postprocessor])

# Example
query_str = "Can you tell me 3 benefits of Amex Gold Card in bullet points?"
response = query_engine.query(query_str)

pprint_response(response,show_source=True)
print('\n-------------------------------------------------------\n')
print(response)
print('-----------------------DONE------------------------')
