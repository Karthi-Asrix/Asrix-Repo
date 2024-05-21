from langchain_community.document_loaders import (
    PyPDFLoader,
    UnstructuredFileLoader
)
from langchain_community.vectorstores import FAISS
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, ConfigDict
from typing import List

import os
import re
import requests

from textgen import TextGen
from utils import doc_load, append_doc_log


class RAGLlm(BaseModel):
    model_url: str | None = Field(default="http://localhost:5000")
    context: str | None = Field(default='data/orient-context.pdf', description="PDF file path on local. (eg. orient-context.pdf, webermeyer-context.pdf)")
    prompt: str | None = Field(default=None, description="User input")

    model_config = ConfigDict(
        protected_namespaces=()
    )


class RAGLlm2(BaseModel):
    model_url: str | None = Field(default="http://localhost:5000")
    character_id: str | None = Field(default="OT-PLC/150424022018")
    contexts: List[str] | None = Field(default=[
        "https://gist.githubusercontent.com/EdwardRayl/3436572afde8ce9e3faf5b7b95356a49/raw/6b25895fce480713560829dec31ac8220ffe5272/gists.txt",
        "https://www.rcrc-resilience-southeastasia.org/wp-content/uploads/2017/12/Contracts-Act-1950.pdf",
        "https://github.com/SheetJS/libreoffice_test-files/blob/master/ooxml-strict/Lorem-ipsum.docx"], 
        description="PDF file path on local. (eg. orient-context.pdf, webermeyer-context.pdf)")
    prompt: str | None = Field(default=None, description="User input")

    model_config = ConfigDict(
        protected_namespaces=()
    )


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)


@app.options("/")
async def options_route():
    return JSONResponse(content="OK")


@app.post("/rag", summary="Production V1 RAG")
async def rag(request_data: RAGLlm):
    """
        Naive RAG implementation
        - Accepts documents in URL and local path
    """

    prompt_template = """
    ### [INST] Instruction: Give only greetings if there is no question. Answer the question based on the context information and if the question can't be answered based on the context, say "I don't know". Here is context to help:

    {context}

    ### QUESTION:
    {question} [/INST]
    """

    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=prompt_template,
    )

    textgen_llm = TextGen(model_url=request_data.model_url, mode="instruct",
                          temperature=0.1, repetition_penalty=1.1,
                          max_new_tokens=1000, truncation_length=32768,
                          do_sample=True)

    model_name = "sentence-transformers/all-mpnet-base-v2"
    embedding_model = HuggingFaceEmbeddings(model_name=model_name)

    loader = PyPDFLoader(request_data.context)
    pages = loader.load_and_split()

    db = FAISS.from_documents(pages, embedding_model)

    retriever_pdf = db.as_retriever(
        search_kwargs={'k': 5}
    )

    rag_chain = (
        {"context": retriever_pdf, "question": RunnablePassthrough()}
        | prompt
        | textgen_llm
    )
    print(pages)
    response = rag_chain.invoke(f"{request_data.prompt}")
    return response


@app.post("/v2/rag", summary="Production V2 RAG")
async def rag2(request_data: RAGLlm):
    """
        Naive RAG with added functionality:
        - Data persistence in local file ./store/
        - Accepts documents in URL and local path
    """

    prompt_template = """
    ### [INST] Instruction: Give only greetings if there is no question. Answer the question based on the context information and if the question can't be answered based on the context, say "I don't know". Here is context to help:

    {context}

    ### QUESTION:
    {question} [/INST]
    """

    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=prompt_template,
    )

    textgen_llm = TextGen(model_url=request_data.model_url, mode="instruct",
                          temperature=0.1, repetition_penalty=1.1,
                          max_new_tokens=1000, truncation_length=32768,
                          do_sample=True)

    model_name = "sentence-transformers/all-mpnet-base-v2"
    embedding_model = HuggingFaceEmbeddings(model_name=model_name)

    match = re.search(r'\/([^\/]+)\.pdf$', request_data.context)
    doc_name = match.group(1) if match else None
    doc_dir = f'./store/{doc_name}'

    if os.path.isdir(doc_dir):
        print(f"Data file: '{doc_name}' already existed.")
        db = FAISS.load_local(doc_dir, embedding_model)

    else:
        print(f"Data file: '{doc_name}' does not exists.")
        loader = PyPDFLoader(request_data.context)
        pages = loader.load_and_split()
        db = FAISS.from_documents(pages, embedding_model)
        db.save_local(doc_dir)

    retriever_pdf = db.as_retriever(search_kwargs={'k': 5})

    rag_chain = (
        {"context": retriever_pdf, "question": RunnablePassthrough()}
        | prompt
        | textgen_llm
    )

    response = rag_chain.invoke(f"{request_data.prompt}")
    return response


@app.post("/v3/rag", summary="Testing RAG V3")
async def rag3(request_data: RAGLlm):
    """
        Naive RAG with added functionality:
        - Data persistence in local file ./store/
        - Only accepts documents in URL
        - Able to handle pdf, doc, docx, txt files
    """

    prompt_template = """
    ### [INST] Instruction: Give only greetings if there is no question. Answer the question based on the context information and if the question can't be answered based on the context, say "I don't know". Here is context to help:

    {context}

    ### QUESTION:
    {question} [/INST]
    """

    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=prompt_template,
    )

    textgen_llm = TextGen(model_url=request_data.model_url, mode="instruct",
                          temperature=0.1, repetition_penalty=1.1,
                          max_new_tokens=1000, truncation_length=32768,
                          do_sample=True)

    model_name = "sentence-transformers/all-mpnet-base-v2"
    embedding_model = HuggingFaceEmbeddings(model_name=model_name)

    match = re.search(r'\/([^\/]+)\.(pdf|txt|docx?)$', request_data.context)
    doc_name = match.group(1) if match else None
    doc_dir = f'./store/{doc_name}'

    if os.path.isdir(doc_dir):
        print(f"Data file: '{doc_name}' already existed.")
        db = FAISS.load_local(doc_dir, embedding_model)

    else:
        print(f"Data file: '{doc_name}' does not exists.")
        temp_dir = './data/temp'
        os.makedirs(temp_dir, exist_ok=True)

        response = requests.get(request_data.context)
        if response.status_code == 200:
            doc_path = os.path.join(temp_dir, os.path.basename(request_data.context))
            with open(doc_path, 'wb') as f:
                f.write(response.content)
            print("File successfully downloaded.")
        else:
            print("Unsuccessful file download.")
            doc_path = request_data.context

        filePDF = re.search(r"pdf$", doc_path)

        if filePDF:  # Checks if file type is pdf, else doc/docx
            print("File type is pdf")
            loader = PyPDFLoader(doc_path)
        else:
            print("File type is doc/docx/txt")
            loader = UnstructuredFileLoader(doc_path)

        pages = loader.load_and_split()
        db = FAISS.from_documents(pages, embedding_model)
        db.save_local(doc_dir)

    retriever_pdf = db.as_retriever(search_kwargs={'k': 5})

    rag_chain = (
        {"context": retriever_pdf, "question": RunnablePassthrough()}
        | prompt
        | textgen_llm
    )

    response = rag_chain.invoke(f"{request_data.prompt}")
    return response


@app.post("/v4/rag", summary="Testing RAG V4")
async def rag4(request_data: RAGLlm2):
    """
        Naive RAG with added functionality:
        - Vector database saved based on character ID and documents for each
          character ID can be tracked inside character/document_logs.txt
        - Only accepts documents in URL
        - Able to handle pdf, doc, docx, txt file types
        - Able to query multiple documents
    """

    prompt_template = """
    ### [INST] Instruction: Give only greetings if there is no question. Answer the question based on the context information and if the question can't be answered based on the context, say "I don't know". Here is context to help:

    {context}

    ### QUESTION:
    {question} [/INST]
    """

    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=prompt_template,
    )

    textgen_llm = TextGen(model_url=request_data.model_url, mode="instruct",
                          temperature=0.1, repetition_penalty=1.1,
                          max_new_tokens=1000, truncation_length=32768,
                          do_sample=True)

    model_name = "sentence-transformers/all-mpnet-base-v2"
    embedding_model = HuggingFaceEmbeddings(model_name=model_name)

    char_dir = f'character/{request_data.character_id}'
    doc_log = f'{char_dir}/document_logs.txt'
    temp_dir = f'./temp/{request_data.character_id}'
    os.makedirs(temp_dir, exist_ok=True)

    if os.path.isdir(char_dir):  # Check if character exists
        print("Character exists.")

        f = open(doc_log, "r")
        cur_list = set()
        for x in f:
            cur_list.add(x.rstrip('\n'))
        f.close()

        new_list = set()
        doc_map = {}
        for idx, value in enumerate(request_data.contexts):
            match = re.search(r'\/([^\/]+)\.(pdf|txt|docx?)$', value)
            doc_name = match.group(1) if match else None
            new_list.add(doc_name)
            # Store doc url and accessible by document name
            doc_map[doc_name] = value

        if new_list == cur_list:
            print(f"Same set: {cur_list} && {new_list}")
            faiss_index = FAISS.load_local(char_dir, embedding_model)

        elif cur_list.issubset(new_list):
            print(f"{cur_list} is a subset of {new_list}")
            faiss_index = FAISS.load_local(char_dir, embedding_model)

            new_doc = new_list - cur_list
            for doc in new_doc:

                response = requests.get(doc_map[doc])
                if response.status_code == 200:
                    doc_path = os.path.join(temp_dir, os.path.basename(doc_map[doc]))
                    with open(doc_path, 'wb') as f:
                        f.write(response.content)
                        print("File successfully downloaded.")

                else:
                    print("Unsuccessful file download")
                    doc_path = doc_map[doc]

                # Only for printing documents name and no other functions
                match = re.search(r'\/([^\/]+)\.(pdf|txt|docx?)$', doc_path)
                doc_name = match.group(1) if match else None
                print(doc_name)
                f = open(doc_log, "a")
                f.write(f'{doc_name}\n')
                f.close()

                filePDF = re.search(r"pdf$", doc_path)
                print(doc_path)
                if filePDF:  # Checks if file type is pdf, else .doc/.docx/.txt
                    loader = PyPDFLoader(doc_path)
                else:
                    loader = UnstructuredFileLoader(doc_path)

                pages = loader.load_and_split()

                faiss_index_i = FAISS.from_documents(pages, embedding_model)
                faiss_index.merge_from(faiss_index_i)
                os.remove(doc_path)
            faiss_index.save_local(char_dir)

        else:
            print(f"Not a subset and different list: {cur_list} && {new_list}")
            with open(doc_log, 'w') as f:
                pass
            print(request_data.contexts)
            for idx, value in enumerate(request_data.contexts):

                response = requests.get(value)
                if response.status_code == 200:
                    doc_path = os.path.join(temp_dir, os.path.basename(value))
                    with open(doc_path, 'wb') as f:
                        f.write(response.content)
                    print("File successfully downloaded.")
                else:
                    print("Unsuccessful file download.")
                    doc_path = value

                # Only for printing documents name and no other functions
                match = re.search(r'\/([^\/]+)\.(pdf|txt|docx?)$', doc_path)
                doc_name = match.group(1) if match else None
                print(doc_name)
                f = open(doc_log, "a")
                f.write(f'{doc_name}\n')
                f.close()

                filePDF = re.search(r"pdf$", doc_path)
                print(doc_path)
                if filePDF:  # Checks if file type is pdf, else .doc/.docx/.txt
                    loader = PyPDFLoader(doc_path)
                else:
                    loader = UnstructuredFileLoader(doc_path)

                pages = loader.load_and_split()

                if idx == 0:
                    faiss_index = FAISS.from_documents(pages, embedding_model)
                else:
                    faiss_index_i = FAISS.from_documents(pages,
                                                         embedding_model)
                    faiss_index.merge_from(faiss_index_i)

                os.remove(doc_path)
            faiss_index.save_local(char_dir)

    else:
        print("Character does not exist.")
        os.makedirs(char_dir)
        for idx, value in enumerate(request_data.contexts):
            response = requests.get(value)
            if response.status_code == 200:
                doc_path = os.path.join(temp_dir, os.path.basename(value))
                with open(doc_path, 'wb') as f:
                    f.write(response.content)
                print("File successfully downloaded.")
            else:
                print("Unsuccessful file download. Resorting to local path.")
                doc_path = value

            # Only for printing documents name and no other functions
            match = re.search(r'\/([^\/]+)\.(pdf|txt|docx?)$', doc_path)
            doc_name = match.group(1) if match else None
            print(doc_name)
            f = open(doc_log, "a+")
            f.write(f'{doc_name}\n')
            f.close()

            filePDF = re.search(r"pdf$", doc_path)
            print(doc_path)
            if filePDF:  # Checks if file type is pdf, else .doc/.docx/.txt
                loader = PyPDFLoader(doc_path)
            else:
                loader = UnstructuredFileLoader(doc_path)

            pages = loader.load_and_split()

            if idx == 0:
                faiss_index = FAISS.from_documents(pages, embedding_model)
            else:
                faiss_index_i = FAISS.from_documents(pages,
                                                     embedding_model)
                faiss_index.merge_from(faiss_index_i)

            os.remove(doc_path)
        faiss_index.save_local(char_dir)

    os.rmdir(temp_dir)

    retriever = faiss_index.as_retriever(search_kwargs={'k': 5})

    rag_chain = ( 
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | textgen_llm
    )

    response = rag_chain.invoke(f"{request_data.prompt}")
    return response


@app.post("/v4-1/rag", summary="Testing RAG V4.1")
async def rag4_1(request_data: RAGLlm2):
    """
        Naive RAG with added functionality:
        - Vector database saved based on character ID and documents for each
          character ID can be tracked inside character/document_logs.txt
        - Only accepts documents in URL
        - Able to handle pdf, doc, docx, txt file types
        - Able to query multiple documents
        - Architecture change for faster response. (Added use of both store
          and character vector database)
    """

    prompt_template = """
    ### [INST] Instruction: Give only greetings if there is no question. Answer the question based on the context information and if the question can't be answered based on the context, say "I don't know". Here is context to help:

    {context}

    ### QUESTION:
    {question} [/INST]
    """

    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=prompt_template,
    )

    textgen_llm = TextGen(model_url=request_data.model_url, mode="instruct",
                          temperature=0.1, repetition_penalty=1.1,
                          max_new_tokens=1000, truncation_length=32768,
                          do_sample=True)

    model_name = "sentence-transformers/all-mpnet-base-v2"
    embedding_model = HuggingFaceEmbeddings(model_name=model_name)

    char_dir = f'character/{request_data.character_id}'
    doc_log = f'{char_dir}/document_logs.txt'
    temp_dir = f'./temp/{request_data.character_id}'
    store_dir = './store'
    os.makedirs(temp_dir, exist_ok=True)

    try:
        f = open(doc_log, "r")
        cur_list = set()
        for x in f:
            cur_list.add(x.rstrip('\n'))
        f.close()
    except FileNotFoundError:
        cur_list = set()

    new_list = set()
    doc_map = {}
    for idx, value in enumerate(request_data.contexts):
        match = re.search(r'\/([^\/]+)\.(pdf|txt|docx?)$', value)
        doc_name = match.group(1) if match else value.replace("/", "_")
        new_list.add(doc_name)
        # Store doc url and accessible by document name
        doc_map[doc_name] = value

    if not os.path.isdir(char_dir) or not cur_list.issubset(new_list):
        print("Character does not exist or list is not a subset.\
              \nReloading all documents...")

        os.makedirs(char_dir, exist_ok=True)
        with open(doc_log, 'w') as f:  # Create a new empty document_logs.txt
            pass

        for idx, value in enumerate(request_data.contexts):
            # Initialise doc_name by getting key from doc_map by value
            doc_index = list(doc_map.values()).index(value)
            doc_name = list(doc_map.keys())[doc_index]

            if os.path.isdir(f"{store_dir}/{doc_name}"):
                print(f"Load local store: {doc_name}")
                append_doc_log(doc_name, doc_log)
                faiss_index_i = FAISS.load_local(f"{store_dir}/{doc_name}",
                                                 embedding_model)
            else:
                pages = doc_load(value, doc_log, temp_dir)
                faiss_index_i = FAISS.from_documents(pages,
                                                     embedding_model)
                faiss_index_i.save_local(f'{store_dir}/{doc_name}')

            if idx == 0:
                faiss_index = faiss_index_i
            else:
                faiss_index.merge_from(faiss_index_i)

        faiss_index.save_local(char_dir)

    else:
        print("Character exists. Same set unless stated.")

        faiss_index = FAISS.load_local(char_dir, embedding_model)

        if new_list != cur_list:
            print(f"Is a subset: {cur_list} SUBSET OF {new_list}\
                  \nAppending...")

            new_doc = new_list - cur_list
            for doc_name in new_doc:
                if os.path.isdir(f"{store_dir}/{doc_name}"):
                    print(f"Load local store: {doc_name}")
                    append_doc_log(doc_name, doc_log)
                    faiss_index_i = FAISS.load_local(f"{store_dir}/{doc_name}",
                                                     embedding_model)
                else:
                    pages = doc_load(doc_map[doc_name], doc_log, temp_dir)
                    faiss_index_i = FAISS.from_documents(pages,
                                                         embedding_model)
                    faiss_index_i.save_local(f'{store_dir}/{doc_name}')

                faiss_index.merge_from(faiss_index_i)

            faiss_index.save_local(char_dir)

    os.rmdir(temp_dir)

    retriever = faiss_index.as_retriever(search_kwargs={'k': 5})

    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | textgen_llm
    )

    response = rag_chain.invoke(f"{request_data.prompt}")
    return response
