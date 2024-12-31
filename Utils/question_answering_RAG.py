from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.messages import HumanMessage
from langchain_core.runnables import Runnable

from Utils.utils import read_file
from dotenv import dotenv_values, find_dotenv
import os
from streamlit.runtime.uploaded_file_manager import UploadedFile
from typing import Generator
import time


# Set environment variables
os.environ["HuggingFace_API"] = dotenv_values(find_dotenv())[
    "HuggingFace_API"
]
os.environ["GOOGLE_API_KEY"] = dotenv_values(find_dotenv())["gemini_api_key"]


def init_llm_model() -> ChatGoogleGenerativeAI:
    """Initializes the ChatGoogleGenerativeAI model."""
    return ChatGoogleGenerativeAI(
        model="gemini-1.5-flash", temperature=0.1, max_tokens=None, timeout=None, max_retries=2
    )


def init_embeddings_model() -> HuggingFaceEmbeddings:
    """Initializes the Hugging Face embeddings model."""
    return HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")



def split_text(text: str, chunk_size: int = 1000, chunk_overlap: int = 100) -> list:
    """Splits the given text into manageable chunks."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return text_splitter.split_text(text)


def create_vector_store(
    uploaded_files: list[UploadedFile], embedding_model: HuggingFaceEmbeddings
) -> FAISS:
    """Creates a FAISS vector store from uploaded files."""
    all_split_texts = []

    for uploaded_file in uploaded_files:
        texts = read_file(uploaded_file)
        all_split_texts.extend(split_text(texts))

    return FAISS.from_texts(all_split_texts, embedding_model)


def create_qa_model(
    vector_store: FAISS,
    llm: ChatGoogleGenerativeAI,
    prompt: ChatPromptTemplate,
    qa_prompt: ChatPromptTemplate,
) -> Runnable:
    """Initializes the QA model."""
    history_aware_retriever = create_history_aware_retriever(
        llm, vector_store.as_retriever(kwargs={"k": 6}), qa_prompt
    )
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    return create_retrieval_chain(history_aware_retriever, question_answer_chain)



def init_prompt() -> tuple[ChatPromptTemplate, ChatPromptTemplate]: ##
    """Initializes prompt templates."""
    qa_system_prompt = """
    Given the chat history and the latest user question, which might reference context in the chat history, 
    create a standalone question that can be understood without the chat history. 
    Do NOT answer the question, 
    just reformulate it if needed and otherwise return it as is.
    """
    system_prompt = """
    You are a knowledgeable and precise assistant tasked with answering queries based only on the provided context. Adhere to the following rules:

    1. Use the supplied context as the exclusive source for your answers. Do not rely on outside knowledge or make assumptions.
    2. If the context does not provide enough information to answer the question, respond explicitly with: 
    "I cannot determine the answer from the given information. Additional resources may help."
    3. When the answer is clear from the context, deliver a concise and focused response, limited to five sentences maximum.

    Provided Context:
    {context}
    """

    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", qa_system_prompt),
            MessagesPlaceholder("chat_history"), # Placeholder for chat history
            ("human", "{input}"),
        ]
    )
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    return prompt, qa_prompt


def qa(text: str, qa_model: Runnable, messages: list) -> Generator[str, str, str]:
    """Generates answers to questions based on the given text."""
    chat_history = []

    for message in messages:
        if message["role"] == "user":
            chat_history.append(HumanMessage(content=message["content"]))
        elif message["role"] == "assistant":
            chat_history.append(message["content"])

    try:
        response = qa_model.invoke({"chat_history": chat_history, "input": text}) # Invoke the QA model with the chat history and the user input
        answer = response["answer"].strip()
    except Exception:
        return None

    for word in answer.split(" "):
        yield word + " "
        time.sleep(0.1)

def init_gemini_model() -> ChatGoogleGenerativeAI:
    """Initializes the Gemini model."""
    return ChatGoogleGenerativeAI(
        model="gemini-1.5-flash", temperature=0.1, max_tokens=None, timeout=None, max_retries=2
    )




def gemini_generate_response(prompt_text: str, gemini_model: ChatGoogleGenerativeAI) -> str:
    """Generates response using the Gemini model."""
    # Ensure the prompt is wrapped in a HumanMessage
    prompt_message = HumanMessage(content=prompt_text) # Wrap the prompt in a HumanMessage
    response = gemini_model([prompt_message])
    return response.content.strip()