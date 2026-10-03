import os
import tempfile

import google.generativeai as genai
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.document_loaders import (
    CSVLoader,
    Docx2txtLoader,
    PyPDFLoader,
    TextLoader,
)
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("GOOGLE_API_KEY not found in .env file")

genai.configure(api_key=api_key)

PERSIST_DIR = os.path.join(os.path.dirname(__file__), "persist")
os.makedirs(PERSIST_DIR, exist_ok=True)

embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-2")

llm = ChatGoogleGenerativeAI(
    model="models/gemini-2.5-flash",
    temperature=0,
    streaming=True,
)

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

IMAGE_EXTS = {"png", "jpg", "jpeg", "gif", "bmp", "webp"}
DOC_EXTS = {"pdf", "txt", "csv", "docx", "md"}


def get_vectorstore() -> Chroma:
    return Chroma(
        persist_directory=PERSIST_DIR,
        embedding_function=embeddings,
        collection_name="rag_collection",
    )


def get_retriever(chat_id: int, k: int = 4):
    vectorstore = get_vectorstore()
    return vectorstore.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={
            "score_threshold": 0.5,
            "k": k,
            "filter": {"chat_id": str(chat_id)},
        },
    )


def _format_docs(docs):
    formatted = []
    for doc in docs:
        source_name = doc.metadata.get("source_name", doc.metadata.get("source", "unknown"))
        page = doc.metadata.get("page", "?")
        formatted.append(f"[Source: {source_name}, Page {page}]\n{doc.page_content}")
    return "\n\n".join(formatted)


def get_relevant_docs(chat_id: int, query: str, k: int = 6):
    retriever = get_retriever(chat_id, k=k)
    return retriever.invoke(query)


def stream_response(chat_id: int, query: str, docs=None):
    if docs is None:
        docs = get_relevant_docs(chat_id, query)

    context = _format_docs(docs) if docs else "No relevant context found."

    prompt = ChatPromptTemplate.from_template(
        "You are a Study Notes assistant. Answer the question strictly using "
        "the provided context from the user's uploaded study materials.\n\n"
        "Rules:\n"
        "1. ONLY use information from the provided context. Do not use outside knowledge.\n"
        "2. If the context does not contain enough information, say: "
        "\"I don't have enough information in your uploaded notes to answer this question.\"\n"
        "3. When referencing information, cite the source inline like [SourceName, p.X].\n"
        "4. Be concise, accurate, and well-structured.\n\n"
        "Context:\n{context}\n\n"
        "Question: {question}\n\n"
        "Answer:"
    )

    chain = prompt | llm

    for chunk in chain.stream({"context": context, "question": query}):
        if chunk.content:
            yield chunk.content


def _add_docs(vectorstore, docs, chat_id, source_name):
    chunks = text_splitter.split_documents(docs)
    for chunk in chunks:
        chunk.metadata["chat_id"] = str(chat_id)
        chunk.metadata["source_name"] = source_name
        raw_page = chunk.metadata.get("page")
        if raw_page is not None:
            chunk.metadata["page"] = int(raw_page) + 1
        elif "page" not in chunk.metadata:
            chunk.metadata["page"] = 1
        for key in list(chunk.metadata):
            if key not in ("chat_id", "source_name", "page", "source"):
                del chunk.metadata[key]
    if chunks:
        vectorstore.add_documents(chunks)
    return len(chunks)


def _save_temp(file, suffix):
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(file.getvalue())
        return tmp.name


def process_document(file, chat_id, source_name):
    ext = source_name.rsplit(".", 1)[-1].lower() if "." in source_name else ""
    suffix = f".{ext}" if ext else ".tmp"

    path = _save_temp(file, suffix)
    try:
        if ext == "pdf":
            loader = PyPDFLoader(path)
        elif ext == "docx":
            loader = Docx2txtLoader(path)
        elif ext == "csv":
            loader = CSVLoader(path)
        else:
            loader = TextLoader(path, encoding="utf-8")
        docs = loader.load()
    finally:
        os.unlink(path)

    vectorstore = get_vectorstore()
    return _add_docs(vectorstore, docs, chat_id, source_name)


def process_image(file, chat_id, source_name):
    from PIL import Image

    img = Image.open(file)
    model = genai.GenerativeModel("models/gemini-2.5-flash")
    response = model.generate_content(["Extract all text from this image", img])
    text = response.text or ""

    doc = Document(page_content=text, metadata={"source": source_name})
    vectorstore = get_vectorstore()
    return _add_docs(vectorstore, [doc], chat_id, source_name)


def process_url(url, chat_id):
    resp = requests.get(
        url,
        timeout=15,
        headers={"User-Agent": "Mozilla/5.0 (compatible; RAGBot/1.0)"},
    )
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "lxml")
    for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
        tag.decompose()
    text = soup.get_text(separator="\n", strip=True)
    text = "\n".join(line for line in text.splitlines() if line.strip())

    doc = Document(page_content=text, metadata={"source": url})
    vectorstore = get_vectorstore()
    return _add_docs(vectorstore, [doc], chat_id, url)


def delete_source_vectors(chat_id, source_name):
    vectorstore = get_vectorstore()
    collection = vectorstore._collection
    results = collection.get(
        where={
            "$and": [
                {"chat_id": str(chat_id)},
                {"source_name": source_name},
            ]
        }
    )
    if results["ids"]:
        collection.delete(ids=results["ids"])


def delete_chat_vectors(chat_id):
    vectorstore = get_vectorstore()
    collection = vectorstore._collection
    results = collection.get(where={"chat_id": str(chat_id)})
    if results["ids"]:
        collection.delete(ids=results["ids"])
