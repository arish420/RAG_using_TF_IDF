import streamlit as st
import sqlite3
import numpy as np
import joblib

from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
import re
import os
from dotenv import load_dotenv

# load_dotenv() 

from dotenv import dotenv_values

config = dotenv_values(".env")


os.environ["GROQ_API_KEY"] = config["GROQ_API_KEY"]


with open("chunks_per_page.md", "r", encoding="utf-8") as f:
    content = f.read()

chunks = [
    item.strip()
    for item in re.split(r"^## Item \d+\s*$", content, flags=re.MULTILINE)
    if item.strip()
]

from spellchecker import SpellChecker
import spacy

# Load once
spell = SpellChecker()
nlp = spacy.load("en_core_web_sm")

def correct_spelling(text):
    words = text.split()
    corrected = []
    for w in words:
        # only correct if spellchecker thinks it's misspelled
        if w.lower() in spell.unknown([w.lower()]):
            fixed = spell.correction(w.lower())
            corrected.append(fixed if fixed else w)
        else:
            corrected.append(w)
    return " ".join(corrected)

def lemmatize(text):
    doc = nlp(text)
    return " ".join([token.lemma_ for token in doc])

def preprocess_query(query):
    step1 = correct_spelling(query)
    step2 = lemmatize(step1)
    return step2

# # Example
# q = "I am runnning fastr to catch the bal"
# print("Original :", q)
# print("Corrected:", correct_spelling(q))
# print("Final    :", preprocess_query(q))


# ============================================================
# CONFIG
# ============================================================

DB_PATH = "rag_without_rai_final.db"
TFIDF_VECTORIZER_PATH = "tfidf_vectorizer_per_page.pkl"
TFIDF_MATRIX_PATH = "tfidf_matrix_per_page.pkl"

TOP_K = 5

tfidf_context = ""
embedding_context = ""
# ============================================================
# STREAMLIT UI
# ============================================================

st.set_page_config(
    page_title="TF-IDF vs Sentence Transformer RAG",
    layout="wide"
)

st.title("TF-IDF vs Sentence Transformer RAG")

query = st.text_input(
    "Enter your question:",
    placeholder="Ask something about your documents..."
)

query =preprocess_query(query)

# ============================================================
# LOAD TF-IDF
# ============================================================

@st.cache_resource
def load_tfidf():

    vectorizer = joblib.load(TFIDF_VECTORIZER_PATH)
    tfidf_matrix = joblib.load(TFIDF_MATRIX_PATH)

    return vectorizer, tfidf_matrix


# ============================================================
# LOAD SENTENCE TRANSFORMER
# ============================================================

@st.cache_resource
def load_embedding_model():

    return SentenceTransformer(
        "all-MiniLM-L6-v2"
    )


# ============================================================
# LOAD SQLITE DATA
# ============================================================

@st.cache_data
def load_chunks_from_db():

    conn = sqlite3.connect(DB_PATH)

    cursor = conn.cursor()

    cursor.execute("""
        SELECT id, text, embedding
        FROM chunks
    """)

    rows = cursor.fetchall()

    conn.close()

    ids = []
    texts = []
    embeddings = []

    for row_id, text, embedding_blob in rows:

        ids.append(row_id)
        texts.append(text)

        embedding = np.frombuffer(
            embedding_blob,
            dtype=np.float32
        )

        embeddings.append(embedding)

    embeddings = np.array(embeddings)

    return ids, texts, embeddings


# ============================================================
# GROQ MODEL
# ============================================================

@st.cache_resource
def load_llm():

    return ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0
    )


# ============================================================
# GENERATE ANSWER
# ============================================================

def generate_answer(llm, query, context):

    prompt = ChatPromptTemplate.from_template("""
You are a helpful RAG assistant.
Answer greetings queries and tell about your role and services that your are providing.


Answer the user's question using ONLY the provided context.

If the answer cannot be found in the context, say:
"I don't know based on the provided context."

Do not make up information.


Context:
{context}

Question:
{query}

Answer:
""")

    chain = prompt | llm

    response = chain.invoke({
        "context": context,
        "query": query
    })

    return response.content


# ============================================================
# TF-IDF RETRIEVAL
# ============================================================

def retrieve_tfidf(query, vectorizer, tfidf_matrix, texts):

    query_vector = vectorizer.transform([query])

    similarities = cosine_similarity(
        query_vector,
        tfidf_matrix
    ).flatten()

    top_indices = np.argsort(
        similarities
    )[::-1][:TOP_K]

    results = []

    for index in top_indices:

        results.append({
            "index": index,
            "text": texts[index],
            "score": similarities[index]
        })

    return results


# ============================================================
# SENTENCE TRANSFORMER RETRIEVAL
# ============================================================

def retrieve_embeddings(
    query,
    embedding_model,
    ids,
    texts,
    embeddings
):

    query_embedding = embedding_model.encode(
        query,
        convert_to_numpy=True
    ).astype(np.float32)

    similarities = cosine_similarity(
        query_embedding.reshape(1, -1),
        embeddings
    ).flatten()

    top_indices = np.argsort(
        similarities
    )[::-1][:TOP_K]

    results = []

    for index in top_indices:

        results.append({
            "id": ids[index],
            "text": texts[index],
            "score": similarities[index]
        })

    return results


# ============================================================
# MAIN
# ============================================================

if query:

    with st.spinner("Retrieving relevant chunks..."):

        try:

            # Load resources
            vectorizer, tfidf_matrix = load_tfidf()

            embedding_model = load_embedding_model()

            ids, texts, embeddings = load_chunks_from_db()

            llm = load_llm()


            # ----------------------------------------------------
            # TF-IDF
            # ----------------------------------------------------

            tfidf_results = retrieve_tfidf(
                query,
                vectorizer,
                tfidf_matrix,
                chunks
            )

            tfidf_context = "\n\n".join(
                f"[Chunk {r['index']}]\n{r['text']}"
                for r in tfidf_results
            )


            # ----------------------------------------------------
            # Sentence Transformer
            # ----------------------------------------------------

            embedding_results = retrieve_embeddings(
                query,
                embedding_model,
                ids,
                texts,
                embeddings
            )

            embedding_context = "\n\n".join(
                f"[Chunk {r['id']}]\n{r['text']}"
                for r in embedding_results
            )
        except:
            st.info("No Relevant Information Retrieved")


    # ========================================================
    # GENERATE BOTH ANSWERS
    # ========================================================

    with st.spinner("Generating answers with Groq..."):
        try:

            tfidf_answer = generate_answer(
                llm,
                query,
                tfidf_context
            )
            # st.write(tfidf_context)
    
            embedding_answer = generate_answer(
                llm,
                query,
                embedding_context
            )
        except:
            pass


    # ========================================================
    # DISPLAY ANSWERS SIDE-BY-SIDE
    # ========================================================

    col1, col2 = st.columns(2)


    # ========================================================
    # TF-IDF COLUMN
    # ========================================================

    with col1:
        try:

            st.subheader("TF-IDF RAG")
    
            st.markdown("### Answer")
    
            st.write(tfidf_answer)
    
            st.markdown("### Retrieved Chunks")
    
            for i, result in enumerate(tfidf_results, 1):
    
                with st.expander(
                    f"Chunk {i} — Score: {result['score']:.4f}"
                ):
    
                    st.write(result["text"])
        except:
            pass


    # ========================================================
    # SENTENCE TRANSFORMER COLUMN
    # ========================================================

    with col2:
        try:

            st.subheader("Sentence Transformer RAG")
    
            st.markdown("### Answer")
    
            st.write(embedding_answer)
    
            st.markdown("### Retrieved Chunks")
    
            for i, result in enumerate(embedding_results, 1):
    
                with st.expander(
                    f"Chunk {i} — Score: {result['score']:.4f}"
                ):
    
                    st.write(result["text"])
        except:
            pass

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("TF-IDF")

        st.markdown("### Context")
        if tfidf_context:

            st.write(tfidf_context)

        # st.markdown("### Retrieved Chunks")

    with col2:
        st.subheader("Sentence-Transformer")

        st.markdown("### Context")
        if embedding_context:
            st.write(embedding_context)


        
