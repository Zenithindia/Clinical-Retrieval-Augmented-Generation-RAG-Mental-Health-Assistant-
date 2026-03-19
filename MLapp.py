%%writefile app.py

import streamlit as st
import faiss
import json
import numpy as np
import torch
import re
import os

from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM

# =====================================================
# PAGE SETTINGS
# =====================================================
st.set_page_config(page_title="🧠 Personal Mental Health Chatbot")
st.title("🧠 Personal Mental Health Chatbot")

DEVICE = "cpu"
RELEVANCE_THRESHOLD = 0.55
TOP_K = 3

# =====================================================
# BASE INDEX DIRECTORY
# =====================================================
BASE_INDEX_PATH = "/content/drive/MyDrive/CB/Nancy_INDEX"
if not os.path.exists(BASE_INDEX_PATH):
    st.error("Nancy_INDEX folder not found.")
    st.stop()

# =====================================================
# CATEGORY ROUTER
# =====================================================
CATEGORY_KEYWORDS = {

    "Questionnaires/depression":[
        "depression","sad","hopeless","worthless","empty","lonely","no motivation"
    ],

    "Questionnaires/anxiety":[
        "anxiety","panic","fear","nervous","overthinking","social anxiety","worry"
    ],

    "Questionnaires/stress and burnout":[
        "stress","burnout","pressure","overwhelmed","tired","exhausted","exam"
    ],

    "Questionnaires/ptsd":[
        "ptsd","trauma","flashback","nightmares","traumatic memory"
    ],

    "Questionnaires/substance abuse":[
        "substance","drug","addiction","alcohol","smoking","dependency"
    ],

    "Questionnaires/suicidal ideation":[
        "suicide","kill myself","end my life","self harm","want to die"
    ],

    "Questionnaires/emotional regulation":[
        "emotional control","control my emotions","anger issues","mood swings"
    ],

    "Questionnaires/marital life":[
        "marriage problem","husband","wife","relationship","marital conflict"
    ],

    "Questionnaires/well being":[
        "wellbeing","life satisfaction","happiness","positive life"
    ],

    "Questionnaires/workplace":[
        "boss","job stress","workplace","office stress","manager problem"
    ],

    "Questionnaires/GENERAL HEALTH":[
        "sleep","health","fatigue","energy","daily functioning"
    ],

    "Health Psychology and Addiction":[
        "addiction","health behavior","substance dependency"
    ],

    "Industrial and Organizational Psychology":[
        "organizational","work productivity","employee stress","work culture"
    ],

    "Counselling and Therapies":[
        "therapy","counselling","cbt","treatment","psychotherapy"
    ],

    "Freud Psychology and Defense mechanisms":[
        "defense mechanism","repression","projection","freud"
    ],

    "Abnormal Psychology & ICD 11":[
        "mental disorder","psychological disorder","diagnosis","icd"
    ],

    "Miscellaneous":[
        "mental health","psychology","emotional problem"
    ]
}

def detect_category(query):
    q=query.lower()
    best_match=None
    best_score=0

    for category,keywords in CATEGORY_KEYWORDS.items():
        score=sum(1 for k in keywords if k in q)

        if score>best_score:
            best_score=score
            best_match=category

    return best_match

# =====================================================
# LOAD MODELS
# =====================================================
@st.cache_resource
def load_embedding():
    return SentenceTransformer("all-MiniLM-L6-v2")

@st.cache_resource
def load_llm():

    model_name="Qwen/Qwen2.5-0.5B-Instruct"

    tokenizer=AutoTokenizer.from_pretrained(model_name)

    model=AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32
    )

    model.to(DEVICE)
    model.eval()

    return tokenizer,model

embedding_model=load_embedding()
tokenizer,model=load_llm()

# =====================================================
# LOAD INDEX
# =====================================================
@st.cache_resource
def load_category_index(category_path):

    full_path=os.path.join(BASE_INDEX_PATH,category_path)

    faiss_path=os.path.join(full_path,"index.faiss")
    chunks_path=os.path.join(full_path,"chunks.json")

    if not os.path.exists(faiss_path) or not os.path.exists(chunks_path):
        return None,None

    index=faiss.read_index(faiss_path)

    with open(chunks_path,"r",encoding="utf-8") as f:
        chunks=json.load(f)

    return index,chunks

def extract_text(chunk):

    if isinstance(chunk,str):
        return chunk

    if isinstance(chunk,dict) and "text" in chunk:
        return chunk["text"]

    return ""

# =====================================================
# RETRIEVAL
# =====================================================
def retrieve_best_chunk(query):

    category=detect_category(query)

    if not category:
        return None,None,0.0

    index,chunks=load_category_index(category)

    if index is None:
        return None,category,0.0

    query_embedding=embedding_model.encode(
        [query],
        normalize_embeddings=True
    ).astype("float32")

    D,I=index.search(query_embedding,TOP_K)

    selected_chunks=[
        extract_text(chunks[idx])
        for score,idx in zip(D[0],I[0])
        if idx>=0 and score>=RELEVANCE_THRESHOLD
    ]

    if not selected_chunks:
        return None,category,float(D[0][0])

    return "\n\n".join(selected_chunks),category,float(D[0][0])

# =====================================================
# CLEAN TEXT
# =====================================================
def clean_generated_text(text):

    text=re.sub(r'\*\*','',text)
    text=re.sub(r'\s+',' ',text)

    return text.strip()

# =====================================================
# SENTENCE SPLIT
# =====================================================
def split_sentences(text):

    sentences=re.split(r'(?<=[.!?])\s+',text)

    return [s.strip() for s in sentences if len(s.strip())>30]

# =====================================================
# BRIEF SUMMARY
# =====================================================
def generate_brief_summary(query):

    prompt=f"""
Summarize the emotional issue in the following question in 2 short sentences.

User question:
{query}

Summary:
"""

    inputs=tokenizer(prompt,return_tensors="pt").to(DEVICE)

    with torch.no_grad():
        outputs=model.generate(
            **inputs,
            max_new_tokens=60,
            temperature=0.3
        )

    generated_tokens=outputs[0][inputs["input_ids"].shape[-1]:]

    summary=tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True
    )

    return clean_generated_text(summary)

# =====================================================
# MOTIVATION (IMPROVED NATURAL VERSION)
# =====================================================
def generate_motivation(query):

    prompt=f"""
You are a compassionate mental health guide.

Write a meaningful motivational reflection for someone facing the following emotional situation.

The reflection should feel natural, supportive, and deeply encouraging.
It can include a small inspiring thought, a life lesson, or a gentle perspective about overcoming struggles.

Important rules:
- Do NOT give instructions or numbered steps.
- Do NOT diagnose mental illness.
- Write in a warm and human tone.
- Focus on hope, resilience, and growth.

Emotional situation:
{query}

Motivational reflection:
"""

    inputs=tokenizer(prompt,return_tensors="pt").to(DEVICE)

    with torch.no_grad():

        outputs=model.generate(
            **inputs,
            max_new_tokens=180,
            temperature=0.75,
            top_p=0.92,
            repetition_penalty=1.15,
            do_sample=True
        )

    generated_tokens=outputs[0][inputs["input_ids"].shape[-1]:]

    text=tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True
    )

    text=clean_generated_text(text)

    text=re.sub(r'["“”]',"",text)
    text=re.sub(r'\d+\)',"",text)
    text=re.sub(r'\s+',' ',text)

    sentences=re.split(r'(?<=[.!?])\s+',text)

    sentences=[s.strip() for s in sentences if len(s.strip())>20]

    sentences=sentences[:8]

    return "\n".join(sentences)

# =====================================================
# FORMAT STRUCTURED ANSWER
# =====================================================
def format_structured_answer(text,brief_summary,motivation):

    text=clean_generated_text(text)

    sentences=split_sentences(text)

    if len(sentences)<6:
        return text

    intro=sentences[0]
    explanation=" ".join(sentences[1:6])
    key_points=sentences[6:9] if len(sentences)>=9 else sentences[2:5]
    impact=sentences[9:12] if len(sentences)>=12 else sentences[-4:-1]
    conclusion=sentences[-1]

    formatted=f"""
**Introduction**

{intro}

**Brief Explanation**

{brief_summary}

**Explanation**

{explanation}

**Key Points**
"""

    for kp in key_points:
        formatted+=f"\n- {kp}"

    formatted+="\n\n**Impact and Coping Strategies**"

    for im in impact:
        formatted+=f"\n- {im}"

    formatted+=f"""

**Conclusion**

{conclusion}

**Motivation**

{motivation}
"""

    return formatted.strip()

# =====================================================
# GENERATION SETTINGS
# =====================================================
GEN_KWARGS=dict(
    max_new_tokens=420,
    temperature=0.4,
    top_p=0.9,
    repetition_penalty=1.2,
    do_sample=True
)

# =====================================================
# GENERATE ANSWER
# =====================================================
def generate_answer(query,context=None):

    system_message=(
        "You are a compassionate mental health support assistant.\n\n"
        "Rules:\n"
        "- Do not diagnose mental illness.\n"
        "- Explain emotional experiences clearly.\n"
        "- Use supportive language.\n"
        "- Provide helpful coping suggestions.\n"
    )

    if context:

        user_message=f"""
Reference material:
{context}

User question:
{query}

Explain the emotional situation clearly.
"""

    else:

        user_message=f"""
User question:
{query}

Explain the emotional challenge in supportive language.
"""

    messages=[
        {"role":"system","content":system_message},
        {"role":"user","content":user_message}
    ]

    prompt=tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs=tokenizer(prompt,return_tensors="pt").to(DEVICE)

    with torch.no_grad():
        outputs=model.generate(**inputs,**GEN_KWARGS)

    generated_tokens=outputs[0][inputs["input_ids"].shape[-1]:]

    response=tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True
    )

    brief_summary=generate_brief_summary(query)

    motivation=generate_motivation(query)

    return format_structured_answer(response,brief_summary,motivation)

# =====================================================
# CHAT UI
# =====================================================
if "chat_history" not in st.session_state:
    st.session_state.chat_history=[]

user_input=st.chat_input("Ask about stress, anxiety, sleep, depression...")

if user_input:

    st.session_state.chat_history.append(
        {"role":"user","content":user_input}
    )

    with st.spinner("Thinking..."):

        chunk,category,score=retrieve_best_chunk(user_input)

        reply=generate_answer(
            user_input,
            chunk if chunk else None
        )

        source_type="📖 RAG Knowledge Base" if chunk else "🤖 Model Response"

    st.session_state.chat_history.append(
        {"role":"assistant","content":reply}
    )

    with st.expander("🔎 Transparency Details"):

        st.write("Detected Category:",category)

        st.write("Similarity Score:",round(score,4))

        if chunk:
            st.write("Retrieved Context Preview:")
            st.write(chunk[:500])
        else:
            st.write("No relevant chunk found.")

        st.write("Answer Source:",source_type)

for msg in st.session_state.chat_history:

    with st.chat_message("user" if msg["role"]=="user" else "assistant"):

        st.markdown(
            f"<span style='font-size:16px'>{msg['content']}</span>",
            unsafe_allow_html=True
        )