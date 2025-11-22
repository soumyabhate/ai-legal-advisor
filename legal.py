#!/usr/bin/env python
# coding: utf-8

import streamlit as st
# import sounddevice as sd

try:
    import sounddevice as sd
    import scipy.io.wavfile
    AUDIO_AVAILABLE = True
except Exception:
    sd = None
    AUDIO_AVAILABLE = False

import numpy as np
import scipy.io.wavfile
import ffmpeg
import shutil
import os
import threading
import queue
from datetime import datetime
from groq import Groq
from io import BytesIO
from elevenlabs.client import ElevenLabs
import base64
import time
from pymongo import MongoClient
import json
import PyPDF2
import re
from io import StringIO

import io
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from gensim import corpora
from gensim.models import LdaModel
from dotenv import load_dotenv 

# Page config
st.set_page_config(page_title="Legal Advisor AI", layout="centered", page_icon="⚖️")

# === Load secrets from .env (local only) ===
load_dotenv()  # .env file se env variables load karega

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")

# API clients
groq_client = Groq(api_key=GROQ_API_KEY)
elevenlabs_client = ElevenLabs(api_key=ELEVENLABS_API_KEY)

# MongoDB Configuration - Using existing setup
MONGO_URI = "mongodb+srv://asingh23:Hitman.13@cluster0.irnjt9h.mongodb.net/?appName=Cluster0"  # Add your MongoDB URI here or use secrets
DB_NAME = "Legal_rag"
COLLECTION_NAME = "legal_chunks"
VECTOR_INDEX_NAME = "606"  # Your existing vector search index name

# Audio config
SAMPLE_RATE = 44100
CHANNELS = 1
INPUT_DEVICE_INDEX = 1

# Initialize MongoDB connection
@st.cache_resource
def init_mongodb():
    """Initialize MongoDB connection and return collection"""
    try:
        # Use hardcoded URI first, then check secrets as fallback
        mongo_uri = MONGO_URI
        
        if not mongo_uri or mongo_uri == "":
            # Try secrets as fallback
            try:
                mongo_uri = st.secrets["MONGO_URI"]
            except:
                st.error("❌ MongoDB URI not configured. Please add it to MONGO_URI variable in the code.")
                return None
        
        if not mongo_uri:
            st.error("❌ MongoDB URI is empty")
            return None
            
        client = MongoClient(mongo_uri)
        db = client[DB_NAME]
        collection = db[COLLECTION_NAME]
        
        # Test connection
        client.admin.command('ping')
        
        return collection
    except Exception as e:
        st.error(f"❌ MongoDB connection error: {str(e)}")
        return None

# Text search in MongoDB (no embeddings needed - uses existing index)
def search_legal_documents(collection, query_text, legal_area, k=3):
    """Search MongoDB using text search on existing indexed documents"""
    try:
        # Use MongoDB text search on indexed fields
        search_query = {
            "$text": {"$search": query_text}
        }
        
        # Add legal area filter if not general
        if legal_area != "General Legal Questions":
            search_query["legal_area"] = legal_area
        
        # Perform text search with relevance score
        results = list(collection.find(
            search_query,
            {"score": {"$meta": "textScore"}}
        ).sort([("score", {"$meta": "textScore"})]).limit(k))
        
        return results
        
    except Exception as e:
        #st.warning(f"Text search error: {str(e)[:100]}")
        # Fallback to simple search without text index
        try:
            search_query = {"$or": [
                {"text": {"$regex": query_text, "$options": "i"}},
                {"content": {"$regex": query_text, "$options": "i"}}
            ]}
            
            if legal_area != "General Legal Questions":
                search_query["legal_area"] = legal_area
                
            results = list(collection.find(search_query).limit(k))
            return results
        except Exception as e2:
            st.error(f"Search failed: {str(e2)[:100]}")
            return []

# Privacy protection function
def redact_sensitive_info(text):
    """Remove or replace sensitive personal information from text"""
    
    redacted_text = text
    redactions_made = []
    
    # 1. SSN patterns (XXX-XX-XXXX, XXXXXXXXX, XXX XX XXXX)
    ssn_pattern = r'\b\d{3}[-\s]?\d{2}[-\s]?\d{4}\b'
    if re.search(ssn_pattern, redacted_text):
        redacted_text = re.sub(ssn_pattern, '[SSN REDACTED]', redacted_text)
        redactions_made.append("Social Security Numbers")
    
    # 2. Credit Card Numbers (16 digits, with or without spaces/dashes)
    cc_pattern = r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b'
    if re.search(cc_pattern, redacted_text):
        redacted_text = re.sub(cc_pattern, '[CREDIT CARD REDACTED]', redacted_text)
        redactions_made.append("Credit Card Numbers")
    
    # 3. Email addresses
    email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    if re.search(email_pattern, redacted_text):
        redacted_text = re.sub(email_pattern, '[EMAIL REDACTED]', redacted_text)
        redactions_made.append("Email Addresses")
    
    # 4. Phone Numbers (various formats)
    phone_pattern = r'\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b'
    if re.search(phone_pattern, redacted_text):
        redacted_text = re.sub(phone_pattern, '[PHONE REDACTED]', redacted_text)
        redactions_made.append("Phone Numbers")
    
    # 5. Date of Birth patterns (MM/DD/YYYY, MM-DD-YYYY, etc.)
    dob_pattern = r'\b(?:0[1-9]|1[0-2])[-/](?:0[1-9]|[12]\d|3[01])[-/](?:19|20)\d{2}\b'
    if re.search(dob_pattern, redacted_text):
        redacted_text = re.sub(dob_pattern, '[DATE REDACTED]', redacted_text)
        redactions_made.append("Dates of Birth")
    
    # 6. Driver's License (varies by state, general pattern)
    dl_pattern = r'\b[A-Z]{1,2}\d{5,8}\b'
    if re.search(dl_pattern, redacted_text):
        redacted_text = re.sub(dl_pattern, '[LICENSE REDACTED]', redacted_text)
        redactions_made.append("License Numbers")
    
    # 7. Passport Numbers (9 digits or letters+digits)
    passport_pattern = r'\b[A-Z]{1,2}\d{6,9}\b'
    if re.search(passport_pattern, redacted_text):
        redacted_text = re.sub(passport_pattern, '[PASSPORT REDACTED]', redacted_text)
        redactions_made.append("Passport Numbers")
    
    # 8. Bank Account Numbers (8-17 digits)
    bank_pattern = r'\b\d{8,17}\b'
    if re.search(bank_pattern, redacted_text):
        redacted_text = re.sub(bank_pattern, '[ACCOUNT# REDACTED]', redacted_text)
        redactions_made.append("Account Numbers")
    
    # 9. IP Addresses
    ip_pattern = r'\b(?:\d{1,3}\.){3}\d{1,3}\b'
    if re.search(ip_pattern, redacted_text):
        redacted_text = re.sub(ip_pattern, '[IP REDACTED]', redacted_text)
        redactions_made.append("IP Addresses")
    
    # 10. Medical Record Numbers (MRN)
    mrn_pattern = r'\b(?:MRN|Medical Record|Patient ID)[\s:#]*[A-Z0-9]{6,12}\b'
    if re.search(mrn_pattern, redacted_text, re.IGNORECASE):
        redacted_text = re.sub(mrn_pattern, '[MEDICAL ID REDACTED]', redacted_text, flags=re.IGNORECASE)
        redactions_made.append("Medical Record Numbers")
    
    return redacted_text, redactions_made

# Session state init
if 'recording' not in st.session_state:
    st.session_state.recording = False
if 'audio_data' not in st.session_state:
    st.session_state.audio_data = []
if 'recording_thread' not in st.session_state:
    st.session_state.recording_thread = None
if 'stop_event' not in st.session_state:
    st.session_state.stop_event = threading.Event()
if 'processing' not in st.session_state:
    st.session_state.processing = False
if 'playing_audio' not in st.session_state:
    st.session_state.playing_audio = False
if 'audio_queue' not in st.session_state:
    st.session_state.audio_queue = queue.Queue()
if 'conversation_history' not in st.session_state:
    st.session_state.conversation_history = []
if 'legal_context' not in st.session_state:
    st.session_state.legal_context = ""
if 'mongo_collection' not in st.session_state:
    st.session_state.mongo_collection = init_mongodb()
if 'uploaded_pdf_text' not in st.session_state:
    st.session_state.uploaded_pdf_text = None
if 'pdf_filename' not in st.session_state:
    st.session_state.pdf_filename = None

# UI header
st.markdown("""
    <h1 style='text-align: center; color: #B9B28A;'>⚖️ AI Legal Advisor</h1>
    <p style='text-align: center;'>Your intelligent legal consultation assistant with knowledge base</p>
    <hr>
""", unsafe_allow_html=True)

# Disclaimer at the end
# st.warning("""
# ⚠️ **Legal Disclaimer**: This AI assistant provides general legal information only. 
# It does NOT constitute legal advice. For specific legal matters, please consult a licensed attorney.
# """)

# Legal area selection
with st.sidebar:
    st.header("⚙️ Configuration")
    
    legal_area = st.selectbox(
        "Legal Area",
        [
            "General Legal Questions",
            "Contract Law",
            "Family Law",
            "Criminal Law",
            "Employment Law",
            "Real Estate Law",
            "Intellectual Property",
            "Immigration Law",
            "Business Law",
            "Personal Injury"
        ]
    )
    
    st.session_state.legal_context = legal_area
    
    jurisdiction = st.selectbox(
        "Jurisdiction (Optional)",
        ["Not Specified", "United States", "United Kingdom", "Canada", "Australia", "India", "European Union"]
    )
    
    # PDF Upload Section
    st.markdown("---")
    st.markdown("### 📄 Upload Legal Document")
    
    uploaded_file = st.file_uploader(
        "Upload PDF (optional)",
        type=['pdf'],
        help="Upload a legal document to ask questions about it"
    )
    
    if uploaded_file is not None:
        if st.button("📖 Process PDF"):
            # Processing silently
                try:
                    # Read PDF
                    pdf_reader = PyPDF2.PdfReader(uploaded_file)
                    pdf_text = ""
                    
                    for page_num in range(len(pdf_reader.pages)):
                        page = pdf_reader.pages[page_num]
                        pdf_text += page.extract_text()
                    
                    # Redact sensitive information
                    redacted_text, redactions = redact_sensitive_info(pdf_text)
                    
                    st.session_state.uploaded_pdf_text = redacted_text
                    st.session_state.pdf_filename = uploaded_file.name
                    
                    st.success(f"✅ PDF processed: {uploaded_file.name}")
                    st.info(f"📊 Extracted {len(pdf_text)} characters from {len(pdf_reader.pages)} pages")
                    
                    # Show what was redacted
                    if redactions:
                        st.warning(f"🔒 Security: Redacted {len(redactions)} types of sensitive information")
                        with st.expander("🔍 See what was redacted"):
                            for item in redactions:
                                st.write(f"• {item}")
                    else:
                        st.info("🔒 No sensitive information detected")
                    
                except Exception as e:
                    st.error(f"Error reading PDF: {str(e)}")
    
    if st.session_state.uploaded_pdf_text:
        st.success(f"📄 Active Document: {st.session_state.pdf_filename}")
        if st.button("🗑️ Remove PDF"):
            st.session_state.uploaded_pdf_text = None
            st.session_state.pdf_filename = None
            st.rerun()

        # Text Analysis Tools
        st.markdown("---")
        st.markdown("### 📊 Document Analysis Tools")

        col1, col2, col3 = st.columns(3)
        def generate_wordcloud_image(text):
            """Generate a word cloud image from text and return as BytesIO buffer"""
            wc = WordCloud(width=1200, height=600, background_color="white",
                        collocations=False, stopwords=set(ENGLISH_STOP_WORDS)).generate(text.lower())
            buf = io.BytesIO()
            plt.figure(figsize=(10, 5))
            plt.imshow(wc, interpolation="bilinear")
            plt.axis("off")
            plt.savefig(buf, format="png", bbox_inches="tight")
            plt.close()
            buf.seek(0)
            return buf
        def getTextLLM(prompt, inp):
            """Get text completion from Groq LLM with streaming"""
            full_prompt = prompt + ".\n USER INPUT:" + inp
            completion = groq_client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": full_prompt}],
                temperature=0.7,
                max_completion_tokens=1500,
                top_p=0.95,
                stream=True,
                stop=None
            )
            txtLLM = ""
            for chunk in completion:
                content = chunk.choices[0].delta.content or ""
                txtLLM += content
            return txtLLM
        def summarize_text(text):
            """Summarize text using LLM with 1:10 compression ratio"""
            return getTextLLM("Summarize this text concisely, 1:10 ratio, include key points, just give me the summary and no pre-text explaination about the summary", text)
        with col1:
            if st.button("📝 Summary"):
                with st.spinner("Generating summary..."):
                    try:
                        summary = summarize_text(st.session_state.uploaded_pdf_text)
                        st.subheader("Summary")
                        st.write(summary)
                    except Exception as e:
                        st.error(f"Error generating summary: {str(e)}")
        def run_lda_on_chunks(chunks, num_topics=6):
            """Run LDA topic modeling on document chunks"""
            texts = [c.page_content for c in chunks]

            # Define stopwords (scikit-learn + custom common words)
            custom_stopwords = set(ENGLISH_STOP_WORDS).union({
                "said", "one", "two", "also", "would", "could", "may", "might"
            })

            # Tokenize and remove stopwords + short words
            tokenized = [
                [w.lower() for w in re.findall(r"\b[a-zA-Z]+\b", t)
                if w.lower() not in custom_stopwords and len(w) > 2]
                for t in texts
            ]

            dictionary = corpora.Dictionary(tokenized)
            corpus = [dictionary.doc2bow(toks) for toks in tokenized]

            if not corpus or not dictionary:
                return None

            lda = LdaModel(corpus=corpus, id2word=dictionary, num_topics=num_topics, passes=2)
            return lda.print_topics(num_topics=num_topics, num_words=8)

        def explain_topics(topics):
            """Convert raw LDA topics into human-friendly labels + explanations."""
            readable = []
            for idx, topic in topics:
                # Extract words from topic string
                words = re.findall(r'"(.*?)"', topic)

                # Filter stopwords/common words again for safety
                filtered_words = [
                    w for w in words if w.lower() not in ENGLISH_STOP_WORDS and len(w) > 2
                ]

                label = ", ".join(filtered_words[:3])
                explanation = f"This topic is mostly about: {', '.join(filtered_words)}."
                readable.append({"Topic #": idx, "Label": label, "Explanation": explanation})
            return readable
        with col2:
            if st.button("🔍 Topic Modeling"):
                with st.spinner("Running topic analysis..."):
                    try:
                        # Import Document class for creating chunks
                        from langchain_core.documents import Document


                        # Create simple chunks from text
                        text_chunks = [st.session_state.uploaded_pdf_text[i:i+1000] 
                                      for i in range(0, len(st.session_state.uploaded_pdf_text), 1000)]
                        chunks = [Document(page_content=chunk) for chunk in text_chunks if chunk.strip()]

                        topics = run_lda_on_chunks(chunks, num_topics=5)
                        if topics:
                            readable_topics = explain_topics(topics)
                            st.subheader("Topics Found")
                            for topic in readable_topics:
                                st.write(f"**{topic['Label']}**")
                                st.caption(topic['Explanation'])
                        else:
                            st.warning("Not enough content for topic modeling")
                    except Exception as e:
                        st.error(f"Error in topic modeling: {str(e)}")

        with col3:
            if st.button("☁️ Word Cloud"):
                with st.spinner("Generating word cloud..."):
                    try:
                        img_buf = generate_wordcloud_image(st.session_state.uploaded_pdf_text)
                        st.image(img_buf, use_container_width=True)
                    except Exception as e:
                        st.error(f"Error generating word cloud: {str(e)}")

    
    # MongoDB settings - only show for Criminal Law
    st.markdown("---")
    st.markdown("### 🗄️ Knowledge Base")
    
    if legal_area == "Criminal Law":
        if st.session_state.mongo_collection is not None:
            try:
                doc_count = st.session_state.mongo_collection.count_documents({})
                st.success(f"✅ Knowledge Base Active")
                st.info(f"📚 {doc_count:,} criminal law documents")
                st.info(f"🗄️ Database: Legal_rag")
            except Exception as e:
                st.warning(f"⚠️ Could not retrieve info: {str(e)[:50]}")
        else:
            st.error("❌ Not connected to knowledge base")
            st.info("Add MONGO_URI to code")
        
        num_results = st.slider("Results to retrieve", 1, 10, 3)
    else:
        st.info("💡 Knowledge base only used for Criminal Law")
        st.caption("Other areas use general legal knowledge")
        num_results = 3  # Default value when not used
    
    st.markdown("---")
    st.markdown("### 📋 Quick Actions")
    if st.button("Clear Conversation"):
        st.session_state.conversation_history = []
        st.rerun()
    
    if st.button("Download Consultation"):
        if st.session_state.conversation_history:
            transcript = "Legal Consultation Transcript\n"
            transcript += f"Legal Area: {legal_area}\n"
            transcript += f"Jurisdiction: {jurisdiction}\n"
            transcript += f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
            
            for msg in st.session_state.conversation_history:
                role = "Client" if msg["role"] == "user" else "Legal Advisor"
                transcript += f"{role}: {msg['text']}\n\n"
                
                if msg.get("sources"):
                    transcript += "Sources:\n"
                    for src in msg["sources"]:
                        transcript += f"  - {src.get('title', 'Unknown')}\n"
                    transcript += "\n"
            
            st.download_button(
                "💾 Download Transcript",
                transcript,
                file_name=f"legal_consultation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain"
            )
    st.markdown("---")
# Debug: List audio devices
    st.markdown("### 🔧 Audio Device Settings")
    with st.expander("🔧 Check Device Settings"):
        if st.button("List Audio Devices"):
            devices = sd.query_devices()
            st.text(str(devices))
            st.info(f"Default input device: {sd.default.device[0]}")
            st.info(f"Default output device: {sd.default.device[1]}")

    # Query all devices
    devices = sd.query_devices()

    # Build readable labels
    input_devices = [
        f"{i}: {d['name']} (inputs: {d['max_input_channels']})"
        for i, d in enumerate(devices) if d['max_input_channels'] > 0
    ]
    output_devices = [
        f"{i}: {d['name']} (outputs: {d['max_output_channels']})"
        for i, d in enumerate(devices) if d['max_output_channels'] > 0
    ]

    # Select input device
    selected_input = st.selectbox("Select Input Device", input_devices)
    # Select output device
    selected_output = st.selectbox("Select Output Device", output_devices)

    if st.button("Set Devices"):
        input_index = int(selected_input.split(":")[0])
        output_index = int(selected_output.split(":")[0])
        sd.default.device = (input_index, output_index)
        st.success(f"Devices set → Input: {selected_input}, Output: {selected_output}")



# 🎙️ Audio Recording Thread
def record_audio_continuous(stop_event, audio_queue):
    """Record audio continuously until stop_event is set"""
    chunk_count = 0
    
    def callback(indata, frames, time_info, status):
        nonlocal chunk_count
        if status:
            print(f"Recording status: {status}")
        audio_queue.put(indata.copy())
        chunk_count += 1

    try:
        print(f"Starting recording with device {INPUT_DEVICE_INDEX}")
        
        with sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=CHANNELS,
            dtype='int16',
            device=INPUT_DEVICE_INDEX,
            callback=callback
        ):
            print("Recording stream opened successfully")
            while not stop_event.is_set():
                sd.sleep(100)
        
        print(f"Recording stopped. Total chunks captured: {chunk_count}")
        audio_queue.put(None)
        
    except Exception as e:
        print(f"Recording error: {e}")
        audio_queue.put(None)

# 🎧 Save to file
def save_recording(audio_queue, filename="client_question.wav"):
    print(f"=== Saving recording ===")
    
    audio_chunks = []
    while True:
        try:
            chunk = audio_queue.get(timeout=1.0)
            if chunk is None:
                break
            audio_chunks.append(chunk)
        except queue.Empty:
            break
    
    print(f"Audio chunks collected: {len(audio_chunks)}")
    
    if not audio_chunks:
        return None

    try:
        audio_array = np.concatenate(audio_chunks, axis=0)
        print(f"Audio duration: {len(audio_array) / SAMPLE_RATE:.2f} seconds")
        
        # Check minimum duration
        min_samples = int(SAMPLE_RATE * 0.3)
        if len(audio_array) < min_samples:
            st.error(f"Recording too short. Please speak for at least 1 second.")
            return None
        
        # Try MP3 conversion with ffmpeg
        ffmpeg_bin = shutil.which("ffmpeg")
        
        if ffmpeg_bin:
            wav_filename = "temp_recording.wav"
            scipy.io.wavfile.write(wav_filename, SAMPLE_RATE, audio_array.squeeze())
            
            try:
                (
                    ffmpeg
                    .input(wav_filename)
                    .output(filename.replace('.wav', '.mp3'), format='mp3', audio_bitrate='192k')
                    .run(cmd=ffmpeg_bin, overwrite_output=True, quiet=True)
                )
                os.remove(wav_filename)
                return filename.replace('.wav', '.mp3')
            except:
                pass
        
        # Fallback to WAV
        scipy.io.wavfile.write(filename, SAMPLE_RATE, audio_array.squeeze())
        return filename
        
    except Exception as e:
        print(f"Error saving: {e}")
        return None

# 🧠 Legal Advisor LLM Response with conditional RAG
def get_legal_advice_with_rag(client_query, legal_area, jurisdiction, collection, num_results=3, pdf_context=None):
    """Generate legal advice, using RAG only for Criminal Law, and PDF context if available"""
    
    sources = []
    context = ""
    
    # Add PDF context if available
    if pdf_context:
        context += f"UPLOADED LEGAL DOCUMENT CONTENT:\n\n{pdf_context[:4000]}\n\n"
        context += "=" * 50 + "\n\n"
    
    # Only use knowledge base for Criminal Law
    if legal_area == "Criminal Law" and collection is not None:
        # Retrieve relevant documents using text search
        retrieved_docs = search_legal_documents(collection, client_query, legal_area, k=num_results)
        
        # Build context from retrieved documents
        if retrieved_docs:
            context += "RELEVANT CRIMINAL LAW INFORMATION FROM KNOWLEDGE BASE:\n\n"
            for i, doc in enumerate(retrieved_docs, 1):
                # Handle different possible field names
                doc_content = doc.get('text', doc.get('content', doc.get('chunk', '')))
                doc_title = doc.get('title', doc.get('source', f'Document {i}'))
                doc_source = doc.get('source', doc.get('metadata', {}).get('source', 'Legal Database'))
                score = doc.get('score', 0)
                
                context += f"[Source {i}: {doc_title}]\n{doc_content}\n\n"
                sources.append({
                    "title": doc_title,
                    "source": doc_source,
                    "score": score,
                    "content": doc_content[:200] + "..."
                })
        else:
            context += "No specific documents found in criminal law knowledge base. Providing general criminal law guidance.\n\n"
    else:
        # For non-Criminal Law areas, use general knowledge
        if not pdf_context:
            context += f"Providing general legal guidance for {legal_area}.\n\n"
    
    # Special instruction for Criminal Law with knowledge base
    criminal_law_restriction = ""
    if legal_area == "Criminal Law" and collection is not None and not pdf_context:
        criminal_law_restriction = """
CRITICAL RESTRICTION FOR CRIMINAL LAW:
- You are STRICTLY a criminal law specialist using ONLY the criminal law knowledge base
- You MUST ONLY answer questions related to CRIMINAL LAW (crimes, criminal procedure, criminal defense, arrests, charges, criminal penalties, criminal rights, etc.)
- If the question is about ANY other area of law (divorce, contracts, real estate, employment, etc.), you MUST respond with:
  "I'm a criminal law specialist and can only answer questions about criminal law using our knowledge base. Your question is about [area of law], which is outside my scope. Please select the appropriate legal area from the dropdown menu or consult an attorney specializing in [area of law]."
- DO NOT provide general knowledge about non-criminal law topics
- DO NOT answer questions outside of criminal law even if you know the answer
- If the question IS about criminal law but not covered in the knowledge base sources, state: "I don't have specific information about this criminal law matter in my knowledge base. Please consult a criminal defense attorney."
- ONLY use information explicitly provided in the [Source] sections above for criminal law questions
"""
    
    system_prompt = f"""You are an experienced legal advisor AI assistant specializing in {legal_area}.

{context}

{criminal_law_restriction}

IMPORTANT GUIDELINES:
1. {f"The user has uploaded a legal document. Use this document to answer their specific questions about it." if pdf_context else ""}
2. {"STRICT: You are a CRIMINAL LAW ONLY specialist. Refuse to answer non-criminal law questions. Only use the criminal law knowledge base sources provided." if legal_area == "Criminal Law" and collection is not None and not pdf_context else "Provide general legal information based on your training"}
3. Provide clear, accurate general legal information
4. Use plain language that non-lawyers can understand
5. Always include relevant disclaimers when appropriate
6. {f'Reference the sources when you use information from them (e.g., "According to [Source 1]..." or "Based on the uploaded document...")' if sources or pdf_context else "Base your response on general legal principles"}
7. Suggest when the client should consult a licensed attorney
8. Be empathetic and professional
9. Keep responses concise (200-300 words unless document analysis requires more detail)
10. If the jurisdiction matters, consider: {jurisdiction}

CRITICAL: You provide general legal information only. You are NOT providing legal advice or creating an attorney-client relationship.

Client's question: {client_query}

Provide a helpful, informative response{"based STRICTLY on criminal law knowledge base sources - refuse to answer if not about criminal law" if legal_area == "Criminal Law" and collection is not None and not pdf_context else "based on the uploaded document, knowledge base, and general legal principles" if pdf_context and sources else "based on the uploaded document and general legal principles" if pdf_context else "based on the knowledge base and general legal principles" if sources else "based on general legal principles"}."""

    try:
        completion = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": client_query}
            ],
            temperature=0.3,
            max_completion_tokens=1500,  # Increased for document analysis
            top_p=0.9
        )
        
        return completion.choices[0].message.content, sources
        
    except Exception as e:
        print(f"LLM Error: {e}")
        st.error(f"Error generating response: {str(e)[:100]}")
        return None, []

# 🗣️ Speech to Text
def stt(local_audio_path):
    """Transcribe audio to text"""
    try:
        with open(local_audio_path, "rb") as f:
            audio_data = BytesIO(f.read())

        transcription = elevenlabs_client.speech_to_text.convert(
            file=audio_data,
            model_id="scribe_v1",
            language_code="en"
        )
        return transcription.text
    except Exception as e:
        st.error(f"Transcription error: {str(e)[:100]}")
        return None

# 📊 Text to Speech
def tts(text, output_filename="advisor_response.mp3"):
    """Convert text to speech"""
    try:
        audio = elevenlabs_client.text_to_speech.convert(
            text=text,
            voice_id="pNInz6obpgDQGcFmaJgB",
            model_id="eleven_multilingual_v2",
            output_format="mp3_44100_128",
        )

        with open(output_filename, "wb") as f:
            for chunk in audio:
                f.write(chunk)
        return output_filename
        
    except Exception as e:
        print(f"TTS Error: {e}")
        # if "quota" in str(e).lower() or "credit" in str(e).lower():
        #     st.warning("⚠️ Text-to-speech quota exceeded. Response shown as text only.")
        return None

# 🔈 Manual Play Audio
def play_audio_manual(file_path):
    """Manually play audio response"""
    try:
        with open(file_path, "rb") as audio_file:
            audio_bytes = audio_file.read()
        audio_base64 = base64.b64encode(audio_bytes).decode()
        st.markdown(f"""
            <audio controls autoplay>
                <source src="data:audio/mp3;base64,{audio_base64}" type="audio/mp3">
            </audio>
        """, unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Audio playback error: {str(e)}")



# Display conversation history
if st.session_state.conversation_history:
    st.markdown("### 💬 Consultation History")
    for i, msg in enumerate(st.session_state.conversation_history):
        if msg["role"] == "user":
            # User message - chatbox style (right-aligned, blue background)
            st.markdown(f'''
                <div style="text-align: right; margin: 10px 0;">
                    <div style="display: inline-block; background-color: #E3F2FD; padding: 12px 16px; border-radius: 15px; max-width: 70%; text-align: left; box-shadow: 0 1px 2px rgba(0,0,0,0.1);">
                        <strong style="color: #1976D2;">👤 You:</strong><br/>
                        <span style="color: #333;">{msg['text']}</span>
                    </div>
                </div>
            ''', unsafe_allow_html=True)
        else:
            # Assistant message - chatbox style (left-aligned, green background)
            st.markdown(f'''
                <div style="text-align: left; margin: 10px 0;">
                    <div style="display: inline-block; background-color: #E8F5E9; padding: 12px 16px; border-radius: 15px; max-width: 70%; text-align: left; box-shadow: 0 1px 2px rgba(0,0,0,0.1);">
                        <strong style="color: #388E3C;">⚖️ Legal Advisor:</strong><br/>
                        <span style="color: #333;">{msg['text']}</span>
                    </div>
                </div>
            ''', unsafe_allow_html=True)

            # Add play audio button if audio file exists
            if msg.get("audio_file") and os.path.exists(msg["audio_file"]):
                if st.button(f"🔊 Play Audio Response", key=f"play_audio_{i}"):
                    play_audio_manual(msg["audio_file"])

            # Show if response used PDF
            if msg.get("used_pdf"):
                st.caption("📄 Answer based on uploaded document")

            # Show sources if available
            if msg.get("sources"):
                with st.expander(f"📚 Sources ({len(msg['sources'])} documents)"):
                    for j, src in enumerate(msg["sources"], 1):
                        st.markdown(f"**{j}. {src['title']}**")
                        st.caption(f"Score: {src.get('score', 0):.3f}")
                        st.text(src.get('content', 'No preview available'))
                        st.markdown("---")
    st.markdown("---")

# 🎛️ Main UI Controls
# Add input mode selection to session state
if 'input_mode' not in st.session_state:
    st.session_state.input_mode = None  # None means not selected yet

# If no input mode is selected, show the choice
if st.session_state.input_mode is None:
    st.markdown("### How would you like to ask your legal question?")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("✍️ Type", use_container_width=True, type="primary"):
            st.session_state.input_mode = "text"
            st.rerun()
    with col2:
        if st.button("🎤 Voice", use_container_width=True, type="primary"):
            st.session_state.input_mode = "voice"
            st.rerun()

# TEXT INPUT MODE
elif st.session_state.input_mode == "text":

    # Text input area with chat-like styling
    user_question = st.text_area(
        "Your Question:",
        placeholder="Type your legal question here...",
        height=100,
        key="text_question_input"
    )

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        submit_col, back_col = st.columns(2)
        with submit_col:
            if st.button("Send", use_container_width=True, type="primary"):
                if user_question.strip():
                    # Process text question silently
                    # Add to conversation history
                    st.session_state.conversation_history.append({
                        "role": "user", 
                        "text": user_question,
                        "timestamp": datetime.now().isoformat()
                    })

                    # Get legal advice
                    legal_response, sources = get_legal_advice_with_rag(
                        user_question, 
                        st.session_state.legal_context,
                        jurisdiction,
                        st.session_state.mongo_collection,
                        num_results,
                        pdf_context=st.session_state.uploaded_pdf_text
                    )

                    if legal_response:
                        # Generate audio response
                        audio_output = tts(legal_response)

                        # Save response with audio file reference
                        response_data = {
                            "role": "assistant",
                            "text": legal_response,
                            "sources": sources,
                            "timestamp": datetime.now().isoformat(),
                            "used_pdf": bool(st.session_state.uploaded_pdf_text)
                        }

                        if audio_output:
                            response_data["audio_file"] = audio_output

                        st.session_state.conversation_history.append(response_data)

                    st.rerun()
                else:
                    st.warning("Please enter a question before submitting.")

        with back_col:
            if st.button("Back", use_container_width=True):
                st.session_state.input_mode = None
                st.rerun()

# VOICE INPUT MODE
elif st.session_state.input_mode == "voice":
    st.markdown("### Ask Your Legal Question via Voice")

    # Cloud pe mic available nahi, isliye sirf info dikhayenge
    if not AUDIO_AVAILABLE:
        st.info(
            "🎙️ Voice recording is only available in the local version of this app. "
            "On Streamlit Cloud, please type your legal question instead."
        )
    else:
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if not st.session_state.recording:
                record_col, back_col = st.columns(2)
                with record_col:
                    if st.button("🎤 Start Recording", use_container_width=True, type="primary"):
                        st.session_state.recording = True
                        st.session_state.stop_event.clear()
                        st.session_state.audio_queue = queue.Queue()
                        st.session_state.recording_thread = threading.Thread(
                            target=record_audio_continuous,
                            args=(st.session_state.stop_event, st.session_state.audio_queue)
                        )
                        st.session_state.recording_thread.start()
                        st.rerun()
                with back_col:
                    if st.button("🔙 Back", use_container_width=True):
                        st.session_state.input_mode = None
                        st.rerun()
            else:
                if st.button("⏹️ Stop & Submit Question", use_container_width=True, type="secondary"):
                    st.session_state.recording = False
                    st.session_state.stop_event.set()

                    if st.session_state.recording_thread:
                        st.session_state.recording_thread.join(timeout=3.0)

                    time.sleep(0.3)

                    audio_file = save_recording(st.session_state.audio_queue, "client_question.wav")
                    if audio_file:
                        client_text = stt(audio_file)
                        if client_text:
                            st.session_state.conversation_history.append({
                                "role": "user",
                                "text": client_text,
                                "timestamp": datetime.now().isoformat()
                            })

                            legal_response, sources = get_legal_advice_with_rag(
                                client_text,
                                st.session_state.legal_context,
                                st.session_state.using_knowledge_base,
                                st.session_state.uploaded_pdf_text,
                            )

                            if legal_response:
                                audio_output = None
                                if st.session_state.use_tts:
                                    audio_output = tts(legal_response)

                                response_data = {
                                    "role": "assistant",
                                    "text": legal_response,
                                    "sources": sources,
                                    "timestamp": datetime.now().isoformat(),
                                    "used_pdf": bool(st.session_state.uploaded_pdf_text)
                                }
                                if audio_output:
                                    response_data["audio_file"] = audio_output

                                st.session_state.conversation_history.append(response_data)
                        else:
                            st.error("Failed to process recording. Please try again.")

                    st.rerun()

# 🔴 Recording Status
if AUDIO_AVAILABLE and st.session_state.recording:
    st.markdown("""
        <div style='text-align: center; padding: 20px; background-color: #FEE2E2; border-radius: 10px;'>
            <h3 style='color: #DC2626;'>🔴 Recording in progress...</h3>
            <p style='color: #991B1B;'>Speak clearly and describe your legal question. Click 'Stop & Submit' when done.</p>
        </div>
    """, unsafe_allow_html=True)

# ℹ️ Instructions
# with st.expander("ℹ️ How to Use This Legal Advisor"):
#     st.markdown("""
#     ### Getting Started
#     1. **Ensure MongoDB connection** - Check the sidebar for connection status
#     2. **Select your legal area** from the sidebar (e.g., Contract Law, Family Law)
#     3. **Optional:** Upload a PDF document to ask specific questions about it
#     4. **Optional:** Choose your jurisdiction for location-specific guidance
#     5. Click **"Ask Your Legal Question"** and speak clearly
#     6. Click **"Stop & Submit Question"** when you're done
#     7. The AI will search the knowledge base and/or analyze your PDF
#     8. **Click "🔊 Play Audio Response"** to hear the answer (saves ElevenLabs credits)
    
#     ### Features
#     - **PDF Document Analysis**: Upload contracts, agreements, or legal documents
#     - **Criminal Law Knowledge Base**: Specialized information for criminal law queries
#     - **Manual Audio Playback**: Play audio responses only when needed
#     - **Source Citations**: See which documents informed the response
#     - **Voice Interface**: Natural conversation with speech-to-text
    
#     ### PDF Upload Use Cases
#     - Review contracts and agreements
#     - Analyze legal documents
#     - Ask questions about specific clauses
#     - Get explanations of legal terminology in your document
#     - Compare your document against general legal principles
    
#     ### Best Practices
#     - Speak clearly and provide relevant context
#     - Be specific about your situation
#     - Upload PDFs for document-specific questions
#     - Review the cited sources for more details
#     - Ask follow-up questions if you need clarification
#     - Download the transcript for your records
    
#     ### Important Notes
#     - This tool provides **general legal information only**
#     - It does **NOT** create an attorney-client relationship
#     - For specific legal advice, consult a licensed attorney
#     - PDF content is processed in-memory and not stored permanently
#     """)

#     # Show sample document structure if connected
#     if st.session_state.mongo_collection is not None:
#         if st.button("Show Sample Document"):
#             try:
#                 sample = st.session_state.mongo_collection.find_one()
#                 if sample:
#                     # Remove _id and embedding for cleaner display
#                     sample_display = {k: v for k, v in sample.items() if k not in ['_id', 'embedding']}
#                     st.json(sample_display)
#                 else:
#                     st.warning("No documents found in collection")
#             except Exception as e:
#                 st.error(f"Error: {str(e)}")

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: #6B7280; padding: 20px;'>
        <p><strong>⚠️ Disclaimer:</strong> This AI provides general legal information, not legal advice.</p>
        <p>Powered by AI • MongoDB Vector Search • ElevenLabs • Groq</p>
        <p style='font-size: 12px; margin-top: 10px;'>
            Always consult with a qualified attorney for your specific legal situation.
        </p>
    </div>
""", unsafe_allow_html=True)
