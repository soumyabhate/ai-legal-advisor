# ⚖️ AI Legal Advisor [Chatbot & Voice Assistant]

**Course** — DATA 622 Natural Language Processing
**Creators:** Arshdeep Singh | Soumya Bhate | Yatharth Kumar
**Instructor:** Prof. Tony Diana

AI Legal Advisor is an interactive legal information assistant built with Streamlit. It helps users **ask legal questions via text or voice**, upload **legal PDFs**, and receive **plain‑language explanations** with optional **audio responses**, while enforcing strong **privacy and disclaimer safeguards**.

> ⚠️ This tool provides **general legal information only** and **does not** provide legal advice or create an attorney–client relationship.

---

## ✨ Key Features

- **Multi‑modal input**
  - ✍️ **Text mode** – type your legal question.
  - 🎤 **Voice mode (local only)** – record a question using your microphone, transcribed via ElevenLabs STT.
- **Legal area & jurisdiction selection**
  - Choose from areas like Contract, Family, Criminal, Employment, IP, etc.
  - Optional jurisdiction selector (US, India, EU, etc.) to guide contextual answers.
- **Criminal law knowledge base (MongoDB)**
  - For **Criminal Law**, the bot uses a curated MongoDB knowledge base (vector/text index) to retrieve relevant chunks and build context for RAG‑style answers.
- **PDF‑aware Q&A**
  - Upload a **legal PDF** (contracts, notices, agreements).
  - The app extracts text, **redacts sensitive PII**, and lets you:
    - Ask document‑specific questions.
    - Generate a **summary**, **topic modeling (LDA)**, and **word cloud** over the document.
- **PII redaction**
  - Regex‑based removal of SSNs, credit cards, emails, phone numbers, DOB, licenses, passports, bank accounts, IPs, and MRNs before processing.
- **LLM‑powered responses (Groq)**
  - Uses **Groq’s Llama 3.3 70B** model for structured, instruction‑following answers with special rules for Criminal Law scope.
- **Text‑to‑Speech responses (ElevenLabs)**
  - Answers can be converted to speech and played back as MP3 inside the app.
- **Consultation transcript export**
  - Download a **full text transcript** of the conversation (with sources and metadata) as a `.txt` file.

---

## 🧱 Tech Stack

- **UI & orchestration:** Streamlit  
- **LLM backend:** Groq `llama-3.3-70b-versatile`  
- **Speech:**
  - ElevenLabs **Text‑to‑Speech** (TTS)
  - ElevenLabs **Speech‑to‑Text** (STT / `scribe_v1`)
- **Database:** MongoDB (criminal law knowledge base, text search)  
- **NLP / analysis:**
  - `wordcloud`, `scikit-learn` stopwords, `gensim` LDA topic modeling
  - `PyPDF2` for PDF text extraction
- **Audio (local):** `sounddevice`, `scipy`, `ffmpeg-python` for recording & MP3 conversion  
- **Config:** `python-dotenv` for local env variables  

---

## 📂 Project Structure

```text
.
├─ legal.py           # Main Streamlit app (UI + logic)
├─ requirements.txt   # Python dependencies
└─ .env               # Local-only secrets (NOT committed to Git)
```

- `legal.py` contains:
  - UI layout (sidebar config, main chat area, PDF analysis tools).
  - Audio recording pipeline for local voice input.
  - PII redaction, MongoDB search, LLM prompt construction, and response rendering.
- `.env` is used only for local development to store API keys (**never** push this file).  

---

## 🔑 Configuration – Environment Variables

Create a `.env` file in the project root (same folder as `legal.py`) and define:

```bash
GROQ_API_KEY=your_groq_api_key_here
ELEVENLABS_API_KEY=your_elevenlabs_api_key_here
MONGO_URI=your_mongodb_connection_string # preferred to keep mongo_uri in the main file for easy access
ELEVEN_VOICE_ID=pNInz6obpgDQGcFmaJgB  # or your preferred ElevenLabs voice ID (can keep this in the main file as well, as there is no risk of exposure)
```

> Do **not** commit `.env` to Git. Add it to `.gitignore` if you haven’t already.

The app loads these values via `python-dotenv` and constructs Groq, ElevenLabs, and MongoDB clients at startup.

---

## 💻 Local Setup & Run

### 1. Clone the repo & create a virtual environment

```bash
git clone <your-repo-url>.git
cd <your-repo-folder>

# Create & activate a venv (example for Windows / bash)
python -m venv .venv
source .venv/bin/activate   # On Windows: .venv\Scripts\activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

(Dependencies are defined in `requirements.txt` and include Streamlit, Groq, ElevenLabs, MongoDB client, NLP libraries, and audio tools.)

> For **audio recording** on some systems, you may need system‑level PortAudio / ffmpeg installed.

### 3. Set up `.env`

Create `.env` as shown above and paste your actual keys/URIs.

### 4. Run the app locally

```bash
streamlit run legal.py
```

- Open the local URL (usually `http://localhost:8501`).
- Voice input & audio playback work fully **on local runs**, as the app can access your microphone and speakers.

---

## 🌐 Deploying on Streamlit Community Cloud

1. Push `legal.py` and `requirements.txt` to GitHub (do **not** push `.env`).  
2. Go to Streamlit Community Cloud and deploy a new app:
   - Select your GitHub repo & branch.
   - Set **main file** to `legal.py`.
3. In the app’s **Settings → Secrets**, add:

```toml
GROQ_API_KEY = "your_groq_api_key_here"
ELEVENLABS_API_KEY = "your_elevenlabs_api_key_here"
```

4. Deploy – Streamlit will install `requirements.txt`, start the app, and give you a public URL like:

```text
https://your-legal-advisor-app.streamlit.app
```

> Note: On cloud, **server-side microphone access is not available**, so the app shows a message that **voice recording is only supported in the local version**. Text chat + PDF analysis + TTS playback remain available.

---

## 🧭 How to Use the App

1. **Select Legal Area & Jurisdiction** in the sidebar.  
2. (Optional) **Upload a PDF** and click **Process PDF** to:
   - Redact sensitive info.
   - Enable summary, topic modeling, and word cloud over the document.
3. Choose **Type** or **Voice** input mode on the main screen.
4. Ask your question:
   - Text mode → type and click **Send**.
   - Voice mode (local) → **Start Recording**, speak, then **Stop & Submit**.
5. Review the **AI Legal Advisor** response, optional **audio playback**, and **sources** if provided.
6. Use **Download Consultation** to export the full transcript (for your records or to share with a real lawyer).

---

## ⚠️ Legal & Ethical Disclaimer

- This system is an **academic / project‑grade AI assistant**.
- It provides **general legal information only**, based on:
  - Public legal principles,
  - The uploaded document (where applicable),
  - And a limited MongoDB knowledge base for criminal law.
- It **does not**:
  - Provide legal advice,
  - Replace a licensed attorney,
  - Or create an attorney–client relationship.

Always consult a qualified lawyer before making decisions about your legal situation.

---

## 🙌 Acknowledgements

- Built as an **NLP project** combining:
  - Streamlit UI,
  - MongoDB knowledge base,
  - Groq LLMs,
  - ElevenLabs speech,
  - And classic NLP tools (topic modeling, word clouds).
- Designed to be **demo‑friendly**, with both local (full) and cloud (text‑first) modes.
- Grateful to Prof. Tony Diana for his constant guidance throughout this project and for equipping us with the skills to achieve this milestone.
