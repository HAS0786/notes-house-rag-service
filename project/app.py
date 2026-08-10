from csv import reader
import os
import pickle
from pydoc import text
import numpy as np
from flask import Flask, request, jsonify
import faiss
from sentence_transformers import SentenceTransformer
from openai import OpenAI

from config import *

# File readers
from pypdf import PdfReader
from docx import Document
from pptx import Presentation
from PIL import Image
import pytesseract
import cv2
import json

import subprocess

from dotenv import load_dotenv
import os

load_dotenv()


def convert_to_pdf(input_path):
    output_dir = os.path.dirname(input_path)
    # libreoffice_path = r"C:\Program Files\LibreOffice\program\soffice.exe"
    libreoffice_path = "soffice"
    result = subprocess.call([
        libreoffice_path,
        '--headless',
        '--convert-to', 'pdf',
        input_path,
        '--outdir', output_dir
    ])

    base = os.path.splitext(input_path)[0]
    pdf_path = base + ".pdf"

    print("RESULT:", result)
    print("PDF PATH:", pdf_path)

    # 🔥 CRITICAL CHECK
    if result != 0 or not os.path.exists(pdf_path):
        raise Exception("PDF conversion failed")

    return pdf_path

# INIT

app = Flask(__name__)

client = OpenAI(
    api_key=OPENAI_API_KEY,
    base_url="https://openrouter.ai/api/v1"
)

pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH

embed_model = SentenceTransformer("all-MiniLM-L6-v2")
DIM = 384


# LOAD / CREATE FAISS

if os.path.exists(FAISS_PATH) and os.path.getsize(FAISS_PATH) > 0:
    try:
        index = faiss.read_index(FAISS_PATH)
        with open(DOCS_PATH, "rb") as f:
            documents = pickle.load(f)
    except:
        index = faiss.IndexFlatL2(DIM)
        documents = []
else:
    index = faiss.IndexFlatL2(DIM)
    documents = []


# SAVE
def save_state():
    faiss.write_index(index, FAISS_PATH)
    with open(DOCS_PATH, "wb") as f:
        pickle.dump(documents, f)


# FILE → TEXT
def extract_text(path):
    ext = os.path.splitext(path)[1].lower()
    

    if ext == ".pdf":
        try:
             reader = PdfReader(path)
        except Exception as e:
            print("PDF READ ERROR:", str(e))
            return ""
        
        text = ""

        for page in reader.pages:
            text += page.extract_text() or ""

        # CLEAN AFTER FULL EXTRACTION
        text = "\n".join([
            line for line in text.split("\n")
            if len(line.strip()) > 10
        ])

        # IF TEXT EMPTY → USE OCR
        if text.strip():
            return text

        print("PDF has no text, using OCR...")

        try:
            import pdf2image

            images = pdf2image.convert_from_path(path)

            ocr_text = ""

            for img in images:
                text = pytesseract.image_to_string(img)
                ocr_text += text + "\n"

            print("OCR TEXT SAMPLE:", ocr_text[:200])

            return ocr_text

        except Exception as e:
            print("OCR ERROR:", str(e))
            return ""

    
    elif ext == ".docx":
        doc = Document(path)
        return "\n".join([p.text for p in doc.paragraphs])

    elif ext == ".pptx":
        prs = Presentation(path)
        text = ""
        for slide in prs.slides:
            for shape in slide.shapes:
                if hasattr(shape, "text"):
                    text += shape.text + "\n"
        return text

    elif ext == ".txt":
        return open(path, "r", encoding="utf-8").read()

    elif ext in [".png", ".jpg", ".jpeg"]:
        try:
            # result = ocr.ocr(path)
            img = cv2.imread(path)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            cv2.imwrite("temp_ocr.jpg", gray)

            text = pytesseract.image_to_string(gray)

            print("OCR OUTPUT:", text[:200])  #  DEBUG

            return text

        except Exception as e:
            print("OCR ERROR:", str(e))
            return ""

    return ""


# CHUNKING

def chunk_text(text):
    import re

    # 🔹 Clean text (VERY IMPORTANT)
    text = text.replace("\n", " ")

    # 🔹 Split by sentences instead of words
    paragraphs = [p.strip() for p in text.split("\n") if len(p.strip()) > 20]
    MAX_CHUNK_SIZE = 1200  
    OVERLAP = 200 
    chunks = []
    current_chunk = ""

    for sentence in paragraphs:
        if len(current_chunk) + len(sentence) < MAX_CHUNK_SIZE:
            current_chunk += " " + sentence
        else:
            chunks.append(current_chunk.strip())
            current_chunk = current_chunk[-OVERLAP:] + "\n" + sentence

    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks
# STORE
def add_chunks(chunks, source):
    clean_chunks = []

    for chunk in chunks:

        #  skip noisy chunks
        if len(chunk.split()) < 15:
            continue

        if chunk.count(" ") < 15:
            continue

        if any(x in chunk for x in ["{", "}", ";", "//"]):
            continue

        clean_chunks.append(chunk)

    if not clean_chunks:
        return

    #  ONLY CLEAN DATA goes to FAISS
    emb = embed_model.encode(clean_chunks, batch_size=8)
    index.add(np.array(emb))

    for chunk in clean_chunks:
        documents.append({
            "text": chunk,
            "source": source
        })

    save_state()

# Keyword search 
def keyword_search(query, top_k=10,session_id=None):
    results = []
    query_words = query.lower().split()

    for doc in documents:
        text = doc["text"].lower()

        #  Strong scoring (names / keywords boost)
        score = sum(8 for word in query_words if word in text)

        #  Extra boost for full phrase match
        if session_id and session_id in doc["source"]:
            score += 10
        if query.lower() in text:
            score += 10

        if score > 0:
            results.append((score, doc["text"]))

    results.sort(reverse=True, key=lambda x: x[0])
    return [r[1] for r in results[:top_k]]
# SEARCH

def search(query, k=TOP_K, source=None, session_id=None):
    if len(documents) == 0:
        return []

    query_lower = query.lower()
    query_words = query_lower.split()

    #  detect factual / short queries (names, roles, etc.)
    is_fact_query = any(w in query_lower for w in [
        "who", "name", "supervisor", "represented",
    "submitted", "author", "developer", "team", "group"
    ]) or len(query_words) <= 5

    # 🔹 1. Semantic search
    q_emb = embed_model.encode([query])
    D, I = index.search(np.array(q_emb), k * 10)

    semantic_results = []

    for idx, dist in zip(I[0], D[0]):

        #  filter weak matches
        if dist > 2.5:
            continue

        doc = documents[idx]

#  SESSION PRIORITY (ADD THIS BLOCK)
        if session_id and session_id in doc["source"]:
            semantic_results.insert(0, doc["text"])
            continue

        # existing filter
        if source and source not in doc["source"]:
            continue

        semantic_results.append(doc["text"])
    # 🔹 2. Keyword search
    keyword_results = keyword_search(query, top_k=10, session_id=session_id)

    # 🔹 3. Exact match boost
    exact_matches = []

    for doc in documents:
        text = doc["text"].lower()

        match_count = sum(1 for word in query_words if word in text)
        if session_id and session_id in doc["source"]:
            match_count += 2

        if match_count > 0:
            for _ in range(match_count * 3):
                exact_matches.append(doc["text"])

    # 🔹 4. Strong keyword candidates (generic fix)
    keyword_strong = []

    for doc in documents:
        text = doc["text"].lower()

        match_count = sum(1 for w in query_words if w in text)

        if match_count >= 2:
            if session_id and session_id in doc["source"]:
                keyword_strong.insert(0, doc["text"])  #  priority
            else:
                keyword_strong.append(doc["text"])

    # 🔹 5. FILTER semantic (important for factual queries)
    if is_fact_query:
        filtered_semantic = []
        for chunk in semantic_results:
            text = chunk.lower()
            if sum(1 for w in query_words if w in text) >= 1:
                filtered_semantic.append(chunk)
    else:
        filtered_semantic = semantic_results
    #  FORCE INCLUDE (SESSION KEYWORD MATCH)
    for doc in documents:
        if session_id and session_id in doc["source"]:
            text = doc["text"].lower()

            if any(w in text for w in query_words):
                filtered_semantic.insert(0, doc["text"])

    # 🔹 6. Merge all
    combined = exact_matches + keyword_strong + filtered_semantic + keyword_results
    combined = list(dict.fromkeys(combined))
    # 🔹 7. Rerank
    scored = []

   

    for chunk in combined:
        text = chunk.lower()

        score = 0

        #  SESSION BOOST (CORRECT PLACE)
        for doc in documents:
            if doc["text"] == chunk and session_id and session_id in doc["source"]:
                score += 20
                break


        # keyword strength
        match_count = sum(1 for word in query_words if word in text)
        score += match_count * 8

        # exact phrase boost
        if query_lower in text:
            score += 40

        #  name boost
        if any(word[0].isupper() for word in query.split()):
            score += 10

        # length penalty
        score -= len(text) * 0.0005

        scored.append((score, chunk))

    scored.sort(reverse=True, key=lambda x: x[0])

    # 🔹 8. Remove duplicates + limit
    #  GROUP BY FILE (NEW LOGIC)
    file_map = {}

    for score, chunk in scored:
        for doc in documents:
            if doc["text"] == chunk:
                src = doc["source"]

                if src not in file_map:
                    file_map[src] = {
                        "score": 0,
                        "chunks": []
                    }

                file_map[src]["score"] += score
                file_map[src]["chunks"].append(chunk)
                break

    #  SORT FILES BY SCORE
    sorted_files = sorted(
        file_map.items(),
        key=lambda x: x[1]["score"],
        reverse=True
    )

    #  BUILD FINAL RESULTS
    final = []

    for src, data in sorted_files:
        for chunk in data["chunks"][:5]:  # max 5 chunks per file
            final.append({
                "text": chunk,
                "source": src
            })

        if len(final) >= k:
            break

    # 🔥 DEBUG
    print("\n🔥 FINAL TOP CHUNKS:")
    for i, item in enumerate(final[:5]):
        print(f"\n{i+1}:", item["text"][:200])

    return final
# OPENAI CALL

def call_openai(prompt, max_tokens=700, json_mode=False):
    params = {
        "model": "openai/gpt-4o-mini",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.8
    }
    
    if json_mode:
        params["response_format"] = {"type": "json_object"}

    res = client.chat.completions.create(**params)

    return res.choices[0].message.content


# PROMPTS

def build_answer_prompt(context, question):
    return f"""
You are a helpful AI assistant.

Answer the question using ONLY the given context.

RULES:
- Give a COMPLETE answer (not short or not too Long)
- Explain clearly in simple language
- If multiple points exist → use bullet points
- Combine information from multiple chunks if needed
- Do NOT say "not found"
- Do NOT guess
- Do NOT use external knowledge if any relevant data find from context
- Answer should feel like a short explanation, not just a line

Context:
{context}

Question:
{question}

Answer:
"""


# def build_quiz_prompt(context, previous_quiz=None):
#     return f"""
# You are a strict JSON generator.

# TASK:
# Generate EXACTLY 5 multiple choice questions (MCQs).

# CRITICAL RULES (DO NOT BREAK):
# - Output ONLY valid JSON
# - DO NOT write any text before or after JSON
# - JSON must start with {{ and end with }}
# - Always return complete JSON
# - If unsure, still return valid JSON with empty quiz []

# FORMAT (STRICT):
# {{
#   "quiz": [
#     {{
#       "question": "Question text",
#       "options": ["Option 1", "Option 2", "Option 3", "Option 4"],
#       "answer": 0,
#       "topic": "General",
#       "explanation": "Explanation of correct answer"
#     }}
#   ]
# }}

# RULES:
# - Exactly 5 questions
# - Each question must be unique
# - options length MUST be 4
# - answer must be index (0,1,2,3)
# - DO NOT include numbering inside options
# - explanation must be clear
# - Regenerate from context if it comes again

# Context:
# {context[:1200]}
# """

def build_quiz_prompt(context, previous_quiz=None):
    previous_quiz = previous_quiz or []

    previous_questions = "\n".join(
        f"- {q.get('question', '')}"
        for q in previous_quiz
        if isinstance(q, dict) and q.get("question")
    )

    regeneration_rules = ""

    if previous_questions:
        regeneration_rules = f"""
IMPORTANT REGENERATION RULES:
- This is a regeneration request.
- Generate completely NEW questions.
- DO NOT repeat any previous question.
- DO NOT simply rephrase a previous question.
- DO NOT ask the same question with different options.
- Try to test different concepts from the provided context.

PREVIOUS QUESTIONS:
{previous_questions}
"""

    return f"""
You are a strict JSON generator.

TASK:
Generate EXACTLY 5 multiple choice questions (MCQs).

CRITICAL RULES:
- Output ONLY valid JSON
- DO NOT write any text before or after JSON
- JSON must start with {{ and end with }}
- Always return complete JSON

RULES:
- Exactly 5 questions
- Each question must be unique
- Each question must have exactly 4 options
- answer must be an index: 0, 1, 2, or 3
- DO NOT include numbering inside options
- explanation must be clear
- Questions must be based ONLY on the provided context
- Cover different concepts from the context where possible

{regeneration_rules}

FORMAT:
{{
  "quiz": [
    {{
      "question": "Question text",
      "options": [
        "Option 1",
        "Option 2",
        "Option 3",
        "Option 4"
      ],
      "answer": 0,
      "topic": "General",
      "explanation": "Explanation of correct answer"
    }}
  ]
}}

Context:
{context[:1200]}
"""
# UPLOAD

indexed_hashes = set()

@app.route("/upload", methods=["POST"])
def upload():
    path = None
    try:
        import hashlib

        file = request.files["file"]

        file_bytes = file.read()
        file_hash = hashlib.sha256(file_bytes).hexdigest()

        file.seek(0)

        if file_hash in indexed_hashes:
            return jsonify({"status": "already indexed"})

        indexed_hashes.add(file_hash)

        path = os.path.join(UPLOAD_DIR, file.filename)
        file.save(path)

        text = extract_text(path)
        print("TEXT LENGTH:", len(text))  
        print("TEXT SAMPLE:", text[:200]) 
        if not text.strip():
            return jsonify({"error": "No text extracted"})

        chunks = chunk_text(text)
        print("CHUNKS COUNT:", len(chunks)) 
        session_id = request.form.get("session_id", "global")

        source = f"{session_id}/{file.filename}"

        add_chunks(chunks, source=source)

        return jsonify({"status": "indexed", "chunks": len(chunks)})

    except Exception as e:
        print("UPLOAD ERROR:", str(e))
        return jsonify({"error": str(e)}), 500

    finally:
        if path and os.path.exists(path):
            os.remove(path)  
        
@app.route("/convert", methods=["POST"])
def convert_file():
    file = request.files.get("file")

    if not file:
        return jsonify({"error": "No file"}), 400

    input_path = os.path.join(UPLOAD_DIR, file.filename)
    file.save(input_path)

    ext = os.path.splitext(input_path)[1].lower()

    try:
        if ext == ".pdf":
    # still check readability later via extract_text
            result = os.path.basename(input_path)
        else:
            pdf_path = convert_to_pdf(input_path)
            result = os.path.basename(pdf_path)

        return jsonify({"pdfPath": result})

    except Exception as e:
        print("CONVERSION ERROR:", str(e))
        return jsonify({"error": str(e)}), 500

    finally:
        # 🔥 ADD THIS (CRITICAL)
        if os.path.exists(input_path):
            os.remove(input_path)
# ASK (RAG)

@app.route("/ask-rag", methods=["POST"])
def ask_rag():
    try:
        data = request.get_json()

        question = data.get("question")
        source = data.get("source")
        session_id = data.get("session_id")

        if not question:
            return jsonify({"error": "Question required"}), 400

        #  1. Retrieve chunks (session-aware search)
        context_chunks = search(
            question,
            source=source,
            session_id=session_id
        )

        print("\n QUESTION:", question)

        print("\n========== RETRIEVED CHUNKS ==========")
        for i, item in enumerate(context_chunks[:5]):
            print(f"\nCHUNK {i+1}:\n", item["text"][:300])
        print("======================================")

        #  2. Split into session + global
        session_chunks = []
        global_chunks = []

        for item in context_chunks:
            text = item["text"]
            source = item["source"]

            if session_id and session_id in source:
                session_chunks.append(text)
            else:
                global_chunks.append(text)

        #  3. Build smart context
        context = ""

        if session_chunks:
            # session priority (main file user is working on)
            # GROUP BY SOURCE (FILE)
            file_context = {}

#  PRIORITIZE SESSION FILE
            for item in context_chunks:
                src = item["source"]
                text = item["text"]

                #  ONLY include session file first
                if session_id and session_id in src:
                    if src not in file_context:
                        file_context[src] = []
                    file_context[src].append(text)

            #  fallback (agar session empty ho)
            if not file_context:
                for item in context_chunks:
                    src = item["source"]
                    text = item["text"]

                    if src not in file_context:
                        file_context[src] = []

                    file_context[src].append(text)
            #  PICK BEST FILE
            best_file = max(file_context.items(), key=lambda x: len(x[1]))[0]

            #  BUILD CONTEXT FROM SAME FILE
            context = "\n\n".join(file_context[best_file][:12])
        else:
            #  fallback to global chunks
            context = "\n\n".join(global_chunks[:6])

        #  4. Weak context detection
        is_weak_context = (
            not context_chunks
            or len(context_chunks) < 2
            or len(context_chunks[0]["text"].split()) < 25
        )

        #  5. GLOBAL FALLBACK (important)
        if is_weak_context and not session_chunks:
            print("Weak context → controlled fallback")

            prompt = f"""
        Answer the question.

        RULES:
        - If you are sure → answer
        - If unsure → say "This is not available in uploaded data"
        - Keep answer short

        Question:
        {question}
        """

            answer = call_openai(prompt, max_tokens=300)

            return jsonify({
                "answer": answer,
                "mode": "fallback/Global"
            })

        # 🔹 6. RAG Answer
        prompt = build_answer_prompt(context, question)
        answer = call_openai(prompt, max_tokens=700)

        return jsonify({
            "answer": answer,
            "mode": "rag"
        })

    except Exception as e:
        print("ASK_RAG ERROR:", str(e))
        return jsonify({"error": str(e)}), 500
#  COMBINED ENDPOINT
@app.route("/ask-ai", methods=["POST"])
def ask_ai():
    try:
        data = request.get_json()
        question = data.get("question")

        if not question:
            return jsonify({"error": "Question required"}), 400

        prompt = f"Answer clearly:\n{question}"
        answer = call_openai(prompt, max_tokens=700)

        return jsonify({"answer": answer})

    except Exception as e:
        print("ASK_AI ERROR:", str(e))
        return jsonify({"error": str(e)}), 500


@app.route("/ask-pdf", methods=["POST"])
def ask_pdf():
    file = request.files.get("file")
    question = request.form.get("question")

    if not file or not question:
        return jsonify({"error": "Missing file or question"}), 400

    temp_path = os.path.join(UPLOAD_DIR, file.filename)
    file.save(temp_path)

    text = extract_text(temp_path)
    if not text.strip():
        return jsonify({"error": "No text extracted"})

    chunks = chunk_text(text)

    # SESSION MEMORY (MAIN FIX)
    session_id = request.form.get("session_id", "temp_session")
    source = f"{session_id}/{file.filename}"

    add_chunks(chunks, source=source)

    # USE GLOBAL SEARCH (NOT TEMP INDEX)
    results = search(question, session_id=session_id)

    relevant_chunks = [r["text"] for r in results[:5]]

    context = "\n".join(relevant_chunks)

    prompt = build_answer_prompt(context, question)
    answer = call_openai(prompt, max_tokens=700)

    os.remove(temp_path)

    return jsonify({"answer": answer})

@app.route("/ask-file", methods=["POST"])
def ask_file():
    file = request.files["file"]

    temp_path = os.path.join(UPLOAD_DIR, file.filename)
    file.save(temp_path)

    # IMPORTANT: USE UNIVERSAL EXTRACT
    text = extract_text(temp_path)

    if not text.strip():
        return jsonify({"answer": "No readable content found"})

    chunks = chunk_text(text)

#  create embeddings
    embeddings = embed_model.encode(chunks)

    temp_index = faiss.IndexFlatL2(embeddings.shape[1])
    temp_index.add(np.array(embeddings))

    question = request.form.get("question")

    q_embedding = embed_model.encode([question])

    D, I = temp_index.search(np.array(q_embedding), k=3)

    relevant_chunks = [chunks[i] for i in I[0]]

    context = "\n".join(relevant_chunks)

    prompt = build_answer_prompt(context, question)
    answer = call_openai(prompt, max_tokens=700)
    os.remove(temp_path) 
    return jsonify({"answer": answer})
# QUIZ


@app.route("/quiz", methods=["POST"])
def quiz():
    topic = request.json["topic"]

    context_chunks = search(topic)
    context = "\n".join(context_chunks)[:600]

    prompt = build_quiz_prompt(context)
  

    for _ in range(2):  # retry twice
        raw = call_openai(prompt, max_tokens=800, json_mode=True)

        try:
            start = raw.find("{")
            end = raw.rfind("}") + 1

            if start == -1 or end == -1:
                continue

            clean = raw[start:end]
            data = json.loads(clean)

            return jsonify(data)

        except:
            continue

    return jsonify({
        "error": "JSON parse failed",
        "raw": raw
    })
    
    # File Based Quiz Generation
# @app.route("/generate-quiz-file", methods=["POST"])
# def generate_quiz_file():
#     file = request.files.get("file")
 
#     if not file:
#         return jsonify({"error": "No file"}), 400

#     # temp_path = "temp_quiz_file"
#     temp_path = os.path.join(UPLOAD_DIR, file.filename)
#     file.save(temp_path)

#     #  Extract text (reuse your function)
#     text = extract_text(temp_path)

#     if not text.strip():
#         return jsonify({"error": "No text found"})

#     chunks = chunk_text(text)[:15]
#     context = "\n".join(chunks)

#     prompt = build_quiz_prompt(context)

#     raw = call_openai(prompt, max_tokens=700, json_mode=True)
#     print("RAW AI OUTPUT:", raw)
#     try:
#         start = raw.find("{")
#         end = raw.rfind("}") + 1
#         clean = raw[start:end]
#         data = json.loads(clean)

#         # ALWAYS RETURN SAME STRUCTURE
#         quiz = data.get("quiz")

#         if not quiz or not isinstance(quiz, list):
#             return jsonify({
#                 "quiz": [],
#                 "error": "Invalid quiz format from AI"
#             })

#         return jsonify({
#             "quiz": quiz
#         })
#     except:
#         print("RAW OUTPUT:", raw)

#         return jsonify({
#             "quiz": [],
#             "error": "JSON parse failed"
#         })

# File Based Quiz Generation
@app.route("/generate-quiz-file", methods=["POST"])
def generate_quiz_file():
    file = request.files.get("file")

    if not file:
        return jsonify({"error": "No file"}), 400

    temp_path = os.path.join(UPLOAD_DIR, file.filename)

    try:
        file.save(temp_path)

        # 🔹 Extract text
        text = extract_text(temp_path)

        if not text.strip():
            return jsonify({
                "quiz": [],
                "error": "No text found"
            }), 400

        # 🔹 Get regeneration information
        regenerate = (
            request.form.get("regenerate", "false").lower()
            == "true"
        )

        previous_quiz_raw = request.form.get(
            "previousQuiz",
            "[]"
        )

        try:
            previous_quiz = json.loads(previous_quiz_raw)

            if not isinstance(previous_quiz, list):
                previous_quiz = []

        except Exception:
            previous_quiz = []

        print("REGENERATE:", regenerate)
        print(
            "PREVIOUS QUIZ QUESTIONS:",
            len(previous_quiz)
        )

        # 🔹 Prepare context
        chunks = chunk_text(text)[:15]
        context = "\n".join(chunks)

        # 🔹 Build prompt
        prompt = build_quiz_prompt(
            context,
            previous_quiz if regenerate else []
        )

        # 🔹 Generate quiz
        raw = call_openai(
            prompt,
            max_tokens=700,
            json_mode=True
        )

        print("RAW AI OUTPUT:", raw)

        # 🔹 Parse JSON
        start = raw.find("{")
        end = raw.rfind("}") + 1

        if start == -1 or end <= start:
            return jsonify({
                "quiz": [],
                "error": "Invalid JSON returned by AI"
            }), 500

        clean = raw[start:end]
        data = json.loads(clean)

        quiz = data.get("quiz")

        if not quiz or not isinstance(quiz, list):
            return jsonify({
                "quiz": [],
                "error": "Invalid quiz format from AI"
            }), 500

        return jsonify({
            "quiz": quiz
        })

    except Exception as e:
        print("GENERATE QUIZ ERROR:", str(e))

        return jsonify({
            "quiz": [],
            "error": str(e)
        }), 500

    finally:
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception:
                pass
# =========================
# RUN
# =========================
if __name__ == "__main__":
    # app.run(debug=True,)
    # app.run(host="0.0.0.0", port=5000, debug=True,use_reloader=False)
    app.run(
    host=os.getenv("FLASK_HOST", "0.0.0.0"),
    port=int(os.getenv("FLASK_PORT", "5000"))
)