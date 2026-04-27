import os
import pickle
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

from paddleocr import PaddleOCR

ocr = PaddleOCR(use_angle_cls=True, lang='en')

def convert_to_pdf(input_path):
    output_dir = os.path.dirname(input_path)

    libreoffice_path = r"C:\Program Files\LibreOffice\program\soffice.exe"

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

# pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH

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
        reader = PdfReader(path)
        return "".join([p.extract_text() or "" for p in reader.pages])

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

            result = ocr.predict("temp_ocr.jpg")

            text = ""

            if result and result[0]:
                for line in result[0]:
                    text += line[1][0] + "\n"

            print("OCR OUTPUT:", text[:200])  # 🔥 DEBUG

            return text

        except Exception as e:
            print("OCR ERROR:", str(e))
            return ""

    return ""


# CHUNKING

def chunk_text(text):
    words = text.split()
    return [" ".join(words[i:i+CHUNK_SIZE]) for i in range(0, len(words), CHUNK_SIZE)]

# STORE
def add_chunks(chunks):
    emb = embed_model.encode(chunks, batch_size=8)
    index.add(np.array(emb))
    documents.extend(chunks)
    save_state()

# SEARCH

def search(query, k=TOP_K):
    if len(documents) == 0:
        return []

    q_emb = embed_model.encode([query])
    D, I = index.search(np.array(q_emb), k)
    return [documents[i] for i in I[0]]


# OPENAI CALL

def call_openai(prompt, max_tokens=700, json_mode=False):
    params = {
        "model": "openai/gpt-4o-mini",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.3
    }
    
    if json_mode:
        params["response_format"] = {"type": "json_object"}

    res = client.chat.completions.create(**params)

    return res.choices[0].message.content


# PROMPTS

def build_answer_prompt(context, question):
    return f"""
You are an AI assistant for students.

Answer the question clearly using the given context.

Rules:
- Give complete answer (no cut sentences)
- Use proper paragraphs
- Use bullet points or numbering if needed
- Do NOT leave incomplete lines
- Keep answer clean and structured

Context:
{context}

Question:
{question}

Answer:
"""
def build_quiz_prompt(context):
    return f"""
You are a strict JSON generator.

TASK:
Generate EXACTLY 5 multiple choice questions (MCQs).

CRITICAL RULES (DO NOT BREAK):
- Output ONLY valid JSON
- DO NOT write any text before or after JSON
- JSON must start with {{ and end with }}
- Always return complete JSON
- If unsure, still return valid JSON with empty quiz []

FORMAT (STRICT):
{{
  "quiz": [
    {{
      "question": "Question text",
      "options": ["Option 1", "Option 2", "Option 3", "Option 4"],
      "answer": 0,
      "topic": "General",
      "explanation": "Explanation of correct answer"
    }}
  ]
}}

RULES:
- Exactly 5 questions
- Each question must be unique
- options length MUST be 4
- answer must be index (0,1,2,3)
- DO NOT include numbering inside options
- explanation must be clear

Context:
{context[:1200]}
"""
# UPLOAD

@app.route("/upload", methods=["POST"])
def upload():
    file = request.files["file"]
    path = os.path.join(UPLOAD_DIR, file.filename)
    file.save(path)

    text = extract_text(path)

    if not text.strip():
        return jsonify({"error": "No text extracted"})

    chunks = chunk_text(text)
    add_chunks(chunks)

    return jsonify({"status": "indexed", "chunks": len(chunks)})


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
            return jsonify({"pdfPath": input_path})

        pdf_path = convert_to_pdf(input_path)

        return jsonify({"pdfPath": pdf_path})

    except Exception as e:
        print("CONVERSION ERROR:", str(e))
        return jsonify({"error": str(e)}), 500

# ASK (RAG)

@app.route("/ask-ai", methods=["POST"])
def ask():
    q = request.json["question"]

    context_chunks = search(q)
    context = "\n".join(context_chunks)[:1500]

    prompt = build_answer_prompt(context, q)
    # answer = call_openai(prompt)
    answer = call_openai(prompt, max_tokens=700)

    return jsonify({"answer": answer})


@app.route("/ask-pdf", methods=["POST"])
def ask_pdf():
    file = request.files.get("file")
    question = request.form.get("question")

    if not file or not question:
        return jsonify({"error": "Missing file or question"}), 400

    # temp_path = "temp.pdf"
    temp_path = os.path.join(UPLOAD_DIR, file.filename)
    file.save(temp_path)

    text = extract_text(temp_path)  
    if not text.strip():
       return jsonify({"error": "No text extracted"})
 
    chunks = chunk_text(text)      

    embeddings = embed_model.encode(chunks)

    temp_index = faiss.IndexFlatL2(embeddings.shape[1])
    temp_index.add(np.array(embeddings))

    q_embedding = embed_model.encode([question])

    D, I = temp_index.search(np.array(q_embedding), k=3)

    relevant_chunks = [chunks[i] for i in I[0]]

    context = "\n".join(relevant_chunks)

    prompt = build_answer_prompt(context, question)
    answer = call_openai(prompt, max_tokens=700)
    return jsonify({"answer": answer})


@app.route("/ask-file", methods=["POST"])
def ask_file():
    file = request.files["file"]

    temp_path = os.path.join(UPLOAD_DIR, file.filename)
    file.save(temp_path)

    # 🔥 IMPORTANT: USE UNIVERSAL EXTRACT
    text = extract_text(temp_path)

    if not text.strip():
        return jsonify({"answer": "No readable content found"})

    chunks = chunk_text(text)

# 🔥 create embeddings
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
@app.route("/generate-quiz-file", methods=["POST"])
def generate_quiz_file():
    file = request.files.get("file")
 
    if not file:
        return jsonify({"error": "No file"}), 400

    # temp_path = "temp_quiz_file"
    temp_path = os.path.join(UPLOAD_DIR, file.filename)
    file.save(temp_path)

    # 🔹 Extract text (reuse your function)
    text = extract_text(temp_path)

    if not text.strip():
        return jsonify({"error": "No text found"})

    chunks = chunk_text(text)[:15]
    context = "\n".join(chunks)

    prompt = build_quiz_prompt(context)

    raw = call_openai(prompt, max_tokens=700, json_mode=True)
    print("RAW AI OUTPUT:", raw)

    try:
        start = raw.find("{")
        end = raw.rfind("}") + 1
        clean = raw[start:end]
        data = json.loads(clean)

        # return jsonify(data)
        # 🔥 ALWAYS RETURN SAME STRUCTURE
        quiz = data.get("quiz")

        if not quiz or not isinstance(quiz, list):
            return jsonify({
                "quiz": [],
                "error": "Invalid quiz format from AI"
            })

        return jsonify({
            "quiz": quiz
        })
    except:
        print("RAW OUTPUT:", raw)

        return jsonify({
            "quiz": [],
            "error": "JSON parse failed"
        })
# =========================
# RUN
# =========================
if __name__ == "__main__":
    # app.run(debug=True,use_reloader=False)
    app.run(host="0.0.0.0", port=5000, debug=True, use_reloader=False)