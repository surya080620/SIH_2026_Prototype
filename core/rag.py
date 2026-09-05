import sqlite3
import requests
import json
import chromadb
import re

CHROMA_PATH = "./chroma_db"
OLLAMA_URL = "http://localhost:11434/api/generate"
# Updated to your downloaded model
MODEL_NAME = "qwen3:4b"

ROLE_PROMPTS = {
    "Consumer": """You are the BIS Consumer Safety Guide.
Tone: Clear, protective, and non-technical.
Focus: How to verify genuine BIS marks (the interlocking S-monogram), check the 8-digit R-number (R-XXXXXXXX), identify substandard or hazardous goods, and report grievances to the BIS helpline. Avoid technical test procedures.""",

    "MSME": """You are the BIS MSME Compliance & Registration Consultant.
Tone: Direct, operational, and practical.
Focus: Step-by-step guidance under Compulsory Registration Scheme (CRS) Scheme-II: testing at BIS-recognized laboratories (report validity 90 days), Form VI submission, 2-year license validity, renewal rules, and the manufacturer transition grace period ending 02 February 2027.""",

    "Student": """You are a Technical Standards Researcher and Academic Advisor.
Tone: Structured, educational, and analytical.
Focus: Standard genealogy, alignment with international standard IEC 62560, the mandate of Electrotechnical Committee ETD 23 (Lamps and Related Equipments), photobiological safety (IS 16108), and legal backing under Section 10(1)(p) of the BIS Act, 1986.""",

    "Manufacturer": """You are the Industrial Regulatory Compliance Lead.
Tone: Authoritative, procedural, and compliance-rigorous.
Focus: Series Approval Guidelines (representative sample testing), Safety Critical Components submission (PCB, driver circuit, thermal protection, caps), 10-day model inclusion rules, and surveillance counter-sampling protocols under the 2012 Compulsory Registration Order."""
}

def generate_dynamic_questions(user_query, role):
    """
    Pass 1: Evaluates intent.
    - General/informational queries -> Returns [] (Answer immediately)
    - Specific standard/license queries -> Returns [] (Answer immediately)
    - Ambiguous certification/compliance queries -> Returns slot-filling questions
    """
    q_lower = user_query.lower().strip()

    # 1. Skip slot filling for general knowledge / overview questions about BIS
    general_intents = [
        "what is bis", "about bis", "who is bis", "what does bis do", 
        "full form of bis", "history of bis", "bureau of indian standards",
        "why bis", "explain bis", "role of bis"
    ]
    if any(intent in q_lower for intent in general_intents) or (len(q_lower.split()) <= 4 and "bis" in q_lower):
        return []

    # 2. Skip slot filling if query already mentions specific standards or products
    direct_entities = ["16102", "led", "lamp", "bulb", "driver", "luminaire", "r-", "licence", "license number"]
    if any(entity in q_lower for entity in direct_entities):
        return []

    # 3. For all other queries, let Ollama evaluate if it's asking about compliance/certification
    eval_prompt = f"""You are an intent classifier for the Bureau of Indian Standards Assistant.
User query: "{user_query}"

Task:
- If the user is asking a general knowledge question (e.g. what is BIS, standard definition, organization structure, greetings), output: {{"needs_clarification": false}}
- If the user is asking how to certify, register, test, or get an Indian Standard for an UNNAMED product, output:
{{
  "needs_clarification": true,
  "questions": [
    {{
      "id": "category",
      "title": "Product Category",
      "hint": "Select your product category",
      "options": ["LED Lighting", "Electronics & IT Goods", "Domestic Appliances", "Other"]
    }},
    {{
      "id": "scheme",
      "title": "Applicable Scheme",
      "hint": "Select scheme if known",
      "options": ["CRS (Scheme-II)", "ISI Mark (Scheme-I)", "Not Sure"]
    }}
  ]
}}"""

    try:
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": MODEL_NAME,
                "prompt": eval_prompt,
                "format": "json",
                "stream": False,
                "options": {"temperature": 0.0}
            },
            timeout=10
        )
        if response.status_code == 200:
            parsed = json.loads(response.json().get("response", "{}"))
            if not parsed.get("needs_clarification", False):
                return []
            return parsed.get("questions", [])
    except Exception as e:
        print(f"Classifier error: {e}")

    # Fallback default: do not block general questions with wizard
    return []

def query_knowledge_base(user_query, role="MSME"):
    """
    Pass 2: Retrieves relevant chunks from ChromaDB and executes
    grounded generation through qwen3:4b.
    """
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = client.get_collection(name="bis_knowledge")

    results = collection.query(query_texts=[user_query], n_results=3)
    chunks = results["documents"][0] if results["documents"] else []
    distances = results["distances"][0] if results.get("distances") and results["distances"] else [0.4]

    best_dist = distances[0] if distances else 0.4
    confidence_score = max(55, min(95, int((1.0 - (best_dist / 2.0)) * 100)))

    # Fetch structured metadata from SQLite
    conn = sqlite3.connect("standards.db")
    cur = conn.cursor()
    cur.execute("SELECT is_number, title, status, last_updated, scheme, transition_deadline, contact_dept, contact_phone, contact_email FROM standards_meta WHERE is_number LIKE '%2026%' LIMIT 1")
    row = cur.fetchone()
    conn.close()

    metadata = {
        "is_number": row[0] if row else "IS 16102 (Part 1): 2026",
        "title": row[1] if row else "Self-ballasted LED lamps for general lighting services",
        "status": row[2] if row else "Active",
        "last_updated": row[3] if row else "02 February 2026",
        "scheme": row[4] if row else "Compulsory Registration Scheme (CRS) - Scheme II",
        "transition_deadline": row[5] if row else "02 February 2027",
        "contact_dept": row[6] if row else "CRS Registration Department",
        "contact_phone": row[7] if row else "+91-11-23230856",
        "contact_email": row[8] if row else "registration@bis.org.in"
    }

    role_prompt = ROLE_PROMPTS.get(role, ROLE_PROMPTS["MSME"])
    context_text = "\n---\n".join(chunks)

    full_prompt = f"""{role_prompt}

Context from Official BIS Gazette & Standards Database:
{context_text}

Metadata State:
Standard: {metadata['is_number']} is {metadata['status']}. 
Notice: 2012 version is officially withdrawn as of 02 August 2026; manufacturer transition extension is valid until {metadata['transition_deadline']}.

User Query: {user_query}

Instructions:
1. Answer the query strictly grounded in the provided context and metadata.
2. Maintain the assigned persona point of view.
3. Conclude with 3 or 4 practical next steps.
4. Keep the response concise and clearly formatted."""

    try:
        res = requests.post(
            OLLAMA_URL,
            json={
                "model": MODEL_NAME,
                "prompt": full_prompt,
                "stream": False,
                "options": {"temperature": 0.2}
            },
            timeout=40
        )
        if res.status_code == 200:
            llm_text = res.json().get("response", "").strip()
        else:
            llm_text = "Ollama returned an unexpected status code. Fallback response generated."
    except Exception:
        llm_text = (
            f"Under the Compulsory Registration Scheme (CRS), self-ballasted LED lamps are governed by {metadata['is_number']}. "
            f"The revised 2026 standard is currently {metadata['status']}. While the 2012 version is withdrawn, registered manufacturers have until {metadata['transition_deadline']} to migrate technical files and sample test reports."
        )

    return {
        "text": llm_text,
        "metadata": metadata,
        "confidence": confidence_score
    }