import sqlite3
import json
import os
import re
from google import genai
from google.genai import types
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from dotenv import load_dotenv
# Vector Store and Local Embeddings Setup
load_dotenv()
CHROMA_PATH = "./chroma_db"
EMBEDDING_MODEL = "nomic-embed-text"
OLLAMA_BASE_URL = "http://localhost:11434"

# Google Gemini API Client Configuration
GEMINI_API_KEY = os.getenv("googleapikey")
print(GEMINI_API_KEY)
client_ai = genai.Client(api_key=GEMINI_API_KEY)
MODEL_ID = "gemini-3.5-flash-lite"

ROLE_PROMPTS = {
    "Consumer": """You are the official BIS Consumer Safety Guide.
Tone: Protective, plain language, and accessible.
Focus: Authentic Standard Marks (the blue/red S-monogram), verifying 8-digit R-numbers (R-XXXXXXXX), recognizing counterfeit products, and filing consumer grievances. Avoid dense engineering jargon.""",

    "MSME": """You are the official BIS MSME Compliance Consultant.
Tone: Direct, procedural, and business-focused.
Focus: CRS Scheme-II regulations, 90-day validity for lab test reports, online Form VI filing, Authorized Indian Representative (AIR) rules, 2-year license cycles, and the 02 February 2027 manufacturer transition deadline.""",

    "Student": """You are a Technical Standardization Academic Specialist.
Tone: Educational, historical, and analytical.
Focus: Statutory history (ISI 1947 to BIS Act 2016), Technical Committees (ETD 23 for Lighting, LITD 07 for IT Equipment), IEC alignment (IEC 62560, IEC 60950-1), and Section 10(1)(p) of the BIS Act 1986.""",

    "Manufacturer": """You are the Industrial Regulatory Compliance Lead.
Tone: Formal, rigorous, and compliance-first.
Focus: Series Approval Guidelines (worst-case sample selection), safety critical components (PCBs, drivers, isolation), 10-day timeline for model additions, market surveillance counter-sampling, and compulsory scrap deformation of non-compliant inventory."""
}

OFFICIAL_HELPLINE = """Official BIS Support Contacts:
- National Helpdesk: 1800-11-1234 / +91-76-6908-9323
- Departmental Email: helpdesk@bis.gov.in / registration@bis.org.in
- CRS Grievance Desk: ddgmscd@bis.gov.in
- Central Headquarters: Bureau of Indian Standards, Manak Bhawan, 9 Bahadur Shah Zafar Marg, New Delhi-110002"""

# Reference URLs categorized by subject area
PORTAL_LINKS = {
    "crs": [
        {"title": "Official BIS CRS Portal (eBIS)", "url": "https://www.crsbis.in/BIS/"},
        {"title": "Know Your Standard (KYS) Directory", "url": "https://www.services.bis.gov.in/php/BIS_2.0/bisconnect/knowyourstandards/indian_standards/isdetails"},
        {"title": "List of BIS Recognized Testing Laboratories", "url": "https://www.crsbis.in/BIS/lab-list.do"}
    ],
    "lighting": [
        {"title": "IS 16102 (Part 1) Technical Guidelines & Notifications", "url": "https://www.crsbis.in/BIS/products.do"},
        {"title": "BIS ETD 23 Committee Scope & Revisions", "url": "https://www.services.bis.gov.in/"},
        {"title": "Official BIS CRS Portal (eBIS)", "url": "https://www.crsbis.in/BIS/"}
    ],
    "it_equipment": [
        {"title": "MeitY Compulsory Registration Orders (CRO)", "url": "https://www.meity.gov.in/esdm/standards"},
        {"title": "IS 13252 Safety Requirements FAQ", "url": "https://www.crsbis.in/BIS/ite-guidelines.do"},
        {"title": "Official BIS CRS Portal (eBIS)", "url": "https://www.crsbis.in/BIS/"}
    ]
}

def build_raw_chunk_fallback(user_query, chunks, metadata):
    """
    Constructs a clear fallback response directly from retrieved vector chunks 
    when the LLM encounters a rate limit (429) or network issue.
    """
    q_lower = user_query.lower()
    
    # Pick relevant reference links based on query content
    if any(k in q_lower for k in ["16102", "led", "lamp", "bulb", "light", "luminaire"]):
        links = PORTAL_LINKS["lighting"]
    elif any(k in q_lower for k in ["13252", "mobile", "phone", "laptop", "power bank", "it equipment"]):
        links = PORTAL_LINKS["it_equipment"]
    else:
        links = PORTAL_LINKS["crs"]

    links_markdown = "\n".join([f"* [{link['title']}]({link['url']})" for link in links])

    # Clean and format the top chunks for user readability
    formatted_chunks = []
    for i, c in enumerate(chunks[:2], start=1):
        # Truncate overly long text and remove extra whitespace
        cleaned = re.sub(r'\s+', ' ', c).strip()
        if len(cleaned) > 420:
            cleaned = cleaned[:420] + "..."
        formatted_chunks.append(f"**Extracted Passage {i}:**\n> \"{cleaned}\"")

    passages_text = "\n\n".join(formatted_chunks) if formatted_chunks else "No relevant passages were extracted from the local vector index."

    fallback_text = (
        "**AI Inference Unavailable (Rate Limit / Connection Notice)**\n\n"
        "The automated synthesis model could not be reached. "
        "Below are the verified regulatory clauses extracted directly from the local knowledge base, "
        "along with relevant metadata and official portals for your query:\n\n"
        f"{passages_text}\n\n"
        f"**Official Governing Standard:** {metadata['is_number']} — {metadata['title']}\n"
        f"* **Conformity Scheme:** {metadata['scheme']}\n"
        f"* **Enforcement Status:** {metadata['status']}\n"
        f"* **Transition Deadline:** {metadata['transition_deadline']}\n\n"
        "**Official Reference & Filing Links:**\n"
        f"{links_markdown}\n\n"
        "**Helpline Escalation:**\n"
        "If you need immediate verification, call the national helpdesk at **1800-11-1234** or email **helpdesk@bis.gov.in**."
    )
    return fallback_text

def handle_conversational_chitchat(user_query, role="MSME"):
    q_clean = user_query.lower().strip().rstrip(".!?")
    greetings = {"hi", "hello", "hey", "good morning", "good evening", "namaste", "greetings"}
    if q_clean in greetings:
        return (
            f"Hello! I am your Bureau of Indian Standards (BIS) AI Assistant, operating in **{role}** mode. "
            "How can I assist you with Indian Standards, CRS certification, or testing compliance today?"
        )
    appreciations = {
        "thanks", "thank you", "thanks for the support", "thank you so much",
        "thanks a lot", "ok thanks", "great thanks", "thank u"
    }
    if q_clean in appreciations:
        return (
            "You are welcome! Feel free to ask whenever you need standard specifications, "
            "licensing procedures, or laboratory testing requirements. Have a productive day!"
        )
    farewells = {"bye", "goodbye", "see you", "exit"}
    if q_clean in farewells:
        return "Goodbye! Do not hesitate to return if you have further compliance questions."
    return None

def classify_intent_with_llm(user_query, role="MSME"):
    system_instruction = """You are an intent classifier for a Bureau of Indian Standards (BIS) AI Assistant.
The system possesses technical data ONLY on:
1. General BIS Overview (BIS Act 2016, ISI history, 10 schemes, governance, Manak Bhawan)
2. IS 16102 Parts 1 & 2 (Self-ballasted LED lamps, related luminaire standards IS 10322 series, IS 16614, ETD 23 committee)
3. IS 13252 Part 1 (IT Equipment: mobile phones, laptops, power banks, LITD 07, IS 16333 Indian language support)
4. CRS Scheme-II Registration, Renewal, lab testing, and Standard Mark labelling.

Classify the user query into ONE of the following intents:
- "CONVERSATIONAL": Greetings, thank-yous, polite closings, or small talk.
- "INFORMATIONAL_QUERY": The user wants an explanation, an overview, a list, or a catalog of standards (e.g. "indian standards related to led lamps", "what standards exist for mobiles", "tell me about IS 16102", "what is BIS").
- "ACTIONABLE_AMBIGUOUS": The user states an actionable intent to manufacture, test, import, or certify a product, but lacks necessary engineering specifications (e.g., "I want to build a led lamp", "how to certify my light", "help me register my IT device").
- "OUT_OF_SCOPE": The query is about an unindexed product or domain (e.g., shampoo, cosmetics, cement, steel, toys, footwear).

Output strictly valid JSON with no markdown backticks:
{
  "intent": "CONVERSATIONAL" | "INFORMATIONAL_QUERY" | "ACTIONABLE_AMBIGUOUS" | "OUT_OF_SCOPE",
  "category": "lighting" | "it_equipment" | "general_bis" | "other",
  "direct_reply": "string (only if intent is CONVERSATIONAL, otherwise empty string)"
}"""

    prompt = f"User Query: {user_query}\nPersona: {role}"
    try:
        response = client_ai.models.generate_content(
            model=MODEL_ID,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                temperature=0.0,
                response_mime_type="application/json"
            )
        )
        return json.loads(response.text.strip())
    except Exception as e:
        q_lower = user_query.lower().strip()
        if any(w in q_lower for w in ["what", "list", "which", "related to", "standards for", "tell me about"]):
            return {"intent": "INFORMATIONAL_QUERY", "category": "lighting", "direct_reply": ""}
        if any(w in q_lower for w in ["make", "build", "manufacture", "register", "apply"]):
            return {"intent": "ACTIONABLE_AMBIGUOUS", "category": "lighting", "direct_reply": ""}
        return {"intent": "INFORMATIONAL_QUERY", "category": "general_bis", "direct_reply": ""}

def generate_dynamic_questions(user_query, role="MSME"):
    if any(k in user_query.lower() for k in ["is 16102", "16102", "is 13252", "13252", "r-"]):
        return []

    classification = classify_intent_with_llm(user_query, role)
    intent = classification.get("intent")
    category = classification.get("category")

    if intent != "ACTIONABLE_AMBIGUOUS":
        return []

    if category == "lighting":
        return [
            {
                "id": "lamp_type",
                "title": "Exact Product Construction",
                "hint": "Select standard option or enter custom specification",
                "allow_custom": True,
                "options": [
                    "Self-ballasted with integrated driver",
                    "LED Luminaire / Fixture",
                    "Standalone LED Driver",
                    "Linear Tube (Double-capped)"
                ]
            },
            {
                "id": "voltage_feature",
                "title": "Supply & Features",
                "hint": "Select standard electrical configuration or enter custom",
                "allow_custom": True,
                "options": [
                    "Mains AC (50V–250V up to 60W)",
                    "Battery-incorporated / Smart LED",
                    "Industrial High Wattage (>60W)"
                ]
            },
            {
                "id": "applicant_type",
                "title": "Applicant Classification",
                "hint": "Select enterprise category or specify status",
                "allow_custom": True,
                "options": [
                    "Domestic Indian Manufacturer",
                    "Importer (AIR Required)",
                    "Trader / Retailer"
                ]
            }
        ]

    if category == "it_equipment":
        return [
            {
                "id": "it_type",
                "title": "Device Category",
                "hint": "Select device type or specify custom",
                "allow_custom": True,
                "options": ["Mobile Phone", "Laptop / Notebook", "Power Bank", "Printer / Scanner"]
            },
            {
                "id": "applicant_type",
                "title": "Applicant Classification",
                "hint": "Select enterprise category",
                "allow_custom": True,
                "options": ["Domestic Indian Manufacturer", "Importer (AIR Required)", "Trader / Retailer"]
            }
        ]

    return []

def query_knowledge_base(user_query, role="MSME"):
    # 1. Immediate conversational check
    chitchat = handle_conversational_chitchat(user_query, role)
    if chitchat:
        return {
            "text": chitchat,
            "metadata": {"status": "Ready", "is_number": "BIS Assistant", "scheme": "N/A"},
            "confidence": 100
        }

    # 2. Local semantic vector retrieval via ChromaDB
    embeddings = OllamaEmbeddings(
        model=EMBEDDING_MODEL,
        base_url=OLLAMA_BASE_URL
    )
    vector_store = Chroma(
        collection_name="bis_knowledge",
        embedding_function=embeddings,
        persist_directory=CHROMA_PATH
    )

    search_results = vector_store.similarity_search_with_score(user_query, k=4)
    chunks = [doc.page_content for doc, _ in search_results]
    distances = [score for _, score in search_results]

    best_dist = distances[0] if distances else 0.4
    confidence_score = max(40, min(97, int((1.0 - (best_dist / 2.0)) * 100)))

    # 3. Retrieve relational standard metadata from SQLite
    q_lower = user_query.lower()
    conn = sqlite3.connect("standards.db")
    cur = conn.cursor()

    if any(k in q_lower for k in ["13252", "it equipment", "mobile", "phone", "laptop", "power bank"]):
        cur.execute("SELECT is_number, title, status, last_updated, scheme, transition_deadline, contact_dept, contact_phone, contact_email FROM standards_meta WHERE is_number LIKE '%13252%' LIMIT 1")
    elif any(k in q_lower for k in ["16102", "led", "lamp", "bulb", "lighting", "luminaire"]):
        cur.execute("SELECT is_number, title, status, last_updated, scheme, transition_deadline, contact_dept, contact_phone, contact_email FROM standards_meta WHERE is_number LIKE '%16102 (Part 1): 2026%' LIMIT 1")
    else:
        cur.execute("SELECT is_number, title, status, last_updated, scheme, transition_deadline, contact_dept, contact_phone, contact_email FROM standards_meta LIMIT 1")

    row = cur.fetchone()
    conn.close()

    metadata = {
        "is_number": row[0] if row else "Indian Standard",
        "title": row[1] if row else "Bureau of Indian Standards Specification",
        "status": row[2] if row else "Active",
        "last_updated": row[3] if row else "Official Gazette",
        "scheme": row[4] if row else "Compulsory Registration Scheme (CRS)",
        "transition_deadline": row[5] if row else "N/A",
        "contact_dept": row[6] if row else "CRS Registration Department",
        "contact_phone": row[7] if row else "+91-11-23230856",
        "contact_email": row[8] if row else "registration@bis.org.in"
    }

    role_instruction = ROLE_PROMPTS.get(role, ROLE_PROMPTS["MSME"])
    context_text = "\n---\n".join(chunks)

    full_prompt = f"""{role_instruction}

STRICT BOUNDARY & COMPLIANCE RULES:
1. You have access ONLY to the ingested BIS technical knowledge base:
   - General BIS Overview: BIS Act 2016, ISI origins, 10 conformity assessment schemes, Manak Bhawan.
   - Lighting & LED Standards: IS 16102 (Part 1): 2026 (Safety), IS 16102 (Part 2): 2026 (Performance), IS 10322 Series (Luminaires), IS 16614 (Part 1): 2026 (Double-capped linear LED lamps), ETD 23 committee, 02 Feb 2027 manufacturer transition deadline.
   - IT Equipment Standards: IS 13252 (Part 1): 2010 (General IT equipment safety, Amendment 2), IS 16333 (Part 3) (Mandatory Indian language support for mobile phones), LITD 07 committee.
   - CRS Registration & Labelling: 90-day lab report validity, 2-year renewal cycle, Standard Mark S-monogram and R-XXXXXXXX format.

2. OUT-OF-CONTEXT CUSTOM INPUT DETECTION:
   Review the user query and custom user-provided slot parameters carefully. If the user has entered custom inputs or asked about products/specifications outside our ingested standards (e.g., shampoo, chemicals, food, cement, steel, automotive parts, organic goods, solar panels, toys):
   - You MUST explicitly highlight in the first paragraph:
     "**Notice on Custom Input:** One or more of your specified parameters falls outside the verified technical scope of this prototype repository."
   - State clearly which parameter is unindexed.
   - Provide the official BIS escalation details:
{OFFICIAL_HELPLINE}

3. IN-SCOPE QUERIES:
   - Ground answers directly in the context and metadata.
   - Provide end-to-end guidance (Standard number, Testing, AIR/Form VI, Labelling, Deadlines).
   - Conclude with 3 clear, actionable next steps.

Ingested Context:
{context_text}

Metadata State:
Standard: {metadata['is_number']} ({metadata['title']})
Status: {metadata['status']} | Scheme: {metadata['scheme']} | Transition Deadline: {metadata['transition_deadline']}

User Query & Clarified Parameters: {user_query}"""

    try:
        response = client_ai.models.generate_content(
            model=MODEL_ID,
            contents=full_prompt,
            config=types.GenerateContentConfig(
                temperature=0.1,
                max_output_tokens=1400
            )
        )
        llm_text = response.text.strip()
    except Exception as e:
        print("\n[AI INFERENCE EXCEPTION — ENGAGING RETRIEVAL FALLBACK]:", e, "\n")
        # Logical fallback: return retrieved chunks + metadata + direct links
        llm_text = build_raw_chunk_fallback(user_query, chunks, metadata)
        confidence_score = 50

    if "outside the verified technical scope" in llm_text.lower() or "no verified technical dataset" in llm_text.lower():
        confidence_score = 35
        metadata["status"] = "Out of Scope / Refer Helpline"

    return {
        "text": llm_text,
        "metadata": metadata,
        "confidence": confidence_score
    }