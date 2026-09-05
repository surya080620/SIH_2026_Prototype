from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import re
import sqlite3
from core.rag import (
    classify_intent_with_llm,
    generate_dynamic_questions,
    query_knowledge_base
)

app = Flask(__name__)
CORS(app)

def lookup_license_db(r_number: str):
    """Direct fast-path lookup for 8-digit BIS CRS Registration Numbers."""
    clean_r = r_number.upper().strip()
    conn = sqlite3.connect("standards.db")
    cur = conn.cursor()
    cur.execute(
        "SELECT r_number, brand, manufacturer, product, is_number, status, valid_till "
        "FROM crs_licenses WHERE r_number = ?",
        (clean_r,)
    )
    row = cur.fetchone()
    conn.close()

    if row:
        return {
            "found": True,
            "r_number": row[0],
            "brand": row[1],
            "manufacturer": row[2],
            "product": row[3],
            "is_number": row[4],
            "status": row[5],
            "valid_till": row[6]
        }
    return {
        "found": False,
        "message": f"No active BIS CRS registration record found for {clean_r}. Exercise caution regarding uncertified or non-conforming goods."
    }

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/assistant")
def assistant():
    return render_template("assistant.html")

@app.route("/api/chat", methods=["POST"])
def api_chat():
    data = request.json or {}
    message = data.get("message", "").strip()
    role = data.get("role", "MSME")
    is_clarified = data.get("is_clarified", False)

    if not message:
        return jsonify({"error": "Empty query received"}), 400

    # 1. License Check Fast-Path (Interception before LLM)
    r_match = re.search(r"\b(R-\d{8})\b", message, re.IGNORECASE)
    if r_match:
        r_no = r_match.group(1).upper()
        lic_data = lookup_license_db(r_no)
        if lic_data.get("found"):
            text_out = (
                f"**Registration {lic_data['r_number']} is VERIFIED**\n\n"
                f"* **Brand:** {lic_data['brand']}\n"
                f"* **Manufacturer:** {lic_data['manufacturer']}\n"
                f"* **Product:** {lic_data['product']}\n"
                f"* **Standard:** {lic_data['is_number']}\n"
                f"* **Status:** {lic_data['status']} (Valid through {lic_data['valid_till']})"
            )
            confidence = 98
            status = lic_data['status']
        else:
            text_out = lic_data.get("message")
            confidence = 35
            status = "Unregistered / Caution"

        return jsonify({
            "action": "PROVIDE_ANSWER",
            "data": {
                "text": text_out,
                "metadata": {
                    "is_number": lic_data.get("is_number", "CRS Registry"),
                    "title": lic_data.get("product", "Registration Record"),
                    "status": status,
                    "scheme": "CRS Scheme-II"
                },
                "confidence": confidence
            }
        })

    # 2. Already Clarified by User -> Direct Grounded RAG Retrieval
    if is_clarified:
        result = query_knowledge_base(user_query=message, role=role)
        return jsonify({
            "action": "PROVIDE_ANSWER",
            "data": result
        })

    # 3. Single-Call Intent Classification via Gemini
    classification = classify_intent_with_llm(message, role=role)
    intent = classification.get("intent")
    category = classification.get("category")
    direct_reply = classification.get("direct_reply", "")

    # Branch A: Pure Conversational Greeting / Thank You / Closing
    if intent == "CONVERSATIONAL" and direct_reply:
        return jsonify({
            "action": "PROVIDE_ANSWER",
            "data": {
                "text": direct_reply,
                "metadata": {
                    "status": "Ready",
                    "is_number": "BIS Conversational Assistant",
                    "scheme": "N/A"
                },
                "confidence": 100
            }
        })

    # Branch B: Actionable Ambiguity -> Emit Clarification Chips
    if intent == "ACTIONABLE_AMBIGUOUS":
        questions = generate_dynamic_questions(message, role=role)
        if questions:
            return jsonify({
                "action": "REQUEST_SLOT_FILL",
                "questions": questions
            })

    # Branch C: Informational, Catalog Lists, Out-of-Scope, or Standard Queries
    result = query_knowledge_base(user_query=message, role=role)
    return jsonify({
        "action": "PROVIDE_ANSWER",
        "data": result
    })

@app.route("/api/verify-license", methods=["GET"])
def verify_license_endpoint():
    r_no = request.args.get("r_number", "").strip().upper()
    if not r_no:
        return jsonify({"found": False, "message": "Please specify an 8-digit R-Number (e.g., R-41013730)."}), 400

    return jsonify(lookup_license_db(r_no))

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)