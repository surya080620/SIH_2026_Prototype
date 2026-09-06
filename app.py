from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import sqlite3
from core.rag import query_knowledge_base, generate_dynamic_questions

# Initialize Flask application with CORS support for cross-origin frontend requests
app = Flask(__name__)
CORS(app)

# Main web interface routes
@app.route("/")
def index():
    # Render primary landing dashboard page
    return render_template("index.html")

@app.route("/assistant")
def assistant():
    # Render interactive BIS AI assistant chat UI
    return render_template("assistant.html")

# Chat bot endpoint processing user inquiries and slot-filling
@app.route("/api/chat", methods=["POST"])
def api_chat():
    # Extract request payload parameters
    data = request.json or {}
    message = data.get("message", "").strip()
    role = data.get("role", "MSME")
    is_clarified = data.get("is_clarified", False)

    # Validate non-empty message input
    if not message:
        return jsonify({"error": "Empty message"}), 400

    # If the user has not completed the slot-filling step, evaluate via LLM
    if not is_clarified and "R-" not in message.upper():
        questions = generate_dynamic_questions(message, role)
        if questions and len(questions) > 0:
            return jsonify({
                "action": "REQUEST_SLOT_FILL",
                "questions": questions
            })

    # Retrieve context and generate grounded answer via RAG pipeline
    result = query_knowledge_base(user_query=message, role=role)
    return jsonify({
        "action": "PROVIDE_ANSWER",
        "data": result
    })

# BIS CRS License registration verification endpoint
@app.route("/api/verify-license", methods=["GET"])
def verify_license():
    # Format and sanitize input R-number
    r_no = request.args.get("r_number", "").strip().upper()
    if not r_no:
        return jsonify({"found": False, "message": "Please specify an 8-digit R-Number (e.g., R-41014095)."}), 400

    # Query SQLite database for active CRS registration records
    conn = sqlite3.connect("standards.db")
    cur = conn.cursor()
    cur.execute("SELECT r_number, brand, manufacturer, product, is_number, status, valid_till FROM crs_licenses WHERE r_number = ?", (r_no,))
    row = cur.fetchone()
    conn.close()

    # Construct response based on database search result
    if row:
        return jsonify({
            "found": True,
            "r_number": row[0],
            "brand": row[1],
            "manufacturer": row[2],
            "product": row[3],
            "is_number": row[4],
            "status": row[5],
            "valid_till": row[6]
        })
    return jsonify({
        "found": False,
        "message": f"No active BIS CRS registration found for {r_no}. Exercise caution regarding non-conforming or counterfeit goods."
    })

if __name__ == "__main__":
    # Launch application server on localhost port 5000
    app.run(host="0.0.0.0", port=5000, debug=True)