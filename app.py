from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import sqlite3
from core.rag import query_knowledge_base, generate_dynamic_questions

app = Flask(__name__)
CORS(app)

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
        return jsonify({"error": "Empty message"}), 400

    # If the user has not completed the slot-filling step, evaluate via LLM
    if not is_clarified and "R-" not in message.upper():
        questions = generate_dynamic_questions(message, role)
        if questions and len(questions) > 0:
            return jsonify({
                "action": "REQUEST_SLOT_FILL",
                "questions": questions
            })

    # Retrieve context and generate grounded answer
    result = query_knowledge_base(user_query=message, role=role)
    return jsonify({
        "action": "PROVIDE_ANSWER",
        "data": result
    })

@app.route("/api/verify-license", methods=["GET"])
def verify_license():
    r_no = request.args.get("r_number", "").strip().upper()
    if not r_no:
        return jsonify({"found": False, "message": "Please specify an 8-digit R-Number (e.g., R-41014095)."}), 400

    conn = sqlite3.connect("standards.db")
    cur = conn.cursor()
    cur.execute("SELECT r_number, brand, manufacturer, product, is_number, status, valid_till FROM crs_licenses WHERE r_number = ?", (r_no,))
    row = cur.fetchone()
    conn.close()

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
    app.run(host="0.0.0.0", port=5000, debug=True)