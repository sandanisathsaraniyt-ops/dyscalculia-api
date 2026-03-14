from typing import final

from flask import Flask, request, jsonify
from flask_cors import CORS
from datetime import datetime
import hashlib
import re
import random
import urllib.parse
import os
import time
import logging
import joblib
import pandas as pd
import numpy as np
from collections import Counter

from database import get_db_connection, save_activity_result
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = Flask(__name__)
CORS(app)
# ================= PERFORMANCE LOGGING =================

@app.before_request
def start_timer():
    request.start_time = time.time()


@app.after_request
def log_request_time(response):
    if hasattr(request, "start_time"):
        duration = time.time() - request.start_time
        logging.info(f"{request.path} | {request.method} | {duration:.4f} seconds")
        print(f"{request.path} loaded in {duration:.4f} seconds")
    return response

# ==========================================================
# ================= LOAD ML MODELS =========================
# ==========================================================
# ==========================================================
# ================= LOAD NEW DYSCALCULIA MODEL (V9) =======
# ==========================================================

DYS_DIR = os.path.join(BASE_DIR, "dyscalculia_v9_pkl")

dys_feature_cols = joblib.load(os.path.join(DYS_DIR, "feature_cols (1).pkl"))
dys_preprocessor = joblib.load(os.path.join(DYS_DIR, "preprocessor.pkl"))
dys_inv_map = joblib.load(os.path.join(DYS_DIR, "inv_label_map (1).pkl"))
dys_model = joblib.load(os.path.join(DYS_DIR, "best_model.pkl"))

print(" New Dyscalculia v9 model loaded")

# ==========================================================
# ================= LOAD ATTENTION MODEL ===================
# ==========================================================
# ==========================================================
# ================= LOAD NEW ATTENTION MODEL ===============
# ==========================================================

ATT_DIR = os.path.join(BASE_DIR, "attention_simple_models_output")

att_preprocessor = joblib.load(os.path.join(ATT_DIR, "preprocessor.pkl"))
att_inv_map = joblib.load(os.path.join(ATT_DIR, "inv_label_map (1).pkl"))
att_model = joblib.load(os.path.join(ATT_DIR, "best_model.pkl"))
att_raw_features = joblib.load(os.path.join(ATT_DIR, "raw_features.pkl"))

print(" New Attention model loaded")




# ==========================================================
# ================= LOAD MEMORY MODEL ======================
# ==========================================================

# ==========================================================
# ================= LOAD NEW MEMORY MODEL ==================
# ==========================================================

MEM_DIR = os.path.join(BASE_DIR, "memory_simple_models_output")

mem_scaler = joblib.load(os.path.join(MEM_DIR, "scaler (1).pkl"))
mem_inv_map = joblib.load(os.path.join(MEM_DIR, "inv_label_map (1).pkl"))
mem_model = joblib.load(os.path.join(MEM_DIR, "best_model.pkl"))
mem_raw_features = joblib.load(os.path.join(MEM_DIR, "raw_features.pkl"))

print(" New Memory model loaded")

# ==========================================================
# ================= PASSWORD ===============================
# ==========================================================

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def is_valid_password(password):
    if len(password) < 8: return False
    if not re.search(r"[A-Z]", password): return False
    if not re.search(r"[a-z]", password): return False
    if not re.search(r"[0-9]", password): return False
    if not re.search(r"[!@#$%^&*(),.?\":{}|<>]", password): return False
    return True

def is_valid_email(email):
    return re.match(r"^[a-zA-Z0-9._%+-]+@gmail\.com$", email)

# ==========================================================
# ================= USERNAME HELPERS =======================
# ==========================================================

def suggest_usernames(base, cursor):
    suggestions = []
    while len(suggestions) < 5:
        suffix = random.randint(100, 9999)
        new_name = f"{base}{suffix}"
        cursor.execute("SELECT 1 FROM responsible_adult WHERE username = ?", (new_name,))
        if not cursor.fetchone():
            suggestions.append(new_name)
    return suggestions

def suggest_child_names(base, cursor, adult_id):
    suggestions = []
    while len(suggestions) < 5:
        suffix = random.randint(100, 9999)
        new_name = f"{base}{suffix}"
        cursor.execute(
            "SELECT 1 FROM child WHERE adult_id = ? AND child_name = ?",
            (adult_id, new_name)
        )
        if not cursor.fetchone():
            suggestions.append(new_name)
    return suggestions

# ==========================================================
# ================= SIGNUP ================================
# ==========================================================

@app.route("/signup", methods=["POST"])
def signup():
    data = request.json
    email = data.get("email")
    username = data.get("username")
    password = data.get("password")

    if not all([email, username, password]):
        return jsonify({"error": "All fields required"}), 400

    if not is_valid_email(email):
        return jsonify({"error": "Invalid email format. Use @gmail.com only"}), 400

    if not is_valid_password(password):
        return jsonify({"error": "Weak password"}), 400

    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT 1 FROM responsible_adult WHERE email = ?", (email,))
    if cursor.fetchone():
        conn.close()
        return jsonify({"error": "Email already exists"}), 409

    cursor.execute("SELECT 1 FROM responsible_adult WHERE username = ?", (username,))
    if cursor.fetchone():
        suggestions = suggest_usernames(username, cursor)
        conn.close()
        return jsonify({"error": "Username exists", "suggestions": suggestions}), 409

    cursor.execute("""
        INSERT INTO responsible_adult (email, username, password_hash)
        VALUES (?, ?, ?)
    """, (email, username, hash_password(password)))

    conn.commit()
    conn.close()

    return jsonify({"message": "Signup successful"}), 201

# ==========================================================
# ================= LOGIN =================================
# ==========================================================

@app.route("/login", methods=["POST"])
def login():
    data = request.json
    email = data.get("email")
    password = data.get("password")

    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute(
        "SELECT password_hash FROM responsible_adult WHERE email = ?",
        (email,)
    )
    user = cursor.fetchone()
    conn.close()

    if not user or user["password_hash"] != hash_password(password):
        return jsonify({"error": "Invalid login"}), 401

    return jsonify({"message": "Login successful"}), 200

# ================= RESET PASSWORD =================
@app.route("/reset-password", methods=["POST"])
def reset_password():
    data = request.json
    email = data.get("email")
    new_password = data.get("new_password")

    if not email or not new_password:
        return jsonify({"error": "Missing fields"}), 400

    # Validate using the same function
    if not is_valid_password(new_password):
        return jsonify({
            "error": "Weak password. Must include uppercase, lowercase, number, symbol, and be 8+ chars"
        }), 400

    conn = get_db_connection()
    cursor = conn.cursor()

    # Check user exists
    cursor.execute(
        "SELECT adult_id FROM responsible_adult WHERE email = ?",
        (email,)
    )
    user = cursor.fetchone()

    if not user:
        conn.close()
        return jsonify({"error": "User not found"}), 404

    # Update password
    cursor.execute(
        "UPDATE responsible_adult SET password_hash = ? WHERE email = ?",
        (hash_password(new_password), email)
    )

    conn.commit()
    conn.close()

    return jsonify({"message": "Password updated successfully"}), 200


# ==========================================================
# ================= ADD CHILD ==============================
# ==========================================================

@app.route("/add-child", methods=["POST"])
def add_child():
    data = request.json
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute(
        "SELECT adult_id FROM responsible_adult WHERE email = ?",
        (data["email"],)
    )
    adult = cursor.fetchone()

    if not adult:
        conn.close()
        return jsonify({"error": "Adult not found"}), 404

    adult_id = adult["adult_id"]

    cursor.execute(
        "SELECT 1 FROM child WHERE adult_id = ? AND child_name = ?",
        (adult_id, data["name"])
    )

    if cursor.fetchone():
        suggestions = suggest_child_names(data["name"], cursor, adult_id)
        conn.close()
        return jsonify({
            "exists": True,
            "message": "Child name already exists",
            "suggestions": suggestions
        }), 200

    cursor.execute("""
        INSERT INTO child (adult_id, child_name, gender, age, grade)
        VALUES (?, ?, ?, ?, ?)
    """, (adult_id, data["name"], data["gender"], data["age"], data["grade"]))

    conn.commit()
    conn.close()

    return jsonify({"message": "Child added"}), 201

# ==========================================================
# ================= CHILD LIST =============================
# ==========================================================

@app.route("/children/<path:parent_email>", methods=["GET"])
def get_children(parent_email):

    parent_email = urllib.parse.unquote(parent_email)

    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute(
        "SELECT adult_id FROM responsible_adult WHERE email = ?",
        (parent_email,)
    )
    adult = cursor.fetchone()

    if not adult:
        conn.close()
        return jsonify([]), 200

    cursor.execute(
        "SELECT child_name FROM child WHERE adult_id = ?",
        (adult["adult_id"],)
    )

    children = [row["child_name"] for row in cursor.fetchall()]
    conn.close()

    return jsonify(children), 200
@app.route("/child/<child_name>", methods=["GET"])
def get_child_details(child_name):
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT child_name, age, grade, gender
        FROM child
        WHERE child_name = ?
    """, (child_name,))
    
    child = cursor.fetchone()
    conn.close()

    if not child:
        return jsonify({"error": "Child not found"}), 404

    return jsonify({
        "name": child["child_name"],
        "age": child["age"],
        "grade": child["grade"],
        "gender": child["gender"]
    }), 200

@app.route("/update-child", methods=["PUT"])
def update_child():
    data = request.json
    old_name = data.get("old_name")
    new_name = data.get("name")
    age = data.get("age")
    grade = data.get("grade")
    gender = data.get("gender")

    conn = get_db_connection()
    cursor = conn.cursor()

    # Check child exists
    cursor.execute(
        "SELECT child_id FROM child WHERE child_name = ?",
        (old_name,)
    )
    child = cursor.fetchone()

    if not child:
        conn.close()
        return jsonify({"error": "Child not found"}), 404

    # Update the child
    cursor.execute("""
        UPDATE child
        SET child_name = ?, age = ?, grade = ?, gender = ?
        WHERE child_name = ?
    """, (new_name, age, grade, gender, old_name))

    conn.commit()
    conn.close()

    return jsonify({"message": "Child updated"}), 200


# ================= SAVE ACTIVITY =================
@app.route('/save-activity', methods=['POST'])
def save_activity():
    data = request.json

    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute(
        "SELECT child_id FROM child WHERE child_name = ?",
        (data["child_name"],)
    )
    child = cursor.fetchone()

    if not child:
        conn.close()
        return jsonify({"error": "Child not found"}), 404

    activity_id = data["activity_id"]
    given = data.get("given_answer")

    # ---------- CORRECT ANSWERS ----------
    correct_answers = {
        1: "5",
        2: "<",
        3: "7",
        4: "නැත",
        5: "7",
        6: "3",
        7: "1",
        8: "1",
        9: "-",
        11: "1",
        13: "1",  
    }

    score = 0
    is_correct = 0

    # ---------- ACTIVITY 10 ----------
    if activity_id == 10:
        correct_set = {"0", "8"}
        if not given:
            score = 0
        elif set(given.split(",")) == correct_set:
            score = 1
            is_correct = 1
        else:
            score = -1

    # ---------- ACTIVITY 12 ----------
    elif activity_id == 12:
        if not given:
            score = 0
        elif given == "3":
            score = 1
            is_correct = 1
        else:
            score = -1

    # ---------- NORMAL ----------
    else:
        if not given:
            score = 0
        elif given == correct_answers.get(activity_id):
            score = 1
            is_correct = 1
        else:
            score = -1

    save_activity_result(
        child["child_id"],
        activity_id,
        given,
        is_correct,
        score,
        1,
        data["time_taken_seconds"]
    )

    conn.close()
    return jsonify({"message": "Saved"}), 200

# ==========================================================
# ================= BUILD ML FEATURES ======================
# ==========================================================



# ==========================================================
# ================= DYS CALCULIA ML ========================
# ==========================================================

def ml_model_dyscalculia(activity_rows):

    if len(activity_rows) < 9:
        return "Not Enough Data"

    sample = {}

    # Build feature dictionary from DB rows
    for a in activity_rows:
        aid = a["activity_id"]
        sample[f"activity{aid}_score"] = a["score"]
        sample[f"activity{aid}_time"] = a["time_taken_seconds"] or 0

    df_in = pd.DataFrame([sample])

    # Ensure all required features exist
    for col in dys_feature_cols:
        if col not in df_in.columns:
            df_in[col] = 0

    # Transform using saved preprocessor
    X_in = dys_preprocessor.transform(df_in[dys_feature_cols].astype(float))

    probs = dys_model.predict_proba(X_in)
    pred_class = np.argmax(probs, axis=1)[0]

    label = dys_inv_map[pred_class]
    confidence = float(probs.max())

    # Map to UI labels
    label_map = {
        "high": "High Risk",
        "mild": "Mild Risk",
        "no": "No Risk"
    }

    return label_map.get(label, label)
# ==========================================================
# ================= ATTENTION ML ===========================
# ==========================================================

# ==========================================================
# ================= NEW ATTENTION ML =======================
# ==========================================================

def ml_model_attention(activity_rows):

    if len(activity_rows) < 2:
        return "Not Enough Data"

    sample = {}

    # Build input features from DB rows
    for a in activity_rows:
        aid = a["activity_id"]
        sample[f"activity{aid}_score"] = a["score"]
        sample[f"activity{aid}_time_sec"] = a["time_taken_seconds"] or 0

    df_in = pd.DataFrame([sample])

    # Ensure all required raw features exist
    for col in att_raw_features:
        if col not in df_in.columns:
            df_in[col] = 0

    # Transform using saved pipeline
    X_new = att_preprocessor.transform(df_in[att_raw_features].astype(float))

    probs = att_model.predict_proba(X_new) if hasattr(att_model, "predict_proba") else None
    preds = att_model.predict(X_new)

    pred_class = preds[0]
    label = att_inv_map[pred_class]

    # Map to UI labels
    label_map = {
        "yes": "Attention Impairment",
        "no": "Normal Attention"
    }

    return label_map.get(label, label)
# ==========================================================
# ================= MEMORY ML ==============================
# ==========================================================

# ==========================================================
# ================= NEW MEMORY ML ==========================
# ==========================================================

def ml_model_memory(activity_rows):

    if len(activity_rows) < 2:
        return "Not Enough Data"

    sample = {}

    # Build input features from DB rows
    for a in activity_rows:
        aid = a["activity_id"]
        sample[f"activity{aid}_score"] = a["score"]
        sample[f"activity{aid}_time_taken"] = a["time_taken_seconds"] or 0

    df_in = pd.DataFrame([sample])

    # Ensure all required raw features exist
    for col in mem_raw_features:
        if col not in df_in.columns:
            df_in[col] = 0

    # Scale features (your new pipeline uses scaler directly)
    X_new = mem_scaler.transform(df_in[mem_raw_features].astype(float))

    probs = mem_model.predict_proba(X_new) if hasattr(mem_model, "predict_proba") else None
    preds = mem_model.predict(X_new)

    pred_class = preds[0]
    label = mem_inv_map[pred_class]

    # Map to UI labels
    label_map = {
        "yes": "Memory Impairment",
        "no": "Normal Memory"
    }

    return label_map.get(label, label)



    
    

# ==========================================================
# ================= VIEW REPORT ============================
# ==========================================================

@app.route("/view-report/<child_name>", methods=["GET"])
def view_report(child_name):

    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT child_id, child_name, age, gender
        FROM child WHERE child_name = ?
    """,(child_name,))
    
    child = cursor.fetchone()

    if not child:
        conn.close()
        return jsonify({"error": "Child not found"}), 404

    cursor.execute("""
SELECT activity_id, given_answer, is_correct, score, time_taken_seconds
FROM activity_results
WHERE child_id = ?
GROUP BY activity_id
ORDER BY activity_id
""",(child["child_id"],))

    activities = cursor.fetchall()
    conn.close()

    dys_data = [a for a in activities if 1 <= a["activity_id"] <= 9]
    att_data = [a for a in activities if 10 <= a["activity_id"] <= 11]
    mem_data = [a for a in activities if 12 <= a["activity_id"] <= 13]



    dys_result = ml_model_dyscalculia(dys_data)
    attention_result = ml_model_attention(att_data)
    memory_result = ml_model_memory(mem_data)

    print("ML Prediction:", dys_result)
    print("ML Attention:", attention_result)
    print("ML Memory:", memory_result)

    return jsonify({
        "child":dict(child),
        "activities":[dict(a) for a in activities],
        "dyscalculia_risk":dys_result,
        "attention_result":attention_result,
        "memory_result":memory_result
    }),200

# ==========================================================
# ================= TEST ===================================
# ==========================================================

@app.route("/")
def home():
    return "API running with ML",200

if __name__ == "__main__":
    app.run(host="0.0.0.0",port=5000)