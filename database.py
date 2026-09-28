"""
CROPLENS AI - SQLite Database Module
Manages:
1. User accounts by Gmail address
2. Scan & Diagnosis history persistence
3. Agricultural Notebook (Field Notes) persistence (Day 1, Day 2, etc.)
100% Offline, standard Python sqlite3, zero external cloud APIs.
"""

import os
import json
import sqlite3
import uuid
import time
import hashlib
from typing import List, Dict, Any, Optional, Tuple

DB_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))
DB_PATH = os.path.join(DB_DIR, "croplens.db")
USER_IMG_DIR = os.path.join(DB_DIR, "user_images")

os.makedirs(DB_DIR, exist_ok=True)
os.makedirs(USER_IMG_DIR, exist_ok=True)


def get_connection():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_database():
    """Initializes the SQLite database tables."""
    conn = get_connection()
    cursor = conn.cursor()

    # Users Table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS users (
        email TEXT PRIMARY KEY,
        created_at TEXT,
        last_login TEXT
    )
    """)

    # Scans / Diagnosis History Table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS scans (
        id TEXT PRIMARY KEY,
        case_id TEXT,
        user_email TEXT,
        timestamp TEXT,
        time_str TEXT,
        date_str TEXT,
        prediction TEXT,
        confidence REAL,
        stress_score REAL,
        image_path TEXT,
        image_label TEXT,
        features_json TEXT,
        probabilities_json TEXT,
        latency_ms REAL,
        FOREIGN KEY(user_email) REFERENCES users(email)
    )
    """)

    # Ensure case_id column exists if table was already created
    try:
        cursor.execute("ALTER TABLE scans ADD COLUMN case_id TEXT")
    except Exception:
        pass

    # Agricultural Notebook Entries Table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS notebook (
        id TEXT PRIMARY KEY,
        user_email TEXT,
        day_title TEXT,
        note_content TEXT,
        crop_tag TEXT,
        created_at TEXT,
        FOREIGN KEY(user_email) REFERENCES users(email)
    )
    """)

    # Ensure profile columns exist on users table
    try:
        cursor.execute("ALTER TABLE users ADD COLUMN name TEXT")
    except Exception:
        pass
    try:
        cursor.execute("ALTER TABLE users ADD COLUMN avatar_emoji TEXT")
    except Exception:
        pass
    try:
        cursor.execute("ALTER TABLE users ADD COLUMN avatar_path TEXT")
    except Exception:
        pass
    try:
        cursor.execute("ALTER TABLE users ADD COLUMN password_hash TEXT")
    except Exception:
        pass
    try:
        cursor.execute("ALTER TABLE notebook ADD COLUMN seed_name TEXT")
    except Exception:
        pass
    try:
        cursor.execute("ALTER TABLE notebook ADD COLUMN work_type TEXT")
    except Exception:
        pass

    conn.commit()
    conn.close()


# -------------------------------------------------------------
# User Authentication & Security Functions (Pure Python SHA-256)
# -------------------------------------------------------------
def hash_password(password: str) -> str:
    """Computes SHA-256 hash for secure local password storage."""
    salt = "croplens_secure_salt_2026"
    return hashlib.sha256((salt + password).encode("utf-8")).hexdigest()


def register_user(
    email: str,
    password: str,
    name: str = "",
    avatar_emoji: str = "👨‍🌾"
) -> Tuple[bool, str]:
    """
    Registers a new user account with secure password hashing.
    Returns (success_boolean, message).
    """
    clean_email = email.strip().lower()
    if not clean_email or "@" not in clean_email or "." not in clean_email:
        return False, "Please enter a valid email address."

    if not password or len(password) < 4:
        return False, "Password must be at least 4 characters long."

    display_name = name.strip() if name and name.strip() else clean_email.split("@")[0].title()
    pwd_hash = hash_password(password)
    now_str = time.strftime("%Y-%m-%d %H:%M:%S")

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT email, password_hash FROM users WHERE email = ?", (clean_email,))
    existing = cursor.fetchone()

    if existing and existing["password_hash"]:
        conn.close()
        return False, "An account with this email already exists. Please Sign In."

    if existing:
        # User existed without password (e.g., from prior scan/demo) - update password
        cursor.execute("""
            UPDATE users SET password_hash = ?, name = ?, avatar_emoji = ?, last_login = ?
            WHERE email = ?
        """, (pwd_hash, display_name, avatar_emoji, now_str, clean_email))
    else:
        cursor.execute("""
            INSERT INTO users (email, password_hash, name, avatar_emoji, created_at, last_login)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (clean_email, pwd_hash, display_name, avatar_emoji, now_str, now_str))

    conn.commit()
    conn.close()
    return True, "Account registered successfully! Welcome to CropLens AI."


def authenticate_user(email: str, password: str) -> Tuple[bool, str]:
    """
    Authenticates a user by email and password.
    Returns (success_boolean, message).
    """
    clean_email = email.strip().lower()
    if not clean_email:
        return False, "Please provide your email."

    # Allow demo bypass
    if clean_email in ["farmer@gmail.com", "farmer@facebook.com"] and (not password or password in ["farmer123", "demo"]):
        get_or_create_user(clean_email)
        return True, "Demo login successful!"

    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT email, password_hash FROM users WHERE email = ?", (clean_email,))
    user = cursor.fetchone()

    if not user:
        conn.close()
        return False, "No account found for this email. Please register via Sign Up."

    stored_hash = user["password_hash"]
    if not stored_hash:
        # If user registered earlier without password, set their password now
        pwd_hash = hash_password(password)
        now_str = time.strftime("%Y-%m-%d %H:%M:%S")
        cursor.execute("UPDATE users SET password_hash = ?, last_login = ? WHERE email = ?", (pwd_hash, now_str, clean_email))
        conn.commit()
        conn.close()
        return True, "Password configured and login successful!"

    if stored_hash != hash_password(password):
        conn.close()
        return False, "Incorrect password. Please try again."

    now_str = time.strftime("%Y-%m-%d %H:%M:%S")
    cursor.execute("UPDATE users SET last_login = ? WHERE email = ?", (now_str, clean_email))
    conn.commit()
    conn.close()
    return True, "Login successful!"



# -------------------------------------------------------------
# User Profile Functions (Name & Profile Picture Customization)
# -------------------------------------------------------------
def get_or_create_user(email: str) -> Dict[str, Any]:
    """Logs in or registers a user via Gmail."""
    email_clean = email.strip().lower()
    now_str = time.strftime("%Y-%m-%d %H:%M:%S")

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT * FROM users WHERE email = ?", (email_clean,))
    user = cursor.fetchone()

    if user:
        cursor.execute("UPDATE users SET last_login = ? WHERE email = ?", (now_str, email_clean))
    else:
        cursor.execute("INSERT INTO users (email, created_at, last_login, name, avatar_emoji) VALUES (?, ?, ?, ?, ?)",
                       (email_clean, now_str, now_str, email_clean.split("@")[0].title(), "👨‍🌾"))

    conn.commit()
    conn.close()
    return get_user_profile(email_clean)


def get_user_profile(email: str) -> Dict[str, Any]:
    """Fetches user profile details (name, email, avatar)."""
    if not email:
        return {"email": "farmer@gmail.com", "name": "Farmer Kumar", "avatar_emoji": "👨‍🌾", "avatar_path": None}

    clean_email = email.strip().lower()
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM users WHERE email = ?", (clean_email,))
    user = cursor.fetchone()
    conn.close()

    if user:
        keys = user.keys()
        name = user["name"] if ("name" in keys and user["name"]) else clean_email.split("@")[0].title()
        avatar_emoji = user["avatar_emoji"] if ("avatar_emoji" in keys and user["avatar_emoji"]) else "👨‍🌾"
        avatar_path = user["avatar_path"] if ("avatar_path" in keys and user["avatar_path"]) else None
        return {
            "email": clean_email,
            "name": name,
            "avatar_emoji": avatar_emoji,
            "avatar_path": avatar_path
        }
    return {
        "email": clean_email,
        "name": clean_email.split("@")[0].title(),
        "avatar_emoji": "👨‍🌾",
        "avatar_path": None
    }


def update_user_profile(
    email: str,
    name: str,
    avatar_emoji: str = "👨‍🌾",
    avatar_bytes: Optional[bytes] = None
) -> Dict[str, Any]:
    """Updates user display name, avatar emoji, and optional avatar image file."""
    if not email:
        return {}
    clean_email = email.strip().lower()
    get_or_create_user(clean_email)

    avatar_filepath = None
    if avatar_bytes:
        av_filename = f"avatar_{clean_email.replace('@', '_').replace('.', '_')}_{int(time.time())}.jpg"
        avatar_filepath = os.path.join(USER_IMG_DIR, av_filename)
        with open(avatar_filepath, "wb") as f:
            f.write(avatar_bytes)

    conn = get_connection()
    cursor = conn.cursor()
    if avatar_filepath:
        cursor.execute(
            "UPDATE users SET name = ?, avatar_emoji = ?, avatar_path = ? WHERE email = ?",
            (name.strip(), avatar_emoji, avatar_filepath, clean_email)
        )
    else:
        cursor.execute(
            "UPDATE users SET name = ?, avatar_emoji = ? WHERE email = ?",
            (name.strip(), avatar_emoji, clean_email)
        )
    conn.commit()
    conn.close()
    return get_user_profile(clean_email)


def delete_scan(scan_id: str):
    """Deletes a specific scan record from database."""
    if not scan_id:
        return
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM scans WHERE id = ? OR case_id = ?", (scan_id, scan_id))
    conn.commit()
    conn.close()


# -------------------------------------------------------------
# Scan History Functions
# -------------------------------------------------------------
def save_scan(
    user_email: str,
    prediction: str,
    confidence: float,
    stress_score: float,
    features: Dict[str, float],
    class_probabilities: Dict[str, float],
    image_bytes: bytes,
    image_label: str,
    latency_ms: float
) -> Dict[str, Any]:
    """Saves a diagnostic scan record and stores image on disk."""
    scan_id = str(uuid.uuid4())[:8]
    time_str = time.strftime("%I:%M %p")
    date_str = time.strftime("%d %b %Y")
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")

    # Save image to user_images directory
    img_filename = f"{scan_id}_{int(time.time())}.jpg"
    img_filepath = os.path.join(USER_IMG_DIR, img_filename)
    with open(img_filepath, "wb") as f:
        f.write(image_bytes)

    case_id = f"CASE-{time.strftime('%Y')}-{scan_id.upper()}"

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
    INSERT INTO scans (
        id, case_id, user_email, timestamp, time_str, date_str,
        prediction, confidence, stress_score, image_path,
        image_label, features_json, probabilities_json, latency_ms
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        scan_id,
        case_id,
        user_email.strip().lower(),
        timestamp,
        time_str,
        date_str,
        prediction,
        confidence,
        stress_score,
        img_filepath,
        image_label,
        json.dumps(features),
        json.dumps(class_probabilities),
        latency_ms
    ))

    conn.commit()
    conn.close()

    return {
        "id": scan_id,
        "case_id": case_id,
        "time": time_str,
        "date": date_str,
        "prediction": prediction,
        "confidence": confidence,
        "stress_score": stress_score,
        "features": features,
        "class_probabilities": class_probabilities,
        "image_path": img_filepath,
        "image_label": image_label,
        "latency_ms": latency_ms
    }


def get_user_scans(user_email: str) -> List[Dict[str, Any]]:
    """Fetches all past scans for the given Gmail address in reverse chronological order."""
    if not user_email:
        return []

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
    SELECT * FROM scans 
    WHERE user_email = ? 
    ORDER BY timestamp DESC
    """, (user_email.strip().lower(),))

    rows = cursor.fetchall()
    scans = []
    for r in rows:
        scans.append({
            "id": r["id"],
            "case_id": r["case_id"] if ("case_id" in r.keys() and r["case_id"]) else f"CASE-{r['date_str'].split()[-1] if r['date_str'] else '2026'}-{r['id'].upper()}",
            "time": r["time_str"],
            "date": r["date_str"],
            "prediction": r["prediction"],
            "confidence": r["confidence"],
            "stress_score": r["stress_score"],
            "features": json.loads(r["features_json"]) if r["features_json"] else {},
            "class_probabilities": json.loads(r["probabilities_json"]) if r["probabilities_json"] else {},
            "image_path": r["image_path"],
            "image_label": r["image_label"],
            "latency_ms": r["latency_ms"]
        })

    conn.close()
    return scans


def get_scan_by_case_id(case_or_scan_id: str) -> Optional[Dict[str, Any]]:
    """Retrieves an official certified scan record by Case ID (e.g. CASE-2026-X8F9B1) or scan ID."""
    if not case_or_scan_id:
        return None

    clean_query = str(case_or_scan_id).strip()
    raw_suffix = clean_query.split("-")[-1].lower() if "-" in clean_query else clean_query.lower()

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
    SELECT * FROM scans 
    WHERE UPPER(case_id) = ? 
       OR UPPER(id) = ? 
       OR LOWER(id) LIKE ? 
    ORDER BY timestamp DESC LIMIT 1
    """, (clean_query.upper(), clean_query.upper(), f"{raw_suffix}%"))

    r = cursor.fetchone()
    conn.close()

    if not r:
        return None

    return {
        "id": r["id"],
        "case_id": r["case_id"] if ("case_id" in r.keys() and r["case_id"]) else f"CASE-{r['date_str'].split()[-1] if r['date_str'] else '2026'}-{r['id'].upper()}",
        "user_email": r["user_email"],
        "time": r["time_str"],
        "date": r["date_str"],
        "timestamp": r["timestamp"],
        "prediction": r["prediction"],
        "confidence": r["confidence"],
        "stress_score": r["stress_score"],
        "features": json.loads(r["features_json"]) if r["features_json"] else {},
        "class_probabilities": json.loads(r["probabilities_json"]) if r["probabilities_json"] else {},
        "image_path": r["image_path"],
        "image_label": r["image_label"],
        "latency_ms": r["latency_ms"]
    }


def clear_user_scans(user_email: str):
    """Clears all scan history for the user."""
    if not user_email:
        return
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM scans WHERE user_email = ?", (user_email.strip().lower(),))
    conn.commit()
    conn.close()


# -------------------------------------------------------------
# Agricultural Notebook (Day 1, Day 2 Notes) Functions
# -------------------------------------------------------------
def get_user_notes(user_email: str) -> List[Dict[str, Any]]:
    """Retrieves all agricultural notebook entries for the user."""
    if not user_email:
        return []
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
    SELECT * FROM notebook 
    WHERE user_email = ? 
    ORDER BY created_at ASC
    """, (user_email.strip().lower(),))

    rows = cursor.fetchall()
    notes = []
    for r in rows:
        keys = r.keys()
        notes.append({
            "id": r["id"],
            "day_title": r["day_title"],
            "note_content": r["note_content"],
            "crop_tag": r["crop_tag"],
            "seed_name": r["seed_name"] if "seed_name" in keys and r["seed_name"] else "",
            "work_type": r["work_type"] if "work_type" in keys and r["work_type"] else "Farm Work",
            "created_at": r["created_at"]
        })

    conn.close()
    return notes


def add_notebook_entry(
    user_email: str,
    day_title: str,
    note_content: str,
    crop_tag: str = "General",
    seed_name: str = "",
    work_type: str = "Farm Work"
) -> Dict[str, Any]:
    """Adds a new day entry into the user's field notebook with seed and work activity tracking."""
    note_id = str(uuid.uuid4())[:8]
    created_at = time.strftime("%d %b %Y, %I:%M %p")

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
    INSERT INTO notebook (id, user_email, day_title, note_content, crop_tag, seed_name, work_type, created_at)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        note_id,
        user_email.strip().lower(),
        day_title.strip(),
        note_content.strip(),
        crop_tag.strip(),
        seed_name.strip(),
        work_type.strip(),
        created_at
    ))

    conn.commit()
    conn.close()

    return {
        "id": note_id,
        "day_title": day_title,
        "note_content": note_content,
        "crop_tag": crop_tag,
        "seed_name": seed_name,
        "work_type": work_type,
        "created_at": created_at
    }


def delete_notebook_entry(note_id: str):
    """Deletes a specific note entry."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM notebook WHERE id = ?", (note_id,))
    conn.commit()
    conn.close()


# Initialize database upon import
init_database()
