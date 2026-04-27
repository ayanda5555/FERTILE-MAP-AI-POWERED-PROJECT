import jwt
import datetime
from functools import wraps
from flask import Blueprint, request, jsonify
from flask_bcrypt import Bcrypt
from config import Config
from database.db import get_db

auth_bp = Blueprint('auth', __name__, url_prefix='/api/auth')
bcrypt = Bcrypt()


# This function protects routes — checks if user is logged in
def token_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        token = None

        if 'Authorization' in request.headers:
            auth_header = request.headers['Authorization']
            if auth_header.startswith('Bearer '):
                token = auth_header.split(' ')[1]

        if not token:
            return jsonify({"error": "Login required"}), 401

        try:
            data = jwt.decode(token, Config.SECRET_KEY, algorithms=["HS256"])
            db = get_db()
            current_user = db.execute(
                'SELECT * FROM users WHERE id = ?', (data['user_id'],)
            ).fetchone()
            db.close()

            if not current_user:
                return jsonify({"error": "User not found"}), 401

        except jwt.ExpiredSignatureError:
            return jsonify({"error": "Token expired. Please login again."}), 401
        except jwt.InvalidTokenError:
            return jsonify({"error": "Invalid token"}), 401

        return f(current_user, *args, **kwargs)
    return decorated


# REGISTER
@auth_bp.route('/register', methods=['POST'])
def register():
    data = request.get_json()

    if not data or not data.get('email') or not data.get('password'):
        return jsonify({"error": "Email and password required"}), 400

    if len(data['password']) < 6:
        return jsonify({"error": "Password must be at least 6 characters"}), 400

    db = get_db()

    existing = db.execute(
        'SELECT id FROM users WHERE email = ?', (data['email'],)
    ).fetchone()

    if existing:
        db.close()
        return jsonify({"error": "Email already registered"}), 409

    password_hash = bcrypt.generate_password_hash(data['password']).decode('utf-8')

    db.execute(
        'INSERT INTO users (email, password_hash, full_name, farm_name) VALUES (?, ?, ?, ?)',
        (data['email'], password_hash, data.get('full_name', ''), data.get('farm_name', ''))
    )
    db.commit()
    db.close()

    return jsonify({"message": "Registration successful!"}), 201


# LOGIN
@auth_bp.route('/login', methods=['POST'])
def login():
    data = request.get_json()

    if not data or not data.get('email') or not data.get('password'):
        return jsonify({"error": "Email and password required"}), 400

    db = get_db()
    user = db.execute(
        'SELECT * FROM users WHERE email = ?', (data['email'],)
    ).fetchone()
    db.close()

    if not user or not bcrypt.check_password_hash(user['password_hash'], data['password']):
        return jsonify({"error": "Invalid email or password"}), 401

    token = jwt.encode(
        {
            'user_id': user['id'],
            'exp': datetime.datetime.utcnow() + datetime.timedelta(days=30)
        },
        Config.SECRET_KEY,
        algorithm="HS256"
    )

    return jsonify({
        "message": "Login successful!",
        "token": token,
        "user": {
            "id": user['id'],
            "email": user['email'],
            "full_name": user['full_name'],
            "farm_name": user['farm_name']
        }
    })


# GET PROFILE
@auth_bp.route('/profile', methods=['GET'])
@token_required
def get_profile(current_user):
    return jsonify({
        "id": current_user['id'],
        "email": current_user['email'],
        "full_name": current_user['full_name'],
        "farm_name": current_user['farm_name'],
        "created_at": current_user['created_at']
    })


# UPDATE PROFILE
@auth_bp.route('/profile', methods=['PUT'])
@token_required
def update_profile(current_user):
    data = request.get_json()
    db = get_db()
    db.execute(
        'UPDATE users SET full_name=?, farm_name=? WHERE id=?',
        (data.get('full_name', ''), data.get('farm_name', ''), current_user['id'])
    )
    db.commit()
    db.close()
    return jsonify({"message": "Profile updated!"})
