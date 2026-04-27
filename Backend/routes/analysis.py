import os
import json
import datetime
from flask import Blueprint, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
from config import Config
from database.db import get_db
from routes.auth import token_required

analysis_bp = Blueprint('analysis', __name__, url_prefix='/api')


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in Config.ALLOWED_EXTENSIONS


# ANALYZE SOIL
@analysis_bp.route('/analyze', methods=['POST'])
@token_required
def analyze_soil(current_user):
    from services.soil_analyzer import analyzer
    from services.fertilizer import get_recommendations

    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files['image']

    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400

    if not allowed_file(file.filename):
        return jsonify({"error": "Invalid file type. Use JPG, PNG, or WEBP"}), 400

    # Save file
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = secure_filename(f"user{current_user['id']}_{timestamp}_{file.filename}")
    filepath = os.path.join(Config.UPLOAD_FOLDER, filename)
    file.save(filepath)

    crop_type = request.form.get('crop_type', 'general')

    # Run AI
    try:
        prediction = analyzer.predict(filepath)
    except Exception as e:
        return jsonify({"error": f"Analysis failed: {str(e)}"}), 500

    recommendations = get_recommendations(prediction['soil_type'], crop_type)

    # Save to database
    db = get_db()
    db.execute(
        '''INSERT INTO analyses
           (user_id, image_path, soil_type, confidence, properties, recommendations, crop_type)
           VALUES (?, ?, ?, ?, ?, ?, ?)''',
        (
            current_user['id'], filename, prediction['soil_type'],
            prediction['confidence'], json.dumps(prediction['properties']),
            json.dumps(recommendations), crop_type
        )
    )
    db.commit()
    db.close()

    return jsonify({
        "prediction": prediction,
        "recommendations": recommendations,
        "image_url": f"/api/uploads/{filename}"
    })


# SERVE UPLOADED IMAGES
@analysis_bp.route('/uploads/<filename>')
def serve_upload(filename):
    return send_from_directory(Config.UPLOAD_FOLDER, filename)
