import json
from flask import Blueprint, jsonify
from database.db import get_db
from routes.auth import token_required

history_bp = Blueprint('history', __name__, url_prefix='/api')


# GET ALL HISTORY
@history_bp.route('/history', methods=['GET'])
@token_required
def get_history(current_user):
    db = get_db()
    analyses = db.execute(
        'SELECT * FROM analyses WHERE user_id = ? ORDER BY created_at DESC',
        (current_user['id'],)
    ).fetchall()
    db.close()

    result = []
    for a in analyses:
        result.append({
            "id": a['id'],
            "soil_type": a['soil_type'],
            "confidence": a['confidence'],
            "properties": json.loads(a['properties']),
            "recommendations": json.loads(a['recommendations']),
            "crop_type": a['crop_type'],
            "notes": a['notes'],
            "image_url": f"/api/uploads/{a['image_path']}",
            "created_at": a['created_at']
        })

    return jsonify(result)


# DELETE ANALYSIS
@history_bp.route('/history/<int:analysis_id>', methods=['DELETE'])
@token_required
def delete_analysis(current_user, analysis_id):
    db = get_db()
    db.execute(
        'DELETE FROM analyses WHERE id = ? AND user_id = ?',
        (analysis_id, current_user['id'])
    )
    db.commit()
    db.close()
    return jsonify({"message": "Analysis deleted"})


# GET STATS
@history_bp.route('/stats', methods=['GET'])
@token_required
def get_stats(current_user):
    db = get_db()

    total = db.execute(
        'SELECT COUNT(*) as count FROM analyses WHERE user_id = ?',
        (current_user['id'],)
    ).fetchone()['count']

    soil_distribution = db.execute(
        '''SELECT soil_type, COUNT(*) as count
           FROM analyses WHERE user_id = ?
           GROUP BY soil_type ORDER BY count DESC''',
        (current_user['id'],)
    ).fetchall()

    recent = db.execute(
        '''SELECT soil_type, confidence, created_at
           FROM analyses WHERE user_id = ?
           ORDER BY created_at DESC LIMIT 10''',
        (current_user['id'],)
    ).fetchall()

    avg_confidence = db.execute(
        'SELECT AVG(confidence) as avg FROM analyses WHERE user_id = ?',
        (current_user['id'],)
    ).fetchone()['avg']

    db.close()

    return jsonify({
        "total_analyses": total,
        "average_confidence": round(avg_confidence or 0, 4),
        "soil_distribution": [
            {"soil_type": s['soil_type'], "count": s['count']}
            for s in soil_distribution
        ],
        "recent_analyses": [
            {"soil_type": r['soil_type'], "confidence": r['confidence'], "date": r['created_at']}
            for r in recent
        ]
    })
