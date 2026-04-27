from flask import Blueprint, send_from_directory
import os
from config import Config

frontend_bp = Blueprint('frontend', __name__)

FRONTEND_DIR = Config.FRONTEND_DIR

@frontend_bp.route('/')
def serve_index():
    return send_from_directory(FRONTEND_DIR, 'index.html')

@frontend_bp.route('/pages/<path:filename>')
def serve_pages(filename):
    return send_from_directory(os.path.join(FRONTEND_DIR, 'pages'), filename)

@frontend_bp.route('/css/<path:filename>')
def serve_css(filename):
    return send_from_directory(os.path.join(FRONTEND_DIR, 'css'), filename)

@frontend_bp.route('/js/<path:filename>')
def serve_js(filename):
    return send_from_directory(os.path.join(FRONTEND_DIR, 'js'), filename)

@frontend_bp.route('/assets/<path:filename>')
def serve_assets(filename):
    return send_from_directory(os.path.join(FRONTEND_DIR, 'assets'), filename)
