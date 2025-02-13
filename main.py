from flask import Flask, request, jsonify, send_file, send_from_directory, Response
from flask_cors import CORS
from werkzeug.utils import secure_filename
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload, MediaIoBaseDownload
from google.oauth2.service_account import Credentials
import boto3
import os
import io
import logging
import json
from PIL import Image
import concurrent.futures
import threading
from queue import Queue
import numpy as np
from scipy.fftpack import dct
from functools import lru_cache
import time
import math
from typing import List, Dict, Any

# Initialize Flask app
app = Flask(_name_)
CORS(app)

STATIC_FOLDER = 'static'
if not os.path.exists(STATIC_FOLDER):
    os.makedirs(STATIC_FOLDER)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(_name_)

# Constants
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
SCOPES = ['https://www.googleapis.com/auth/drive', 'https://www.googleapis.com/auth/drive.readonly']
MAX_IMAGE_SIZE = 15 * 1024 * 1024  # 15MB
REKOGNITION_MAX_SIZE = 5 * 1024 * 1024  # 5MB AWS limit
BATCH_SIZE = 5  # Size of each batch
NUM_PARALLEL_BATCHES = 4  # Number of batches to process in parallel
CACHE_TTL = 3600

# Initialize cache
class ImageCache:
    def _init_(self):
        self.cache = {}
        self.lock = threading.Lock()

    def get(self, key):
        with self.lock:
            if key in self.cache:
                data, timestamp = self.cache[key]
                if time.time() - timestamp < CACHE_TTL:
                    return data
                else:
                    del self.cache[key]
            return None

    def set(self, key, value):
        with self.lock:
            self.cache[key] = (value, time.time())

image_cache = ImageCache()

# Thread-local storage
thread_local = threading.local()

class OptimizedImageProcessor:
    def _init_(self, max_workers=4):
        self.max_workers = max_workers
        self.process_queue = Queue()
        self.results = []
        self.thread_pool = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)
        self.processed_hashes = set()

    @staticmethod
    def optimize_image(img_bytes, target_size=(800, 800), quality=85):
        try:
            img = Image.open(io.BytesIO(img_bytes))

            if img.mode in ('RGBA', 'LA'):
                background = Image.new('RGB', img.size, (255, 255, 255))
                background.paste(img, mask=img.split()[-1])
                img = background

            img.thumbnail(target_size, Image.LANCZOS)
            
            output = io.BytesIO()
            img.save(output, format='JPEG', quality=quality, optimize=True, progressive=True)
            return output.getvalue()
        except Exception as e:
            raise Exception(f"Image optimization failed: {str(e)}")

    @staticmethod
    @lru_cache(maxsize=1000)
    def get_image_hash(image_bytes):
        img = Image.open(io.BytesIO(image_bytes))
        img = img.convert('L').resize((8, 8), Image.LANCZOS)
        pixels = np.array(img.getdata(), dtype=np.float64).reshape((8, 8))
        dct_transformed = np.abs(dct(dct(pixels, axis=0), axis=1))
        return hash(dct_transformed.tobytes())

# Routes
@app.route('/')
def home():
    return send_file('index.html')

@app.route('/static/<path:path>')
def send_static(path):
    return send_from_directory('static', path)

@app.route('/set-folder-id', methods=['POST'])
def set_folder_id():
    global DRIVE_FOLDER_ID
    data = request.json
    DRIVE_FOLDER_ID = data.get('folderId')
    try:
        folder_metadata = get_drive_service().files().get(fileId=DRIVE_FOLDER_ID, fields="name").execute()
        return jsonify({'success': True, 'folderName': folder_metadata.get('name')})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/search-face', methods=['POST'])
def search_face():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']
    if file.filename == '' or not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type'}), 400

    try:
        image_bytes = file.read()
        if len(image_bytes) > MAX_IMAGE_SIZE:
            return jsonify({'error': 'Image size must be less than 15MB'}), 400

        return Response(
            stream_results(process_image_stream(image_bytes)),
            mimetype='application/x-ndjson'
        )

    except Exception as e:
        logger.error(f"Error in search_face: {e}")
        return jsonify({'error': str(e)}), 500


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Initialize services
try:
    credentials = Credentials.from_service_account_file('credentials.json', scopes=SCOPES)
    drive_service = build('drive', 'v3', credentials=credentials)
    
    aws_creds = json.load(open('credentials.json'))['aws']
    rekognition_client = boto3.client('rekognition',
        aws_access_key_id=aws_creds['aws_access_key_id'],
        aws_secret_access_key=aws_creds['aws_secret_access_key'],
        region_name=aws_creds['region']
    )
    
    DRIVE_FOLDER_ID = None
    
except Exception as e:
    logger.error(f"Error during initialization: {e}")
    raise

if _name_ == '_main_':
    app.run(host='0.0.0.0', port=5000, debug=False)
