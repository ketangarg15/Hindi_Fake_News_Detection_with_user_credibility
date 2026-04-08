import sys
import os
# Add parent directory to path so we can import model_pipeline
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from flask import Flask, render_template, request, jsonify
from model_pipeline import FakeNewsPipeline

app = Flask(__name__)

# Initialize pipeline (pointing to the models directory in the root)
pipeline = FakeNewsPipeline(model_dir="../models/")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    text = data.get('text', '')
    mode = data.get('mode', 'website')
    
    user_data = None
    if mode == 'social_media':
        user_data = {
            'name': data.get('name', ''),
            'screen_name': data.get('screen_name', ''),
            'statuses_count': int(data.get('statuses_count', 0)),
            'followers_count': int(data.get('followers_count', 0)),
            'friends_count': int(data.get('friends_count', 0)),
            'location': data.get('location', ''),
            'verified': data.get('verified', False)
        }
    
    result = pipeline.predict(text, mode=mode, user_data=user_data)
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True, port=5000)
