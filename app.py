from flask import Flask, render_template, request
import os
import random
import torch
import timm
import numpy as np
from PIL import Image
from torchvision import transforms
from torch import nn
import cv2

app = Flask(__name__)

# Define directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "video")
PROCESSED_FRAMES_FOLDER = os.path.join(BASE_DIR, "test3")
HEATMAP_FOLDER = os.path.join(BASE_DIR, "gradcam")
MODEL_PATH = os.path.join(BASE_DIR, "model.pt")

# Ensure directories exist
for folder in [UPLOAD_FOLDER, PROCESSED_FRAMES_FOLDER, HEATMAP_FOLDER]:
    os.makedirs(folder, exist_ok=True)

# Model class
class MyModel(nn.Module):
    def __init__(self, model_path):
        super(MyModel, self).__init__()
        self.model = timm.create_model('resnet50', pretrained=True)
        self.model.fc = nn.Linear(self.model.fc.in_features, 2)
        self.load_model(model_path)

    def load_model(self, model_path):
        checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
        self.model.load_state_dict(checkpoint, strict=False)
        self.model.eval()

    def forward(self, x):
        return self.model(x)

# Image transformation
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Extract frames from video
def extract_n_frames(video_path, output_folder, num_frames=20):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_indices = [int(i * total_frames / num_frames) for i in range(num_frames)]
    frame_paths = []
    
    for i, frame_idx in enumerate(frame_indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if ret:
            frame_path = os.path.join(output_folder, f"frame_{i:02d}.jpg")
            cv2.imwrite(frame_path, frame)
            frame_paths.append(frame_path)
    
    cap.release()
    return frame_paths

# Process video for DeepFake detection
def process_video(video_path, model_path, num_frames=10):
    model = MyModel(model_path)
    class_names = ['real', 'deepfake']
    
    frame_paths = extract_n_frames(video_path, PROCESSED_FRAMES_FOLDER, num_frames)
    predictions = []

    for frame_path in frame_paths:
        image = Image.open(frame_path).convert('RGB')
        image_tensor = transform(image).unsqueeze(0)
        with torch.no_grad():
            output = model(image_tensor)
            _, predicted_class = torch.max(output, 1)
            predictions.append(class_names[predicted_class.item()])

    selected_frames = random.sample(frame_paths, min(5, len(frame_paths)))

    return max(set(predictions), key=predictions.count), {label: predictions.count(label) for label in set(predictions)}

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        video_file = request.files['video']
        if video_file:
            video_path = os.path.join(UPLOAD_FOLDER, video_file.filename)
            video_file.save(video_path)
            final_prediction, prediction_counts = process_video(video_path, MODEL_PATH)
            
            # Cleanup after processing
            os.remove(video_path)
            # Clear processed frames and heatmaps
            clear_folder(PROCESSED_FRAMES_FOLDER)
            clear_folder(HEATMAP_FOLDER)

            return render_template('result.html', prediction=final_prediction, counts=prediction_counts)
    
    return render_template('upload.html')

# Function to clear folder content
def clear_folder(folder):
    for file in os.listdir(folder):
        file_path = os.path.join(folder, file)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
        except Exception as e:
            print(f"Failed to delete {file_path}: {e}")

if __name__ == '__main__':
    app.run(debug=True)
