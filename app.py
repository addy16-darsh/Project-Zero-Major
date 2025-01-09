import os
import cv2
import torch
import timm
import numpy as np
import random
from glob import glob
from torch import nn
from torchvision import transforms
from PIL import Image
from flask import Flask, request, render_template, send_from_directory

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

# Clear folder contents
def clear_folder(folder):
    for file in os.listdir(folder):
        file_path = os.path.join(folder, file)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
        except Exception as e:
            print(f"Failed to delete {file_path}: {e}")

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

# Generate Grad-CAM heatmaps
def generate_gradcam(model, img_tensor, target_layer):
    gradients, activations = None, None

    def backward_hook(module, grad_input, grad_output):
        nonlocal gradients
        gradients = grad_output[0]

    def forward_hook(module, input, output):
        nonlocal activations
        activations = output

    handle_f = target_layer.register_forward_hook(forward_hook)
    handle_b = target_layer.register_full_backward_hook(backward_hook)
    
    output = model(img_tensor)
    class_idx = torch.argmax(output).item()
    model.zero_grad()
    output[0, class_idx].backward()
    
    pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])
    for i in range(activations.shape[1]):
        activations[:, i, :, :] *= pooled_gradients[i]
    
    heatmap = torch.mean(activations, dim=1).squeeze().cpu().detach().numpy()
    heatmap = np.maximum(heatmap, 0) / (np.max(heatmap) + 1e-10)
    return np.uint8(255 * heatmap)

# Overlay heatmap on image
def overlay_heatmap(image, heatmap):
    heatmap_resized = cv2.resize(heatmap, (image.size[0], image.size[1]))
    heatmap_colored = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(np.array(image), 0.6, heatmap_colored, 0.4, 0)
    return overlay

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
    for i, frame_path in enumerate(selected_frames):
        image = Image.open(frame_path).convert('RGB')
        image_tensor = transform(image).unsqueeze(0)
        heatmap = generate_gradcam(model.model, image_tensor, model.model.layer4)
        overlay = overlay_heatmap(image, heatmap)
        heatmap_path = os.path.join(HEATMAP_FOLDER, f"heatmap_{i}.jpg")
        cv2.imwrite(heatmap_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

    return max(set(predictions), key=predictions.count), {label: predictions.count(label) for label in set(predictions)}

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        video_file = request.files['video']
        if video_file:
            video_path = os.path.join(UPLOAD_FOLDER, video_file.filename)
            video_file.save(video_path)
            final_prediction, prediction_counts = process_video(video_path, MODEL_PATH)
            
            os.remove(video_path)  
            clear_folder(PROCESSED_FRAMES_FOLDER)
            clear_folder(HEATMAP_FOLDER)

            return render_template('result.html', prediction=final_prediction, counts=prediction_counts)
    
    return render_template('upload.html')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)


