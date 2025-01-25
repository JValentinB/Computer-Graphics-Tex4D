import os 
import torch 

from flask import Flask, request, jsonify, send_file
from flask_socketio import SocketIO, emit

from werkzeug.utils import secure_filename
from process_pipeline import process_mesh_with_preloaded_models
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, DDPMScheduler

app = Flask(__name__)
socketio = SocketIO(app,
    cors_allowed_origins="*",
)

UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'outputs'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Configuration
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER


# Preload models
print("Initializing models...")
controlnet_normal = ControlNetModel.from_pretrained("lllyasviel/control_v11p_sd15_normalbae", variant="fp16", torch_dtype=torch.float16)
pipe = StableDiffusionControlNetPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5", controlnet=controlnet_normal, torch_dtype=torch.float16, image_encoder=None
)
pipe.scheduler = DDPMScheduler.from_config(pipe.scheduler.config)


# Remove unnecessary components
if "image_encoder" in pipe.components:
    del pipe.components["image_encoder"]
    del pipe.image_encoder

print("Models initialized.\n")
# _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _

@socketio.on('connect')
def handle_connect():
    print('Client connected')

@socketio.on('disconnect')
def handle_disconnect():
    print('Client disconnected')

@socketio.on('connect_error')
def handle_connect_error(error):
    print(f'Connection error: {error}')

# @socketio.on('process')
@app.route('/process', methods=['POST'])
def process(data, file):
    
    print("Received a mesh!\n")
    if 'mesh' not in file:
        emit('error', {'error': 'Mesh file missing'})
        return
   
    mesh_file = file['mesh']
    prompt = data.get('prompt', '')
    steps = int(data.get('inference_steps', 10))
    print(f"Prompt: {prompt}\nSteps: {steps}")
    
    if not prompt or prompt == '':
        emit('error', {'error': 'Prompt missing'})
        return

    if not mesh_file:
        emit('error', {'error': 'Mesh file missing'})
        return
    
    # Save uploaded file
    filename = secure_filename(mesh_file.filename)
    upload_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    mesh_file.save(upload_path)
    
    # Define output path for the texture
    output_path = os.path.join(app.config['OUTPUT_FOLDER'], f'texture.png')
    
    def send_progress_update(progress):
        print(f"___Sending progress: {progress}%")
        emit('progress_update', {'progress': progress})

    # Process the mesh using preloaded models and user-provided prompt
    try:
        print("Starting SyncMVD...")
        result = process_mesh_with_preloaded_models(pipe, upload_path, output_path, prompt, steps, send_progress_update)
        emit('result', result)  # Emit the result back to the client
    except Exception as e:
        emit('error', {'error': str(e)})
    
    # # Return the textured mesh
    # return send_file(output_path, as_attachment=True)
    


if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=5000)
