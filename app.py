import os 
import torch
import io
import zipfile
import time
import uuid
import yaml
import shutil
import numpy as np

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
controlnet = ControlNetModel.from_pretrained("lllyasviel/control_v11f1p_sd15_depth", variant="fp16", torch_dtype=torch.float16)
pipe = StableDiffusionControlNetPipeline.from_pretrained(
	"runwayml/stable-diffusion-v1-5", controlnet=controlnet, torch_dtype=torch.float16
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
def process():
    print("Received a mesh!\n")
    if 'mesh' not in request.files:
        return jsonify({'error': 'Mesh file missing'}), 400
   
    mesh_file = request.files['mesh']
    prompt = request.form.get('prompt', '') 
    steps = int(request.form.get('inference_steps', 10))
    print(f"Prompt: {prompt}\nSteps: {steps}")
    
    if not prompt or prompt == '':
        return jsonify({'error': 'Prompt missing'}), 400
   
    if not mesh_file:
        return jsonify({'error': 'Missing mesh file'}), 400
    
    # Save uploaded file
    filename = secure_filename(mesh_file.filename)
    upload_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    mesh_file.save(upload_path)
    
    # Define output path for the texture
    output_path = os.path.join(app.config['OUTPUT_FOLDER'], f'texture.png')
    
    def send_progress_update(progress):
        print(f"___Sending progress: {progress}%")
        socketio.emit('progress_update', {'progress': progress})

    # Process the mesh using preloaded models and user-provided prompt
    try:
        print("Starting SyncMVD...")
        process_mesh_with_preloaded_models(pipe, upload_path, output_path, prompt, steps, send_progress_update)
        
        results_dir = os.path.join(os.path.dirname(output_path), 'results')
    
        if not os.path.exists(results_dir):
            return {"error": "Results directory not found"}, 404
    
        files_to_zip = [f for f in os.listdir(results_dir) if os.path.splitext(f)[1] in{".obj", ".mtl", ".png"}]
    
        if not files_to_zip:
            return {"error": "No valid files found in results directory"}, 404
    
        zip_buffer = io.BytesIO()
    
        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
            for filename in files_to_zip:
                file_path = os.path.join(results_dir, filename)
                zip_file.write(file_path, filename)
    
        zip_buffer.seek(0)
        return send_file(zip_buffer, mimetype="application/zip", as_attachment=True, download_name="results.zip")
    
        #return jsonify(result)

        # Simulate progress update
        # for i in range(101):
        #     time.sleep(0.1)
        #     send_progress_update(i / 100)
            
        # return jsonify({'result': 'success'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    
    # # Return the textured mesh
    # return send_file(output_path, as_attachment=True)
    
@app.route('/process_animated', methods=['POST'])
def process_animated():
    print("Received an animated mesh!\n")

    # Extract prompt and inference steps
    prompt = request.form.get("prompt", "No prompt provided")
    inference_steps = int(request.form.get("inference_steps", 1))

    # Initialize storage for different file types
    meshes = {}
    depths = {}
    views = {}

    # Process uploaded files
    for key, file in request.files.items():
        print(f"Processing file: {key}")
        if key.startswith("mesh_"):
            keyframe = int(key.split("_")[1])  # Extract keyframe
            meshes[keyframe] = file.read()  # Read mesh content as bytes
        
        elif key.startswith("depth_"):
            parts = key.split("_")  # Format: depth_{view}_{keyframe}
            view, keyframe = int(parts[1]), int(parts[2])
            depths[(view, keyframe)] = file.read()  # Read depth image as bytes
        
        elif key.startswith("view_"):
            view = int(key.split("_")[1])  # Extract view number
            views[view] = np.load(file)  # Load .npy file directly into an array

    print(f"View matrices: {views}\n")

    return jsonify({
        "result": "success",
        "mesh_count": len(meshes),
        "depth_count": len(depths),
        "view_count": len(views),
    })
    
@app.route('/process_sequence', methods=['POST'])
def process_sequence():
    print("Received meshes!\n")
    print(request)
    print(request.files)
    temp_dir = os.path.join(app.config['UPLOAD_FOLDER'], str(uuid.uuid4()))
    os.makedirs(temp_dir)

    # Check if the request contains files
    if 'config' not in request.files or 'mesh' not in request.files:
        return jsonify({"error": "Missing config or mesh in the request"}), 400
    
    # Get the config YAML file
    config_file = request.files['config']
    if not config_file.filename.endswith('.yaml'):
        return jsonify({"error": "Config file must be a YAML file"}), 400
    config_path = os.path.join(temp_dir, 'config.yaml')
    config_file.save(config_path)
    
    # Get the OBJ files
    obj_files = request.files.getlist('mesh')
    if not obj_files:
        return jsonify({"error": "No OBJ files provided"}), 400
    
    # Create a subdirectory for the OBJ files
    obj_folder = os.path.join(temp_dir, 'meshes')
    if not os.path.exists(obj_folder):
        os.makedirs(obj_folder)
    
    # Save the OBJ files
    for obj_file in obj_files:
        if obj_file.filename.endswith('.obj'):
            obj_path = os.path.join(obj_folder, obj_file.filename)
            obj_file.save(obj_path)
        else:
            return jsonify({"error": f"File {obj_file.filename} is not a valid .obj file"}), 400

    # Load the YAML file
    try:
        with open(config_path, 'r') as yaml_file:
            config_data = yaml.safe_load(yaml_file)
    except yaml.YAMLError as e:
        return jsonify({"error": f"Error parsing YAML file: {str(e)}"}), 400

    # Define output path for the texture
    output_path = os.path.join(app.config['OUTPUT_FOLDER'], f'texture.png')
    
    def send_progress_update(progress):
        print(f"___Sending progress: {progress * 100}%")
        socketio.emit('progress_update', {'progress': progress})

    try:
        print("Starting SyncMVD...")
        prompt = config_data.get('prompt', 'Default prompt')
        steps = config_data.get('steps', 20)
        process_mesh_with_preloaded_models(pipe, obj_folder, output_path, prompt, steps, send_progress_update)
        
        results_dir = os.path.join(os.path.dirname(output_path), 'results')
    
        if not os.path.exists(results_dir):
            return {"error": "Results directory not found"}, 404
    
        files_to_zip = [f for f in os.listdir(results_dir) if os.path.splitext(f)[1] in{".obj", ".mtl", ".png"}]
    
        if not files_to_zip:
            return {"error": "No valid files found in results directory"}, 404
    
        zip_buffer = io.BytesIO()
    
        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
            for filename in files_to_zip:
                file_path = os.path.join(results_dir, filename)
                zip_file.write(file_path, filename)
    
        zip_buffer.seek(0)
        return send_file(zip_buffer, mimetype="application/zip", as_attachment=True, download_name="results.zip")
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

    finally:
        send_progress_update(1)
        # Delete the temporary directory and its contents after processing is complete
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
    


if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=7340)
