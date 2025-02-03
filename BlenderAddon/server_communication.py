import bpy
import requests
import threading
import base64
import sys
import io
import zipfile
sys.executable

from .utils import *

try: 
    import socketio
    import yaml
except:
    import pip
    import sys
    modules_path = bpy.utils.user_resource("SCRIPTS", path="modules")
    sys.path.append(modules_path)
    pip.main(['install', 'python-socketio', 'websocket-client', 'pyyaml', '--target', modules_path])
        
    import socketio
    import yaml



SERVER_URL = "http://localhost:7340" 
sio = socketio.Client()

def send_single_mesh(mesh_path, prompt, inference_steps, keyframe=0):
    """Send a single mesh file to the server."""
    with open(mesh_path, 'rb') as mesh_file:
        files = {
            'mesh': ('mesh.obj', mesh_file, 'application/octet-stream'),
        }

        # Send the request
        data = {'prompt': prompt, 'inference_steps': inference_steps}
        try:
            response = requests.post(SERVER_URL + "/process", files=files, data=data)
            response.raise_for_status()

            response_data = response.json()

            # Save the texture returned from the server
            output_texture_path = os.path.join(os.path.dirname(mesh_path), f'texture_{keyframe:05d}.png')
            with open(output_texture_path, 'wb') as f:
                f.write(base64.b64decode(response_data['texture']))
            print(f"Texture saved at: {output_texture_path}")

            import_mesh_from_data(response_data['mesh'])

        except requests.exceptions.RequestException as e:
            print("Error communicating with server:", e)


        # Close file handles
        for file in files.values():
            file[1].close()

def send_minimal_data(context, mesh_dir, prompt, inference_steps, latent_tex_size, rgb_tex_size, view_matrices):
    """ Send the mesh files and config file to the server."""
    output_dir = context.scene.output_directory

    config_dict = {
        'mesh': 'mesh',
        'mesh_config_relative': True,
        'use_mesh_name' : False,
        'prompt': prompt,
        'steps': inference_steps,
        'cond_type': "depth",
        'seed' : 1,
        'mesh_scale' : 1,
        'tex_fast_preview': True,
        'view_fast_preview' :  True, 
        'latent_tex_size': latent_tex_size,
        'rgb_tex_size': rgb_tex_size
    }
    config_filename = os.path.join(mesh_dir, "config.yaml")

    # Save the config dictionary to the YAML file
    with open(config_filename, 'w') as config_file:
        yaml.dump(config_dict, config_file)

    # Open mesh files
    mesh_paths = [os.path.join(mesh_dir, f) for f in os.listdir(mesh_dir) if f.endswith(".obj")]
    obj_files = [('mesh', open(path, 'rb')) for path in mesh_paths]
    config_file = ('config', open(config_filename, 'rb'))

    # Save the view_matrices to a temporary .npy file within the mesh_dir
    view_file_path = os.path.join(mesh_dir, 'view_matrices.npy')
    np.save(view_file_path, view_matrices)

    # Open the .npy file for view_matrices
    view_file = ('view_matrices', open(view_file_path, 'rb'))

    # Send the request
    try:
        response = requests.post(SERVER_URL + "/process_sequence", files=obj_files + [config_file, view_file])
        response.raise_for_status()

        if response.status_code == 200:
            # Create the save directory if it doesn't exist
            os.makedirs(output_dir, exist_ok=True)
    
            # Unzip the contents directly from memory
            with zipfile.ZipFile(io.BytesIO(response.content), 'r') as zip_ref:
                zip_ref.extractall(output_dir)
    except requests.exceptions.RequestException as e:
        print("Error communicating with server:", e)
    finally:
        # Ensure all files are closed after the request
        config_file[1].close()
        for file in obj_files:
            file[1].close()
        view_file[1].close()
        os.remove(view_file_path)  # Clean up the temporary .npy file
            
    

def send_meshes_and_prompt(
        context, 
        mesh_path, 
        prompt="", 
        inference_steps=1, 
        latent_tex_size=512, 
        rgb_tex_size=512,
        view_matrices=None
    ):
    scene = context.scene

    def redraw_ui():
            bpy.ops.wm.redraw_timer(type='DRAW_WIN_SWAP', iterations=1)
            return None  # Returning None stops the timer
    
    def handle_progress_update(data):
        scene.is_connecting = False
        # scene = bpy.context.scene  # Ensure you get the correct scene
        progress = data.get('progress', 0)  
        scene.model_progress = progress
        
        if progress >= 1.0:
            sio.disconnect()
            scene.model_progress = 0.0
            
        bpy.app.timers.register(redraw_ui, first_interval=0.01)
    sio.on('progress_update', handle_progress_update)

    
    print("Connecting to server...")
    if sio.connected:
        sio.disconnect()
    sio.connect(SERVER_URL, wait_timeout=120)
    
    scene.is_connecting = True
    redraw_ui()
    
    """Send the mesh files to the server."""
    def task():
        try:
            send_minimal_data(bpy.context, mesh_path, prompt, inference_steps, latent_tex_size, rgb_tex_size, view_matrices)
        except Exception as e:
            print("Error processing data:", e)

    thread = threading.Thread(target=task)
    thread.start()
    

@sio.on('connect')
def handle_connect():
    print("Connected to server")
    print("Connection transport:", sio.transport)
    
@sio.on('disconnect')
def handle_disconnect():
    print("Disconnected from server")
    bpy.context.scene.model_progress = 0.0

@sio.on('connect_error')
def handle_connect_error(error):
    print("Connection error:", error)

@sio.on('error')
def handle_socket_error(error):
    print("Socket error:", error)