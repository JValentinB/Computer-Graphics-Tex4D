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

def send_minimal_data(context, mesh_dir, prompt, inference_steps):
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
    'view_fast_preview' :  True
   }
    config_filename = os.path.join(mesh_dir, "config.yaml")

    # Save the config dictionary to the YAML file
    with open(config_filename, 'w') as config_file:
        yaml.dump(config_dict, config_file)

    
    # Open mesh files
    mesh_paths = [os.path.join(mesh_dir, f) for f in os.listdir(mesh_dir) if f.endswith(".obj")]
    obj_files = [('mesh', open(path, 'rb')) for path in mesh_paths]
    config_file = ('config', open(config_filename, 'rb'))

    # Send the request
    try:
        response = requests.post(SERVER_URL + "/process_sequence", files=obj_files + [config_file])
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
        for file in obj_files.values():
            file.close()
            
def send_all_data(context, mesh_dir, prompt, inference_steps):
    """ Send the mesh files, text prompt, inference steps, as well as view files and depth images to the server."""
    output_dir = context.scene.output_directory
    
    files = {}
    data = {'prompt': prompt, 'inference_steps': inference_steps}
    
    # Open mesh files
    mesh_paths = [os.path.join(mesh_dir, f) for f in os.listdir(mesh_dir) if f.endswith(".obj")]
    for mesh_path in mesh_paths:
        filename = os.path.basename(mesh_path)
        match = re.search(r"mesh_(\d+)\.obj", filename)
        if match:
            keyframe = int(match.group(1))
        else:
            print(f"Warning: Could not parse keyframe from {filename}")
            continue

        files[f"mesh_{keyframe}"] = open(mesh_path, 'rb')  # Keep file open
    
    # Open depth images
    depth_paths = [os.path.join(output_dir, f) for f in os.listdir(output_dir) if f.endswith(".png")]
    for depth_path in depth_paths:
        filename = os.path.basename(depth_path)
        match = re.search(r"view_(\d+)_keyframe_(\d+)\.png", filename)
        if match:
            view = int(match.group(1))
            keyframe = int(match.group(2))
        else:
            print(f"Warning: Could not parse keyframe from {filename}")
            continue

        files[f"depth_{view}_{keyframe}"] = open(depth_path, 'rb')
    
    # Open view matrices
    view_paths = [os.path.join(output_dir, f) for f in os.listdir(output_dir) if f.startswith("view_") and f.endswith(".npy")]
    for view_path in view_paths:
        filename = os.path.basename(view_path)
        match = re.search(r"view_(\d+)\.npy", filename)
        if match:
            view = int(match.group(1))
        else:
            print(f"Warning: Could not parse view number from {filename}")
            continue

        files[f"view_{view}"] = open(view_path, 'rb')

    # Send the request
    try:
        response = requests.post(SERVER_URL + "/process_animated", files=files, data=data)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        print("Error communicating with server:", e)
    finally:
        # Ensure all files are closed after the request
        for file in files.values():
            file.close()


    

def send_meshes_and_prompt(context, mesh_path, prompt, inference_steps):
    scene = context.scene

    def handle_progress_update(data):
        # scene = bpy.context.scene  # Ensure you get the correct scene
        progress = data.get('progress', 0)  
        scene.model_progress = progress

        # Use bpy.app.timers to schedule a redraw on the main thread
        def redraw_ui():
            bpy.ops.wm.redraw_timer(type='DRAW_WIN_SWAP', iterations=1)
            return None  # Returning None stops the timer
        
        if progress >= 1.0:
            sio.disconnect()
            scene.model_progress = 0.0
            
        bpy.app.timers.register(redraw_ui, first_interval=0.01)
    sio.on('progress_update', handle_progress_update)

    
    print("Connecting to server...")
    if sio.connected:
        sio.disconnect()
    sio.connect(SERVER_URL, wait_timeout=120)
    
    
    """Send the mesh files to the server."""
    def task():
        try:
            if bpy.context.scene.image_sequence:
                # for filename in os.listdir(mesh_path):
                #     if filename.endswith('.obj'):
                #         file_path = os.path.join(mesh_path, filename)
                #         keyframe = extract_keyframe_number(filename)
                #         send_single_mesh(file_path, prompt, inference_steps, keyframe)

                # texture_paths = [file for file in os.listdir(mesh_path) if file.endswith('.png')]
                # current_frame = bpy.context.scene.frame_current
                # total_frames = bpy.context.scene.frame_end

                # material = bpy.context.active_object.active_material
                # setup_texture_interpolation(material, texture_paths, current_frame, total_frames)
                # bpy.app.handlers.frame_change_post.append(update_switch_factor)

                send_minimal_data(bpy.context, mesh_path, prompt, inference_steps)
                #send_all_data(bpy.context, mesh_path, prompt, inference_steps)
            else:
                # Prepare the files to send
                send_single_mesh(mesh_path, prompt, inference_steps)
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