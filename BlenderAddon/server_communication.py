import bpy
import requests
import threading
import base64
import sys
sys.executable

from .utils import *

try: 
    import socketio
except:
    import pip
    import sys
    modules_path = bpy.utils.user_resource("SCRIPTS", path="modules")
    sys.path.append(modules_path)
    pip.main(['install', 'python-socketio', 'websocket-client', '--target', modules_path])
        
    import socketio



SERVER_URL = "http://localhost:7341" 
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
    sio.connect(SERVER_URL, wait_timeout=120)
    
    
    """Send the mesh files to the server."""
    def task():
        try:
            if bpy.context.scene.image_sequence:
                for filename in os.listdir(mesh_path):
                    if filename.endswith('.obj'):
                        file_path = os.path.join(mesh_path, filename)
                        keyframe = extract_keyframe_number(filename)
                        send_single_mesh(file_path, prompt, inference_steps, keyframe)

                texture_paths = [file for file in os.listdir(mesh_path) if file.endswith('.png')]
                current_frame = bpy.context.scene.frame_current
                total_frames = bpy.context.scene.frame_end

                material = bpy.context.active_object.active_material
                setup_texture_interpolation(material, texture_paths, current_frame, total_frames)
                bpy.app.handlers.frame_change_post.append(update_switch_factor)
            else:
                # Prepare the files to send
                send_single_mesh(mesh_path, prompt, inference_steps)
        except Exception as e:
            print("Error processing data:", e)

    thread = threading.Thread(target=task)
    thread.start()
    
# handle the result
# @sio.on('result')
# def handle_result(data):
#     print(f"Processing complete: {data}")
#     texture_data = data.get('texture', None)
#     if texture_data:
#         # Save the texture as a PNG file
#         output_texture_path = f'texture_{data["keyframe"]:05d}.png'
#         with open(output_texture_path, 'wb') as f:
#             f.write(base64.b64decode(texture_data))
#         print(f"Texture saved at: {output_texture_path}")
        
#         # Optionally, import the mesh from the processed result (if needed)
#         import_mesh_from_data(data['mesh'])
    
#     print("Disconnecting from server...")
#     sio.disconnect()
    

@sio.on('connect')
def handle_connect():
    print("Connected to server")
    print("Connection transport:", sio.transport)
    
@sio.on('disconnect')
def handle_disconnect():
    print("Disconnected from server")

@sio.on('connect_error')
def handle_connect_error(error):
    print("Connection error:", error)

@sio.on('error')
def handle_socket_error(error):
    print("Socket error:", error)