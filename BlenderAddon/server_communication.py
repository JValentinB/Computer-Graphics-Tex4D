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
    pip.main(['install', 'python-socketio', '--target', modules_path])
    
    try: 
        import socketio
    except:
        print("Failed to install socketio")



SERVER_URL = "http://localhost:5000" 
sio = socketio.Client()

def send_single_mesh(mesh_path, prompt, inference_steps, keyframe=0):
    """Send a single mesh file to the server via socket."""
    with open(mesh_path, 'rb') as mesh_file:
        files = {'mesh': (os.path.basename(mesh_path), mesh_file)}
        data = {
            'prompt': prompt,
            'inference_steps': inference_steps,
            'keyframe': keyframe
        }
        sio.emit('process', data=data, file=files)
        print(f"Mesh sent: {mesh_path}")

def send_meshes_and_prompt(context, mesh_path, prompt, inference_steps):
    scene = context.scene

    def handle_progress_update(data):
        print(f"Received progress update: {data['progress']}")
        scene.model_progress = data['progress']
        bpy.ops.wm.redraw_timer(type='DRAW_WIN_SWAP', iterations=1)

    # Define handler for receiving the result (processed texture)
    def handle_result(data):
        print(f"Processing complete: {data}")
        texture_data = data.get('texture', None)
        if texture_data:
            # Save the texture as a PNG file
            output_texture_path = os.path.join(os.path.dirname(mesh_path), f'texture_{data["keyframe"]:05d}.png')
            with open(output_texture_path, 'wb') as f:
                f.write(base64.b64decode(texture_data))
            print(f"Texture saved at: {output_texture_path}")
            
            # Optionally, import the mesh from the processed result (if needed)
            import_mesh_from_data(data['mesh'])

    sio.on('progress_update', handle_progress_update)
    sio.on('result', handle_result)  # Handle the final result (texture and mesh)

    print("Connecting to server...")
    sio.connect(SERVER_URL, wait_timeout=120)
    print("Connected to server")
    
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
        finally:
            sio.disconnect()

    task()