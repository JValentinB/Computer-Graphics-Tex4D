import requests
import threading
import base64

from .utils import *

SERVER_URL = "http://127.0.0.1:5000/process"  # Replace with your server endpoint

def send_single_mesh(mesh_path, prompt, inference_steps, keyframe=0):
    """Send a single mesh file to the server."""
    with open(mesh_path, 'rb') as mesh_file:
        files = {
            'mesh': ('mesh.obj', mesh_file, 'application/octet-stream'),
        }

        # Send the request
        data = {'prompt': prompt, 'inference_steps': inference_steps}
        try:
            response = requests.post(SERVER_URL, files=files, data=data)
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

def send_meshes_and_prompt(mesh_path, prompt, inference_steps):
    """Send the mesh files to the server."""
    def task():
        try:
            if(bpy.context.scene.image_sequence):
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

    # Create and start a thread

    thread = threading.Thread(target=task)
    thread.start()
