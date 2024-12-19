import bpy
import requests
import json
import os

# Configuration
SERVER_URL = "http://127.0.0.1:5000/process"  # Replace with your server endpoint

def export_active_mesh(filepath):
    """Export the active mesh as an OBJ file."""
    obj = bpy.context.active_object
    if obj is None or obj.type != 'MESH':
        print("No active mesh object found.")
        return False
    
    bpy.ops.wm.obj_export(filepath=filepath)
    return True

def send_mesh_and_prompt(mesh_path, prompt):
    """Send the mesh file and text prompt to the server."""
    with open(mesh_path, 'rb') as mesh_file:
        files = {'mesh': ('mesh.obj', mesh_file, 'application/octet-stream')}
        data = {'prompt': prompt}
        try:
            response = requests.post(SERVER_URL, files=files, data=data)
            response.raise_for_status()
            print("Response from server:", response.json())
        except requests.exceptions.RequestException as e:
            print("Error communicating with server:", e)

def main():
    # Ask user for a text prompt
    prompt = "Rubiks cube"
    if not prompt:
        print("No prompt provided. Exiting.")
        return
    
    # Temporary file path for exporting the mesh
    temp_dir = bpy.app.tempdir
    mesh_path = os.path.join(temp_dir, "active_mesh.obj")
    
    # Export the active mesh
    if export_active_mesh(mesh_path):
        print(f"Mesh exported to {mesh_path}")
        
        # Send the mesh and text prompt to the server
        send_mesh_and_prompt(mesh_path, prompt)
        
        # Clean up: delete the temporary file
        try:
            os.remove(mesh_path)
        except OSError as e:
            print("Error deleting temporary file:", e)

if __name__ == "__main__":
    main()
