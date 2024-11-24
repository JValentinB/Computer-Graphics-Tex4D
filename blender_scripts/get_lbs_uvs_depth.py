import bpy
import numpy as np
import os
import random
from mathutils import Matrix, Vector
from math import radians

# Parameters
OUTPUT_DIR = "E:\\Master\\Semester_2\\ComputerGraphics\\Project\\output"  # Change to your desired output directory
K = 10                # Number of keyframes
V = 5                 # Number of random camera poses
DEPTH_RESOLUTION = (512, 512)  # Depth image resolution
UV_LAYOUT_PATH = os.path.join(OUTPUT_DIR, "uv_layout")
UV_ATLAS_PATH = os.path.join(OUTPUT_DIR, "uv_atlas")
LBS_WEIGHTS_PATH = os.path.join(OUTPUT_DIR, "lbs_weights.npy")
armature_name = "Armature"

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Extract K Keyframes
def get_keyframes(k):
    start_frame = bpy.context.scene.frame_start
    end_frame = bpy.context.scene.frame_end
    return np.linspace(start_frame, end_frame, k, dtype=int)

# Uniformly Sample V Random Camera Poses
def random_camera_poses(v, radius=10.0):
    poses = []
    for _ in range(v):
        phi = random.uniform(0, 2 * np.pi)
        theta = random.uniform(0, np.pi)
        x = radius * np.sin(theta) * np.cos(phi)
        y = radius * np.sin(theta) * np.sin(phi)
        z = radius * np.cos(theta)
        camera_loc = Vector((x, y, z))
        look_at = Vector((0, 0, 0))
        up = Vector((0, 0, 1))
        forward = (look_at - camera_loc).normalized()
        right = forward.cross(up).normalized()
        up = right.cross(forward)
        camera_matrix = Matrix((
            right.to_4d(),
            up.to_4d(),
            (-forward).to_4d(),
            camera_loc.to_4d(),
        )).transposed()
        poses.append(camera_matrix)
    return poses

# Create Depth Images
def render_depth_image(frame, camera_matrix, output_path):
    scene = bpy.context.scene

    # Add a camera
    bpy.ops.object.camera_add()
    camera = bpy.context.object
    camera.matrix_world = camera_matrix
    scene.camera = camera

    # Set render settings
    scene.render.engine = 'CYCLES'
    scene.render.image_settings.file_format = 'OPEN_EXR'
    scene.render.image_settings.color_mode = 'BW'
    scene.render.resolution_x = DEPTH_RESOLUTION[0]
    scene.render.resolution_y = DEPTH_RESOLUTION[1]
    scene.frame_set(frame)

    # Render depth pass
    scene.use_nodes = True
    tree = scene.node_tree
    tree.nodes.clear()
    render_layers = tree.nodes.new(type='CompositorNodeRLayers')
    depth_output = tree.nodes.new(type='CompositorNodeOutputFile')
    depth_output.base_path = output_path

    # Ensure proper connection of the depth pass node
    tree.links.new(render_layers.outputs['Depth'], depth_output.inputs['Image'])

    # Run the render
    bpy.ops.render.render(write_still=True)

    # Clean up the camera object after rendering
    bpy.data.objects.remove(camera)

# Generate UV Map
def create_uv_map_for_mesh(mesh_obj, output_path):
    bpy.context.view_layer.objects.active = mesh_obj
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.uv.smart_project(angle_limit=66, island_margin=0.03)
    bpy.ops.object.mode_set(mode='OBJECT')
    bpy.ops.uv.export_layout(filepath=output_path, size=(2048, 2048))
    
# Generate UV Atlas   
def generate_uv_atlas(meshes, output_path):
    bpy.ops.object.select_all(action='DESELECT')
    
    for mesh in meshes:
        mesh.select_set(True)

    bpy.ops.object.join()  # Join meshes into one for UV packing
    combined_obj = bpy.context.view_layer.objects.active
    bpy.ops.uv.export_layout(filepath=output_path, size=(2048, 2048))
    
# Create LBS Weights
def extract_lbs_weights(mesh_obj):
    vertex_groups = mesh_obj.vertex_groups
    num_vertices = len(mesh_obj.data.vertices)
    num_bones = len(vertex_groups)
    weights = np.zeros((num_vertices, num_bones), dtype=np.float32)
    for v in mesh_obj.data.vertices:
        for g in v.groups:
            group_index = g.group
            weight = g.weight
            weights[v.index, group_index] = weight
    weights = weights / weights.sum(axis=1, keepdims=True)  # Normalize
    return weights
    
camera_poses = random_camera_poses(V)    
keyframes = get_keyframes(K)

# Render all depth images
for frame in keyframes:
    for i, camera_matrix in enumerate(camera_poses):
        output_path = os.path.join(OUTPUT_DIR, f"depth_frame{frame}_view{i}.exr")
        render_depth_image(frame, camera_matrix, output_path)
        
        
armature = bpy.data.objects.get(armature_name)
meshes = [obj for obj in bpy.data.objects if obj.type == 'MESH' and obj.find_armature() == armature]

  
for mesh in meshes:
    uv_map_path = os.path.join(OUTPUT_DIR, f"{mesh.name}_uv_map.png")
    create_uv_map_for_mesh(mesh, uv_map_path)
    lbs_weights_path = os.path.join(OUTPUT_DIR, f"{mesh.name}_lbs_weights.npy")
    weights = extract_lbs_weights(mesh)
    np.save(lbs_weights_path, weights)
generate_uv_atlas(meshes, UV_ATLAS_PATH)


print("Script completed successfully!")