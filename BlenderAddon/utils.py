import bpy
import os
import tempfile

import numpy as np
from mathutils import Vector, Matrix
import re

def log_message(message, level='INFO'):
    levels = {
        'INFO': bpy.ops.report({'INFO'}, message),
        'WARNING': bpy.ops.report({'WARNING'}, message),
        'ERROR': bpy.ops.report({'ERROR'}, message),
    }
    levels.get(level.upper(), levels['INFO'])

def get_keyframes(k):
    start_frame = bpy.context.scene.frame_start
    end_frame = bpy.context.scene.frame_end
    return np.linspace(start_frame, end_frame, k, dtype=int)
def random_camera_views(v, radius=10.0):
    views = []
    phi = (1 + np.sqrt(5)) / 2  # Golden ratio
    for i in range(v):
        z = 1 - (2 * i) / (v - 1)  # z-coordinate from top to bottom of the sphere
        theta = 2 * np.pi * i / phi  # Angle around the sphere based on golden ratio
        x = np.sqrt(1 - z**2) * np.cos(theta)
        y = np.sqrt(1 - z**2) * np.sin(theta)
        camera_loc = Vector((x * radius, y * radius, z * radius))

        # Create a camera matrix to point to the origin (0, 0, 0)
        look_at = Vector((0, 0, 0))
        forward = (look_at - camera_loc).normalized()
        
        up = Vector((0, 0, 1))  # Default up vector
        if abs(forward.dot(up)) > 0.999:  # Handle degeneracy at poles
            up = Vector((1, 0, 0))  # Choose a different up vector
            
        right = forward.cross(up).normalized()
        up = right.cross(forward)
        camera_matrix = Matrix((
            right.to_4d(),
            up.to_4d(),
            (-forward).to_4d(),
            camera_loc.to_4d(),
        )).transposed()
        views.append(camera_matrix)
    return views    
def create_camera(view, camera_idx):
    """
    Create a temporary camera at a specified view and return the camera object.
    """
    cam_data = bpy.data.cameras.new(name=f"Camera_v{camera_idx}")
    cam_obj = bpy.data.objects.new(name=f"Camera_v{camera_idx}", object_data=cam_data)
    bpy.context.collection.objects.link(cam_obj)
    cam_obj.matrix_world = view
    return cam_obj



def get_absolute_path(path):
    """ Ensure the path is absolute. Handle Blender's relative path format. """
    if path.startswith("//"): 
        blend_path = bpy.path.abspath("//") 
        return os.path.join(blend_path, path[2:]) 
    return os.path.abspath(path) 

def export_weights(self, context, output_directory):
    # Collect weights from the selected object (armature or mesh)
    obj = context.object
    if not obj or obj.type != 'MESH':
        self.report({'WARNING'}, "No mesh object selected")
        return
    
    if not obj.vertex_groups:
        self.report({'WARNING'}, "Object has no vertex groups")
        return
    
    weights = []
    for v in obj.data.vertices:
        vg_weights = [0] * len(obj.vertex_groups)
        for g in v.groups:
            vg_weights[g.group] = g.weight
        weights.append(vg_weights)
    
    weights_array = np.array(weights)
    weights_file_path = os.path.join(output_directory, "LBS_weights.npy")
    np.save(weights_file_path, weights_array)
    print(f"LBS weights exported to {weights_file_path}...")

def export_uv_maps(self, context, output_directory):
    obj = context.object
    if not obj or obj.type != 'MESH':
        self.report({'WARNING'}, "No mesh object selected")
        return
    
    # Make sure the object has UV maps
    if not obj.data.uv_layers:
        self.report({'WARNING'}, "Object has no UV maps")
        return
    
    uv_maps = {}
    for uv_layer in obj.data.uv_layers:
        uv_coords = []
        for face in obj.data.polygons:
            for loop_index in face.loop_indices:
                uv_coords.append(uv_layer.data[loop_index].uv)
        uv_maps[uv_layer.name] = np.array(uv_coords)
    
    # Save UV maps to .npy files
    for uv_name, uv_coords in uv_maps.items():
        uv_file_path = os.path.join(output_directory, f"uv_map_{uv_name}.npy")
        np.save(uv_file_path, uv_coords)
        print(f"UV map '{uv_name}' exported to {uv_file_path}...")

def export_depth_images(self, context, output_directory):
    scene = context.scene
    obj = context.object
    scene.depth_progress = 0.0
    bpy.ops.wm.redraw_timer(type='DRAW_WIN_SWAP', iterations=1)

    # Ensure there is a selected object
    if not obj:
        self.report({'ERROR'}, "No object selected.")
        return

    # Define view offsets (relative to the object)
    num_camera_views = context.scene.num_views
    distance = context.scene.camera_distance
    views = random_camera_views(num_camera_views, radius=distance)

    # Export depth images for each view
    num_keyframes = context.scene.num_keyframes
    keyframes = get_keyframes(num_keyframes)

    current_render = 0
    total_renders = len(views) * len(keyframes)
    for v, view_matrix in enumerate(views):
        camera = create_camera(view_matrix, v)

        path = os.path.join(output_directory, f"view_{v}.npy")
        np.save(path, view_matrix)

        for keyframe in keyframes:
            # Update progress
            current_render += 1
            scene.depth_progress = current_render / total_renders
            bpy.ops.wm.redraw_timer(type='DRAW_WIN_SWAP', iterations=1)

            scene.frame_set(keyframe)
            render_depth(self, camera, keyframe, v, output_directory)

    scene.depth_progress = 0.0
    bpy.ops.wm.redraw_timer(type='DRAW_WIN_SWAP', iterations=1)
            
def render_depth(self, camera, keyframe, view_number, output_directory):
    scene = bpy.context.scene
    scene.view_layers[0].use_pass_z = True # Ensure depth pass is enabled

    # Configure compositor nodes
    scene.use_nodes = True
    tree = scene.node_tree
    tree.nodes.clear()

    # Add render layers node
    render_layer = tree.nodes.new('CompositorNodeRLayers')

    # Add file output node
    output_node = tree.nodes.new('CompositorNodeOutputFile')
    # output_node.format.file_format = 'OPEN_EXR'
    output_node.format.file_format = 'PNG'  # Use PNG format
    output_node.format.color_mode = 'BW'  # Grayscale output
    output_node.format.color_depth = '16'  # Use 16-bit precision
    
    # get camera distance from origin
    camera_distance = camera.location.length
    # Create map value node to normalize depth
    map_range = tree.nodes.new(type='CompositorNodeMapRange')
    map_range.inputs[1].default_value = 0                   # Set the min depth value
    map_range.inputs[2].default_value = camera_distance * 2 # Set the max depth value
    map_range.inputs[3].default_value = 1                   # Set the min output value
    map_range.inputs[4].default_value = 0                   # Set the max output value
    map_range.use_clamp = True

    # Set the output path for the depth image
    output_node.base_path = output_directory
    # output_node.file_slots[0].path = f"keyframe_{keyframe}_view_{view_number}.exr"
    output_node.file_slots[0].path = f"view_{view_number}_keyframe_"

    # Link the depth pass to the file output node
    try:
        tree.links.new(render_layer.outputs['Depth'], map_range.inputs[0])
        tree.links.new(map_range.outputs[0], output_node.inputs[0])
    except KeyError:
        self.report({'ERROR'}, "Depth output not found in Render Layers. Ensure depth pass is enabled.")
        return

    if camera and camera.type == 'CAMERA':
        scene.camera = camera  # Set camera for the scene
        bpy.context.view_layer.objects.active = camera  # Make sure the camera is the active object
    else:
        self.report({'ERROR'}, "Invalid camera object provided.")
        return
    
    # Render the scene and save the file
    try:
        bpy.ops.render.render(write_still=True)
        print(f"Depth image exported for keyframe {keyframe}, view {view_number}")
    except Exception as e:
        self.report({'ERROR'}, f"Error rendering depth image: {str(e)}")
        return


def export_active_mesh(obj, filepath):
    """Export the active mesh as an OBJ file."""
    bpy.context.view_layer.objects.active = obj
    if obj is None or obj.type != 'MESH':
        print("No active mesh object found.")
        return False
    
    bpy.ops.wm.obj_export(filepath=filepath)
    return True

def export_animated_mesh(obj, mesh_dir):
    num_keyframes = bpy.context.scene.num_keyframes
    K = get_keyframes(num_keyframes)
    for keyframe in K:
        output_path = os.path.join(mesh_dir, f"mesh_{keyframe:05d}.obj")
        bpy.context.scene.frame_set(keyframe)
        bpy.ops.object.select_all(action='DESELECT')
        obj.select_set(True)
        bpy.context.view_layer.objects.active = obj
        bpy.ops.object.duplicate()
        duplicate = bpy.context.view_layer.objects.active

        bpy.ops.object.convert(target='MESH')

        #bpy.ops.export_scene.obj(
            #filepath=output_path,
            #use_selection=True,
            #use_mesh_modifiers=True
        #)
        bpy.ops.wm.obj_export(filepath=output_path)

        bpy.data.objects.remove(duplicate)
    

    
def import_mesh_from_data(mesh_data):
    bpy.ops.preferences.addon_enable(module="io_scene_obj")
    
    # Save the mesh data to a temporary file
    with tempfile.NamedTemporaryFile(suffix=".obj", delete=False) as temp_mesh_file:
        temp_mesh_file.write(mesh_data.encode('utf-8'))  # Assuming the mesh is a string
        temp_mesh_path = temp_mesh_file.name

    # Import the mesh into Blender
    bpy.ops.import_scene.obj(filepath=temp_mesh_path)

    # Get the imported object (assumes the last imported object is your target)
    imported_obj = bpy.context.selected_objects[-1]

    # Set the object as active
    bpy.context.view_layer.objects.active = imported_obj
    imported_obj.select_set(True)
    
def extract_keyframe_number(filename):
    """
    Extract the keyframe number from an .obj filename.
    Assumes the filename contains a pattern like '_XXXXX.obj'.
    """
    match = re.search(r'_(\d+)\.obj$', filename)
    if match:
        return int(match.group(1))  # Return the keyframe number as an integer
    return None  # Return None if no match is found 
    
def setup_texture_interpolation(material, texture_paths, current_frame, total_frames):
    """
    Setup the custom switch shader node for texture interpolation.
    """
    # Ensure the material has a node tree
    if not material.use_nodes:
        material.use_nodes = True
    nodes = material.node_tree.nodes
    links = material.node_tree.links

    # Add the custom switch shader node
    custom_node = nodes.new("CustomSwitchShaderNode")
    custom_node.location = (200, 200)

    # Add a Principled BSDF node if not already present
    bsdf_node = None
    for node in nodes:
        if node.type == 'BSDF_PRINCIPLED':
            bsdf_node = node
            break
    if not bsdf_node:
        bsdf_node = nodes.new("ShaderNodeBsdfPrincipled")
        bsdf_node.location = (400, 200)

    # Connect the custom node to the Principled BSDF
    links.new(custom_node.outputs["Interpolated Output"], bsdf_node.inputs["Base Color"])

    # Dynamically add image inputs to the custom node
    for idx, texture_path in enumerate(texture_paths):
        bpy.ops.node.add_image_input({'node': custom_node})
        img_texture = bpy.data.images.load(texture_path)
        input_socket = custom_node.node_tree.interface.items_tree[f"Image_{idx + 1} Texture"]
        input_socket.default_value = img_texture

    # Connect switch factor to an input
    switch_factor_node = nodes.new("ShaderNodeValue")
    switch_factor_node.location = (0, 200)
    switch_factor_node.outputs[0].default_value = current_frame / total_frames
    links.new(switch_factor_node.outputs[0], custom_node.inputs["Switch Factor"])

def update_switch_factor(scene):
    material = bpy.context.active_object.active_material
    nodes = material.node_tree.nodes
    node = bpy.context.active_object.active_material.node_tree.nodes.get('CustomSwitchShaderNode')
    if node:
        switch_factor_node = [n for n in nodes if n.name == "Switch Factor"][0]
        current_frame = scene.frame_current
        total_frames = scene.frame_end
        switch_factor_node.outputs[0].default_value = current_frame / total_frames
        


# def apply_texture_to_active_object(obj, texture_path):
#     """Apply the RGB texture to the active object in Blender."""
#     if obj is None or obj.type != 'MESH':
#         print("No active mesh object selected.")
#         return

#     # Create a new material with nodes enabled
#     mat = bpy.data.materials.new(name="TexturedMaterial")
#     mat.use_nodes = True
#     bsdf = mat.node_tree.nodes["Principled BSDF"]

#     # Add an image texture node
#     tex_image = mat.node_tree.nodes.new('ShaderNodeTexImage')
#     tex_image.image = bpy.data.images.load(texture_path)

#     # Connect the image texture to the material
#     mat.node_tree.links.new(bsdf.inputs['Base Color'], tex_image.outputs['Color'])

#     # Assign the material to the object
#     if obj.data.materials:
#         obj.data.materials[0] = mat
#     else:
#         obj.data.materials.append(mat)

#     print("Texture applied to the active object.")
    
# def apply_uvs_to_mesh(obj, uv_coords):
#     """Apply UV coordinates to a Blender mesh."""
#     mesh = obj.data
    
#     # Create new UV layer if it doesn't exist
#     if not mesh.uv_layers:
#         mesh.uv_layers.new(name='UVMap')
    
#     uv_layer = mesh.uv_layers.active.data
    
#     # Convert UV coordinates to Blender's format and apply them
#     for poly in mesh.polygons:
#         for loop_idx in poly.loop_indices:
#             vertex_idx = mesh.loops[loop_idx].vertex_index
#             uv_layer[loop_idx].uv = uv_coords[vertex_idx]
    
#     # Mark mesh as updated
#     mesh.update()