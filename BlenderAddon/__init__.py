bl_info = {
    "name": "Tex4D",
    "blender": (4, 3, 0),
    "category": "Object",
    "description": "Exports LBS weights as numpy arrays, depth images for keyframes, and UV maps.",
}

import bpy
import os

from mathutils import Vector  
from .utils import *


class ExportPanel(bpy.types.Panel):
    bl_label = "Export Tools"
    bl_idname = "VIEW3D_PT_export_panel"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Export Tools"

    def draw(self, context):
        layout = self.layout
        scene = context.scene
        
        layout.prop(scene, "num_keyframes")
        layout.prop(scene, "num_views")
        layout.prop(scene, "camera_distance")
        
        layout.separator()
        layout.prop(scene, "inference_steps")
        layout.prop(scene, "prompt")
        
        layout.separator()
        layout.prop(scene, "output_directory")
        layout.operator("object.export_data", text="Export")

class ExportOperator(bpy.types.Operator):
    bl_idname = "object.export_data"
    bl_label = "Export Data"
    
    def execute(self, context):
        scene = context.scene
        obj = context.object
        if obj is None or obj.type != 'MESH':
            self.report({'ERROR'}, "No mesh object selected.")
            return {'CANCELLED'}
        
        # Get output directory from the scene settings
        output_directory = scene.output_directory
        if not output_directory:
            self.report({'ERROR'}, "Output directory is not set.")
            return {'CANCELLED'}
        
        # Normalize the path to an absolute path
        output_directory = get_absolute_path(output_directory)
        
        if not os.path.exists(output_directory):
            try:
                os.makedirs(output_directory)
            except Exception as e:
                self.report({'ERROR'}, f"Failed to create directory: {str(e)}")
                return {'CANCELLED'}
        
        # Print prompt text to console
        prompt = scene.prompt
        print(f"Prompt text: {prompt}")
        
        export_weights(self, context, output_directory)
        export_uv_maps(self, context, output_directory)
        export_depth_images(self, context, output_directory)
        
        # Temporary file path for exporting the mesh
        temp_dir = bpy.app.tempdir
        mesh_path = os.path.join(temp_dir, "active_mesh.obj")
        
        # Export the active mesh
        if export_active_mesh(obj, mesh_path):
            print(f"Mesh exported to {mesh_path}")
            
            steps = scene.inference_steps
            # Send the mesh and text prompt to the server
            send_mesh_and_prompt(obj, mesh_path, prompt, steps)
            
            # Clean up: delete the temporary file
            try:
                os.remove(mesh_path)
            except OSError as e:
                print("Error deleting temporary file:", e)
            
        return {'FINISHED'}
    
    
def register():
    bpy.utils.register_class(ExportPanel)
    bpy.utils.register_class(ExportOperator)
    
    # Add a property for the prompt and output directory
    bpy.types.Scene.prompt = bpy.props.StringProperty(name="Prompt", default="")
    bpy.types.Scene.output_directory = bpy.props.StringProperty(name="Output Directory", default="", subtype='DIR_PATH')
    
    bpy.types.Scene.num_keyframes = bpy.props.IntProperty(
        name="k",
        description="Number of keyframes to extract",
        default=1,
        min=1
    )
    bpy.types.Scene.num_views = bpy.props.IntProperty(
        name="v",
        description="Number of random camera views",
        default=1,
        min=1
    )
    bpy.types.Scene.camera_distance = bpy.props.FloatProperty(
        name="Camera Distance",
        description="Distance from the camera to the origin",
        default=10.0
    )
    
    bpy.types.Scene.inference_steps = bpy.props.IntProperty(
        name="Inference Steps",
        description="Number of inference steps the model should take",
        default=10,
        min=1
    )

def unregister():
    bpy.utils.unregister_class(ExportPanel)
    bpy.utils.unregister_class(ExportOperator)
    
    # Remove the properties
    del bpy.types.Scene.prompt
    del bpy.types.Scene.output_directory
    del bpy.types.Scene.num_keyframes
    del bpy.types.Scene.num_views
    del bpy.types.Scene.camera_distance

if __name__ == "__main__":
    register()
