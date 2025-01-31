bl_info = {
    "name": "TEX4D",
    "blender": (4, 3, 0),
    "category": "Object",
    "description": "Exports LBS weights as numpy arrays, depth images for keyframes, and UV maps.",
}

import bpy
import os
 
from .tex_anim_node import *
from .utils import *
from .server_communication import *
from .camera_views import *


class Tex4DPanel(bpy.types.Panel):
    bl_label = "Tex4D"
    bl_idname = "VIEW3D_PT_tex4d_panel"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Tex4D"

    def draw(self, context):
        layout = self.layout
        scene = context.scene
        
        layout.prop(scene, "selected_object", text="Object")
        layout.separator()
        
        layout.prop(scene, "view_count")
        layout.prop(scene, "radius")
        layout.prop(scene, "coverage")
        layout.separator()
        
        layout.prop(scene, "num_keyframes")
        layout.prop(scene, "output_directory")
        layout.operator("object.export_data", text="Generate Input Data")
        
        if (scene.depth_progress > 0.0):
            layout.progress(text=f"Depth Images Progress: {(scene.depth_progress*100):.0f}%", factor=scene.depth_progress, type='BAR')
        
        layout.separator()
        layout.prop(scene, "inference_steps")
        layout.prop(scene, "image_sequence")
        layout.prop(scene, "prompt")
        layout.separator()
        
        
        if (scene.model_progress > 0.0):
            layout.progress(text=f"Model Progress: {(scene.model_progress*100):.0f}%", factor=scene.model_progress, type='BAR')
            layout.operator("object.cancel", text="Cancel", icon='CANCEL')
        else: 
            layout.operator("object.tex4d_start", text="Generate")

class ExportOperator(bpy.types.Operator):
    bl_idname = "object.export_data"
    bl_label = "Export Tex4D Input Data"
    
    def execute(self, context):
        scene = context.scene
        
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
        
        export_weights(self, context, output_directory)
        export_uv_maps(self, context, output_directory)
        export_depth_images(self, context, output_directory)            
        return {'FINISHED'}
    
class Tex4DStartOperator(bpy.types.Operator):
    bl_idname = "object.tex4d_start"
    bl_label = "Generate"
    
    def execute(self, context):
        scene = context.scene
        scene.model_progress = 0.0
        
        obj = context.object
        if obj is None or obj.type != 'MESH':
            self.report({'ERROR'}, "No mesh object selected.")
            return {'CANCELLED'}
        
        # Print prompt text to console
        prompt = scene.prompt
        print(f"Prompt text: {prompt}")
                
        scene.temp_dir = bpy.app.tempdir

        if(scene.image_sequence):
            mesh_dir = scene.temp_dir
            if(export_animated_mesh(obj, mesh_dir)):
                print(f"Mesh exported to {mesh_dir}")
            
                steps = scene.inference_steps
                # Send the mesh and text prompt to the server
                send_meshes_and_prompt(context, mesh_dir, prompt, steps)

        else: 
            mesh_path = os.path.join(scene.temp_dir, "mesh.obj")
        
            # Export the active mesh
            if export_active_mesh(obj, mesh_path):
                print(f"Mesh exported to {mesh_path}")
            
                steps = scene.inference_steps
                # Send the mesh and text prompt to the server
                send_meshes_and_prompt(context, mesh_path, prompt, steps)
        return {'FINISHED'}
    
class CancelOperator(bpy.types.Operator):
    bl_idname = "object.cancel"
    bl_label = "Cancel"
    
    def execute(self, context):
        scene = context.scene
        scene.model_progress = 0.0
        return {'FINISHED'}
    
    
def register():
    bpy.utils.register_class(Tex4DPanel)
    bpy.utils.register_class(ExportOperator)
    bpy.utils.register_class(Tex4DStartOperator)
    bpy.utils.register_class(CancelOperator)
    bpy.utils.register_class(CameraViewOperator)
    
    # Add property to choose an object
    bpy.types.Scene.selected_object = bpy.props.PointerProperty(
        name="Object", 
        type=bpy.types.Object  # Reference to the Object type
    )
    # Add properties to control camera views
    bpy.types.Scene.view_count = bpy.props.IntProperty(
        name="Camera Views",
        default=2,
        min=2, max=100, 
        update=update_handler
    )
    bpy.types.Scene.radius = bpy.props.FloatProperty(
        name="Camera Radius",
        default=10.0,
        min=0.0,
        update=update_handler
    )
    bpy.types.Scene.coverage = bpy.props.FloatProperty(
        name="Camera Coverage",
        default=1.0,
        min=0.0, max=1.0,
        update=update_handler
    )
    
    bpy.types.Scene.num_keyframes = bpy.props.IntProperty(
        name="Number of Keyframes",
        description="Number of keyframes to extract",
        default=5,
        min=1
    )
    bpy.types.Scene.output_directory = bpy.props.StringProperty(name="Output Directory", default="", subtype='DIR_PATH')
    
    
    # Add a property for the prompt and output directory
    bpy.types.Scene.prompt = bpy.props.StringProperty(name="Prompt", default="")
    
    bpy.types.Scene.inference_steps = bpy.props.IntProperty(
        name="Inference Steps",
        description="Number of inference steps the model should take",
        default=10,
        min=1
    )
    bpy.types.Scene.image_sequence = bpy.props.BoolProperty(
        name="Image Sequence",
        description="Wether to generate a single texture or texture sequence",
        default=False
    )
    bpy.types.Scene.temp_dir = bpy.props.StringProperty(name="Temp Dir", default="", subtype='DIR_PATH')
    
    # Depth images progress bar
    bpy.types.Scene.depth_progress = bpy.props.FloatProperty(
        name="Depth Images Progress",
        description="Progress of depth image export",
        default=0.0,
        min=0.0,
        max=1.0,
        subtype='PERCENTAGE'
    )
    # Model progress bar
    bpy.types.Scene.model_progress = bpy.props.FloatProperty(
        name="Model Progress",
        description="Progress of model export",
        default=0.0,
        min=0.0,
        max=1.0,
        subtype='PERCENTAGE'
    )
    
    # Register the custom shader node
    bpy.utils.register_class(AnimateTextureNode)
    bpy.utils.register_class(NODE_OT_AddImageInput)
    bpy.types.NODE_MT_add.append(menu_func)

def unregister():
    bpy.utils.unregister_class(Tex4DPanel)
    bpy.utils.unregister_class(ExportOperator)
    bpy.utils.unregister_class(Tex4DStartOperator)
    bpy.utils.unregister_class(CancelOperator)
    bpy.utils.unregister_class(CameraViewOperator)
    
    bpy.utils.unregister_class(AnimateTextureNode)
    bpy.utils.unregister_class(NODE_OT_AddImageInput)
    bpy.types.NODE_MT_add.remove(menu_func)
    
    # Remove the properties
    del bpy.types.Scene.view_count
    del bpy.types.Scene.radius
    
    del bpy.types.Scene.prompt
    del bpy.types.Scene.output_directory
    
    del bpy.types.Scene.num_keyframes
    del bpy.types.Scene.inference_steps
    del bpy.types.Scene.image_sequence
    del bpy.types.Scene.depth_progress
    del bpy.types.Scene.model_progress

def menu_func(self, context):
    self.layout.operator("node.add_node", text="Switch Textures").type = "AnimateTextureNode"


if __name__ == "__main__":
    register()

