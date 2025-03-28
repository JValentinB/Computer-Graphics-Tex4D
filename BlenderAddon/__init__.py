bl_info = {
    "name": "TEX4D",
    "blender": (4, 3, 0),
    "category": "Object",
    "description": "Generate 3D Textures based on a text prompt and animated mesh sequence.",
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
        layout.separator()

        box1 = layout.box()  # Create a boxed section
        box1.prop(scene, "view_count")
        box1.prop(scene, "distance")
        # box1.prop(scene, "scale")

        col = box1.column(align=True)  # Ensures rows are tightly packed
        col.prop(scene, "coverage_top")
        col.prop(scene, "coverage_bottom")
        box1.separator()

        box1.prop(scene, "num_keyframes")

        # if scene.depth_progress > 0.0:
        #     box1.progress(text=f"Depth Images Progress: {(scene.depth_progress * 100):.0f}%", factor=scene.depth_progress, type='BAR')
        # else:
        #     box1.operator("object.export_data", text="Generate Input Data")

        layout.separator()
        layout.separator()

        box2 = layout.box()  # Create a boxed section
        box2.prop(scene, "inference_steps")
        box2.prop(scene, "latent_tex_size")
        box2.prop(scene, "rgb_tex_size")
        col = box2.column(align=True)  # Create a column to reduce spacing
        col.label(text="Prompt")
        col.prop(scene, "prompt", text="")
        box2.prop(scene, "output_directory")
        if (scene.model_progress > 0.0):
            box2.progress(text=f"Model Progress: {(scene.model_progress*100):.0f}%", factor=scene.model_progress, type='BAR')
            box2.operator("object.cancel", text="Cancel", icon='CANCEL')
        elif scene.is_connecting:
            box2.operator("object.tex4d_start", text="Connecting...")
        else: 
            box2.operator("object.tex4d_start", text="Generate")
        

        layout.separator()
        layout.separator()
        layout.operator("object.apply_texture", text="Apply generated texture")

class ExportOperator(bpy.types.Operator):
    bl_idname = "object.export_data"
    bl_label = "Export Tex4D Input Data"
    bl_description = "Generate and export depth images and view matrices for Tex4D."
    
    def execute(self, context):
        print("Exporting data...")
        scene = context.scene
        
        # Get output directory from the scene settings
        output_directory = scene.output_directory
        if not output_directory:
            self.report({'ERROR'}, "Output directory is not set.")
            return {'CANCELLED'}
        
        # Normalize the path to an absolute path
        output_directory = get_absolute_path(output_directory)
        # delete all files from the output directory
        for file in os.listdir(output_directory):
            os.remove(os.path.join(output_directory, file))
        
        if not os.path.exists(output_directory):
            try:
                os.makedirs(output_directory)
            except Exception as e:
                self.report({'ERROR'}, f"Failed to create directory: {str(e)}")
                return {'CANCELLED'}
        
        export_weights(self, context, output_directory)
        export_uv_maps(self, context, output_directory)
        export_depth_images(self, context, output_directory)
        scene.is_input_generated = True        
        return {'FINISHED'}
    
class Tex4DStartOperator(bpy.types.Operator):
    bl_idname = "object.tex4d_start"
    bl_label = "Generate"
    bl_description = "Send all data and prompt to the server."
    
    def execute(self, context):
        print("Sending data and prompt...")
        scene = context.scene
        scene.model_progress = 0.0
        
        # if not scene.is_input_generated:
        #     self.report({'ERROR'}, "Input data not generated. Please generate input data first.")
        #     return {'CANCELLED'}
        
        obj = scene.selected_object if scene.selected_object else context.view_layer.objects.active
        if obj is None or obj.type != 'MESH':
            self.report({'ERROR'}, "No mesh object selected.")
            return {'CANCELLED'}
        
        view_matrices = get_view_matrices(self, context)
        
        # Print prompt text to console
        prompt = scene.prompt
        print(f"Prompt text: {prompt}")
                
        scene.temp_dir = bpy.app.tempdir

        mesh_dir = scene.temp_dir
        export_animated_mesh(obj, mesh_dir)
        print(f"Mesh exported to {mesh_dir}")
    
        steps = scene.inference_steps
        # Send the mesh and text prompt to the server
        send_meshes_and_prompt(
            context, 
            mesh_dir, 
            prompt, 
            steps, 
            scene.latent_tex_size, 
            scene.rgb_tex_size, 
            view_matrices
        )
        return {'FINISHED'}
    
class ApplyTextureOperator(bpy.types.Operator):
    bl_idname = "object.apply_texture"
    bl_label = "Apply generated texture"
    bl_description = "Apply the generated Tex4D texture, if available"
    
    def execute(self, context):
        scene = context.scene

        file_exists = True

        for i in range(scene.num_keyframes):
             if not os.path.exists(os.path.join(scene.output_directory, f"textured_{i:02}.png")):
                 file_exists = False

        if not scene.is_texture_generated and not file_exists:
            self.report({'ERROR'}, "Texture has not been generated yet.")
            return {'CANCELLED'}
        
        material = create_animated_material()
        obj = scene.selected_object if scene.selected_object else context.view_layer.objects.active
        if obj and obj.type == 'MESH':
            if obj.data.materials:
                obj.data.materials[0] = material
            else:
                obj.data.materials.append(material)
        else:
            self.report({'ERROR'}, "No suitable object selected.")
            return {'CANCELLED'}
        
        scene.is_input_generated = False      
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
    bpy.utils.register_class(ApplyTextureOperator)
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
    bpy.types.Scene.distance = bpy.props.FloatProperty(
        name="Camera Distance",
        default=10.0,
        min=0.0,
        update=update_handler
    )
    bpy.types.Scene.scale = bpy.props.FloatProperty(
        name="Camera Scale",
        default=1.0,
        min=0.0,
        update=update_handler
    )
    bpy.types.Scene.coverage_top = bpy.props.FloatProperty(
        name="Camera Coverage Top",
        default=1.0,
        min=0.0, max=1.0,
        update=update_handler
    )
    bpy.types.Scene.coverage_bottom = bpy.props.FloatProperty(
        name="Camera Coverage Bottom",
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
    bpy.types.Scene.is_input_generated = bpy.props.BoolProperty(name="Is Input Generated", default=False)
    bpy.types.Scene.is_texture_generated = bpy.props.BoolProperty(name="Is Texture Generated", default=False)
    
    # Add a property for the prompt and output directory
    bpy.types.Scene.prompt = bpy.props.StringProperty(name="Prompt", default="")
    
    bpy.types.Scene.inference_steps = bpy.props.IntProperty(
        name="Inference Steps",
        description="Number of inference steps the model should take",
        default=10,
        min=1
    )
    bpy.types.Scene.latent_tex_size = bpy.props.IntProperty(
        name="Latent Texture Size",
        description="Size of the latent texture of the model. Smaller size saves VRAM",
        default=256,
        min=32, max=2048
    )
    bpy.types.Scene.rgb_tex_size = bpy.props.IntProperty(
        name="RGB Texture Size",
        description="Size of the resulting RGB texture.",
        default=512,
        min=64, max=2048
    )

    bpy.types.Scene.temp_dir = bpy.props.StringProperty(name="Temp Dir", default="", subtype='DIR_PATH')
    



    # Depth images progress bar
    # bpy.types.Scene.depth_progress = bpy.props.FloatProperty(
    #     name="Depth Images Progress",
    #     description="Progress of depth image export",
    #     default=0.0,
    #     min=0.0,
    #     max=1.0,
    #     subtype='PERCENTAGE'
    # )
    # Model progress bar
    bpy.types.Scene.is_connecting = bpy.props.BoolProperty(name="Is Connecting", default=False)
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
    bpy.utils.unregister_class(ApplyTextureOperator)
    bpy.utils.unregister_class(CancelOperator)
    bpy.utils.unregister_class(CameraViewOperator)
    
    bpy.utils.unregister_class(AnimateTextureNode)
    bpy.utils.unregister_class(NODE_OT_AddImageInput)
    bpy.types.NODE_MT_add.remove(menu_func)
    
    # Remove the properties
    del bpy.types.Scene.view_count
    del bpy.types.Scene.distance
    del bpy.types.Scene.scale
    del bpy.types.Scene.coverage_top
    del bpy.types.Scene.coverage_bottom
    
    del bpy.types.Scene.prompt
    del bpy.types.Scene.output_directory
    
    del bpy.types.Scene.num_keyframes
    del bpy.types.Scene.inference_steps
    del bpy.types.Scene.latent_tex_size
    del bpy.types.Scene.rgb_tex_size

    # del bpy.types.Scene.depth_progress
    del bpy.types.Scene.model_progress
    del bpy.types.Scene.is_input_generated
    del bpy.types.Scene.is_texture_generated

def menu_func(self, context):
    self.layout.operator("node.add_node", text="Switch Textures").type = "AnimateTextureNode"


if __name__ == "__main__":
    register()

