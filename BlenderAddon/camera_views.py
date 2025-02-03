import bpy
import numpy as np
import os

from mathutils import Vector, Matrix
from .utils import random_camera_views


class CameraViewOperator(bpy.types.Operator):
    bl_idname = "object.camera_view_operator"
    bl_label = "Camera View Operator"
    bl_options = {'REGISTER', 'UNDO'}  # 'UNDO' allows adjustments after execution

    def execute(self, context):
        scene = context.scene
        
        # Store current values to prevent reset
        current_views = scene.view_count
        current_distance = scene.distance
        current_scale = scene.scale
        current_coverage_top = scene.coverage_top
        current_coverage_bottom = scene.coverage_bottom
        
        target_object = context.scene.selected_object if context.scene.selected_object else context.view_layer.objects.active
        if target_object is None:
            self.report({'WARNING'}, "No object selected or active.")
            return {'CANCELLED'}
        world_location = target_object.matrix_world.to_translation()
        
        setup_cameras(current_views, current_distance, current_scale, current_coverage_top, current_coverage_bottom, world_location)    
        return {'FINISHED'}
        
def update_handler(self, context):
    print("Updating cameras...")
    bpy.ops.object.camera_view_operator()
    return None      

def create_camera_parent(center_location):
    """ Create an empty sphere as the parent for all cameras """
    empty = bpy.data.objects.new(name="Camera_Parent", object_data=None)
    empty.location = center_location
    bpy.context.scene.collection.objects.link(empty)
    return empty
 
def create_camera(view, camera_idx, collection, parent=None, ortho_scale=1.0):
    """ Create an orthographic camera and parent it to the given object if provided """
    cam_data = bpy.data.cameras.new(name=f"Camera_{camera_idx}")
    # cam_data.type = 'ORTHO'  # Set camera to orthographic
    # cam_data.ortho_scale = ortho_scale  # Set orthographic scale

    cam_obj = bpy.data.objects.new(name=f"Camera_{camera_idx}", object_data=cam_data)

    # Link to the scene collection
    collection.objects.link(cam_obj)

    # Set transformation
    cam_obj.matrix_world = view

    # Parent to empty for rotation control
    if parent:
        cam_obj.parent = parent

    return cam_obj

def setup_cameras(v, distance, scale, coverage_top, coverage_bottom, center_location):
    """ Setup the camera system with a parent empty for rotation """
    collection = get_or_create_collection("Camera_Views")

    # Remove old cameras & empty if they exist
    old_parent = bpy.data.objects.get("Camera_Parent")
    if old_parent:
        bpy.data.objects.remove(old_parent, do_unlink=True)
    for obj in list(collection.objects):
        bpy.data.objects.remove(obj, do_unlink=True)

    # Create new empty at center location
    parent_empty = create_camera_parent(center_location)

    # Generate new camera views
    views = random_camera_views(v, distance, coverage_top, coverage_bottom, Vector((0, 0, 0)))
    for i, view in enumerate(views):
        create_camera(view, i, collection, parent=parent_empty, ortho_scale=scale)

    return parent_empty

def get_or_create_collection(name):
    """ Get or create a collection with the given name """
    if name in bpy.data.collections:
        return bpy.data.collections[name]
    
    collection = bpy.data.collections.new(name)
    bpy.context.scene.collection.children.link(collection)
    return collection