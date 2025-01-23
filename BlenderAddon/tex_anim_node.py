import bpy

class AnimateTextureNode(bpy.types.ShaderNodeCustomGroup):
    bl_name = "AnimateTextureNode"
    bl_label = "Texture Animation Node"
    
    def init(self, context):
        # Create a new node group
        self.node_tree = bpy.data.node_groups.new(self.bl_name, 'ShaderNodeTree')
        
        # Create the interface for inputs and outputs
        self.node_tree.interface.new_socket("Switch Factor", in_out='INPUT', socket_type='NodeSocketFloat')
        self.node_tree.interface.new_socket("Interpolated Output", in_out='OUTPUT', socket_type='NodeSocketColor')
        
        # Add input and output nodes to the group
        input_node = self.node_tree.nodes.new("NodeGroupInput")
        input_node.location = (-400, 0)
        output_node = self.node_tree.nodes.new("NodeGroupOutput")
        output_node.location = (400, 0)

        # Create default nodes and connections
        self._create_default_nodes()
        self._create_time_group_node()

    def _create_default_nodes(self):
        """Initialize the node group with default nodes."""
        node_tree = self.node_tree
        
        # Create math node to handle the first range (0-1)
        math_node = node_tree.nodes.new("ShaderNodeMath")
        math_node.operation = 'SUBTRACT'
        math_node.location = (-200, 100)
        math_node.name = "Math_1"
        
        # Create the first mix node
        mix_node = node_tree.nodes.new("ShaderNodeMixRGB")
        mix_node.location = (0, 0)
        mix_node.name = "Mix_1"

        # Add the first two image inputs
        self._add_image_input("Image_1", mix_node, 1)
        self._add_image_input("Image_2", mix_node, 2)

        # Link the switch factor through math node to mix node
        input_node = node_tree.nodes["Group Input"]
        output_node = node_tree.nodes["Group Output"]
        
        # Connect switch factor to math node
        node_tree.links.new(input_node.outputs["Switch Factor"], math_node.inputs[0])
        # Set the subtraction value to 0 for first range
        math_node.inputs[1].default_value = 0
        # Connect math node to mix factor
        node_tree.links.new(math_node.outputs[0], mix_node.inputs["Fac"])
        
        # Connect mix output to group output
        node_tree.links.new(mix_node.outputs["Color"], output_node.inputs["Interpolated Output"])

    def _add_image_input(self, name, mix_node, input_idx):
        """Add a new image input socket and connect it."""
        socket_name = f"{name} Texture"
        self.node_tree.interface.new_socket(socket_name, in_out='INPUT', socket_type='NodeSocketColor')
        input_node = self.node_tree.nodes["Group Input"]
        self.node_tree.links.new(input_node.outputs[socket_name], mix_node.inputs[input_idx])
        
    def _create_time_group_node(self):
        self.time_node_tree = bpy.data.node_groups.new("TimeNodeGroup", 'ShaderNodeTree')
        
        # Create nodes for current frame, start frame and end frame
        current_frame_node = self.time_node_tree.nodes.new("ShaderNodeAttribute")
        current_frame_node.attribute_type = "VIEW_LAYER"
        current_frame_node.attribute_name = "frame_current"
        current_frame_node.location = (-400, 0)
        
        start_frame_node = self.time_node_tree.nodes.new("ShaderNodeAttribute")
        start_frame_node.attribute_type = "VIEW_LAYER"
        start_frame_node.attribute_name = "frame_start"
        start_frame_node.location = (-400, -200)
        
        end_frame_node = self.time_node_tree.nodes.new("ShaderNodeAttribute")
        end_frame_node.attribute_type = "VIEW_LAYER"
        end_frame_node.attribute_name = "frame_end"
        end_frame_node.location = (-400, 200)
        
        # First subtract start frame from current frame
        subtract_node = self.time_node_tree.nodes.new("ShaderNodeMath")
        subtract_node.operation = 'SUBTRACT'
        subtract_node.location = (-200, 0)
        self.time_node_tree.links.new(current_frame_node.outputs[2], subtract_node.inputs[0])
        self.time_node_tree.links.new(start_frame_node.outputs[2], subtract_node.inputs[1])
        
        # Then subtract start frame from end frame to get total range
        range_node = self.time_node_tree.nodes.new("ShaderNodeMath")
        range_node.operation = 'SUBTRACT'
        range_node.location = (-200, 200)
        self.time_node_tree.links.new(end_frame_node.outputs[2], range_node.inputs[0])
        self.time_node_tree.links.new(start_frame_node.outputs[2], range_node.inputs[1])
        
        # Finally divide to normalize between 0 and 1
        divide_node = self.time_node_tree.nodes.new("ShaderNodeMath")
        divide_node.operation = 'DIVIDE'
        divide_node.location = (0, 0)
        self.time_node_tree.links.new(subtract_node.outputs[0], divide_node.inputs[0])
        self.time_node_tree.links.new(range_node.outputs[0], divide_node.inputs[1])
        
        # Create output node
        self.time_node_tree.interface.new_socket("Time", in_out='OUTPUT', socket_type='NodeSocketFloat')
        output_node = self.time_node_tree.nodes.new("NodeGroupOutput")
        output_node.location = (200, 0)
        self.time_node_tree.links.new(divide_node.outputs[0], output_node.inputs["Time"])
        
        
        
    def draw_buttons(self, context, layout):
        layout.operator("node.add_image_input", text="Add Image Input")

class NODE_OT_AddImageInput(bpy.types.Operator):
    """Operator to dynamically add an image input socket."""
    bl_idname = "node.add_image_input"
    bl_label = "Add Image Input"
    
    def execute(self, context):
        node = context.node
        if isinstance(node, AnimateTextureNode):
            # Find the last mix node
            mix_nodes = [n for n in node.node_tree.nodes if n.name.startswith("Mix")]
            if mix_nodes:
                last_mix = mix_nodes[-1]
                
                # Add a new math node for range calculation
                math_node = node.node_tree.nodes.new("ShaderNodeMath")
                math_node.operation = 'SUBTRACT'
                math_node.location = (last_mix.location.x - 200, last_mix.location.y + 100)
                math_node.name = f"Math_{len(mix_nodes) + 1}"
                # Set the subtraction value based on the number of existing mix nodes
                math_node.inputs[1].default_value = len(mix_nodes)
                
                # Add a new mix node
                new_mix = node.node_tree.nodes.new("ShaderNodeMixRGB")
                new_mix.location = (last_mix.location.x + 200, last_mix.location.y)
                new_mix.name = f"Mix_{len(mix_nodes) + 1}"
                
                # Connect nodes
                input_node = node.node_tree.nodes["Group Input"]
                output_node = node.node_tree.nodes["Group Output"]
                
                # Connect switch factor to new math node
                node.node_tree.links.new(input_node.outputs["Switch Factor"], math_node.inputs[0])
                # Connect math node to mix factor
                node.node_tree.links.new(math_node.outputs[0], new_mix.inputs["Fac"])
                
                # Connect previous mix output to new mix input
                node.node_tree.links.new(last_mix.outputs["Color"], new_mix.inputs[1])
                # Connect new mix output to group output
                node.node_tree.links.new(new_mix.outputs["Color"], output_node.inputs["Interpolated Output"])
                
                # Get number of existing inputs from interface
                input_idx = len([socket for socket in node.node_tree.interface.items_tree if socket.in_out == 'INPUT']) - 1
                
                # Add a new image input
                node._add_image_input(f"Image_{input_idx + 1}", new_mix, 2)
            else:
                # No mix nodes exist; initialize the first one
                node._create_default_nodes()
        return {'FINISHED'}
    
    