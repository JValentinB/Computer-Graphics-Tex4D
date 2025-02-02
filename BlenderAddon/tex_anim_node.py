import bpy

class AnimateTextureNode(bpy.types.ShaderNodeCustomGroup):
    bl_name = "AnimateTextureNode"
    bl_label = "Texture Animation Node"

    
    def init(self, context):
        self.node_tree = bpy.data.node_groups.new(self.bl_name, 'ShaderNodeTree')
        self._create_time_group_node()
        
        # Interface
        self.node_tree.interface.new_socket("Interpolated Output", in_out='OUTPUT', socket_type='NodeSocketColor')
        
        # Nodes
        input_node = self.node_tree.nodes.new("NodeGroupInput")
        input_node.location = (-800, 0)
        output_node = self.node_tree.nodes.new("NodeGroupOutput")
        output_node.location = (800, 0)


        self._create_first_texture_setup()

    def _create_first_texture_setup(self):
        """Initial setup with 2 textures and 2 mix nodes (including loop)"""
        tree = self.node_tree
        
        # First texture input
        self.node_tree.interface.new_socket("Image_1 Texture", in_out='INPUT', socket_type='NodeSocketColor')
        self.node_tree.interface.new_socket("Image_2 Texture", in_out='INPUT', socket_type='NodeSocketColor')
        
        time_group_node = self.node_tree.nodes.new("ShaderNodeGroup")
        time_group_node.node_tree = bpy.data.node_groups.get("TimeNodeGroup")
        time_group_node.location = (-800, 200)

        # Math nodes for factor calculation
        multiply = tree.nodes.new("ShaderNodeMath")
        multiply.operation = 'MULTIPLY'
        multiply.location = (-400, 200)
        multiply.name = "multiply_node"
        
        subtract = tree.nodes.new("ShaderNodeMath")
        subtract.operation = 'SUBTRACT'
        subtract.location = (-100, 100)
        subtract.use_clamp = True

        # First mix node
        mix = tree.nodes.new("ShaderNodeMixRGB")
        mix.location = (-100, -100)
        mix.name = "Mix_1"

        # Links
        tree.links.new(time_group_node.outputs[0], multiply.inputs[0])
        tree.links.new(multiply.outputs[0], subtract.inputs[0])
        tree.links.new(subtract.outputs[0], mix.inputs['Fac'])
        
        # Connect textures
        group_input = tree.nodes["Group Input"]
        tree.links.new(group_input.outputs["Image_1 Texture"], mix.inputs[1])
        tree.links.new(group_input.outputs["Image_2 Texture"], mix.inputs[2])

        # Connect to output
        tree.links.new(mix.outputs["Color"], self.node_tree.nodes["Group Output"].inputs["Interpolated Output"])

        # Initialize with 2 textures = 2 mix nodes (including loop)
        self._add_loop_mix()
        self._update_math_nodes(2)

        
    def _add_loop_mix(self):
        """Add final mix node that loops back to first texture"""
        tree = self.node_tree
        group_input = tree.nodes["Group Input"]
        last_mix = [n for n in tree.nodes if n.name.startswith("Mix")][-1]

        # New math nodes for loop
        multiply = tree.nodes.get("multiply_node")
        
        subtract = tree.nodes.new("ShaderNodeMath")
        subtract.operation = 'SUBTRACT'
        subtract.location = last_mix.location.x + 400, last_mix.location.y + 200
        subtract.use_clamp = True

        # New mix node for loop
        loop_mix = tree.nodes.new("ShaderNodeMixRGB")
        loop_mix.location = last_mix.location.x + 400, last_mix.location.y
        loop_mix.name = f"Mix_loop"

        # Link math
        tree.links.new(multiply.outputs[0], subtract.inputs[0])
        tree.links.new(subtract.outputs[0], loop_mix.inputs['Fac'])

        # Link textures
        tree.links.new(last_mix.outputs["Color"], loop_mix.inputs[1])
        tree.links.new(group_input.outputs["Image_1 Texture"], loop_mix.inputs[2])

        # Update output connection
        tree.links.new(loop_mix.outputs["Color"], tree.nodes["Group Output"].inputs["Interpolated Output"])

    def _update_math_nodes(self, texture_count):
        """Update all math nodes for current texture count"""
        tree = self.node_tree
        
        mix_nodes = [n for n in tree.nodes if n.name.startswith("Mix")]
        # Loop through all mix nodes and update associated math nodes
        for idx, mix_node in enumerate(mix_nodes):
            correct_index = idx 
            if idx > 0:
                correct_index = idx - 1
            if mix_node.name == "Mix_loop":
                correct_index = len(mix_nodes) - 1
                
            # Find the connected subtract node (which is connected to the Fac input of the mix node)
            subtract_node = None
            multiply_node = None
            
            for link in mix_node.inputs["Fac"].links:  # Directly access the input's links
                subtract_node = link.from_node
                break 
            
            # If subtract node is found, find the multiply node connected to the subtract input
            if subtract_node:
                for link in subtract_node.inputs[0].links:  # Directly access the input's links
                    multiply_node = link.from_node
                    break
            
            # Now that we have the correct subtract and multiply nodes, update their values
            if subtract_node and multiply_node:
                multiply_node.inputs[1].default_value = texture_count
                subtract_node.inputs[1].default_value = correct_index

    def _add_image_input(self, name):
        """Add new texture input and setup nodes"""
        tree = self.node_tree
        group_input = tree.nodes["Group Input"]
        
        # Add new socket
        self.node_tree.interface.new_socket(f"{name} Texture", in_out='INPUT', socket_type='NodeSocketColor')
        
        # Get last non-loop mix node
        mix_nodes = [n for n in tree.nodes if n.name.startswith("Mix") and "loop" not in n.name]
        last_mix = mix_nodes[-1] if mix_nodes else None

        # Create new math nodes
        multiply = tree.nodes.get("multiply_node")
        
        subtract = tree.nodes.new("ShaderNodeMath")
        subtract.operation = 'SUBTRACT'
        subtract.location = last_mix.location.x + 400, last_mix.location.y + 200
        subtract.use_clamp = True

        # New mix node
        new_mix = tree.nodes.new("ShaderNodeMixRGB")
        new_mix.location = last_mix.location.x + 400, last_mix.location.y
        new_mix.name = f"Mix_{len(mix_nodes)+1}"

        # Link math
        tree.links.new(multiply.outputs[0], subtract.inputs[0])
        tree.links.new(subtract.outputs[0], new_mix.inputs['Fac'])

        # Link textures
        tree.links.new(last_mix.outputs["Color"], new_mix.inputs[1])
        socket_index = len([s for s in tree.interface.items_tree if s.in_out == 'INPUT' and "Image" in s.name])
        tree.links.new(group_input.outputs[socket_index - 1], new_mix.inputs[2])

        # Reconnect loop mix
        loop_mix = tree.nodes.get("Mix_loop")
        if loop_mix:
            for link in loop_mix.inputs["Fac"].links:  # Directly access the input's links
                link.from_node.location = new_mix.location.x + 400, new_mix.location.y + 200
                break 
            loop_mix.location = new_mix.location.x + 400, new_mix.location.y
            tree.nodes["Group Output"].location = loop_mix.location.x + 300, loop_mix.location.y
            
            tree.links.new(new_mix.outputs["Color"], loop_mix.inputs[1])
            
            

        # Update texture count in all math nodes
        texture_count = len([s for s in tree.interface.items_tree if s.in_out == 'INPUT' and "Image" in s.name])
        self._update_math_nodes(texture_count)

    def _create_time_group_node(self):
        if "TimeNodeGroup" not in bpy.data.node_groups:
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
    bl_idname = "node.add_image_input"
    bl_label = "Add Image Input"
    
    def execute(self, context):
        node = context.node
        if isinstance(node, AnimateTextureNode):
            # Count existing textures
            texture_count = len([s for s in node.node_tree.interface.items_tree 
                               if s.in_out == 'INPUT' and "Image" in s.name])
            
            # Add new image input
            node._add_image_input(f"Image_{texture_count + 1}")
            
            # If first added texture (making total 3), create loop mix
            # if texture_count == 2:
            #     node._add_loop_mix()

        return {'FINISHED'}