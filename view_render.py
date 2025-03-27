import bpy
import os

# 清空场景
bpy.ops.wm.read_factory_settings(use_empty=True)

# 设定文件路径
obj_path = "/path/to/your/test0001.obj"  # 修改为你的OBJ文件路径
output_path = "/path/to/output/rendered_image.png"  # 设定渲染输出路径

# 导入OBJ文件
bpy.ops.import_scene.obj(filepath=obj_path)

# 获取导入的对象
obj = bpy.context.selected_objects[0]
obj.name = "Imported_Mesh"

# 设定渲染引擎为 Cycles
bpy.context.scene.render.engine = 'CYCLES'

# 创建相机
def create_camera(name, location, rotation):
    cam_data = bpy.data.cameras.new(name)
    cam = bpy.data.objects.new(name, cam_data)
    bpy.context.collection.objects.link(cam)
    cam.location = location
    cam.rotation_euler = rotation
    return cam

# 定义三个相机的位置和角度
cameras = [
    create_camera("Camera1", (2, 0, 1), (1.1, 0, 1.57)),  # 红色
    create_camera("Camera2", (-2, 0, 1), (1.1, 0, -1.57)),  # 绿色
    create_camera("Camera3", (0, 2, 1), (1.1, 0, 3.14)),  # 蓝色
]

# 设定当前相机
bpy.context.scene.camera = cameras[0]

# 创建新材质
mat = bpy.data.materials.new(name="Camera_Visibility_Material")
mat.use_nodes = True
nodes = mat.node_tree.nodes
links = mat.node_tree.links

# 清空默认节点
for node in nodes:
    nodes.remove(node)

# 添加 Principled BSDF
bsdf = nodes.new(type='ShaderNodeBsdfPrincipled')
bsdf.location = (400, 0)

# 创建颜色输入
color_r = nodes.new(type='ShaderNodeRGB')
color_r.outputs[0].default_value = (1, 0, 0, 1)  # 红色
color_r.location = (-200, 100)

color_g = nodes.new(type='ShaderNodeRGB')
color_g.outputs[0].default_value = (0, 1, 0, 1)  # 绿色
color_g.location = (-200, 0)

color_b = nodes.new(type='ShaderNodeRGB')
color_b.outputs[0].default_value = (0, 0, 1, 1)  # 蓝色
color_b.location = (-200, -100)

# 创建 Mix RGB 混合节点
mix_rgb_1 = nodes.new(type='ShaderNodeMixRGB')
mix_rgb_1.blend_type = 'ADD'
mix_rgb_1.location = (0, 50)
mix_rgb_2 = nodes.new(type='ShaderNodeMixRGB')
mix_rgb_2.blend_type = 'ADD'
mix_rgb_2.location = (200, 50)

# 连接颜色
links.new(color_r.outputs[0], mix_rgb_1.inputs[1])
links.new(color_g.outputs[0], mix_rgb_1.inputs[2])
links.new(mix_rgb_1.outputs[0], mix_rgb_2.inputs[1])
links.new(color_b.outputs[0], mix_rgb_2.inputs[2])

# 连接最终颜色到 BSDF
links.new(mix_rgb_2.outputs[0], bsdf.inputs['Base Color'])

# 添加输出节点
output = nodes.new(type='ShaderNodeOutputMaterial')
output.location = (600, 0)
links.new(bsdf.outputs['BSDF'], output.inputs['Surface'])

# 赋予材质到 Mesh
if len(obj.data.materials):
    obj.data.materials[0] = mat
else:
    obj.data.materials.append(mat)

# 渲染图像
bpy.context.scene.render.filepath = output_path
bpy.ops.render.render(write_still=True)

print(f"Rendered image saved at: {output_path}")
