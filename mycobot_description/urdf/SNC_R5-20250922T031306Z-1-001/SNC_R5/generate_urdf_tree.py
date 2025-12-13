from ikpy.urdf.utils import get_urdf_tree

dot, urdf_tree = get_urdf_tree('Humanoid_SNC.urdf', root_element='base_link')
# If Graphviz 'dot' is not available, save the DOT source so you can render it elsewhere
with open('SNC_R4.dot', 'w', encoding='utf-8') as f:
    f.write(dot.source)
print("Wrote SNC_R4.dot. Install Graphviz to render to PDF/PNG (or render elsewhere).")