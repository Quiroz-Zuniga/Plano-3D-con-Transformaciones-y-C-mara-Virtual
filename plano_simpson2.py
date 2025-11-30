# plano_simpson_open3d.py
# Plano 3D estilo Google Earth de la casa de los Simpson
# Librerías: Open3D (pip install open3d)
# Autor: ChatGPT (versión mejorada desde tu código)

import open3d as o3d
import numpy as np


# ---------------------------------------------------------
# Utilidades de creación básica
# ---------------------------------------------------------
def create_box(origin, size, color=[0.8, 0.8, 0.8]):
    x, y, z = origin
    dx, dy, dz = size

    box = o3d.geometry.TriangleMesh.create_box(dx, dy, dz)
    box.translate((x, y, z))
    box.paint_uniform_color(color)
    box.compute_vertex_normals()
    return box


def create_wall(p1, p2, height=3, thickness=0.15, color=[0.72, 0.52, 0.36]):
    x1, y1 = p1
    x2, y2 = p2

    v = np.array([x2 - x1, y2 - y1])
    length = np.linalg.norm(v)

    # Caja base
    wall = o3d.geometry.TriangleMesh.create_box(length, thickness, height)
    wall.paint_uniform_color(color)

    # Rotación y posición
    angle = np.arctan2(v[1], v[0])
    R = wall.get_rotation_matrix_from_xyz((0, 0, angle))

    wall.rotate(R, center=(0, 0, 0))
    wall.translate((x1, y1, 0))

    return wall


# ---------------------------------------------------------
# Clase principal del plano
# ---------------------------------------------------------
class CasaSimpson3D:
    def __init__(self):
        self.objects = []

    def add(self, obj):
        self.objects.append(obj)

    def floor(self):
        piso = create_box(
            origin=(0, 0, 0),
            size=(16, 10, 0.1),
            color=[0.95, 0.95, 0.90]
        )
        self.add(piso)

    def paredes_exteriores(self):
        walls = [
            ((0, 0), (16, 0)),
            ((16, 0), (16, 10)),
            ((16, 10), (0, 10)),
            ((0, 10), (0, 0)),
        ]
        for p1, p2 in walls:
            self.add(create_wall(p1, p2, height=6))

    def paredes_interiores(self):
        interiores = [
            ((6,3),(0,3)), ((0,7),(6,7)), ((10,7),(16,7)),
            ((10,3),(14,3)), ((6,3),(10,3)), ((6,7),(10,7)),
            ((14,3),(14,7)), ((6,3),(6,7)),
            ((6,1.5),(6,3)), ((6,5),(14,5)), ((6,9),(6,10))
        ]
        for p1, p2 in interiores:
            self.add(create_wall(p1, p2, height=6, thickness=0.12))

    def muebles(self):
        # Cama
        self.add(create_box((1, 7, 0), (2, 1, 0.5), color=[0.6, 0.3, 0.6]))

        # Cocina
        self.add(create_box((13, 7, 0), (2, 1.2, 1.0), color=[0.8, 0.1, 0.1]))

        # Mesa
        self.add(create_box((7, 6, 0), (1.5, 1.0, 0.75), color=[0.2, 0.2, 0.5]))

        # Lámparas
        self.add(create_box((8, 4, 0), (0.3, 0.3, 1.0), color=[1, 1, 0.5]))
        self.add(create_box((3, 8, 0), (0.3, 0.3, 1.0), color=[1, 1, 0.5]))

        # Alfombras
        self.add(create_box((14, 3.5, 0), (2.9, 1.2, 0.05), color=[0.0, 0.3, 0.0]))
        self.add(create_box((3, 3, 0), (2.5, 1.5, 0.05), color=[1.0, 0.6, 0.8]))

        # Puerta
        self.add(create_box((0, 4, 0), (0.9, 0.1, 2.1), color=[1.0, 0.7, 0.5]))

    def render(self):
        scene = o3d.visualization.Visualizer()
        scene.create_window(width=1280, height=900)

        for obj in self.objects:
            scene.add_geometry(obj)

        # Cámara estilo Google Earth (vista orbital)
        view = scene.get_view_control()
        view.set_front([0.6, -1, 0.7])
        view.set_up([0, 0, 1])
        view.set_lookat([8, 5, 0])
        view.set_zoom(0.45)

        # Luz ambiental
        opt = scene.get_render_option()
        opt.mesh_show_back_face = True
        opt.background_color = np.array([0.8, 0.9, 1.0])

        scene.run()
        scene.destroy_window()


# ---------------------------------------------------------
# Ejecutar plano
# ---------------------------------------------------------
if __name__ == "__main__":
    casa = CasaSimpson3D()

    casa.floor()
    casa.paredes_exteriores()
    casa.paredes_interiores()
    casa.muebles()

    casa.render()
