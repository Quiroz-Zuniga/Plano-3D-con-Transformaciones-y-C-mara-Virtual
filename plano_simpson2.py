# plano_simpson_open3d.py
# Plano 3D estilo Google Earth de la casa de los Simpson
# Librerías: Open3D (pip install open3d)
# Autor: Ruben Quiroz and Jennifer Gigccella

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

# Creación de paredes
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
            ((7, 0), (12, 0)),
            ((16, 7), (16, 10)),
            ((16, 10), (0, 10)),
            ((0, 10), (0, 0)),
        ]
        for p1, p2 in walls:
            self.add(create_wall(p1, p2, height=3))

    def paredes_interiores(self):
        interiores = [
            ((6,3),(0,3)), ((0,7),(6,7)), ((10,7),(16,7)),
            ((10,3),(14,3)), ((6,3),(10,3)), ((6,7),(10,7)),
            ((14,3),(14,7)), ((6,3),(6,7)),
            ((6,1.5),(6,3)), ((6,5),(14,5)), ((6,9),(6,10))
        ] 
        for p1, p2 in interiores:
            self.add(create_wall(p1, p2, height=3, thickness=0.12))

    def muebles(self):
        # Cama
        self.add(create_box((12, 7, 0), (2, 1.2, 0.5), color=[0.6, 0.3, 0.6]))

        # Cocina
        self.add(create_box((1, 7, 0), (2, 1, 0.5), color=[0.8, 0.1, 0.1]))

        # Mesa
        self.add(create_box((7, 6, 0), (1.5, 1.0, 0.75), color=[0.2, 0.2, 0.5]))

        # Lámparas
        self.add(create_box((8, 4, 0), (0.3, 0.3, 1.0), color=[1, 1, 0.5]))
        self.add(create_box((3, 8, 0), (0.3, 0.3, 1.0), color=[1, 1, 0.5]))

        # Alfombras
        self.add(create_box((14, 3.5, 0), (2.9, 1.2, 0.05), color=[0.0, 0.3, 0.0]))

        # Puerta
        self.add(create_box((11, 3, 0), (0.9, 0.1, 2.1), color=[1.0, 1.0, 1.0])) 

        # Estufa gris oscuro en la cocina
        self.add(create_box((0, 5, 0), (0.8, 0.8, 0.6), color=[0.3, 0.3, 0.3]))

        # Escalera
        for i in range(6):
            self.add(create_box((6,3.5 + i*0.3, 0), (1, 0.1, 0.3), color=[0.5, 0.25, 0.0]))  # escalones de la escalera

        # Mueble pequeño (silla) - color rosado-violeta
        self.add(create_box((12, 2, 0), (0.6, 0.6, 0.9), color=[0.8, 0.4, 0.8]))  # Silla pequeña en rosado-violeta

        # Mueble grande (armario) - color amarillo fuerte
        self.add(create_box((8, 2, -0), (2, 1, 2.5), color=[1.0, 1.0, 0.0]))  # Armario grande en amarillo fuerte
        
        
            # ---------------------------------------------------------
    # PUERTA EXTERIOR (curva con borde y pomo)
    # ---------------------------------------------------------
    def puerta_exterior(self, x=0, y=0, z=0):
        """
        Puerta exterior 
        Cambia x,y,z para mover la puerta.
        """

        # ------------ cuerpo de la puerta ------------
        puerta = create_box(
            origin=(x, y, z),
            size=(0.12, 1.0, 2.1),
            color=[0.88, 0.40, 0.25]   # naranja rojizo
        )
        self.add(puerta)

        # ------------ marco de madera ------------
        marco = create_box(
            origin=(x - 0.05, y - 0.05, z - 0.05),
            size=(0.20, 1.10, 2.25),
            color=[0.35, 0.18, 0.04]
        )
        self.add(marco)

        # ------------ pomo superior ------------
        pomo1 = create_box(
            origin=(x + 0.15, y + 0.60, z + 1.20),
            size=(0.08, 0.08, 0.08),
            color=[0.75, 0.75, 1.0]
        )
        self.add(pomo1)

        # ------------ pomo inferior ------------
        pomo2 = create_box(
            origin=(x + 0.15, y + 0.60, z + 0.80),
            size=(0.08, 0.08, 0.08),
            color=[0.75, 0.75, 1.0]
        )
        self.add(pomo2)
        
            # ---------------------------------------------------------
    # PUERTA INTERIOR ARCO (con banda verde/naranja)
    # ---------------------------------------------------------
    def puerta_interior_arco(self, x=0, y=0, z=0):
        """
        Puerta interior con arco y banda decorada — estilo imagen 2.
        Cambia x,y,z para moverla.
        """

        # --- pared donde va el arco ---
        arco = create_box(
            origin=(x, y, z),
            size=(0.12, 1.0, 2.0),
            color=[0.90, 0.60, 0.65]  # rosado interior clásico
        )
        self.add(arco)

        # --- arco superior (semi-curvo simulado) ---
        arco_superior = create_box(
            origin=(x - 0.02, y, z + 1.7),
            size=(0.16, 1.0, 0.35),
            color=[0.80, 0.45, 0.50]  # rosado más oscuro
        )
        self.add(arco_superior)

        # --- banda decorada verde ---
        banda1 = create_box(
            origin=(x + 0.01, y + 0.10, z + 0.80),
            size=(0.06, 0.80, 0.25),
            color=[0.0, 0.75, 0.0]
        )
        self.add(banda1)

        # --- banda decorada naranja ---
        banda2 = create_box(
            origin=(x + 0.01, y + 0.10, z + 0.55),
            size=(0.06, 0.80, 0.25),
            color=[0.90, 0.55, 0.15]
        )
        self.add(banda2)

        # Simulación de transparencia (colores suaves para "transparente")
        arco.paint_uniform_color([0.9, 0.9, 0.9])  # Color muy claro para simular transparencia
        arco_superior.paint_uniform_color([0.8, 0.8, 0.8])  # Color suave para simular opacidad

    
        #______________Garaje_______________________
    def garaje(self):
        # --- Coordenadas base EXACTAS en la pared ((16,7),(16,10)) ---
        base_x = 16        # pared frontal
        base_y = 7.278         # inicio de la pared
        ancho = 2.5         # horizontal (en eje Y)
        largo = 2.5          # sale hacia afuera en +X
        altura = 1.5

        # --- Portón del garaje ---
        porton = create_box(
            origin=(base_x, base_y, 0),
            size=(0.05, ancho, 2.0),   
            color=[0.75, 0.55, 0.30]
        )
        self.add(porton)

        # --- Paneles horizontales del portón ---
        for i in range(1, 3):
            linea = create_box(
                origin=(base_x + 0.01, base_y, 0.1 + i * 0.6),
                size=(0.06, ancho, 0.05),
                color=[0.55, 0.35, 0.20]
            )
            self.add(linea)

        # --- Ventana superior centrada sobre el portón ---
        ventana = create_box(
            origin=(base_x + 0, base_y + ancho/2 - 0.25, 2.1),
            size=(0.05, 0.5, 0.5),
            color=[0.65, 0.85, 1.0]
        )
        self.add(ventana)

        # --- Techo a dos aguas ---
        # --- Techo a dos aguas corregido ---
        cumbrera_z = altura + 0.3    # altura de la línea del techo
        mitad = ancho / 2             # mitad del ancho total

        # --- Panel izquierdo ---
        techo_izq = create_box(
            origin=(base_x - 0.5 , base_y - 0.30 , altura + 1.5),
            size=(mitad , largo - 0.3, 0.1),
            color=[0.35, 0.20, 0.05]
        )
        # rotación desde la cumbrera (borde superior del panel)
        techo_izq.rotate(
            techo_izq.get_rotation_matrix_from_xyz((0.45, 0, 0)),
            center=(base_x + mitad, base_y, cumbrera_z)
        )
        self.add(techo_izq)

        # --- Panel derecho ---
        techo_der = create_box(
            origin=(base_x -0.5, base_y + 0.20  , altura + 2.5),
            size=(mitad, largo - 0.3, 0.1),
            color=[0.30, 0.17, 0.04]
        )
        # rotación desde la cumbrera (borde superior del panel)
        techo_der.rotate(
            techo_der.get_rotation_matrix_from_xyz((-0.45, 0, 0)),
            center=(base_x + mitad, base_y, cumbrera_z)
        )
        self.add(techo_der) 
            # --- Relleno entre los dos techos ---
        # PUNTO: Centro entre los dos tejados
        pivote_x = base_x + ancho / 2
        pivote_y = base_y + largo / 2
        pivote_z = altura + 0.01   # pequeña el     evación para evitar clipping

        relleno = create_box(
            origin=(pivote_x, pivote_y, pivote_z),
            size=(ancho, 0.05, 1.5),     # AJUSTABLE
            color=[0.20, 0.12, 0.04]
        )

        # --- Rotación del relleno ---
        # Girar para que coincida con la inclinación del techo
        relleno.rotate(
            relleno.get_rotation_matrix_from_xyz((0, np.radians(-25), 0)),
            center=(pivote_x, pivote_y, pivote_z)
        )

        self.add(relleno)


#___________Fin Garaje____________________
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
    casa.garaje()
    # Puerta exterior (solo cambia x,y,z para colocarla)
    casa.puerta_exterior(x=13.9, y=3.5, z=0)
    casa.puerta_exterior(x=0.05, y=1.5, z=0)
    # Puerta interior arco
    casa.puerta_interior_arco(x=6, y=9, z=0)
    casa.puerta_interior_arco(x=6, y=6, z=0) 
    casa.render() 

