# plano_simpson.py
# Plano 3D estático estilo "Casa de los Simpson"
# Requisitos: python3, matplotlib, numpy
# Instalación rápida:
#   pip install matplotlib numpy

from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.pyplot as plt
import numpy as np

class Objeto3D:
    def __init__(self, ax):
        self.ax = ax

    def cuboid(self, origin, size, color=(0.8,0.8,0.8), alpha=1.0, edgecolor='k'):
        x, y, z = origin
        dx, dy, dz = size
        X = [x, x+dx]
        Y = [y, y+dy]
        Z = [z, z+dz]
        pts = [(xi, yi, zi) for xi in X for yi in Y for zi in Z]
        faces = [
            [pts[0], pts[1], pts[3], pts[2]],
            [pts[4], pts[5], pts[7], pts[6]],
            [pts[0], pts[1], pts[5], pts[4]],
            [pts[2], pts[3], pts[7], pts[6]],
            [pts[0], pts[2], pts[6], pts[4]],
            [pts[1], pts[3], pts[7], pts[5]],
        ]
        poly = Poly3DCollection(faces, facecolors=color,
                                linewidths=0.2, edgecolors=edgecolor, alpha=alpha)
        self.ax.add_collection3d(poly)

    def wall(self, p1, p2, height=3.0, thickness=1.1, color=(0.80,0.65,0.45), alpha=1):
        x1, y1 = p1
        x2, y2 = p2
        v = np.array([x2-x1, y2-y1])
        if np.allclose(v, 0): return
        n = np.array([-v[1], v[0]])
        n = n / np.linalg.norm(n) * (thickness/2)
        corners = [
            (x1 + n[0], y1 + n[1]),
            (x1 - n[0], y1 - n[1]),
            (x2 - n[0], y2 - n[1]),
            (x2 + n[0], y2 + n[1]),
        ]
        faces = []
        for i in range(4):
            xA,yA = corners[i]
            xB,yB = corners[(i+1)%4]
            faces.append([(xA,yA,0),(xB,yB,0),(xB,yB,height),(xA,yA,height)])
        poly = Poly3DCollection(faces, facecolors=color,
                                linewidths=0.2, edgecolors='k', alpha=alpha)
        self.ax.add_collection3d(poly)

    def floor(self, origin, size, color=(0.95,0.95,0.90), alpha=1.0):
        x,y,z = origin
        dx,dy = size
        verts = [[(x,y,z),(x+dx,y,z),(x+dx,y+dy,z),(x,y+dy,z)]]
        poly = Poly3DCollection(verts, facecolors=color,
                                linewidths=0.2, edgecolors='k', alpha=alpha)
        self.ax.add_collection3d(poly)

    def stairs(self, origin, step_size=(1.0,0.3,0.18), steps=6, direction='forward', color=(0.55,0.4,0.3)):
        x,y,z = origin
        dx, dy, dz = step_size
        for i in range(steps):
            xi = x + i*dx if direction=='forward' else x
            yi = y + i*dy if direction=='side' else y + i*dy
            zi = z + i*dz
            self.cuboid((xi, yi, zi), (dx, dy, dz), color=color, alpha=1.0)

    # Muebles básicos con colores actualizados
    def cama(self, origin, size=(2.0,1.0,0.5), color=(0.8,0.6,0.9)):
        base_color = (0.5,0.3,0.5)  # base de la cama
        colchón_color = color
        x,y,z = origin
        dx, dy, dz = size
        self.cuboid((x, y, z), (dx, dy, dz*0.6), color=base_color)
        self.cuboid((x, y, z+dz*0.6), (dx, dy, dz*0.4), color=colchón_color)

    def cocina(self, origin, size=(2.0,1.2,1.0), color=(0.8,0,0)):
        x,y,z = origin
        dx, dy, dz = size
        self.cuboid((x, y, z), (dx, dy, dz), color=color)
        estufa_color = (0.5,1.0,0.5)
        self.cuboid((x+0.1, y+0.1, z+dz), (dx*0.8, dy*0.8, 0.3), color=estufa_color)

    def mesa(self, origin, size=(1.5,1.0,0.75), color=(0.1,0.1,0.5)):
        x,y,z = origin
        dx, dy, dz = size
        self.cuboid((x, y, z), (dx, dy, dz*0.05), color=color)
        pata_color = (0.4,0.25,0.1)
        pata_size = (0.1,0.1,dz*0.75)
        patas = [
            (x, y, z - pata_size[2]),
            (x+dx-pata_size[0], y, z - pata_size[2]),
            (x, y+dy-pata_size[1], z - pata_size[2]),
            (x+dx-pata_size[0], y+dy-pata_size[1], z - pata_size[2]),
        ]
        for p in patas:
            self.cuboid(p, pata_size, color=pata_color)

    def lampara(self, origin, size=(0.3,0.3,1.0), color=(1,1,0.5)):
        x,y,z = origin
        dx, dy, dz = size
        self.cuboid((x, y, z), (dx, dy, dz*0.8), color=(0.2,0.2,0.2))
        self.cuboid((x+dx*0.1, y+dy*0.1, z+dz*0.8), (dx*0.8, dy*0.8, dz*0.2), color=color)

    def arbol_pequeno(self, origin, trunk_height=0.5, trunk_radius=0.15, crown_size=0.6, trunk_color=(0.1,0.05,0), crown_color=(0.4,1.0,0.4)):
        x,y,z = origin
        self.cuboid((x-trunk_radius/2, y-trunk_radius/2, z), (trunk_radius, trunk_radius, trunk_height), color=trunk_color)
        self.cuboid((x-crown_size/2, y-crown_size/2, z+trunk_height), (crown_size, crown_size, crown_size), color=crown_color)

    def puerta(self, origin, size=(0.9,0.05,2.1), color=(1.0,0.7,0.5)):
        x,y,z = origin
        dx, dy, dz = size
        self.cuboid((x, y, z), (dx, dy, dz), color=color)

def crear_plano(out_path="house_simpson_colores.png"):
    fig = plt.figure(figsize=(12,8))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_box_aspect((2.5,3.5,0.6))
    obj = Objeto3D(ax)

    obj.floor((0,0,0), (16,10), color=(0.97,0.96,0.9))

    walls = [((7,0),(12,0)),((16,7),(16,10)),((16,10),(0,10)),((0,10),(0,0))]
    for w in walls:
        obj.wall(w[0], w[1], height=6.0, thickness=0.15, color=(0.72,0.52,0.36))

    interiores = [
        ((6,3),(0,3)), ((0,7),(6,7)), ((10,7),(16,7)), ((10,3),(14,3)),
        ((6,3),(10,3)), ((6,7),(10,7)), ((14,3),(14,7)), ((6,3),(6,7)),
        ((6,1.5),(6,3)), ((6,5),(14,5)), ((6,9),(6,10))
    ]
    for w in interiores:
        obj.wall(w[0], w[1], height=6.0, thickness=0.12, color=(0.72,0.52,0.36))

    obj.stairs(origin=(6,3,0.3), step_size=(1.0,0.3,0.18), steps=6, direction='forward', color=(0.55,0.4,0.3))

    # Alfombra verde oscuro
    obj.cuboid((14, 3.5, 0), (2.9, 1.2, 0.05), color=(0.0, 0.3, 0.0))
    # Alfombra rosada nueva
    obj.cuboid((3, 3, 0), (2.5,1.5,0.05), color=(1.0,0.6,0.8))

    # Muebles
    obj.cama((1, 7, 0))                   
    obj.cocina((13, 7, 0))                
    obj.mesa((7, 6, 0))                   
    obj.lampara((8, 4, 0))                
    obj.lampara((3, 8, 0))                

    # Árboles pequeños alrededor de la alfombra
    obj.arbol_pequeno((16, 5.5, 0))
    obj.arbol_pequeno((16, 3.2, 0))

    # Puerta melón
    obj.puerta((0, 4, 0))

    # Ajustes visuales
    ax.set_xlim(-1,17)
    ax.set_ylim(-1,11)
    ax.set_zlim(0,9)
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.set_zlabel('z (m)')
    ax.view_init(elev=25, azim=-45)
    ax.grid(False)
    ax.set_xticks(np.arange(0,17,1))
    ax.set_yticks(np.arange(0,11,1))
    ax.set_zticks(np.arange(0,10,1))

    plt.title("Casa Simpson 3D con muebles y colores actualizados")
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    print(f"Plano guardado en: {out_path}")
    plt.show()

if __name__ == "__main__":
    crear_plano()

