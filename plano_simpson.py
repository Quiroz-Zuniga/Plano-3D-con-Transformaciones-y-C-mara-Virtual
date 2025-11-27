# plano_simpson.py
# Plano 3D estático estilo "Casa de los Simpson"
# Requisitos: python3, matplotlib, numpy
# Instalación rápida:
#   pip install matplotlib numpy

from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.pyplot as plt
import numpy as np
import os

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
        poly = Poly3DCollection(faces, facecolors=color, linewidths=0.2, edgecolors=edgecolor, alpha=alpha)
        self.ax.add_collection3d(poly)

    def wall(self, p1, p2, height=3.0, thickness=0.1, color=(0.9,0.9,0.9), alpha=1.0):
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
        poly = Poly3DCollection(faces, facecolors=color, linewidths=0.2, edgecolors='k', alpha=alpha)
        self.ax.add_collection3d(poly)

    def floor(self, origin, size, color=(0.95,0.95,0.95), alpha=1.0):
        x,y,z = origin
        dx,dy = size
        verts = [[(x,y,z),(x+dx,y,z),(x+dx,y+dy,z),(x,y+dy,z)]]
        poly = Poly3DCollection(verts, facecolors=color, linewidths=0.2, edgecolors='k', alpha=alpha)
        self.ax.add_collection3d(poly)

    def stairs(self, origin, step_size=(1.0,0.3,0.18), steps=6, direction='forward', color=(0.6,0.5,0.4)):
        x,y,z = origin
        dx, dy, dz = step_size
        for i in range(steps):
            xi = x + i*dx if direction=='forward' else x
            yi = y + i*dy if direction=='side' else y + i*dy
            zi = z + i*dz
            self.cuboid((xi, yi, zi), (dx, dy, dz), color=color, alpha=1.0)

def crear_plano(out_path="house_simpson.png"):
    fig = plt.figure(figsize=(12,8))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_box_aspect((1.5,1,0.6))
    obj = Objeto3D(ax)

    # Base
    house_origin = (0,0,0)
    house_size = (16,10)
    obj.floor(house_origin, house_size, color=(0.97,0.96,0.9))

    # Muros exteriores
    walls = [((0,0),(16,0)),((16,0),(16,10)),((16,10),(0,10)),((0,10),(0,0))]
    for w in walls:
        obj.wall(w[0], w[1], height=3.0, thickness=0.15, color=(0.9,0.9,0.95))

    # Muros interiores aproximados
    obj.wall((10,0),(10,4), height=3.0, thickness=0.15)
    obj.wall((0,6),(6,6), height=3.0, thickness=0.15)
    obj.wall((6,6),(6,10), height=3.0, thickness=0.15)
    obj.wall((10,4),(16,4), height=3.0, thickness=0.15)
    obj.wall((6,6),(10,6), height=3.0, thickness=0.12)
    obj.wall((6,4),(10,4), height=3.0, thickness=0.12)

    # Muebles (ejemplos)
    obj.cuboid((11.0,0.6,0.0),(3.2,1.2,0.9), color=(0.8,0.4,0.4))  # sofa
    obj.cuboid((13.0,2.0,0.0),(0.8,0.8,0.35), color=(0.6,0.4,0.2))  # mesa centro
    obj.cuboid((10.2,0.2,0.0),(0.6,0.6,1.2), color=(0.2,0.2,0.25))  # mueble TV

    # Cocina
    obj.cuboid((0.2,6.2,0.0),(5.6,0.6,0.95), color=(0.8,0.8,0.8))
    obj.cuboid((3.0,7.2,0.0),(1.4,1.0,0.75), color=(0.7,0.5,0.3))
    obj.cuboid((5.0,9.0,0.0),(0.8,0.6,1.8), color=(0.9,0.9,0.95))

    # Dormitorios
    obj.cuboid((11.2,4.4,0.0),(2.4,2.0,0.6), color=(0.4,0.6,0.8))
    obj.cuboid((14.0,4.2,0.0),(1.2,1.0,1.6), color=(0.7,0.6,0.5))
    obj.cuboid((11.2,7.0,0.0),(2.6,2.0,0.6), color=(0.4,0.6,0.6))
    obj.cuboid((14.0,7.0,0.0),(1.0,1.2,0.8), color=(0.6,0.5,0.4))

    # Escaleras
    obj.stairs((6.4,1.0,0.0), step_size=(0.8,0.35,0.18), steps=7, direction='forward', color=(0.55,0.4,0.3))

    # Techo indicativo
    # roof_verts = [
    #     [(0,0,3.0),(16,0,3.0),(12,5,4.0),(4,5,4.0)],
    #     [(4,5,4.0),(12,5,4.0),(16,10,3.0),(0,10,3.0)]
    # ]
    # roof = Poly3DCollection(roof_verts, facecolors=(0.7,0.2,0.2), alpha=0.5, edgecolors='k', linewidths=0.2)
    # ax.add_collection3d(roof)

    # Ajustes visuales
    ax.set_xlim(-1,17)
    ax.set_ylim(-1,11)
    ax.set_zlim(0,4.5)
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.set_zlabel('z (m)')
    ax.view_init(elev=25, azim=-45)
    ax.grid(False)
    ax.set_xticks(np.arange(0,17,1))
    ax.set_yticks(np.arange(0,11,1))
    ax.set_zticks([0,1,2,3,4])

    plt.title("Plano 3D (estático) - Casa estilo 'Los Simpson' (planta aproximada)")

    # Guardar imagen
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    print("Plano guardado en:", out_path)
    plt.show()

if __name__ == "__main__":
    crear_plano(out_path="house_simpson.png")
