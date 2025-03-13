import os
import glob
import cv2
import json
import math
import numpy as np
import networkx as nx
from skimage.morphology import skeletonize
from scipy.signal import convolve2d
from sklearn.cluster import DBSCAN

# -----------------------------------------
#       PARÁMETROS PRINCIPALES
# -----------------------------------------
# Distancia para fusionar píxeles de ramificación (DBSCAN)
EPSILON_DBSCAN = 5
MIN_SAMPLES = 1

# Tolerancia para la simplificación RDP (mayor = menos nodos, más recto)
RDP_EPSILON = 2.0

# -----------------------------------------
#       COLORES (BGR para OpenCV)
# -----------------------------------------
COLOR_EXTREMO       = (0,   255, 0)       # Verde
COLOR_BIFURCACION   = (0,   0,   255)     # Rojo
COLOR_TRIFURCACION  = (255, 0,   0)       # Azul
COLOR_INTERMEDIO    = (128, 128, 128)     # Gris
COLOR_ARISTA        = (0,   255, 255)     # Amarillo

# ---------------------------------------------------
#        LECTURA, ESQUELETONIZACIÓN Y GRADO
# ---------------------------------------------------
def load_image(path):
    """Lee la imagen en escala de grises y la binariza."""
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    _, binary = cv2.threshold(img, 127, 1, cv2.THRESH_BINARY)
    return binary.astype(bool)

def get_skeleton(binary_img):
    """Obtiene el esqueleto a partir de la imagen binaria."""
    return skeletonize(binary_img)

def compute_degree(skel):
    """
    Cuenta cuántos vecinos (8-conectividad) tiene cada píxel True.
    Devuelve una matriz 'degree' con el conteo de vecinos para cada posición.
    """
    kernel = np.array([[1,1,1],
                       [1,0,1],
                       [1,1,1]], dtype=np.uint8)
    skel_int = skel.astype(np.uint8)
    degree = convolve2d(skel_int, kernel, mode='same', boundary='fill', fillvalue=0)
    return degree

def get_node_candidates(skel, degree):
    """
    Devuelve:
      - endpoints: píxeles con grado == 1
      - junctions: píxeles con grado >= 3
    """
    endpoints = np.argwhere((skel == True) & (degree == 1))
    junctions = np.argwhere((skel == True) & (degree >= 3))
    return endpoints, junctions

# ---------------------------------------------------
#        CLASIFICACIÓN ROBUSTA DE RAMIFICACIONES
# ---------------------------------------------------
def get_neighbors(r, c, shape):
    """Vecinos en 8-conectividad de (r,c) dentro de 'shape'."""
    neighbors = []
    for dr in [-1, 0, 1]:
        for dc in [-1, 0, 1]:
            if dr == 0 and dc == 0:
                continue
            rr, cc = r + dr, c + dc
            if 0 <= rr < shape[0] and 0 <= cc < shape[1]:
                neighbors.append((rr, cc))
    return neighbors

def classify_cluster(cluster_points, skel):
    """
    Dado un conjunto de píxeles 'cluster_points' que forman un clúster de ramificación,
    determinamos cuántas 'ramas' reales salen del clúster:
      - Hacemos BFS desde cada píxel externo (vecino del clúster) ignorando el clúster.
      - Si se unen BFS, es la misma rama.
      - Contamos cuántas ramas distintas hay.
    Retorna:
      - "bifurcacion" si hay 2 ramas
      - "trifurcacion" si hay 3
      - "trifurcacion" (o "multifurcacion") si hay >= 4
        (puedes personalizar si quieres distinguir 4, 5, etc.)
    """
    cluster_set = set(cluster_points)
    
    # 1) Encontrar vecinos "externos" (en el esqueleto pero fuera del clúster)
    outside_neighbors = []
    for (r, c) in cluster_points:
        for nr, nc in get_neighbors(r, c, skel.shape):
            if (nr, nc) not in cluster_set and skel[nr, nc]:
                outside_neighbors.append((nr, nc))
    
    # Quitar duplicados
    outside_neighbors = list(set(outside_neighbors))
    
    # 2) BFS para contar cuántas ramas independientes hay
    visited = set()
    branches = 0
    
    for pix in outside_neighbors:
        if pix in visited:
            continue
        # Nueva rama
        branches += 1
        
        # Expandimos BFS ignorando el clúster
        queue = [pix]
        while queue:
            cur = queue.pop()
            if cur in visited:
                continue
            visited.add(cur)
            
            # Vecinos en el esqueleto, fuera del clúster
            for nn in get_neighbors(cur[0], cur[1], skel.shape):
                if nn not in cluster_set and skel[nn[0], nn[1]] and nn not in visited:
                    queue.append(nn)
    
    # 3) Clasificación según el número de ramas
    if branches == 2:
        return "bifurcacion"
    elif branches == 3:
        return "trifurcacion"
    elif branches < 2:
        # Caso raro: cluster con 1 rama (puede ser un "endpoint grueso")
        return "bifurcacion"  # o "endpoint especial"
    else:
        # 4 o más ramas
        return "trifurcacion"  # O "multifurcacion" si prefieres

def merge_junctions(junctions, skel):
    """
    Fusiona píxeles de ramificación cercanos con DBSCAN y
    clasifica cada clúster según el número de ramas reales (BFS).
    Retorna:
      - merged_nodes: lista de centroides (row, col)
      - node_types: lista de strings ('bifurcacion', 'trifurcacion', etc.)
    """
    if len(junctions) == 0:
        return [], []
    
    clustering = DBSCAN(eps=EPSILON_DBSCAN, min_samples=MIN_SAMPLES).fit(junctions)
    labels = clustering.labels_
    
    merged_nodes = []
    node_types = []
    
    for label in np.unique(labels):
        cluster_points = junctions[labels == label]
        # Calcular el centroid
        centroid = np.mean(cluster_points, axis=0)
        centroid = np.rint(centroid).astype(int)
        
        # Clasificar cuántas ramas salen de este clúster
        cluster_list = [tuple(pt) for pt in cluster_points]
        tipo = classify_cluster(cluster_list, skel)
        
        merged_nodes.append(tuple(centroid))
        node_types.append(tipo)
    
    return merged_nodes, node_types

# ---------------------------------------------------
#   RDP (Ramer–Douglas–Peucker) para "rectificar"
# ---------------------------------------------------
def perpendicular_distance(p, a, b):
    """Distancia perpendicular del punto p al segmento [a, b]."""
    px, py = float(p[0]), float(p[1])
    ax, ay = float(a[0]), float(a[1])
    bx, by = float(b[0]), float(b[1])
    
    ABx = bx - ax
    ABy = by - ay
    APx = px - ax
    APy = py - ay
    
    ab_len = ABx*ABx + ABy*ABy
    if ab_len == 0:
        return math.dist(p, a)
    
    t = (APx*ABx + APy*ABy) / ab_len
    if t < 0:
        return math.dist(p, a)
    elif t > 1:
        return math.dist(p, b)
    else:
        proj = (ax + t*ABx, ay + t*ABy)
        return math.dist((px, py), proj)

def rdp_simplify(points, epsilon):
    """
    Implementación recursiva de Ramer–Douglas–Peucker.
    points: lista de (r, c)
    epsilon: tolerancia (mayor = menos puntos)
    """
    if len(points) < 3:
        return points
    
    first = points[0]
    last = points[-1]
    
    max_dist = 0
    max_idx = 0
    for i in range(1, len(points)-1):
        dist = perpendicular_distance(points[i], first, last)
        if dist > max_dist:
            max_dist = dist
            max_idx = i
    
    if max_dist > epsilon:
        left_part = rdp_simplify(points[:max_idx+1], epsilon)
        right_part = rdp_simplify(points[max_idx:], epsilon)
        return left_part[:-1] + right_part
    else:
        return [first, last]

# ---------------------------------------------------
#       TRAZADO DE CAMINOS Y CREACIÓN DE GRAFO
# ---------------------------------------------------
def trace_path(start_node, neighbor, skel, node_set, visited):
    """
    Avanza píxel a píxel por el esqueleto desde 'start_node'
    hasta encontrar otro nodo o no poder continuar.
    Retorna la lista de píxeles recorridos (incluye nodos).
    """
    path = [start_node]
    prev = start_node
    current = neighbor
    visited.add(current)
    path.append(current)
    
    while True:
        nbrs = [
            pt for pt in get_neighbors(current[0], current[1], skel.shape)
            if skel[pt[0], pt[1]] and pt != prev
        ]
        if len(nbrs) == 0:
            return path
        for nb in nbrs:
            if nb in node_set:
                path.append(nb)
                return path
        next_pt = None
        for nb in nbrs:
            if nb not in visited:
                next_pt = nb
                break
        if next_pt is None:
            return path
        visited.add(next_pt)
        path.append(next_pt)
        prev = current
        current = next_pt

def build_graph(skel, endpoints, merged_junctions, junction_types):
    """
    Construye un grafo:
      - Nodos: endpoints, bifurcaciones/trifurcaciones fusionadas
      - Aristas: cada tramo de nodo a nodo se simplifica con RDP
                 y se crean nodos intermedios para vértices extra.
    """
    G = nx.Graph()
    node_dict = {}
    node_id = 0
    
    # 1) Insertar endpoints
    for pt in endpoints:
        coord = tuple(pt)
        node_dict[coord] = {
            'id': node_id,
            'type': 'extremo',
            'coord': coord
        }
        node_id += 1
    
    # 2) Insertar ramificaciones fusionadas
    for i, pt in enumerate(merged_junctions):
        coord = tuple(pt)
        if coord in node_dict:
            continue
        node_dict[coord] = {
            'id': node_id,
            'type': junction_types[i],
            'coord': coord
        }
        node_id += 1
    
    node_set = set(node_dict.keys())
    visited = set()
    edges = []
    
    # 3) Trazar cada rama
    for node_coord, node_data in list(node_dict.items()):
        r, c = node_coord
        nbrs = [
            pt for pt in get_neighbors(r, c, skel.shape)
            if skel[pt[0], pt[1]]
        ]
        for nb in nbrs:
            if nb not in node_set and nb not in visited:
                path_pixels = trace_path(node_coord, nb, skel, node_set, visited)
                if len(path_pixels) < 2:
                    continue
                end_node = path_pixels[-1]
                if end_node not in node_set:
                    continue
                source_id = node_dict[node_coord]['id']
                target_id = node_dict[end_node]['id']
                
                # Polilínea completa (incluyendo nodos)
                full_path = path_pixels
                # Aplicar RDP
                simplified = rdp_simplify(full_path, RDP_EPSILON)
                
                # Crear nodos intermedios para vértices nuevos
                new_vertices = []
                for sp in simplified:
                    if sp not in node_set:
                        new_id = len(node_dict)
                        node_dict[sp] = {
                            'id': new_id,
                            'type': 'intermedio',
                            'coord': sp
                        }
                        node_set.add(sp)
                    new_vertices.append(sp)
                
                # Conectar cada par consecutivo
                for i_v in range(len(new_vertices) - 1):
                    src_coord = new_vertices[i_v]
                    dst_coord = new_vertices[i_v+1]
                    src_id = node_dict[src_coord]['id']
                    dst_id = node_dict[dst_coord]['id']
                    
                    edges.append({
                        'source': src_id,
                        'target': dst_id,
                        'intermediate_pixels': []
                    })
    
    # 4) Construir el grafo en NetworkX
    for nd in node_dict.values():
        G.add_node(nd['id'], **nd)
    for e in edges:
        G.add_edge(e['source'], e['target'], intermediate_pixels=e['intermediate_pixels'])
    
    return G, node_dict, edges

# ---------------------------------------------------
#       GUARDAR EN JSON Y VISUALIZAR
# ---------------------------------------------------
def save_json(node_dict, edges, out_path):
    data = {"nodes": [], "edges": []}
    
    for nd in node_dict.values():
        data["nodes"].append({
            "id": int(nd["id"]),
            "row": int(nd["coord"][0]),
            "col": int(nd["coord"][1]),
            "type": nd["type"]
        })
    
    for e in edges:
        data["edges"].append({
            "source": int(e["source"]),
            "target": int(e["target"]),
            "intermediate_pixels": e["intermediate_pixels"]  # aquí están vacíos, ya que RDP "rompe" la curva
        })
    
    with open(out_path, "w") as f:
        json.dump(data, f, indent=4)

def visualize_graph(skel, node_dict, edges, out_img_path):
    skel_vis = np.uint8(skel) * 255
    skel_vis = cv2.cvtColor(skel_vis, cv2.COLOR_GRAY2BGR)
    
    # Dibujar aristas (rectas) en amarillo
    for e in edges:
        src_id = e['source']
        dst_id = e['target']
        src_coord = None
        dst_coord = None
        
        # Buscar coords
        for nd in node_dict.values():
            if nd['id'] == src_id:
                src_coord = nd['coord']
            elif nd['id'] == dst_id:
                dst_coord = nd['coord']
        
        if src_coord is None or dst_coord is None:
            continue
        r1, c1 = src_coord
        r2, c2 = dst_coord
        cv2.line(skel_vis, (c1, r1), (c2, r2), COLOR_ARISTA, 1)
    
    # Dibujar nodos
    for nd in node_dict.values():
        r, c = nd['coord']
        tipo = nd['type']
        if tipo == 'extremo':
            color = COLOR_EXTREMO
        elif tipo == 'bifurcacion':
            color = COLOR_BIFURCACION
        elif tipo == 'trifurcacion':
            color = COLOR_TRIFURCACION
        else:
            color = COLOR_INTERMEDIO
        
        cv2.circle(skel_vis, (c, r), 3, color, -1)
    
    cv2.imwrite(out_img_path, skel_vis)

def process_image(img_path, output_json_dir, output_img_dir):
    binary_img = load_image(img_path)
    skel = get_skeleton(binary_img)
    degree = compute_degree(skel)
    
    endpoints, junctions = get_node_candidates(skel, degree)
    # Fusionar y clasificar robustamente las ramificaciones
    merged_junctions, junction_types = merge_junctions(junctions, skel)
    
    # Construir grafo con RDP
    G, node_dict, edges = build_graph(skel, endpoints, merged_junctions, junction_types)
    
    base_name = os.path.splitext(os.path.basename(img_path))[0]
    json_path = os.path.join(output_json_dir, base_name + ".json")
    img_out_path = os.path.join(output_img_dir, base_name + "_skeleton.png")
    
    save_json(node_dict, edges, json_path)
    visualize_graph(skel, node_dict, edges, img_out_path)
    print(f"Procesada {img_path} -> JSON: {json_path}, IMG: {img_out_path}")

def main():
    input_dir = "datos/etiquetas"
    output_json_dir = "output/json"
    output_img_dir = "output/imagenes"
    os.makedirs(output_json_dir, exist_ok=True)
    os.makedirs(output_img_dir, exist_ok=True)
    
    # Extensiones de imágenes
    extensions = ["*.pgm", "*.png", "*.jpg", "*.jpeg"]
    img_paths = []
    for ext in extensions:
        img_paths.extend(glob.glob(os.path.join(input_dir, ext)))
    
    for img_path in img_paths:
        process_image(img_path, output_json_dir, output_img_dir)

if __name__ == "__main__":
    main()
