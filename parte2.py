import os
import json
import cv2
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from skimage.morphology import skeletonize
from sklearn.cluster import DBSCAN

# Parámetros ajustables
EPSILON_DBSCAN = 5         # radio en píxeles para fusionar nodos cercanos
RDP_EPSILON = 2            # tolerancia para la simplificación de trayectorias con RDP
INPUT_DIR = 'datos/etiquetas'
OUTPUT_JSON_DIR = 'output/json'
OUTPUT_IMG_DIR = 'output/imagenes'

# Aseguramos que existan los directorios de salida
os.makedirs(OUTPUT_JSON_DIR, exist_ok=True)
os.makedirs(OUTPUT_IMG_DIR, exist_ok=True)

def rdp(points, epsilon):
    """
    Implementación recursiva del algoritmo Ramer-Douglas-Peucker.
    points: lista de puntos [ [row, col], ... ]
    epsilon: tolerancia (en píxeles)
    Devuelve una lista simplificada de puntos.
    """
    if len(points) < 3:
        return points

    def point_line_distance(point, start, end):
        if np.allclose(start, end):
            return np.linalg.norm(np.array(point) - np.array(start))
        else:
            num = abs((end[1] - start[1])*(start[0] - point[0]) - 
                      (start[1] - point[1])*(end[0] - start[0]))
            den = np.linalg.norm(np.array(end) - np.array(start))
            return num / den

    start, end = points[0], points[-1]
    dmax = 0
    index = 0
    for i in range(1, len(points)-1):
        d = point_line_distance(points[i], start, end)
        if d > dmax:
            index = i
            dmax = d

    if dmax >= epsilon:
        rec_results1 = rdp(points[:index+1], epsilon)
        rec_results2 = rdp(points[index:], epsilon)
        return rec_results1[:-1] + rec_results2
    else:
        return [start, end]

def build_skeleton_graph(skel):
    """
    Construye un grafo (usando networkx) a partir de la imagen esqueletizada.
    Cada píxel True se agrega como nodo (con coordenadas (row, col))
    y se conectan con sus vecinos (8-conectividad).
    """
    G = nx.Graph()
    rows, cols = skel.shape
    neighbors = [(-1, -1), (-1, 0), (-1, 1),
                 (0, -1),           (0, 1),
                 (1, -1),  (1, 0),  (1, 1)]
    
    ys, xs = np.where(skel)
    for y, x in zip(ys, xs):
        G.add_node((y, x))
    
    for y, x in zip(ys, xs):
        for dy, dx in neighbors:
            ny, nx_ = y + dy, x + dx
            if 0 <= ny < rows and 0 <= nx_ < cols and skel[ny, nx_]:
                G.add_edge((y, x), (ny, nx_))
    return G

def get_candidate_nodes(G):
    """
    Devuelve una lista de nodos candidatos a ser nodos clave (grado != 2).
    Endpoints (grado=1) y ramificaciones (grado>=3).
    """
    candidates = []
    for node in G.nodes():
        if G.degree(node) != 2:
            candidates.append(node)
    return candidates

def extract_paths(G, candidate_set):
    """
    Extrae caminos (trayectorias) entre nodos candidatos en el grafo G.
    Se hace un DFS desde cada candidato a lo largo de nodos de grado 2
    hasta llegar a otro candidato.
    Retorna una lista de caminos, cada uno es [(row, col), ...].
    """
    paths = []
    visited_edges = set()

    def dfs(current, previous, path):
        if current in candidate_set and len(path) > 0:
            return path + [current]
        for neighbor in G.neighbors(current):
            edge = tuple(sorted([current, neighbor]))
            if neighbor == previous or edge in visited_edges:
                continue
            visited_edges.add(edge)
            res_path = dfs(neighbor, current, path + [current])
            if res_path is not None:
                return res_path
        return None

    for cand in candidate_set:
        for neighbor in G.neighbors(cand):
            edge = tuple(sorted([cand, neighbor]))
            if edge in visited_edges:
                continue
            visited_edges.add(edge)
            path = dfs(neighbor, cand, [cand])
            if path is not None and path[-1] != cand:
                paths.append(path)
    return paths

def cluster_nodes(candidates, eps):
    """
    Usa DBSCAN para agrupar nodos candidatos cercanos.
    Devuelve:
      - clusters: diccionario { nodo: cluster_label }
      - centroids: diccionario { cluster_label: (row, col) }
    """
    if not candidates:
        return {}, {}
    coords = np.array(candidates)
    clustering = DBSCAN(eps=eps, min_samples=1).fit(coords)
    labels = clustering.labels_
    clusters = {tuple(candidates[i]): labels[i] for i in range(len(candidates))}
    centroids = {}
    for label in set(labels):
        pts = coords[labels == label]
        centroid = tuple(np.mean(pts, axis=0).astype(int))
        centroids[label] = centroid
    return clusters, centroids

def classify_node(degree):
    """
    Clasifica el nodo según su número de ramas (degree).
    """
    if degree == 1:
        return "extremo"
    elif degree == 2:
        return "bifurcacion"
    elif degree == 3:
        return "bifurcacion"
    else:
        return "trifurcacion"

def process_image(image_path):
    """
    Procesa una imagen: esqueletiza, extrae caminos, fusiona candidatos con DBSCAN,
    aplica RDP, construye el grafo con nodos intermedios y genera la visualización.
    Retorna (grafo_dict, vis_img).
    """
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Error al leer {image_path}")
        return None, None

    _, bin_img = cv2.threshold(img, 127, 1, cv2.THRESH_BINARY)
    skel = skeletonize(bin_img).astype(np.uint8)
    G = build_skeleton_graph(skel)
    candidate_nodes = get_candidate_nodes(G)
    candidate_set = set(candidate_nodes)
    paths = extract_paths(G, candidate_set)

    clusters, centroids = cluster_nodes(candidate_nodes, EPSILON_DBSCAN)
    candidate_to_merged = {node: centroids[clusters[node]] for node in candidate_nodes}

    merged_nodes = {c: {"id": None, "row": c[0], "col": c[1], "type": None, "degree": 0} for c in centroids.values()}
    intermediate_nodes = {}
    connection_count = {c: 0 for c in merged_nodes}
    edges = []

    for path in paths:
        if len(path) < 2:
            continue
        start_coord = candidate_to_merged.get(path[0], path[0])
        end_coord   = candidate_to_merged.get(path[-1], path[-1])
        if start_coord == end_coord:
            continue
        pts = [[p[0], p[1]] for p in path]
        simplified = rdp(pts, RDP_EPSILON)
        inter_coords = [tuple(p) for p in simplified[1:-1]]
        edges.append({
            "source": start_coord,
            "target": end_coord,
            "intermediate_pixels": inter_coords
        })
        connection_count[start_coord] += 1
        connection_count[end_coord]   += 1

    nodes_list = []
    id_counter = 0
    for coord, data in merged_nodes.items():
        deg = connection_count.get(coord, 0)
        data["id"] = id_counter
        data["type"] = classify_node(deg)
        data["degree"] = deg
        nodes_list.append({
            "id": id_counter,
            "row": int(data["row"]),
            "col": int(data["col"]),
            "type": data["type"]
        })
        merged_nodes[coord] = data
        id_counter += 1

    def get_or_create_intermediate_node(pt):
        if pt not in intermediate_nodes:
            intermediate_nodes[pt] = {
                "id": None,
                "row": pt[0],
                "col": pt[1],
                "type": "intermedio"
            }
        return intermediate_nodes[pt]

    final_edges = []
    for e in edges:
        src_coord = e["source"]
        tgt_coord = e["target"]
        inter_coords = e["intermediate_pixels"]
        chain = [src_coord] + inter_coords + [tgt_coord]
        chain_ids = []
        for i, pt in enumerate(chain):
            if i == 0:
                chain_ids.append(merged_nodes[pt]["id"])
            elif i == len(chain) - 1:
                chain_ids.append(merged_nodes[pt]["id"])
            else:
                node_data = get_or_create_intermediate_node(pt)
                chain_ids.append(None)
        for i, pt in enumerate(chain):
            if i in (0, len(chain)-1):
                continue
            if intermediate_nodes[pt]["id"] is None:
                intermediate_nodes[pt]["id"] = id_counter
                nodes_list.append({
                    "id": id_counter,
                    "row": pt[0],
                    "col": pt[1],
                    "type": "intermedio"
                })
                chain_ids[i] = id_counter
                id_counter += 1
            else:
                chain_ids[i] = intermediate_nodes[pt]["id"]
        for i in range(len(chain_ids)-1):
            final_edges.append({
                "source": chain_ids[i],
                "target": chain_ids[i+1]
            })

    grafo_dict = {
        "nodes": nodes_list,
        "edges": final_edges
    }

    vis_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    for edge in final_edges:
        src_node = next((n for n in nodes_list if n["id"] == edge["source"]), None)
        tgt_node = next((n for n in nodes_list if n["id"] == edge["target"]), None)
        if src_node and tgt_node:
            pt1 = (src_node["col"], src_node["row"])
            pt2 = (tgt_node["col"], tgt_node["row"])
            cv2.line(vis_img, pt1, pt2, (0,255,255), 1)
    for node in nodes_list:
        center = (node["col"], node["row"])
        if node["type"] == "extremo":
            color = (0,255,0)
        elif node["type"] == "bifurcacion":
            color = (0,0,255)
        elif node["type"] == "trifurcacion":
            color = (255,0,0)
        else:
            color = (128,128,128)
        cv2.circle(vis_img, center, 3, color, -1)

    return grafo_dict, vis_img

def save_results(grafo_dict, vis_img, base_name):
    """
    Guarda el grafo en JSON y la imagen de visualización.
    Se añade un conversor para transformar np.int64 a int nativo.
    """
    json_path = os.path.join(OUTPUT_JSON_DIR, base_name + ".json")
    with open(json_path, "w") as f:
        json.dump(grafo_dict, f, indent=2, default=lambda o: int(o) if isinstance(o, np.int64) else o)
    
    img_path = os.path.join(OUTPUT_IMG_DIR, base_name + ".png")
    cv2.imwrite(img_path, vis_img)
    print(f"Procesado {base_name}: JSON y PNG guardados.")

def main():
    for file in os.listdir(INPUT_DIR):
        if file.lower().endswith(".pgm"):
            image_path = os.path.join(INPUT_DIR, file)
            print(f"Procesando {file}...")
            grafo_dict, vis_img = process_image(image_path)
            if grafo_dict is not None:
                base_name = os.path.splitext(file)[0]
                save_results(grafo_dict, vis_img, base_name)

if __name__ == "__main__":
    main()
