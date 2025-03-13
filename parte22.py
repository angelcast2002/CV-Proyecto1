import os
import glob
import cv2
import json
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from skimage.morphology import skeletonize
from scipy.signal import convolve2d
from sklearn.cluster import DBSCAN

# ----------- COLORES EN BGR -----------
COLOR_EXTREMO = (0, 255, 0)       # Verde
COLOR_BIFURCACION = (0, 0, 255)   # Rojo
COLOR_TRIFURCACION = (255, 0, 0)  # Azul
COLOR_INTERMEDIO = (128, 128, 128)# Gris
COLOR_ARISTA = (0, 255, 255)      # Amarillo

# Umbrales y parámetros ajustables
EPSILON = 5         # Distancia máxima en píxeles para fusionar ramificaciones con DBSCAN
MIN_SAMPLES = 1     # Para DBSCAN
MAX_SEGMENT_LEN = 20  # Máxima distancia (en píxeles) sin insertar un nodo intermedio

def load_image(path):
    """Lee la imagen en escala de grises y la umbraliza a binaria."""
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    # Si tu ground truth es 0 y 255, con un threshold de 127 basta.
    _, binary = cv2.threshold(img, 127, 1, cv2.THRESH_BINARY)
    return binary.astype(bool)

def get_skeleton(binary_img):
    """Obtiene el esqueleto de la imagen binaria."""
    return skeletonize(binary_img)

def compute_degree(skel):
    """Convoluciona para contar cuántos vecinos (8-conectividad) tiene cada píxel del esqueleto."""
    kernel = np.array([[1,1,1],
                       [1,0,1],
                       [1,1,1]], dtype=np.uint8)
    skel_int = skel.astype(np.uint8)
    degree = convolve2d(skel_int, kernel, mode='same', boundary='fill', fillvalue=0)
    return degree

def get_node_candidates(skel, degree):
    """
    Identifica:
      - Endpoints: píxeles con grado == 1
      - Ramificaciones: píxeles con grado >= 3
    """
    endpoints = np.argwhere((skel == True) & (degree == 1))
    junctions = np.argwhere((skel == True) & (degree >= 3))
    return endpoints, junctions

def merge_junctions(junctions, degree):
    """
    Fusiona ramificaciones muy cercanas con DBSCAN.
    Cada cluster se reduce a un único 'centroid'.
    Luego, clasifica como bifurcación (max_deg == 3) o trifurcación (max_deg >= 4).
    """
    if len(junctions) == 0:
        return [], []
    
    clustering = DBSCAN(eps=EPSILON, min_samples=MIN_SAMPLES).fit(junctions)
    labels = clustering.labels_
    merged_nodes = []
    node_types = []

    for label in np.unique(labels):
        cluster_points = junctions[labels == label]
        centroid = np.mean(cluster_points, axis=0)
        centroid = np.rint(centroid).astype(int)
        # Revisa el grado máximo dentro del cluster
        max_deg = 0
        for pt in cluster_points:
            d = degree[pt[0], pt[1]]
            if d > max_deg:
                max_deg = d
        
        # Clasificación según grado máximo
        if max_deg == 3:
            nodo_tipo = "bifurcacion"
        else:
            nodo_tipo = "trifurcacion"
        
        merged_nodes.append(tuple(centroid))
        node_types.append(nodo_tipo)

    return merged_nodes, node_types

def get_neighbors(r, c, shape):
    """Devuelve los vecinos (8-conectividad) válidos de un píxel (r, c)."""
    neighbors = []
    for dr in [-1, 0, 1]:
        for dc in [-1, 0, 1]:
            if dr == 0 and dc == 0:
                continue
            rr, cc = r + dr, c + dc
            if 0 <= rr < shape[0] and 0 <= cc < shape[1]:
                neighbors.append((rr, cc))
    return neighbors

def trace_path(start_node, next_pixel, skel, node_set, visited, node_dict, max_segment_len):
    """
    Avanza píxel a píxel hasta llegar a otro nodo.
    En el camino, puede insertar nodos intermedios si la trayectoria supera 'max_segment_len'.
    
    Retorna:
      - path (lista de nodos intermedios + el nodo final)
      - end_node (el nodo final encontrado o None si no encuentra)
    """
    path_pixels = [start_node]  # incluye el nodo inicial como referencia
    prev = start_node
    current = next_pixel
    length_count = 0  # para medir longitud del segmento recorrido

    while True:
        visited.add(current)
        path_pixels.append(current)
        length_count += 1
        
        # Busca vecinos en el esqueleto que no sean el píxel anterior
        nbrs = [pt for pt in get_neighbors(current[0], current[1], skel.shape) 
                if skel[pt[0], pt[1]] and pt != prev]
        
        if len(nbrs) == 0:
            # Se quedó sin camino
            return path_pixels, None
        
        # Si alguno de los vecinos es un nodo, detenemos la marcha
        for nb in nbrs:
            if nb in node_set:
                # Hemos llegado a otro nodo
                path_pixels.append(nb)
                return path_pixels, nb
        
        # De lo contrario, seguimos con un vecino que no haya sido visitado
        next_pt = None
        for nb in nbrs:
            if nb not in visited:
                next_pt = nb
                break
        
        if next_pt is None:
            # Todos los vecinos ya se visitaron, fin de la rama
            return path_pixels, None
        
        prev = current
        current = next_pt
        
        # --- Inserción de nodo intermedio si excede 'max_segment_len' ---
        if length_count >= max_segment_len:
            # Creamos un nodo intermedio
            mid_coord = current
            new_id = len(node_dict)  # ID nuevo
            node_dict[mid_coord] = {
                'id': new_id,
                'type': 'intermedio',
                'coord': mid_coord
            }
            # Lo añadimos a node_set para que se reconozca como nodo
            node_set.add(mid_coord)
            # Reiniciamos el conteo de longitud
            length_count = 0
            # Retornamos el camino hasta este nodo intermedio,
            # y él se convierte en 'nodo final' de este sub-tramo
            path_pixels.append(mid_coord)
            return path_pixels, mid_coord

def build_graph(skel, endpoints, merged_junctions, junction_types, max_segment_len=20):
    """
    Construye el grafo a partir de:
      - 'endpoints' (nodos extremos)
      - 'merged_junctions' (bifurcaciones/trifurcaciones)
    y traza aristas entre nodos.
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
    
    # 2) Insertar nodos de ramificación fusionados
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
    
    # Conjunto de nodos (coordenadas) para búsquedas rápidas
    node_set = set(node_dict.keys())
    
    # 3) Trazado de aristas (branch tracing)
    visited = set()
    edges = []
    
    for node_coord, node_data in list(node_dict.items()):
        r, c = node_coord
        # Vecinos en el esqueleto
        nbrs = [pt for pt in get_neighbors(r, c, skel.shape) if skel[pt[0], pt[1]]]
        
        # Por cada vecino que no sea un nodo, iniciamos un trazado
        for nb in nbrs:
            if nb not in node_set and nb not in visited:
                path_pixels, end_node = trace_path(
                    start_node=node_coord,
                    next_pixel=nb,
                    skel=skel,
                    node_set=node_set,
                    visited=visited,
                    node_dict=node_dict,
                    max_segment_len=max_segment_len
                )
                
                if end_node is not None:
                    # Tenemos un nodo final
                    source_id = node_dict[node_coord]['id']
                    target_id = node_dict[end_node]['id']
                    # Quitamos el primer y último píxel, que son nodos
                    # El resto son píxeles intermedios del trayecto
                    intermediate = path_pixels[1:-1] if len(path_pixels) > 2 else []
                    edges.append({
                        'source': source_id,
                        'target': target_id,
                        'intermediate_pixels': intermediate
                    })
    
    # 4) Construir el grafo en NetworkX
    for nd in node_dict.values():
        G.add_node(nd['id'], **nd)
    for e in edges:
        G.add_edge(e['source'], e['target'], intermediate_pixels=e['intermediate_pixels'])
    
    return G, node_dict, edges

def save_json(node_dict, edges, out_path):
    """Convierte a JSON serializable (int nativo) y guarda en disco."""
    data = {"nodes": [], "edges": []}
    
    for nd in node_dict.values():
        data["nodes"].append({
            "id": int(nd["id"]),
            "row": int(nd["coord"][0]),
            "col": int(nd["coord"][1]),
            "type": nd["type"]
        })
    
    for e in edges:
        inter_list = []
        for px in e["intermediate_pixels"]:
            inter_list.append([int(px[0]), int(px[1])])
        
        data["edges"].append({
            "source": int(e["source"]),
            "target": int(e["target"]),
            "intermediate_pixels": inter_list
        })
    
    with open(out_path, "w") as f:
        json.dump(data, f, indent=4)

def visualize_graph(skel, node_dict, edges, out_img_path):
    """
    Genera una imagen en color que muestra:
      - El esqueleto
      - Las aristas en amarillo
      - Nodos en verde/rojo/azul/gris, según su tipo.
    """
    skel_vis = np.uint8(skel) * 255
    skel_vis = cv2.cvtColor(skel_vis, cv2.COLOR_GRAY2BGR)
    
    # 1) Dibujar aristas
    for e in edges:
        # Obtener coordenadas (r, c)
        src = None
        tgt = None
        for nd in node_dict.values():
            if nd['id'] == e['source']:
                src = nd['coord']
            elif nd['id'] == e['target']:
                tgt = nd['coord']
        
        if src is None or tgt is None:
            continue
        
        # Arma la secuencia de puntos a dibujar
        pts = [src] + e["intermediate_pixels"] + [tgt]
        for i in range(len(pts) - 1):
            r1, c1 = pts[i]
            r2, c2 = pts[i+1]
            cv2.line(skel_vis, (c1, r1), (c2, r2), COLOR_ARISTA, 1)
    
    # 2) Dibujar nodos
    for nd in node_dict.values():
        r, c = nd['coord']
        if nd['type'] == 'extremo':
            color = COLOR_EXTREMO
        elif nd['type'] == 'bifurcacion':
            color = COLOR_BIFURCACION
        elif nd['type'] == 'trifurcacion':
            color = COLOR_TRIFURCACION
        else:
            # "intermedio"
            color = COLOR_INTERMEDIO
        
        cv2.circle(skel_vis, (c, r), 3, color, -1)
    
    cv2.imwrite(out_img_path, skel_vis)

def process_image(img_path, output_json_dir, output_img_dir):
    """Procesa una imagen binaria (pgm, png, etc.) y genera el grafo + visualización."""
    binary_img = load_image(img_path)
    skel = get_skeleton(binary_img)
    degree = compute_degree(skel)
    
    # 1) Identificar endpoints y ramificaciones
    endpoints, junctions = get_node_candidates(skel, degree)
    
    # 2) Fusionar ramificaciones cercanas
    merged_junctions, junction_types = merge_junctions(junctions, degree)
    
    # 3) Construir el grafo con subdivisión (nodos intermedios)
    G, node_dict, edges = build_graph(skel, endpoints, merged_junctions, junction_types,
                                      max_segment_len=MAX_SEGMENT_LEN)
    
    # 4) Guardar resultados
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
    
    # Buscar imágenes con extensiones comunes
    extensions = ["*.pgm", "*.png", "*.jpg", "*.jpeg"]
    img_paths = []
    for ext in extensions:
        img_paths.extend(glob.glob(os.path.join(input_dir, ext)))
    
    for img_path in img_paths:
        process_image(img_path, output_json_dir, output_img_dir)

if __name__ == "__main__":
    main()
