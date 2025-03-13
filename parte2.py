import cv2
import numpy as np
import os
import json

from skimage.morphology import skeletonize

###################################
# Parámetros globales ajustables
###################################
UMBRAL_ENDPOINT = 1
UMBRAL_BIFURCACION = 3
UMBRAL_TRIFURCACION = 4

# Umbral de ángulo (en grados) para decidir cuándo crear un nodo intermedio
# por curvatura. Ejemplo: 30 grados
UMBRAL_CURVATURA = 47.0

###################################
# Funciones auxiliares
###################################

def load_image(path):
    """Lee la imagen en escala de grises y la binariza."""
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"No se pudo cargar la imagen: {path}")
    _, bin_img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    return bin_img

def skeletonize_image(bin_img):
    """Convierte la imagen binaria a booleano y la esqueletiza con skimage."""
    bool_img = (bin_img > 0)
    skel = skeletonize(bool_img)
    return skel

def get_neighbors_8(x, y, rows, cols):
    """Devuelve los vecinos en 8 direcciones de (x, y) dentro de los límites."""
    for dx in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            if dx == 0 and dy == 0:
                continue
            nx_, ny_ = x + dx, y + dy
            if 0 <= nx_ < rows and 0 <= ny_ < cols:
                yield nx_, ny_

def count_neighbors(skel, x, y):
    """Cuenta cuántos vecinos (8-conectividad) están activos en skel[x, y]."""
    cnt = 0
    for nx_, ny_ in get_neighbors_8(x, y, *skel.shape):
        if skel[nx_, ny_]:
            cnt += 1
    return cnt

def detect_main_nodes(skel):
    """
    Detecta nodos principales (endpoint, bifurcacion, trifurcacion) basados
    en la cantidad de vecinos. Devuelve un diccionario:
      (x, y) -> 'endpoint' | 'bifurcacion' | 'trifurcacion'
    """
    rows, cols = skel.shape
    main_nodes = {}
    for x in range(rows):
        for y in range(cols):
            if skel[x, y]:
                n_vec = count_neighbors(skel, x, y)
                if n_vec <= UMBRAL_ENDPOINT:
                    main_nodes[(x, y)] = 'endpoint'
                elif n_vec == UMBRAL_BIFURCACION:
                    main_nodes[(x, y)] = 'bifurcacion'
                elif n_vec >= UMBRAL_TRIFURCACION:
                    main_nodes[(x, y)] = 'trifurcacion'
    return main_nodes

def extract_paths_between_main_nodes(skel, main_nodes):
    """
    Recorre el esqueleto y extrae todos los caminos que conectan nodos principales.
    Cada camino es una lista de píxeles [(x1,y1), (x2,y2), ...] que inicia y termina
    en un nodo principal (o se corta si no encuentra otro).
    
    Retorna una lista de caminos. Cada camino es una lista de (x, y).
    
    Estrategia:
      - Para cada nodo principal, hacemos un DFS/BFS para buscar otro nodo principal,
        siguiendo únicamente píxeles del esqueleto que no sean (a priori) un nodo principal
        (excepto al final).
      - Marcamos como visitadas las “ramas” para no duplicar caminos.
    """
    visited = set()
    paths = []
    rows, cols = skel.shape
    
    # Convertimos main_nodes en un set para consultas rápidas
    main_set = set(main_nodes.keys())
    
    def dfs_path(start, visited_global):
        """DFS desde start hasta encontrar otro nodo principal o un callejón sin salida."""
        stack = [(start, [start])]  # (pixel_actual, camino_actual)
        found_paths = []
        
        while stack:
            current, path = stack.pop()
            
            # Si current es un nodo principal y no es el inicio, cerramos el camino
            if current != start and current in main_set:
                found_paths.append(path)
                continue
            
            # Explorar vecinos
            for nx_, ny_ in get_neighbors_8(current[0], current[1], rows, cols):
                if not skel[nx_, ny_]:
                    continue  # no es parte del esqueleto
                if (nx_, ny_) in path:
                    continue  # ya en el camino actual
                # Para evitar que arranque un DFS inverso desde un punto ya visitado en otro camino,
                # podríamos marcar visited_global, pero hay que diseñar la lógica cuidadosamente.
                
                # Extendemos el camino
                new_path = path + [(nx_, ny_)]
                stack.append(((nx_, ny_), new_path))
        
        return found_paths
    
    # Para no duplicar caminos, llevamos un registro de pares (min_node, max_node).
    used_pairs = set()
    
    for node_coord in main_set:
        # Iniciamos un DFS para encontrar trayectos
        result_paths = dfs_path(node_coord, visited)
        
        for p in result_paths:
            # p inicia en node_coord y termina en otro nodo principal
            start_node = p[0]
            end_node   = p[-1]
            # Ordenamos la tupla de nodos para no duplicar
            pair = tuple(sorted([start_node, end_node]))
            if pair not in used_pairs:
                used_pairs.add(pair)
                paths.append(p)
    
    return paths

def angle_between_three_points(p1, p2, p3):
    """
    Dado tres puntos (x1,y1), (x2,y2), (x3,y3),
    calcula el ángulo en 'p2' entre el vector p2->p1 y p2->p3.
    Retorna el ángulo en grados [0..180].
    """
    import math
    
    # Vectores
    v1 = (p1[0] - p2[0], p1[1] - p2[1])  # p2->p1
    v2 = (p3[0] - p2[0], p3[1] - p2[1])  # p2->p3
    
    # Producto punto y magnitudes
    dot = v1[0]*v2[0] + v1[1]*v2[1]
    mag1 = math.sqrt(v1[0]**2 + v1[1]**2)
    mag2 = math.sqrt(v2[0]**2 + v2[1]**2)
    if mag1*mag2 == 0:
        return 0  # Uno de los vectores es 0 => no hay ángulo definido
    
    cos_angle = dot / (mag1*mag2)
    # Evitar problemas de precisión numérica
    cos_angle = max(min(cos_angle, 1.0), -1.0)
    angle = np.degrees(np.arccos(cos_angle))
    return angle

def insert_curvature_nodes(path, main_nodes, angle_threshold=UMBRAL_CURVATURA):
    """
    Dado un camino de píxeles (path) que inicia y termina en nodos principales,
    inserta nodos intermedios cuando la curvatura (cambio de ángulo) supere
    angle_threshold.
    
    Retorna la lista de nodos (x,y) resultantes. El primero y último
    son los nodos principales originales. Entre ellos se insertan
    nodos intermedios en las curvas.
    """
    if len(path) < 3:
        return path  # no hay espacio para medir ángulos
    
    # El primer y último pixel son nodos principales
    final_nodes = [path[0]]
    
    for i in range(1, len(path)-1):
        p_prev = path[i-1]
        p_curr = path[i]
        p_next = path[i+1]
        
        # Si p_curr es un nodo principal, lo agregamos de todos modos
        if p_curr in main_nodes and p_curr not in (path[0], path[-1]):
            final_nodes.append(p_curr)
            continue
        
        # Calculamos el ángulo en p_curr
        ang = angle_between_three_points(p_prev, p_curr, p_next)
        if ang <= 180.0 - angle_threshold:
            # Significa que hay una "curva" significativa
            final_nodes.append(p_curr)
        # Si no supera el umbral, seguimos de largo
    
    final_nodes.append(path[-1])
    return final_nodes

def build_graph_with_curvature(skel, angle_threshold=UMBRAL_CURVATURA):
    """
    Construye un grafo donde:
      - Nodos principales: endpoints, bifurcaciones, trifurcaciones
      - Se conectan trayectos. A lo largo de cada trayecto, se insertan
        nodos intermedios en curvas que superen el umbral.
    
    Retorna:
      nodes_list: [{id, x, y, type}, ...]
      edges_list: [(id_n1, id_n2), ...]
    """
    main_nodes_dict = detect_main_nodes(skel)  # (x,y)-> 'endpoint'/'bifurcacion'/'trifurcacion'
    paths = extract_paths_between_main_nodes(skel, main_nodes_dict)
    
    # Para construir el grafo final
    nodes_list = []
    edges_list = []
    
    # Mapeo (x,y) -> id, para no duplicar nodos
    node_id_counter = 0
    coord_to_id = {}
    
    def add_node(coord, node_type):
        nonlocal node_id_counter
        if coord not in coord_to_id:
            coord_to_id[coord] = node_id_counter
            nodes_list.append({
                'id': node_id_counter,
                'x': float(coord[0]),
                'y': float(coord[1]),
                'type': node_type
            })
            node_id_counter += 1
        else:
            # Si ya existe, podríamos querer “ascender” su tipo si ahora descubrimos que es un principal
            existing_id = coord_to_id[coord]
            # Buscamos en nodes_list
            for nd in nodes_list:
                if nd['id'] == existing_id:
                    # Si era intermedio y ahora sabemos que es endpoint/bif/trif => actualizar
                    if nd['type'] == 'intermedio' and node_type != 'intermedio':
                        nd['type'] = node_type
                    break
    
    # 1) Recorremos cada camino, insertamos nodos intermedios por curvatura
    for path in paths:
        # Insertar nodos por curvatura
        new_path_nodes = insert_curvature_nodes(path, main_nodes_dict, angle_threshold)
        
        # 2) Añadir estos nodos al grafo
        #    El tipo vendrá de main_nodes_dict si es principal, sino 'intermedio'
        typed_nodes = []
        for c in new_path_nodes:
            node_type = main_nodes_dict.get(c, 'intermedio')  # si no está, es intermedio
            add_node(c, node_type)
            typed_nodes.append(c)
        
        # 3) Conectar en edges_list en orden
        for i in range(len(typed_nodes)-1):
            c1 = typed_nodes[i]
            c2 = typed_nodes[i+1]
            id1 = coord_to_id[c1]
            id2 = coord_to_id[c2]
            edge = tuple(sorted((id1, id2)))
            if edge not in edges_list:
                edges_list.append(edge)
    
    return nodes_list, edges_list

def process_image(path, output_json_dir, output_img_dir):
    """Procesa la imagen, construye el grafo y genera JSON + imagen coloreada."""
    # 1) Cargar y esqueletizar
    bin_img = load_image(path)
    skel = skeletonize_image(bin_img)
    
    # 2) Construir grafo con curvatura
    nodes_list, edges_list = build_graph_with_curvature(skel, UMBRAL_CURVATURA)
    
    # 3) Extraer listas de nodos por tipo
    endpoints = []
    bifurcaciones = []
    trifurcaciones = []
    intermedios = []
    for nd in nodes_list:
        if nd['type'] == 'endpoint':
            endpoints.append(nd['id'])
        elif nd['type'] == 'bifurcacion':
            bifurcaciones.append(nd['id'])
        elif nd['type'] == 'trifurcacion':
            trifurcaciones.append(nd['id'])
        else:
            intermedios.append(nd['id'])
    
    # 4) Armar JSON
    edges_json = [{'source': s, 'target': t} for (s, t) in edges_list]
    graph_data = {
        'nodes': nodes_list,
        'endpoints': endpoints,
        'bifurcaciones': bifurcaciones,
        'trifurcaciones': trifurcaciones,
        'intermedios': intermedios,
        'edges': edges_json
    }
    base_name = os.path.splitext(os.path.basename(path))[0]
    json_path = os.path.join(output_json_dir, base_name + ".json")
    with open(json_path, 'w') as f:
        json.dump(graph_data, f, indent=2)
    
    # 5) Generar imagen coloreada
    out_img_path = os.path.join(output_img_dir, base_name + "_colored.png")
    generate_colored_image(skel, nodes_list, edges_list, out_img_path)

def generate_colored_image(skel, nodes_list, edges_list, out_path):
    """
    Crea una imagen RGB con:
      - Aristas en amarillo
      - endpoint en verde
      - bifurcacion en rojo
      - trifurcacion en azul
      - intermedio en gris
    """
    rows, cols = skel.shape
    # Convertimos el esqueleto a 3 canales
    skel_rgb = np.dstack([skel*255, skel*255, skel*255]).astype(np.uint8)
    
    # Colores en BGR (OpenCV)
    color_map = {
        'endpoint': (0, 255, 0),       # verde
        'bifurcacion': (0, 0, 255),    # rojo
        'trifurcacion': (255, 0, 0),   # azul
        'intermedio': (128, 128, 128)  # gris
    }
    color_edges = (0, 255, 255)       # amarillo
    
    # Diccionario para acceder rápido a (row, col)
    node_dict = {n['id']: (int(n['x']), int(n['y']), n['type']) for n in nodes_list}
    
    import cv2
    
    # Dibujamos aristas
    for (n1, n2) in edges_list:
        x1, y1, _ = node_dict[n1]
        x2, y2, _ = node_dict[n2]
        cv2.line(skel_rgb, (y1, x1), (y2, x2), color_edges, 1)
    
    # Dibujamos nodos
    for nd in nodes_list:
        nid = nd['id']
        x_, y_ = int(nd['x']), int(nd['y'])
        t_ = nd['type']
        c_ = color_map.get(t_, (255, 255, 255))
        cv2.circle(skel_rgb, (y_, x_), 2, c_, -1)
    
    cv2.imwrite(out_path, skel_rgb)

def main():
    input_dir = "datos/etiquetas"
    output_json_dir = "salida/json"
    output_img_dir = "salida/imagenes"
    
    os.makedirs(output_json_dir, exist_ok=True)
    os.makedirs(output_img_dir, exist_ok=True)
    
    for file_name in os.listdir(input_dir):
        if file_name.lower().endswith((".pgm", ".png", ".jpg", ".jpeg")):
            path = os.path.join(input_dir, file_name)
            print("Procesando:", path)
            try:
                process_image(path, output_json_dir, output_img_dir)
            except Exception as e:
                print(f"Error procesando {path}: {e}")

if __name__ == "__main__":
    main()
