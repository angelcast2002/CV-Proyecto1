# Proyecto 1 - Visión por Computadora

## Autores
- **Ángel Castellanos**
- **Diego Morales**
- **Alejandro Azurdia**

**Universidad del Valle de Guatemala**  
**Computer Vision - 2025**  
**Fecha de entrega: 13 de marzo de 2025**

---

## 1. Implementación de Algoritmos de Binarización


---

## 2. Discretización de la Estructura Arterial

Este proyecto tiene como objetivo construir un **grafo estructurado** a partir de imágenes binarizadas de estructuras arteriales en formato `.pgm`. Se implementaron algoritmos para **esqueletización**, **detección de nodos clave**, **construcción del grafo** y **visualización** del resultado. 

A continuación, se explican en detalle las ideas detrás del código final:

### **2.1. Esqueletización**
Para reducir la estructura arterial a un solo píxel de grosor, se utiliza la función `skeletonize` de `skimage.morphology`. Este proceso conserva la conectividad del grafo sin modificar su topología original. La imagen binaria de entrada se procesa para obtener un esqueleto donde:
- Los vasos sanguíneos quedan representados como líneas de un píxel de ancho.
- Se preserva la conectividad de la red arterial.

### **2.2. Detección de Nodos Clave**
Se identifican los puntos de interés dentro del esqueleto para definir los nodos del grafo:
- **Extremos (endpoints)**: píxeles con exactamente un vecino.
- **Bifurcaciones**: píxeles con tres vecinos.
- **Trifurcaciones (o multifurcaciones)**: píxeles con cuatro o más vecinos.

Además, se aplica **DBSCAN** (Density-Based Spatial Clustering) para fusionar nodos cercanos y evitar múltiples nodos en una misma intersección. Esto mejora la representación topológica de la estructura arterial.

### **2.3. Construcción del Grafo**
Cada nodo detectado se conecta mediante **aristas** siguiendo el esqueleto de la imagen:
- Se realiza un recorrido **DFS (Depth-First Search)** para identificar caminos entre nodos clave.
- Las trayectorias entre nodos se simplifican mediante el **algoritmo de Ramer-Douglas-Peucker (RDP)** para reducir la cantidad de puntos intermedios sin perder precisión en la representación del vaso sanguíneo.
- Se agregan **nodos intermedios** en las trayectorias curvadas para mejorar la precisión del grafo.

### **2.4. Exportación y Visualización**
El grafo se almacena en formato JSON con la siguiente estructura:
```json
{
  "nodes": [
    { "id": 0, "row": 10, "col": 20, "type": "extremo" },
    { "id": 1, "row": 50, "col": 30, "type": "bifurcacion" }
  ],
  "edges": [
    { "source": 0, "target": 1, "intermediate_pixels": [[15, 22], [20, 25]] }
  ]
}
```
También se genera una imagen con la visualización del grafo superpuesto a la imagen original:
- **Nodos extremos** → Verde
- **Bifurcaciones** → Rojo
- **Trifurcaciones** → Azul
- **Nodos intermedios** → Gris
- **Aristas** → Amarillo

### **2.5. Parámetros Ajustables**
El código permite modificar ciertos parámetros clave:
- `EPSILON_DBSCAN`: radio en píxeles para fusionar nodos cercanos (valor recomendado: 5).
- `RDP_EPSILON`: nivel de simplificación de trayectorias (valor recomendado: 2).

### **2.6. Procesamiento en Lote**
El algoritmo procesa automáticamente todas las imágenes en la carpeta `datos/etiquetas/` y genera:
- **Archivos JSON** en `output/json/` con la estructura del grafo.
- **Imágenes de visualización** en `output/imagenes/` mostrando el grafo superpuesto a la imagen original.

---

## 3. Algoritmos Utilizados y Adaptaciones

### **3.1. Esqueletización de Imágenes**
- **Descripción:** Reduce las estructuras arteriales a su forma central de un solo píxel de grosor.
- **Modificaciones:** Se ajustaron parámetros en `skeletonize` de `skimage` para preservar la conectividad arterial.
- **Fuente:** [Esqueletización de una imagen](https://es.scribd.com/document/444850937/Esqueletizacion-de-una-imagen)

### **3.2. Algoritmo de Ramer–Douglas–Peucker (RDP)**
- **Descripción:** Simplifica trayectorias reduciendo puntos intermedios sin perder la forma esencial.
- **Modificaciones:** Ajuste de `RDP_EPSILON` para encontrar un equilibrio entre precisión y reducción de complejidad.
- **Fuente:** [Algoritmo de Ramer–Douglas–Peucker - Wikipedia](https://es.wikipedia.org/wiki/Algoritmo_de_Ramer%E2%80%93Douglas%E2%80%93Peucker)

### **3.3. Algoritmo DBSCAN (Density-Based Spatial Clustering)**
- **Descripción:** Agrupa nodos cercanos evitando redundancias en bifurcaciones y extremos.
- **Modificaciones:** Ajuste del parámetro `EPSILON_DBSCAN` para definir la distancia óptima en imágenes médicas.
- **Fuente:** [DBSCAN - Wikipedia](https://es.wikipedia.org/wiki/DBSCAN)

