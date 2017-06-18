import numpy as np
from numba import jit
from scipy.misc import imread
import scipy.ndimage as nimg
import networkx as nx
import matplotlib.pyplot as plt
from skimage import data
from skimage.morphology import skeletonize

def mark(img): # mark the array use (0, 1, 2)
    core = np.array([[1,1,1],[1,0,1],[1,1,1]])
    msk = img>0
    nimg.filters.convolve(msk, core, output=img)
    img *= msk
    img[img==1] = 3
    img[img==2] = 1
    img[img>=3] = 2

@jit # fill a node (may be two or more points)
def fill(img, r,c, num, buf):
    back = img[r,c]
    img[r,c] = num
    buf[0,0]=r;buf[0,1]=c
    cur = 0; s = 1;
    while True:
        r=buf[cur,0]; c=buf[cur,1]
        for rr in (-1,0,1):
            for cc in (-1, 0, 1):
                if img[r+rr, c+cc]==back:
                    img[r+rr, c+cc] = num
                    buf[s,0] = r+rr
                    buf[s,1] = c+cc
                    s+=1
                    #print('one')
        cur += 1
        if cur==s:break
    return buf[:s].copy()

@jit # trace the edge and use a buffer, then buf.copy, if use [] numba not works
def trace(img, r,c, buf):
    c1 = 0; c2 = 0
    nc = 0; nr = 0
    cur = 0
    while True:
        buf[cur,0] = r; buf[cur,1] = c
        img[r,c] = 0
        cur += 1
        for rr in (-1,0,1):
            for cc in (-1, 0, 1):
                if img[rr+r, cc+c] >= 10:
                    if c1==0:c1=img[rr+r, cc+c]
                    else: c2 = img[rr+r, cc+c]
                    #print('got', img[rr+r, cc+c])
                if img[rr+r, cc+c] == 1:
                    nr = rr+r; nc = cc+c
                    #print('new',nr,nc)
        r = nr; c = nc
        if c2!=0:break
    return (c1-10, c2-10, buf[:cur].copy())
    
@jit # parse the image then get the nodes and edges
def parse_struc(img):
    pts = np.array(np.where(img==2)).T

    buf = np.zeros((10,2), dtype=np.int)
    num = 10
    nodes = []
    for i in pts:
        if img[i[0], i[1]] == 2:
            nds = fill(img, i[0], i[1], num, buf)
            num += 1
            nodes.append(nds)
            
    buf = np.zeros((1000, 2), dtype=np.int)
    edges = []
    for i in pts:
        r = i[0]; c = i[1]
        for rr in (-1,0,1):
            for cc in (-1, 0, 1):
                if img[r+rr, c+cc]==1:
                    edge = trace(img, r+rr, c+cc, buf)
                    edges.append(edge)
    return nodes, edges

# use nodes and edges build a networkx graph
def build_graph(nodes, edges):
    graph = nx.Graph()
    for i in range(len(nodes)):
        graph.add_node(i, pts=nodes[i])
    for s,e,pts in edges:
        l = np.linalg.norm(pts[1:]-pts[:-1], axis=1).sum()
        graph.add_edge(s,e, pts=pts, weight=l)
    return graph

# draw the graph
def draw_graph(img, graph):
    for idx in graph.nodes():
        pts = graph.node[idx]['pts']
        img[pts[:,0], pts[:,1]] = 255
    for (s, e) in graph.edges():
        pts = graph[s][e]['pts']
        img[pts[:,0], pts[:,1]] = 128

# ====================== test ========================
def test_draw_graph(img, graph):
    draw_graph(img, graph)
    plt.imshow(img)
    plt.show()
    
def test_shortest_path(img, graph):
    path = nx.shortest_path(graph, 0, 7, weight='weight')
    draw_graph(img, graph)
    for i in range(1, len(path)):
        pts = graph[path[i-1]][path[i]]['pts']
        img[pts[:,0], pts[:,1]] = 255
    print(path)
    plt.imshow(img)
    plt.show()
    
if __name__ == '__main__':
    img = data.horse()
    ske = skeletonize(True^img).astype(np.uint16)
    plt.imshow(ske)
    plt.show()
    mark(ske)
    nodes, edges = parse_struc(ske)
    graph = build_graph(nodes, edges)

    test_draw_graph(ske, graph)
    test_shortest_path(ske, graph)
