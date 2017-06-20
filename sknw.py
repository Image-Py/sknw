import numpy as np
from numba import jit
from scipy.misc import imread
import scipy.ndimage as nimg
import networkx as nx
import matplotlib.pyplot as plt
from skimage import data
from skimage.morphology import skeletonize

# get neighbors d index
def neighbors(shape):
    dim = len(shape)
    block = np.ones([3]*dim)
    block[tuple([1]*dim)] = 0
    idx = np.where(block>0)
    idx = np.array(idx, dtype=np.uint8).T
    idx = np.array(idx-[1]*dim)
    acc = np.cumprod((1,)+shape[::-1][:-1])[::-1]
    return np.dot(idx, acc)

@jit # my mark
def mark(img): # mark the array use (0, 1, 2)
    nbs = neighbors(img.shape)
    img = img.ravel()
    for p in range(len(img)):
        if img[p]==0:continue
        s = 0
        for dp in nbs:
            if img[p+dp]!=0:s+=1
        if s==2:img[p]=1
        else:img[p]=2

@jit # trans index to r, c...
def idx2rc(idx, acc):
    rst = np.zeros((len(idx), len(acc)), dtype=np.uint16)
    for i in range(len(idx)):
        for j in range(len(acc)):
            rst[i,j] = idx[i]//acc[j]
            idx[i] -= rst[i,j]*acc[j]
    return rst
    
@jit # fill a node (may be two or more points)
def fill(img, p, num, nbs, acc, buf):
    back = img[p]
    img[p] = num
    buf[0] = p
    cur = 0; s = 1;
    
    while True:
        p = buf[cur]
        for dp in nbs:
            cp = p+dp
            if img[cp]==back:
                img[cp] = num
                buf[s] = cp
                s+=1
        cur += 1
        if cur==s:break
    return idx2rc(buf[:s], acc)

@jit # trace the edge and use a buffer, then buf.copy, if use [] numba not works
def trace(img, p, nbs, acc, buf):
    c1 = 0; c2 = 0
    newp = 0
    cur = 0

    b = p==97625
    while True:
        buf[cur] = p
        img[p] = 0
        cur += 1
        for dp in nbs:
            cp = p + dp
            if img[cp] >= 10:
                if c1==0:c1=img[cp]
                else: c2 = img[cp]
            if img[cp] == 1:
                newp = cp
        p = newp
        if c2!=0:break
    return (c1-10, c2-10, idx2rc(buf[:cur], acc))
   
@jit # parse the image then get the nodes and edges
def parse_struc(img):
    nbs = neighbors(img.shape)
    acc = np.cumprod((1,)+img.shape[::-1][:-1])[::-1]
    img = img.ravel()
    pts = np.array(np.where(img==2))[0]
    buf = np.zeros(20, dtype=np.int64)
    num = 10
    nodes = []
    for p in pts:
        if img[p] == 2:
            nds = fill(img, p, num, nbs, acc, buf)
            num += 1
            nodes.append(nds)
    
    buf = np.zeros(10000, dtype=np.int64)
    edges = []
    for p in pts:
        for dp in nbs:
            if img[p+dp]==1:
                edge = trace(img, p+dp, nbs, acc, buf)
                edges.append(edge)
    return nodes, edges
    
# use nodes and edges build a networkx graph
def build_graph(nodes, edges):
    graph = nx.Graph()
    for i in range(len(nodes)):
        graph.add_node(i, pts=nodes[i], o=nodes[i].mean(axis=0))
    for s,e,pts in edges:
        l = np.linalg.norm(pts[1:]-pts[:-1], axis=1).sum()
        graph.add_edge(s,e, pts=pts, weight=l)
    return graph

def build_sknw(ske):
    mark(ske)
    nodes, edges = parse_struc(ske.copy())
    return build_graph(nodes, edges)
    
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
    os = [graph.node[i]['o'] for i in graph.nodes()]
    os = np.array(os)
    plt.plot(os[:,1], os[:,0], 'r.')
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
    from time import time
    
    img = data.horse()
    ske = skeletonize(~img).astype(np.uint16)
    ske1, ske2 = ske.copy(), ske.copy()

    start = time()
    build_sknw(ske1)
    print(time()-start)

    start = time()
    build_sknw(ske2)
    print(time()-start)
    #test_draw_graph(ske, graph)
    #test_shortest_path(ske, graph)

    from skan import csr
    ske1, ske2 = ske.copy(), ske.copy()
    start = time()
    csr.skeleton_to_csgraph(ske1)
    print(time()-start)
    start = time()
    csr.skeleton_to_csgraph(ske2)
    print(time()-start)

    ''' 
    img = imread('ske.png')
    
    nbs = neighbors(img.shape)
    buf = np.zeros(100000, dtype=np.uint64)
    start = time()

    fill(img.copy(), 10100, 120, nbs, acc, buf)
    #fill(img.copy(), 100, 100, 120, buf)
    print(time()-start)
    start = time()
    fill(img, 10100, 120, nbs, acc, buf)
    #fill(img, 100, 100, 120, buf)
    print(time()-start)
    plt.imshow(img)
    plt.show()
    

    #print(neighbors((10,5)))

    '''
