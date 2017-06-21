from .sknw import mark, parse_struc, build_graph, draw_graph, build_sknw

__version__ = '0.1'

__all__ = ['mark', 'parse_struc', 'build_graph', 'draw_graph', 'build_sknw']


def test():
    from skimage.morphology import skeletonize
    import numpy as np
    from skimage import data
    import matplotlib.pyplot as plt

    img = data.horse()
    ske = skeletonize(~img).astype(np.uint16)
    graph = build_sknw(ske)
    plt.imshow(img, cmap='gray')
    for (s,e) in graph.edges():
        ps = graph[s][e]['pts']
        plt.plot(ps[:,1], ps[:,0], 'green')

    node, nodes = graph.node, graph.nodes()
    ps = np.array([node[i]['o'] for i in nodes])
    plt.plot(ps[:,1], ps[:,0], 'r.')
    plt.title('Build Graph')
    plt.show()
