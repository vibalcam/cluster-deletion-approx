import networkx as nx
import numpy as np
from networkx.algorithms.matching import maximal_matching
import scipy.io
import pandas as pd
import time


# names_graphs = [
#     # "KarateA",
#     # "dolphinsA",
#     # "lesmisA",
#     # "polbooksA",
#     # "adjnounA",
#     # "footballA",

#     "Harvard500A",
#     "Erdos991A",
#     "celegansneuralA",
#     "Netscience",
#     "celegansmetabolicA",
#     "RogetA",
#     "SmaGriA",
#     "standard_emailA",
#     "standard_polblogsA",
#     "standard_ca-GrQc",
#     "standard_caHepThA",
#     "standard_EmailEnronA",
#     "standard_condmat2005A",
#     "standard_ca-AstroPhA",
#     "standard_loc-Brightkite",
# ]

lb_graphs = {
    # "KarateA" : ("", None, None),
    # "dolphinsA" : ("", None, None),
    # "lesmisA" : ("", None, None),
    # "polbooksA" : ("", None, None),
    # "adjnounA" : ("", None, None),
    # "footballA" : ("", None, None),

    "Netscience": ("NetScience", 315, 669),
    "Erdos991A": ("Erdos991", 674, 1338),
    "celegansmetabolicA": ("Celegans-Metabolic", 966, 1917),
    "Harvard500A": ("Harvard500", 776, 1548),
    "celegansneuralA": ("Celegans-Neural", 1062, 2117),
    "RogetA": ("Roget", 1788, 3571),
    "SmaGriA": ("SmaGri", 2410, 4811),
    "emailA": ("Email", 2616, 5169),
    "polblogsA": ("PolBlogs", 8336, 16660),
}


def det_cd(g):
    # Build Gallai graph
    gallai = build_gallai(g)
    # get maximal matching
    m = maximal_matching(gallai)
    # get vertex cover from maximal matching
    cv = [item for t in m for item in t]
    # get STC labeling from vertex cover on Gallai
    stc = cv

    # set values for pivot
    w = lambda i,j: (1,0) if g.has_edge(i,j) else (0,np.inf)
    b = lambda i,j: 1 if get_gallai_node(i,j) in stc else 0
    g2 = g.copy()

    # remove edges in STC labeling
    g2.remove_edges_from(stc)

    return det_pivot(g2, w, b)


def det_pivot(g, w, b):
    # S = set()
    clus = np.zeros(len(g.nodes))
    n_clus = 0
    
    # todo stop when no nodes or when no edges
    # todo when stop, should rest of nodes be added to S or not

    # while not nx.is_empty(g):
    # while there are unclustered nodes
    while np.sum(clus == 0) > 0:
        n_clus += 1
        num = np.zeros(clus.shape)
        den = np.zeros(clus.shape)
        
        # update rule for wedge centered at k
        # if edge opposite to node n does exist, then Tk+
        # if edge opposite to node n does not exist, then Tk-
        def update_wedge_values(i,j,k):
            # update values for k (center), Tk-
            num[k] += w(i,j)[1]     # should be 0 or inf
            assert w(i,j)[1] == 0 or np.isinf(w(i,j)[1]), "Value should be 0 or inf"
            den[k] += b(i,j)

            # update values for other nodes, Tk+
            num[i] += w(j,k)[0]
            den[i] += b(j,k)        # should be 0
            assert b(j,k) == 0, "Value should be 0"
            num[j] += w(i,k)[0]
            den[j] += b(i,k)        # should be 0
            assert b(i,k) == 0, "Value should be 0"

        # search wedges centered at k for every k in G
        for k in g.nodes:
            neighbors_k = list(g.neighbors(k))
            for i in neighbors_k:
                for j in neighbors_k:
                    if i != j and not g.has_edge(i,j):
                        # open wedge centered at k found
                        update_wedge_values(i,j,k)

        # for i,k in g.edges:
        #     # check for wedges centered at k
        #     for j in g.neighbors(k):
        #         if j != i and not g.has_edge(i,j):
        #             # then we have a open wedge centered at k
        #             update_wedge_values(i,j,k)
        #     # check for wedges centered at i
        #     for j in g.neighbors(i):
        #         if j != k and not g.has_edge(k,j):
        #             # then we have a open wedge centered at i
        #             update_wedge_values(j,k,i)
        # select pivot
        p = num/den
        # if np.isnan(p).all():
        if not np.isfinite(p).any():
            assert g.number_of_edges() == 0, "Number of edges should be 0"
            p = list(g.nodes)[0]
        else:
            p = np.nanargmin(p)
        
        # # form cluster
        # c = set(g.neighbors(p))
        # c.add(p)
        # S.update(c)

        # # update graph
        # g.remove_nodes_from(c)

        # form cluster and update graph
        c = set(g.neighbors(p))
        c.add(p)
        for k in c:
            clus[k] = n_clus
            g.remove_node(k)
    
    return clus
        

def build_gallai(g):
    gallai=nx.Graph()
    # for each edge (i,j) add a node
    # for each open wedge centered at k, edge between vik and vjk
    
    for k in g.nodes:
        neighbors_k = list(g.neighbors(k))
        # search wedges centered at k for every k in G
        for i in neighbors_k:
            # add node (k,i)
            gallai.add_node(get_gallai_node(i,k))
            for j in neighbors_k:
                if i != j and not g.has_edge(i,j):
                    # open wedge centered at k found
                    gallai.add_edge(get_gallai_node(i,k), get_gallai_node(j,k))

    # for i,k in g.edges:
    #     # add node for edge
    #     gallai.add_node(get_gallai_node(i,k))
    #     # check for wedges centered at k
    #     for j in g.neighbors(k):
    #         if j != i and not g.has_edge(i,j):
    #             # then we have a open wedge centered at k
    #             gallai.add_edge(get_gallai_node(i,k), get_gallai_node(j,k))
    #     # check for wedges centered at i
    #     for j in g.neighbors(i):
    #         if j != k and not g.has_edge(k,j):
    #             # then we have a open wedge centered at i
    #             gallai.add_edge(get_gallai_node(i,j), get_gallai_node(i,k))
    
    return gallai


def get_gallai_node(i,j):
    return min(i,j), max(i,j)


def check_cd(g, clus):
    n_edges = 0
    for k in range(1, clus.max().astype(int)):
        c = np.where(clus == k)[0].tolist()
        # subgraph of g with nodes c
        sub_g = g.subgraph(c)
        # check if complete
        if len(c) > 1 and nx.density(sub_g) != 1:
            return False, None
        # accumulate number of edges
        n_edges += sub_g.number_of_edges()

        # for i in range(len(c)):
            # for j in range(i+1, len(c)):
            #     if not g.has_edge(c[i],c[j]):
            #         n_edges += 1
            #         return False

    # return number of edges deleted and whether this is a cluster
    return True, g.number_of_edges() - n_edges


if __name__== "__main__" :
    results = []
    # for ng, g_name in enumerate(names_graphs):
    for ng, g_name in enumerate(lb_graphs.keys()):

        print(f"Graph {ng} of {len(lb_graphs)}")

        # load from matlab file
        mat = scipy.io.loadmat(f"graphs/{g_name}.mat")
        graph = nx.from_scipy_sparse_array(mat['A'])

        # graph = nx.karate_club_graph()

        # graph = nx.Graph()
        # graph.add_edge(1,2)
        # graph.add_edge(2,3)
        # graph.add_edge(1,3)
        # graph.add_edge(1,4)

        # relabel nodes to consecutive integers and undirected graph
        graph = nx.convert_node_labels_to_integers(graph.to_undirected())

        # run algorithm
        tic = time.time()
        clus = det_cd(graph)
        toc = time.time()
        # check clustering feasible cd and cd objective
        is_cd, n_del_edges = check_cd(graph, clus)
        assert is_cd, "Clustering not feasible CD"

        # print(f"Number of edges deleted: {n_del_edges}")
        # break

        # results
        g_info = lb_graphs[g_name]
        results.append([g_info[0], n_del_edges, g_info[1], g_info[2], toc-tic])

    df = pd.DataFrame(results, columns=["Graph", "ub", "lb", "ubpaper", "Runtime"])
    # calculate ratios
    df["Ratio"] = df.ub / df.lb
    df["RatioPaper"] = df.ubpaper / df.lb
    # show results
    print(df.to_latex(escape=True, decimal=".", index=False))
    print(df)


#                 Graph     ub    lb  ubpaper   Runtime     Ratio  RatioPaper
# 0          NETSCIENCE    546   315      669  0.506141  1.733333    2.123810
# 1            Erdos991   1353   674     1338  0.226215  2.007418    1.985163
# 2  Celegans-Metabolic   1933   966     1917  0.606921  2.001035    1.984472
# 3          Harvard500   1535   776     1548  0.735813  1.978093    1.994845
# 4     Celegans-Neural   2130  1062     2117  0.304325  2.005650    1.993409
# 5               Roget   3583  1788     3571  0.600492  2.003915    1.997204
# 6              SmaGri   4856  2410     4811  1.432339  2.014938    1.996266
# 7               Email   5230  2616     5169  2.532014  1.999235    1.975917
# 8            PolBlogs  16690  8336    16660  9.858507  2.002159    1.998560


#                 Graph     ub    lb  ubpaper   Runtime     Ratio  RatioPaper
# 0          NETSCIENCE    571   315      669  0.173008  1.812698    2.123810
# 1            Erdos991   1289   674     1338  0.243515  1.912463    1.985163
# 2  Celegans-Metabolic   1894   966     1917  0.867969  1.960663    1.984472
# 3          Harvard500   1546   776     1548  0.843995  1.992268    1.994845
# 4     Celegans-Neural   2090  1062     2117  0.939052  1.967985    1.993409
# 5               Roget   3518  1788     3571  2.212625  1.967562    1.997204
# 6              SmaGri   4802  2410     4811  6.013700  1.992531    1.996266
# 7               Email   5178  2616     5169  6.488147  1.979358    1.975917
# 8            PolBlogs  16608  8336    16660  8.515046  1.992322    1.998560
