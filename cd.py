import networkx as nx
import numpy as np
from networkx.algorithms.matching import maximal_matching
import scipy.io
import pandas as pd
import time


lb_graphs = {
    "KarateA" : ("Karate", 36, 71),
    "dolphinsA" : ("Dolphins", 72, 143),
    "lesmisA" : ("Les-Miserables", 92, 187),
    "polbooksA" : ("PolBooks", 211, 414),
    "adjnounA" : ("Adjnoun", 211, 422),
    "footballA" : ("Football", 256, 538),

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

    # set weight and budget values for pivot
    w = lambda i,j: (1,0) if g.has_edge(i,j) else (0,np.inf)
    b = lambda i,j: 1 if get_gallai_node(i,j) in stc else 0
    g2 = g.copy()

    # remove edges in STC labeling
    g2.remove_edges_from(stc)
    # run deterministic pivot
    return det_pivot(g2, w, b, g)


def det_pivot(g, w, b, g_org):
    clus = np.zeros(len(g.nodes))
    n_clus = 0

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
            num[k] += w(i,j)[1]
            assert w(i,j)[1] == 0, "Value should be "
            den[k] += b(i,j)
            assert b(i,j) == 1, "Value should be 1"

            # update values for other nodes, Tk+
            num[i] += w(j,k)[0]
            assert w(j,k)[0] == 1, "Value should be 1"
            den[i] += b(j,k)
            assert b(j,k) == 0, "Value should be 0"

            num[j] += w(i,k)[0]
            assert w(i,k)[0] == 1, "Value should be 1"
            den[j] += b(i,k)
            assert b(i,k) == 0, "Value should be 0"

        # search wedges centered at k for every k in G
        n_wedges = 0 # just to check when no wedges found
        for k in g.nodes:
            neighbors_k = list(g.neighbors(k))
            for idx1 in range(len(neighbors_k)):
                i = neighbors_k[idx1]
                for idx2 in range(idx1+1, len(neighbors_k)):
                    j = neighbors_k[idx2]
                    if i != j and not g.has_edge(i,j):
                        n_wedges += 1
                        # open wedge centered at k found
                        update_wedge_values(i,j,k)
                        # check open wedge is a triangle in original graph
                        assert g_org.has_edge(i,j)

        # select pivot
        p = num/den
        if n_wedges == 0:
            p = list(g.nodes)[0]
        else:
            assert np.isfinite(p).any()
            p = np.nanargmin(p)
        # p = np.random.choice(g.nodes()).item()

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
        for idx1 in range(len(neighbors_k)):
            i = neighbors_k[idx1]
            # add node (k,i)
            gallai.add_node(get_gallai_node(i,k))
            for idx2 in range(idx1+1, len(neighbors_k)):
                j = neighbors_k[idx2]
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

    # return number of edges deleted and whether this is a cluster
    return True, g.number_of_edges() - n_edges


if __name__== "__main__" :
    results = []
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
        results.append([g_info[0], g_info[1], n_del_edges, g_info[2], toc-tic, graph.number_of_nodes(), graph.number_of_edges()])

    df = pd.DataFrame(results, columns=["Graph", "LB", "UB Det", "UB Rand", "Runtime", "|V|", "|E|"])
    # calculate ratios
    df["Ratio Det"] = df["UB Det"] / df["LB"]
    df["Ratio Rand"] = df["UB Rand"] / df["LB"]
    # reorder results
    df = df[["Graph", "LB", "UB Det", "UB Rand", "Ratio Det", "Ratio Rand", "|V|", "|E|", "Runtime"]]

    # show results
    print(df.round(2).to_latex(escape=True, decimal=".", index=False))
    print(df)


# 0               Karate    36      71       71   1.972222    1.972222    34     78  0.005993
# 1             Dolphins    72     145      143   2.013889    1.986111    62    159  0.015078
# 2       Les-Miserables    92     164      187   1.782609    2.032609    77    254  0.039277
# 3             PolBooks   211     424      414   2.009479    1.962085   105    441  0.041942
# 4              Adjnoun   211     418      422   1.981043    2.000000   112    425  0.064109
# 5             Football   256     498      538   1.945312    2.101562   115    613  0.128026
# 6           NetScience   315     546      669   1.733333    2.123810   379    914  0.453962
# 7             Erdos991   674    1353     1338   2.007418    1.985163   446   1413  0.217738
# 8   Celegans-Metabolic   966    1933     1917   2.001035    1.984472   453   2025  0.479926
# 9           Harvard500   776    1535     1548   1.978093    1.994845   500   2043  0.571453
# 10     Celegans-Neural  1062    2130     2117   2.005650    1.993409   297   2148  0.254527
# 11               Roget  1788    3583     3571   2.003915    1.997204   994   3640  0.775708
# 12              SmaGri  2410    4856     4811   2.014938    1.996266  1024   4916  1.253226
# 13               Email  2616    5230     5169   1.999235    1.975917  1133   5451  2.304043
# 14            PolBlogs  8336   16690    16660   2.002159    1.998560  1222  16714  5.550754


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
