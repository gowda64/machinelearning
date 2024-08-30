def astarAlgo(startnode, endnode):
    open_set=set(startnode)
    closed_set=set()
    g={} #disatnce node form start to that node
    parents={}
    g[startnode]=0 #since starting node
    parents[startnode]=startnode

    while len(open_set)>0:
        n=None
        for v in open_set:
            if n==None or g[v]+heuristic(v) < g[n]+heuristic(n):
                n=v
        if n==endnode or Graph_nodes[n]==None:
            pass
        else:
            for(m,weight) in get_neigbours(n):
                if m not in open_set and m not in closed_set:
                    open_set.add(m)
                    parents[m]=n
                    g[m]=g[n]+weight
                else:
                    if g[m]>g[n]+weight:
                        g[m]=g[n]+weight
                        parents[m]=n
                        if m in closed_set:
                            closed_set.remove(m)
                            open_set.add(m)
        if n==None:
            print("the path does not exist")
            return None
        
        if n==endnode:
            path=[]
            while parents[n]!=n:
                path.append(n)
                n=parents[n]
            path.append(startnode)
            path.reverse()
            print("path found: {}".format(path))
            return path
        open_set.remove(n)
        closed_set.add(n)
    print("path does not exist")
    return None

def get_neigbours(v):
    if v in Graph_nodes:
        return Graph_nodes[v]
    else:
        return None
    
def heuristic(n):
    H_dist={
        "A":11,
        "B":6,
        "C":99,
        "D":1,
        "E":7,
        "G":0,
    }
    return H_dist[n]

Graph_nodes={
    'A':[('B',2),("E",3)],
    'B':[('C',1),("G",9)],
    'C':None,
    "E":[("D",6)],
    'D':[('G',1)],
}

astarAlgo("A","G")













