

import os

import pickle
import numpy as np 
from mod.dsa.dend_fun_0.help_graph import graph_cylinder_heatmap


class save:
    def __init__(self):
        self.ripp,self.Lrip=None,None
        self.others=None
        self.res_area,self.res_ripley,self.rall,self.count_spine=None,None,None,None


class get_riplet(graph_cylinder_heatmap):
    def __init__(self,pid,spine_path,):
        pass
        graph_cylinder_heatmap.__init__(self,pid=pid,spine_path=spine_path)
        self.save=save()        


    def compute(self,  
                shaft_points, 
                neck_points,
                radius,
                point_a,
                point_b,   
                width=800,
                height=700,
                tf_ripley=True,
                param_ripley=None,
                save=None,
                den_Circon=2,
                size_monte=500,
                N_interpolate=1000,
                N_plot=40,
                 ):
        pass
        self.width=width
        self.height=height
        import numpy as np 


        size_monte,N_interpolate,N_plot=[int(mm) for mm in (size_monte,N_interpolate,N_plot)]
        print('[[[[[[ripley]]]]]]',den_Circon,size_monte,N_interpolate,N_plot)


    
    # if tf_ripley:

        
        # from mod.dsa.dend_fun_0.ripley import ripley_all,generate_points,ripley_all_save,monte_carlos_ripley,monte_carlos_area,monte_carlos_area_incremental,monte_carlos_ripley_incremental,monte_carlos_ripley_poisson_incremental,get_distance
        # import numpy as np


        # dist=get_distance(save,list_index=list_index)
        a = point_a
        b = point_b
        R = radius
        # points = generate_points(40, R)   # your cylinder points
        # points=centroid_proj_cylder

        Circon=(2*np.pi*R)
        Circon=save['mesh_shaft'].area/R
        rall=np.linspace(.01,int(Circon/den_Circon),N_plot)[1:]
        if param_ripley is None:
            param_ripley=dict(
                    N_plot=N_plot, 
                    N_interpolate=N_interpolate,
                    rall=rall)
        # self.ripp,self.Lrip=ripley_all(points,a,b,R,**param_ripley)
        self.save.ripp,self.save.Lrip,intensity=ripley_all_save(save,R,**param_ripley)
        intensity['idx']={}
        self.save.others=dict(mesh_shaft=save['mesh_shaft'],
                        intensity=intensity,
                        ) 


        self.save.count_spine=len(shaft_points)
        # res_ripley = monte_carlos_ripley_incremental(save, R, rall, size_monte=size_monte, K_obs=self.ripp)
        self.save.res_ripley = monte_carlos_ripley_poisson_incremental(save, R, rall, size_monte=size_monte, K_obs=self.save.ripp)
        self.save.res_area   = monte_carlos_area_incremental(save, R=None, rall=rall, size_monte=size_monte, K_obs=self.save.ripp)
        self.save.rall=param_ripley['rall']

    def __call__(self,
                save_data=True,
                den_Circon=None,
                size_monte=None,
                N_interpolate=None,
                N_plot=None,):
        save=self.coordinates() 
        if save is None:
            return None
        if (len(save['center'])==0)or(len(save['close'])==0):
            return None
        self.compute(shaft_points=np.array(save['center']),
                            neck_points=np.array(save['close']),
                            point_a=save['point_a'], 
                            point_b=save['point_b'], 
                            radius=save['radius'],
                            width=save['width'],
                            height=save['height'],
                            save=save,
                            den_Circon=den_Circon,
                            size_monte=size_monte,
                            N_interpolate=N_interpolate,
                            N_plot=N_plot,
                            ) 
        if save_data:
            with open(os.path.join(self.spine_path,'riplet.pkl'), 'wb') as f:
                pickle.dump(self.save, f) 
        else:
            return  self.save















import numpy as np

def build_frame(a, b):
    axis = b.astype(float) - a.astype(float)
    L = np.linalg.norm(axis)
    axis /= L

    tmp = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(tmp, axis)) > 0.9:
        tmp = np.array([0.0, 1.0, 0.0])

    n1 = np.cross(axis, tmp)
    n1 /= np.linalg.norm(n1)
    n2 = np.cross(axis, n1)

    return axis, n1, n2


def to_frame(P, a, axis, n1, n2):
    v = P - a
    return np.array([np.dot(v, n1),
                     np.dot(v, n2),
                     np.dot(v, axis)])
 



 


 
def integrand(t,R,alpha,beta):
    return np.sqrt(
        1.0 +
        (R * (-alpha * np.sin(t) + beta * np.cos(t)))**2
    )

def simpson(f, a_, b_, N,R,alpha,beta):
    if N % 2:
        N += 1
    x = np.linspace(a_, b_, N + 1)
    y = f(x,R,alpha,beta)
    h = (b_ - a_) / N
    return h/3 * (y[0] + y[-1] +
                    4*np.sum(y[1:-1:2]) +
                    2*np.sum(y[2:-1:2]))

def arc_length(integrand, t1, t2,R,alpha,beta, N=2000):
    if t2 < t1:
        t2 += 2*np.pi

    t = np.linspace(t1, t2, N)
    f = integrand(t,R=R,alpha=alpha,beta=beta)

    return np.trapz(f, t)
    # return np.trapezoid(f, t)


def ellipse_distance(P1, P2, a, b, R, N=1000):
    axis, n1, n2 = build_frame(a, b)

    x1, y1, z1 = to_frame(P1, a, axis, n1, n2)
    x2, y2, z2 = to_frame(P2, a, axis, n1, n2)

    t1, t2 = np.arctan2(y1, x1), np.arctan2(y2, x2)

    denom = np.sin(t2 - t1)
    if abs(denom) < 1e-12:
        # raise ValueError("Degenerate configuration")
        return 0

    alpha = (z1 * np.sin(t2) - z2 * np.sin(t1)) / denom
    beta  = (z2 * np.cos(t1) - z1 * np.cos(t2)) / denom

    L1 = arc_length(integrand, t1, t2, N=N, R=R, alpha=alpha, beta=beta)
    L2 = arc_length(integrand, t2, t1 + 2*np.pi, N=N, R=R, alpha=alpha, beta=beta)

    return min(L1, L2)




def pairwise_ellipse_distances(points, a, b, R, N=1000):
    n = len(points)
    D = np.zeros((n, n))

    for i in range(n):
        for j in range(i+1, n):
            d = ellipse_distance(points[i], points[j], a, b, R, N=N)
            D[i, j] = d
            D[j, i] = d   

    return D


def ripley_all(points,a,b,R,N_plot=40,N_interpolate=1000,rall=None):
    Circon=(2*np.pi*R)
    A=Circon*np.linalg.norm(a-b)
    N=points.__len__()
    ripp,Lrip=[],[]
    dist=pairwise_ellipse_distances(points, a, b, R, N=N_interpolate)
    if rall is None:
        rall=np.linspace(.01,Circon/4,N_plot)
    for r in rall: 
        K=(sum(sum(dist<r))-N)*A/(N*(N-1))
        ripp.append(K)
        Lrip.append((K/np.pi)**(1/2)-r)
    return np.array(ripp),np.array(Lrip)

'''

def ripley_all_save(save,R,N_plot=40,rall=None,N_interpolate=None,list_index=None,dist=None,intensity=None):
    # llo=np.linalg.norm(save['point_a']-save['point_b'])
    Circon=(2*np.pi*R)
    Circon=save['mesh_shaft'].area/R

    # A=Circon*llo  
    A=save['area_shaft']
    if list_index is None:
        list_index=save['shaft_close_index']
    N=list_index.__len__()
    ripp,Lrip=[],[]
    if dist is None:
        dist,intensity=get_distance(save,list_index=list_index)
    if rall is None:
        rall=np.linspace(.01,Circon/2,N_plot)[1:]
    for r in rall: 
        K=(sum(sum(dist<=r))-N)*A/(N*(N-1))
        ripp.append(K)
        # Lrip.append((K/np.pi)**(1/2)-r)
        Lrip.append(K/Circon-r)
    return np.array(ripp),np.array(Lrip),intensity

'''


def ripley_all_save(save,R,N_plot=40,rall=None,N_interpolate=None,list_index=None,dist=None,intensity=None):
    # llo=np.linalg.norm(save['point_a']-save['point_b'])
    Circon=(2*np.pi*R)
    Circon=save['mesh_shaft'].area/R
    mesh = save['mesh_shaft']

    # A=Circon*llo  
    A=save['area_shaft']
    if list_index is None:
        list_index=save['shaft_close_index']
    N=list_index.__len__()
    ripp,Lrip=[],[]
    if rall is None:
        rall=np.linspace(.01,Circon/2,N_plot)[1:]
    if dist is None:
        dist,intensity=get_distance(save,list_index=list_index,cutoff=rall[-1])
    summ=0 
    aA=A/(N*(N-1)) 
    for r in rall: 
        K=0
        for i in range(N):
            src = list_index[i]  
            # print('[[[[[[ intens ]]]]]]',intensity['idx'][src]['dist'])
            dij=sum(intensity['idx'][src]['dist']<=r)-1
            # distG=intensity['idx'][src]['distG']
            # indd=[v for v, dv in distG.items() if dv <= r]
            # mesh_sub=submesh_from_vertex_indices(mesh,indd) 
            K+=dij#/mesh_sub.area
            
            # ={mm:[] for mm in ['distG','pathG','dist']}
 
        K=K*aA
        ripp.append(K)
        # Lrip.append((K/np.pi)**(1/2)-r)
        Lrip.append(K/Circon-r)
    return np.array(ripp),np.array(Lrip),intensity




def random_point_on_cylinder(R, z_range=(0, 10)):
    theta = 2 * np.pi * np.random.rand()
    z = np.random.uniform(*z_range)

    return np.array([
        R * np.cos(theta),
        R * np.sin(theta),
        z
    ])


def generate_points(n, R, z_range=(0, 10)):
    return np.array([random_point_on_cylinder(R, z_range) for _ in range(n)])




import numpy as np
import networkx as nx
'''

def get_distance(save,list_index=None):
    mesh=save['mesh_shaft']
    G = nx.Graph() 
    for f in mesh.faces:
        for a, b in [(f[0],f[1]), (f[1],f[2]), (f[2],f[0])]:
            w = np.linalg.norm(mesh.vertices[a] - mesh.vertices[b])
            G.add_edge(a, b, weight=w)


    if list_index is None:
        list_index=save['shaft_close_index']
    NN=list_index.__len__()
    # path={ii:{jj:None for jj in range(NN)} for ii in range(NN)}
    dist=np.zeros((NN,NN))
    for ii in range(NN):
        for jj in range(ii+1,NN):
            dist[ii,jj]=dist[jj,ii]=nx.shortest_path_length(G,
                                source=list_index[ii],
                                target=list_index[jj],
                                weight='weight')

    return dist


'''




def get_distance(save, list_index=None,cutoff=None):
    mesh = save['mesh_shaft']
    # intensity=-5*np.ones_like(mesh.vertices[:,0])
    intensities={mm:[] for mm in ['intensity','path','distance','idx']}
    # intensities['intensity']=intensity    
    intensities['idx']={}

    G = nx.Graph()
    for f in mesh.faces:
        a, b, c = f
        for u, v in [(a,b), (b,c), (c,a)]:
            w = np.linalg.norm(mesh.vertices[u] - mesh.vertices[v])
            G.add_edge(u, v, weight=w)
 
    if list_index is None:
        list_index = save['shaft_close_index']

    N = len(list_index)
    dist = np.zeros((N, N))
    for i in range(N):
        src = list_index[i]  
        intensities['idx'][src]={mm:[] for mm in ['distG','pathG','dist']}
 
    for i in range(N):
        src = list_index[i] 
        # d = nx.single_source_dijkstra_path_length(G, src, weight='weight') 
        d, path = nx.single_source_dijkstra(G, src,weight='weight',cutoff=cutoff) 
        # intensities['idx'][src]['distG']=d
        # intensities['idx'][src]['pathG']=path
        # {mm:[] for mm in ['dist','path']}

        for j in range(i+1, N):
            list_index_j=list_index[j]
            dist[i, j] = dist[j, i] = d.get(list_index_j,cutoff+100)
            path_j = path.get(list_index_j,[])
            if len(path_j)>0: 
                intensities['path'].append(path_j)
                intensities['distance'].append(dist[i,j])
    for i in range(N):
        src = list_index[i] 
        intensities['idx'][src]['dist']=dist[i,:]
 
    return dist,intensities




from scipy.spatial import cKDTree

def monte_carlos_ripley(save,R,N_plot=40,rall=None,N_interpolate=None,list_index=None,size_monte=10,K_obs=None):
    mesh_shaft=save['mesh_shaft']
    NN = len(save['shaft_close_index']) 
    area = np.zeros_like(rall)

    param_ripley = dict(
        N_plot=N_plot,
        N_interpolate=None,
        rall=rall
    )

    tree = cKDTree(mesh_shaft.vertices)
    areas=np.zeros((size_monte,len(rall)))

    for ii in range(size_monte):
            
        points = mesh_shaft.sample(NN)  
      #    list_index = np.argmin(np.linalg.norm(points[:, None] - mesh_shaft.vertices[None, :], axis=2), axis=1)
        list_index = tree.query(points)[1]
        dist=get_distance(save,list_index=list_index,cutoff=rall[-1])
        ripp, _ = ripley_all_save(save, R, **param_ripley, list_index=list_index,dist=dist)

        area += ripp

        areas[ii,:]=ripp
        
    area /= size_monte
    pvals,p_cluster,p_regular=None,None,None
    p_global_2tail,p_global_cluster,p_global_regular=None,None,None
    if K_obs is not None:
        # pvals = ( 1 +np.sum(np.abs(areas - ripps)>= np.abs(K_obs - ripps),axis=0 ) ) / (size_monte+ 1)

        p_cluster = (1 + np.sum(areas >= K_obs, axis=0)) / (size_monte + 1)

        p_regular = (1 + np.sum(areas <= K_obs, axis=0)) / (size_monte + 1)


        p_2tail = 2 * np.minimum(p_cluster, p_regular)
        pvals = np.minimum(p_2tail, 1.0)

        p_global_2tail = ( 1 +np.sum(np.sum(np.square(areas - area),axis=1)>= np.sum(np.square(K_obs - area)) ) ) / (size_monte+ 1)
        p_global_cluster = ( 1 +np.sum(np.sum(areas,axis=1)>= np.sum(K_obs) ) ) / (size_monte+ 1)
        p_global_regular = ( 1 +np.sum(np.sum(areas,axis=1)<= np.sum(K_obs) ) ) / (size_monte+ 1)


    return area,pvals,p_cluster,p_regular,[p_global_2tail,p_global_cluster,p_global_regular]





   


import trimesh

def submesh_from_vertex_indices(mesh, vertex_indices):
    vertex_indices = np.array(vertex_indices)

    old_to_new = -np.ones(len(mesh.vertices), dtype=int)
    old_to_new[vertex_indices] = np.arange(len(vertex_indices)) 
    mask = np.all(old_to_new[mesh.faces] != -1, axis=1)
    new_faces = old_to_new[mesh.faces[mask]]
 
    new_vertices = mesh.vertices[vertex_indices]

    return trimesh.Trimesh(vertices=new_vertices, faces=new_faces, process=False)



def vertices_within_distance(G, src, distance): 
    dist = nx.single_source_dijkstra_path_length(G, src, weight='weight') 
    return [[v for v, dv in dist.items() if dv <= d] for d in distance]


def get_area_sum(save,dist, list_index=None):
    mesh = save['mesh_shaft']
 
    G = nx.Graph()
    for f in mesh.faces:
        a, b, c = f
        for u, v in [(a,b), (b,c), (c,a)]:
            w = np.linalg.norm(mesh.vertices[u] - mesh.vertices[v])
            G.add_edge(u, v, weight=w)
 
    if list_index is None:
        list_index = save['shaft_close_index']

    N = len(list_index) 
    area=0
    for i in range(N):
        src = list_index[i]  


        indd=vertices_within_distance(G, src, dist)
        mesh_sub=submesh_from_vertex_indices(mesh,indd) 
        area+=mesh_sub.area

    return area/N



def monte_carlos_area(save,R=None,N_plot=40,rall=None,N_interpolate=None,list_index=None,size_monte=10,K_obs=None):
    mesh=save['mesh_shaft']
    # NN = len(save['shaft_close_index']) 
    # ripps = np.zeros_like(rall)

    G = nx.Graph()
    for f in mesh.faces:
        a, b, c = f
        for u, v in [(a,b), (b,c), (c,a)]:
            w = np.linalg.norm(mesh.vertices[u] - mesh.vertices[v])
            G.add_edge(u, v, weight=w)

    param_ripley = dict(
        N_plot=N_plot,
        N_interpolate=None,
        rall=rall
    )

    tree = cKDTree(mesh.vertices)
    points = mesh.sample(2**size_monte) 
      #    list_index = np.argmin(np.linalg.norm(points[:, None] - mesh_shaft.vertices[None, :], axis=2), axis=1)
    list_index = tree.query(points)[1]
    rall=param_ripley['rall']
    area=np.zeros_like(rall)
    areass=np.zeros((len(list_index),len(rall)))
    for ii,src in enumerate(list_index):
        indx=vertices_within_distance(G=G,src=src,distance=rall)
        ar=np.array([submesh_from_vertex_indices(mesh,ind).area for ind in indx])
        areass[ii,:]=ar
        area+=ar
    
    # for ii,dis in enumerate(param_ripley['rall']):
    #     area[ii]=get_area_sum(save,dis, list_index=list_index)
    area /=len(list_index)

    pp=np.sum((area-K_obs)**2)
    pvals,p_cluster,p_regular=None,None,None
    p_global_2tail,p_global_cluster,p_global_regular=None,None,None
    if K_obs is not None:
         # pvals = ( 1 +np.sum(np.abs(areas - area)>= np.abs(K_obs - area),axis=0 ) ) / (size_monte+ 1)
        areas=areass[:2**ii,:]
        p_cluster = (1 + np.sum(areas >= K_obs, axis=0)) / (size_monte + 1)

        p_regular = (1 + np.sum(areas <= K_obs, axis=0)) / (size_monte + 1)

        p_2tail = 2 * np.minimum(p_cluster, p_regular)
        pvals = np.minimum(p_2tail, 1.0)

        p_regular = (1 + np.sum((areas-area)**2 <= (areas-pp)**2, axis=0)) / (size_monte + 1)

        p_global_2tail = ( 1 +np.sum(np.sum(np.square(areas - area),axis=1)>= np.sum(np.square(K_obs - area)) ) ) / (size_monte+ 1)
        p_global_cluster = ( 1 +np.sum(np.sum(areas,axis=1)>= np.sum(K_obs) ) ) / (size_monte+ 1)
        p_global_regular = ( 1 +np.sum(np.sum(areas,axis=1)<= np.sum(K_obs) ) ) / (size_monte+ 1)


    return area,pvals,p_cluster,p_regular,[p_global_2tail,p_global_cluster,p_global_regular]




def monte_carlos_area_incremental(
    save,
    rall,
    size_monte=200,
    R=None,
    K_obs=None
):
    mesh = save['mesh_shaft']

    # Build graph
    G = nx.Graph()
    for f in mesh.faces:
        a, b, c = f
        for u, v in [(a, b), (b, c), (c, a)]:
            w = np.linalg.norm(mesh.vertices[u] - mesh.vertices[v])
            G.add_edge(u, v, weight=w)

    # Sample points
    tree = cKDTree(mesh.vertices)
    points = mesh.sample(size_monte)
    list_index = tree.query(points)[1]

    rall = np.asarray(rall)
    areas = np.zeros((size_monte, len(rall)))
 
    for ii, src in enumerate(list_index):
        indx = vertices_within_distance(G=G, src=src, distance=rall)
        ar = np.array([submesh_from_vertex_indices(mesh, ind).area for ind in indx])
        areas[ii, :] = ar
 
    area_mean = np.mean(areas, axis=0)

    # If no observed curve, return only the mean
    if K_obs is None:
        return area_mean, None

    # Storage for incremental results
    results = {
        "pvals": [],
        "p_cluster": [],
        "p_regular": [],
        "p_global_2tail": [],
        "p_global_cluster": [],
        "p_global_regular": [],
        "area_mean": [],
        "M_values": []
    }

    # Incremental Monte-Carlo
    for M in range(5, size_monte + 1):   # start at 5 for stability
        A = areas[:M, :]

        area_mean = np.mean(A, axis=0)
        # Per-radius p-values
        p_cluster = (1 + np.sum(A >= K_obs, axis=0)) / (M + 1)
        p_regular = (1 + np.sum(A <= K_obs, axis=0)) / (M + 1)
        p_2tail = np.minimum(2 * np.minimum(p_cluster, p_regular), 1.0)

        # Global tests
        D_obs = np.sum((K_obs - area_mean)**2)
        D_sim = np.sum((A - area_mean)**2, axis=1)

        p_global_2tail = (1 + np.sum(D_sim >= D_obs)) / (M + 1)
        p_global_cluster = (1 + np.sum(np.sum(A, axis=1) >= np.sum(K_obs))) / (M + 1)
        p_global_regular = (1 + np.sum(np.sum(A, axis=1) <= np.sum(K_obs))) / (M + 1)

        # Save increment
        results["pvals"].append(p_2tail)
        results["p_cluster"].append(p_cluster)
        results["p_regular"].append(p_regular)
        results["p_global_2tail"].append(p_global_2tail)
        results["p_global_cluster"].append(p_global_cluster)
        results["p_global_regular"].append(p_global_regular)
        results["M_values"].append(M)
        results["area_mean"].append(area_mean)

    return results


def monte_carlos_ripley_incremental(
    save,
    R,
    rall,
    size_monte=200,
    K_obs=None
):
    mesh = save['mesh_shaft']
    NN = len(save['shaft_close_index'])

    # Prepare storage
    areas = np.zeros((size_monte, len(rall)))

    param_ripley = dict(
        N_plot=40,
        N_interpolate=None,
        rall=rall
    )

    tree = cKDTree(mesh.vertices)

    # --- Monte-Carlo simulations ---
    for ii in range(size_monte): 
        points = mesh.sample(NN)
        list_index = tree.query(points)[1]
 
        dist,_ = get_distance(save, list_index=list_index,cutoff=rall[-1])

        # Compute Ripley curve
        ripp, _ ,_= ripley_all_save(save, R, **param_ripley,
                                  list_index=list_index, dist=dist)

        areas[ii, :] = ripp

    # Mean Ripley curve
    area_mean = np.mean(areas, axis=0)

    # If no observed curve, return only mean
    if K_obs is None:
        return area_mean, None

    # --- Incremental convergence storage ---
    results = {
        "pvals": [],
        "p_cluster": [],
        "p_regular": [],
        "p_global_2tail": [],
        "p_global_cluster": [],
        "p_global_regular": [],
        "area_mean": [],
        "M_values": []
    }

    # --- Incremental Monte-Carlo convergence ---
    for M in range(5, size_monte + 1):

        A = areas[:M, :]

        # Per-radius p-values
        p_cluster = (1 + np.sum(A >= K_obs, axis=0)) / (M + 1)
        p_regular = (1 + np.sum(A <= K_obs, axis=0)) / (M + 1)
        p_2tail = np.minimum(2 * np.minimum(p_cluster, p_regular), 1.0)

        area_mean = np.mean(A, axis=0)
        # Global tests
        D_obs = np.sum((K_obs - area_mean)**2)
        D_sim = np.sum((A - area_mean)**2, axis=1)

        p_global_2tail = (1 + np.sum(D_sim >= D_obs)) / (M + 1)
        p_global_cluster = (1 + np.sum(np.sum(A, axis=1) >= np.sum(K_obs))) / (M + 1)
        p_global_regular = (1 + np.sum(np.sum(A, axis=1) <= np.sum(K_obs))) / (M + 1)

        # Save increment
        results["pvals"].append(p_2tail)
        results["p_cluster"].append(p_cluster)
        results["p_regular"].append(p_regular)
        results["p_global_2tail"].append(p_global_2tail)
        results["p_global_cluster"].append(p_global_cluster)
        results["p_global_regular"].append(p_global_regular)
        results["M_values"].append(M)
        results["area_mean"].append(area_mean)

    return results







def monte_carlos_ripley_poisson_incremental(
    save,
    R,
    rall,
    size_monte=200,
    lambda_intensity=None,  # points per unit area
    K_obs=None
):
    mesh = save['mesh_shaft']

    # Mesh area
    A_mesh = mesh.area
 
    if lambda_intensity is None:
        NN_obs = len(save['shaft_close_index'])
        lambda_intensity = NN_obs / A_mesh  # points per unit area

    # Storage
    areas = np.zeros((size_monte, len(rall)))

    param_ripley = dict(
        N_plot=40,
        N_interpolate=None,
        rall=rall
    )

    tree = cKDTree(mesh.vertices)

    for ii in range(size_monte): 
        # N ~ Poisson(lambda * area)
        N=0
        while N<2:
            N = np.random.poisson(lambda_intensity * A_mesh)
 

        points = mesh.sample(N)
        list_index = tree.query(points)[1]

        # Distances
        dist,intensity = get_distance(save, list_index=list_index,cutoff=rall[-1])

        # Ripley curve
        ripp,_,_= ripley_all_save(
            save, R, **param_ripley,
            list_index=list_index, 
            dist=dist,
            intensity=intensity
        )


        areas[ii, :] = ripp

    # Mean Ripley curve
    area_mean = np.mean(areas, axis=0)

    # If no observed curve, return only mean
    if K_obs is None:
        return area_mean, None

    # --- Incremental convergence storage ---
    results = {
        "pvals": [],
        "p_cluster": [],
        "p_regular": [],
        "p_global_2tail": [],
        "p_global_cluster": [],
        "p_global_regular": [],
        "area_mean": [],
        "M_values": []
    }

    # --- Incremental Monte-Carlo convergence ---
    for M in range(5, size_monte + 1):

        A = areas[:M, :]

        # Per-radius p-values
        p_cluster = (1 + np.sum(A >= K_obs, axis=0)) / (M + 1)
        p_regular = (1 + np.sum(A <= K_obs, axis=0)) / (M + 1)
        p_2tail = np.minimum(2 * np.minimum(p_cluster, p_regular), 1.0)

        area_mean = np.mean(A, axis=0)
        # Global tests
        D_obs = np.sum((K_obs - area_mean)**2)
        D_sim = np.sum((A - area_mean)**2, axis=1)

        p_global_2tail = (1 + np.sum(D_sim >= D_obs)) / (M + 1)
        p_global_cluster = (1 + np.sum(np.sum(A, axis=1) >= np.sum(K_obs))) / (M + 1)
        p_global_regular = (1 + np.sum(np.sum(A, axis=1) <= np.sum(K_obs))) / (M + 1)

        # Save increment
        results["pvals"].append(p_2tail)
        results["p_cluster"].append(p_cluster)
        results["p_regular"].append(p_regular)
        results["p_global_2tail"].append(p_global_2tail)
        results["p_global_cluster"].append(p_global_cluster)
        results["p_global_regular"].append(p_global_regular)
        results["M_values"].append(M)
        results["area_mean"].append(area_mean)

    return results















 

'''

from ripley import ripley_all,generate_points
import numpy as np



a = np.array([0., 0., 0.])
b = np.array([0., 0., 10.])
R = 2.0
points = generate_points(40, R)   # your cylinder points



Circon=(2*np.pi*R)
N_plot=40
rall=np.linspace(.01,Circon/4,N_plot)
ripp,Lrip=ripley_all(points,a,b,R,N_plot=40,N_interpolate=1000)

















from plotly.subplots import make_subplots
import plotly.graph_objects as go

fig = make_subplots(
    rows=1,
    cols=2,
    subplot_titles=("Ripley's K", "Ripley's L")
)

fig.add_trace(
    go.Scatter(
        x=rall,
        y=ripp,
        mode='lines+markers',
        name='K(r)'
    ),
    row=1,
    col=1
)

fig.add_trace(
    go.Scatter(
        x=rall,
        y=np.pi*rall**2,
        mode='lines',
        name='πr²'
    ),
    row=1,
    col=1
)

fig.add_trace(
    go.Scatter(
        x=rall,
        y=Lrip,
        mode='lines+markers',
        name='L(r)-r'
    ),
    row=1,
    col=2
)

fig.add_hline(
    y=0,
    row=1,
    col=2,
    line_dash='dash'
)

fig.update_layout(
    height=500,
    width=1000,
    template='plotly_white'
)

fig.show()



# print(dist)
# 
'''