import numpy as np 
import plotly.graph_objs as go
import plotly.io as pio
  
def plotly_scatter(points, marker=None, mode='markers', color='red', symbol=None, size=2, opacity=1.0, name=None, showlegend=True): 
    dim = points.shape[1]
    if marker is None:
        markerr = dict(
            size=size,
            color=color,
            symbol=symbol,
            opacity=opacity  ,
        )
    else:
        markerr = marker 
        if 'opacity' not in markerr:
            markerr['opacity'] = opacity

    if dim == 3:
        return go.Scatter3d(
            x=points[:, 0],
            y=points[:, 1],
            z=points[:, 2],
            mode=mode,
            marker=markerr,
            name=name,
            showlegend=showlegend
        ) 
    else:
        return go.Scatter(
            x=points[:, 0],
            y=points[:, 1], 
            mode=mode,
            marker=markerr,
            name=name,
            showlegend=showlegend,
        )



def Lines_plot(v1,v2):
    dim = v1.shape[1]
    x = np.column_stack((v1[:, 0], v2[:, 0])).flatten()
    y = np.column_stack((v1[:, 1], v2[:, 1])).flatten()
    if dim ==3:
        z = np.column_stack((v1[:, 2], v2[:, 2])).flatten()
 
    x = np.insert(x, slice(2, None, 2), None)
    y = np.insert(y, slice(2, None, 2), None)
    if dim == 3:
        z = np.insert(z, slice(2, None, 2), None)
        return np.column_stack((x,y,z))
    else:
        return  np.column_stack((x,y))


def plotly_lines(points,points_proj,dash='dash',color='grey',width=2,mode='lines',showlegend=False): 
    connecting_lines = []
    for i in range(points.shape[0]):
        connecting_lines.append(go.Scatter3d(
            x=[points[i, 0], points_proj[i, 0]],
            y=[points[i, 1], points_proj[i, 1]],
            z=[points[i, 2], points_proj[i, 2]],
            mode=mode,
            line=dict(color=color, width=width, dash=dash),
            showlegend=showlegend,
        ))
    return connecting_lines
  



import plotly.graph_objs as go
import plotly.io as pio
from plotly.subplots import make_subplots
import numpy as np

def create_scatter_plot(cluu,dend, geo_cluster ): 
    geo_cluster.Cluster_PCA(cluster_index=cluu)
    geo_cluster.Lines()

    intensity = dend.mean_curv()[cluu]
    vertices_ = dend.vertices[cluu]
    point = geo_cluster.cluster_pca_points
    line_points_1 = geo_cluster.pca_line_1
    line_points_2 = geo_cluster.pca_line_2
    line_points_3 = geo_cluster.pca_line_3
    far_point = geo_cluster.cluster_farthest_point
    close_point = geo_cluster.cluster_closest_point

    scatter = [
        plotly_scatter(points=vertices_, color=intensity),
        plotly_scatter(points=line_points_1, color='blue'),
        plotly_scatter(points=line_points_2, color='black'),
        plotly_scatter(points=line_points_3, color='black'),
        plotly_scatter(points=point, color='green'),
        plotly_scatter(points=far_point.reshape(1, -1), color='yellow', size=7),
        plotly_scatter(points=close_point.reshape(1, -1), color='orange', size=7),
        plotly_scatter(points=point[-1].reshape(1, -1), color='grey', size=7),
    ]
    
    return scatter

def create_subplots(clu, clu_new ,dend,geo_cluster , width=500, height=400):
    # Create subplots
    fig = make_subplots(rows=1, 
                        cols=2, 
                        specs=[[{'type': 'scatter3d'}, {'type': 'scatter3d'}]],
                        horizontal_spacing=0.0)
 
    scatter_1 = create_scatter_plot(clu, dend,geo_cluster )
    for scatter in scatter_1:
        fig.add_trace(scatter, row=1, col=1)
 
    scatter_2 = create_scatter_plot(clu_new, dend,geo_cluster )
    for scatter in scatter_2:
        fig.add_trace(scatter, row=1, col=2)

    # Update layout
    fig.update_layout(
        width=2 * width,  
        height=height
    )

    return fig


 
import os
import numpy as np
















def get_configs(x0=None,
                step=100,
                step_test=None,
                window=10,
                process_std_init=2.1,
                process_std_true=2,
                process_std=0.001,
                measurement_std=0.01,
                measurement_std_true=0.1,
                dt=0.001,
                weight=0.1,
                d_nam=None,
                dyna=None,
                **kward): 
    Model_all={} 
    dnn_modes=[]
    KB=None
    # for d_nam in dyna:
    # for step in steps:
    #     for window in [10,step]: 
    mnm=0
    inten_pinn_index=[]
    inten_pinn_index.append([])
    inten_pinn_index.append([])
    base_features_index=[]
    base_features_index.append( [mnm+i for i in [0-mnm,2-mnm,7,8,9,10,11,12,13,14,15]])
    base_features_index.append([])
    pre_opts=[]
    pre_opts.append('pre')
    pre_opts.append('pre')
    for state_dim,obs_dim,inten,base,pre_opt in zip([3,3  ],
                                                    [3,2 ],
                                                    inten_pinn_index,
                                                    base_features_index,
                                                    pre_opts,
                                                    ):
        # mo=f'{d_nam}___s{state_dim}_o{obs_dim}_w{window}_s{step}'
        mo=f'DNN_{obs_dim}'
        
        step=step if step_test is None else step_test
        param={
            mm:{} for mm in ['par_0','train_param','train_fun','base_features_index','inten_pinn_index']
        } 
        param['train_param']['epoc']=10
        param['train_param']['avg_train']=5 
        dnn_modes.append(mo)
        Model_all[mo]=dict(
                        data_sufix= mo,
                        dest_sufix= mo, 
                        inten_pinn_index=inten,
                        base_features_index=base,
                        pre_opt=pre_opt,

                        param=param,
                )
 


    return Model_all,dyna,dnn_modes
















dynadic={}

def app_dict_param(param,nam_gen=None,navbar=None,):
    nam_gen=nam_gen if nam_gen is not None else param['nam_gen']['param'] 
    action=param["action"]["param"]
    file_path_org=param['file_path_org']['param']
    file_path_data=param['file_path_data']['param']
    path_heads_show=param['path_heads_show']['param']
    path_dirs_show=param['path_dirs_show']['param'] 

    nam=f'{nam_gen}_{action}'
    os.makedirs(file_path_data, exist_ok=True)  
    from mod.dsa.neld_fun_0.main_0_help import get_dict_param 
    LorenzParam=param['dyna param']['param'] 
    nam_gens=nam_gen.split('_')
    d_nam=nam_gen if len(nam_gens)<2 else '_'.join(nam_gens[:2])
    if d_nam not in dynadic:
        dynadic[d_nam]=dict(
                            # param=dict(
                            #     sigma = LorenzParam['sigma'],
                            #     rho = LorenzParam['rho'],
                            #     beta =LorenzParam['beta'], 
                            #     gamma =LorenzParam['gamma'], 
                            #     ),
                        param=LorenzParam,
                        # fun=gforce.get_force(name=nam_gen,type='force_single'),
                        # dyn_jacobian=gforce.get_force(name=nam_gen,type='jacobian'),
                        # obs_jacobian=gforce.get_force(name=nam_gen,type='jacobian'),
                    )
    print('[[[[[[[[[[pppp]]]]]]]]]]',d_nam,LorenzParam)
    configs,dyna,dnn_modes=get_configs(d_nam=d_nam,
                                       dyna=dynadic,
                                       **LorenzParam)

    dict_param=get_dict_param(nam=nam,
                        n_step = 0,    
                        dnn_modes=dnn_modes, 
                        configs=configs,
                        file_path_org=file_path_org,
                        file_path_data=file_path_data,
                        path_heads_show=path_heads_show,
                        path_dirs_show=path_dirs_show,
                        dyna=dyna,
                        navbar=navbar,
                        )
    return dict_param






def app_get_data_all(param,nam_gen=None,gdas=None):
    from mod.dsa.neld_fun_0.main_0_help import get_data_all
    nam_gen=nam_gen if nam_gen is not None else param['nam_gen']['param'] 
    action=param["action"]["param"]
    file_path_org=param['file_path_org']['param']
    file_path_data=param['file_path_data']['param']
    path_heads_show=param['path_heads_show']['param']
    path_dirs_show=param['path_dirs_show']['param'] 
    dynaParam=param['dyna param']['param']
    neld_names_chld={}
    action='train'
    nam=f'{nam_gen}_{action}' 
    if gdas is None or nam not in gdas.neld_data: 
        obj_org_path= os.path.join(file_path_org,'data_initial',nam_gen)
        # nbr_train_sample=dynaParam.get('nbr_train_sample',10)
        # obj_list=np.arange(nbr_train_sample) 
        neld_names = [
            mm for mm in os.listdir(obj_org_path) 
            if mm not in ['.DS_Store'] and os.path.isdir(os.path.join(obj_org_path, mm))
        ]
        # neld_names = [f'd{str(i).zfill(3)}' for i in obj_list]  
        neld_last = [f'{de}_'  for de in neld_names]
        neld_first= [ f'{de}' for de in neld_names] 
        
        neld_path_inits=[nam for _ in range(len(neld_names))]
        neld_names_chld[nam]= dict(   
                        neld_names=neld_names , 
                        obj_org_path=obj_org_path,
                        neld_last=neld_last ,
                        neld_first=neld_first ,
                        neld_path_inits=neld_path_inits,  
                        data_studied=action,
                        )   
    action='test' 
    nam=f'{nam_gen}_{action}'

    if gdas is None or nam not in gdas.neld_data:
        # path_heads_show.extend([f'rnn_KAL___{nam_gen}',f'tfm_KAL___{nam_gen}'])
        # obj_list=np.arange(5) 
 
        # nbr_train_sample=dynaParam.get('nbr_train_sample',10)
        # obj_list=np.arange(nbr_train_sample) 
        neld_names = [
            mm for mm in os.listdir(obj_org_path) 
            if mm not in ['.DS_Store'] and os.path.isdir(os.path.join(obj_org_path, mm))
        ] 
        neld_last = [f'{de}_'  for de in neld_names]
        neld_first= [ f'{de}' for de in neld_names]  

        neld_path_inits=[nam for _ in range(len(neld_names))]
        neld_names_chld[nam]= dict(  
                        # neld_namess=neld_namess ,
                        neld_names=neld_names , 
                        obj_org_path=obj_org_path,
                        neld_last=neld_last ,
                        neld_first=neld_first ,
                        neld_path_inits=neld_path_inits,  
                        data_studied=action,
        name_spine_id='sp',
        name_head_id='hsp',
        name_neck_id='nsp',
        name_shaft_id='shsp',
                        
                        )  
    if len(neld_names_chld)>0:
        neld_data=None if gdas is  None else gdas.neld_data 
        gdas=get_data_all(names_dic=neld_names_chld,
                        neld_data= neld_data ,
                        file_path_data=file_path_data,
                        cpath=[nam_gen, action],)   

    return gdas














