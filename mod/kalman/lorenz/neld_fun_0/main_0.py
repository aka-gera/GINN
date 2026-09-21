# -*- coding: utf-8 -*-
# Import required libraries

import sys
import os
 
import pickle
# import dash
from dash import dcc, html, dash_table, Input, Output, State, callback 
import dash_bootstrap_components as dbc
import numpy as np 
from mod.kalman.lorenz.neld_fun_0.main_0_help import get_navs_bar
from mod.kalman.lorenz.neld_fun_0.help_funn import safe_path_join

from  mod.kalman.lorenz.neld_fun_0.main_0_help import box_style,dropdown_style,dropdown_options_style

import time
import sys,os
file_path_org=os.getcwd()   
file_path_org=safe_path_join(file_path_org,'files') 

DTYPE='float32'  

from neld_data_all import paraws,params

from neld_data_all import general_nam_main,file_path_datab,file_path_orgb,github_link

from mod.kalman.lorenz.neld_fun_0.help_model_pred import model_pred 
from mod.kalman.lorenz.neld_fun_0.get_path import get_path_train,get_data_mode  
from mod.kalman.lorenz.apps.side_bar import  dnn_page
file_path_parent=file_path_org 


from mod.kalman.lorenz.neld_fun_0.main_0_help import dropdown_callback

mdrp=dropdown_callback()


def algorithm_param(file_path_org,
                    file_path_data,
                    neld_names_all=None,
                    nam='meshes',
                    n_step = 200,
                    weight=3.,
                    weight2=1.,
                    size_threshold=100,
                    path_heads=None, 
                    dnn_modes=None,
                    path_dirs=None,
                    path_list=None,
                    path_display=None,
                    tf_restart=False,
                    kmean_n_run=50,
                    kmean_max_iter=600,
                    param_dic=None,
                    path_dir=None,
                    data_dir=None,  
                    model_type_data =None,
                    path_display_dic=None,
                    path_heads_show=None,
                    path_dirs_show=None,
                    nam_gen_show=None,
                    path_hmod_dir=None,
                    path_dirs_weig=None,
                    head_navbar=None,
                    path_heads_show_nami=None,
					wrap_method='alpha_wrap',
                    # path_display_dic=None,
                    pre_portion=None,
                    train_neld_param=None,
                    configs=None,
                    gdas=None,
                    data_studied=None,
                    tname=None,
                    navbar=None,
                    **kward,
                    ):
    if param_dic is None: 
        param_dic={
            hh:{
                kk:True for kk in ['get_pinn_features','get_wrap','get_scale','get_smooth','get_skeleton']
            }
            for hh in ['tf_restart','get_info']
        }
    if path_heads is None:
        path_heads=['pinn', ]
        path_heads=['pinn','rpinn','ML','pinn_new','unet3d']
    # if dnn_modes is None:
    #     dnn_modes=['DNN-0','DNN-1','DNN-2' , 'DNN-3', 'DNN-5']
    if path_list is None:
        path_list=['hmod_path','smod_path',  ] 
    if path_dirs is None:
        path_dirs=['save',  ] 
    if path_display is None:
        path_display=['dest_hmod_path','dest_smod_path',]
        path_display=['dest_smod_path_pre','dest_smod_path','dest_hmod_path']
    
    print('[[[[[[[[[[[-------]]]]]]]]]]result]',nam_gen_show)
    if nam_gen_show is None:
    #     # print('[[[[[------------------]]]]]',file_path_org)
    #     file_path_org=file_path_org or safe_path_join(os.getcwd(),'files')
    #     print('[[[[[---------file_path_data---------]]]]]',file_path_data)
    #     file_path_data=file_path_data if file_path_data is not None else safe_path_join(file_path_org,'data_initial')
        path_diroi=os.path.join(file_path_org,'data_initial')
        if not os.path.exists(path_diroi):
            ttname=os.path.basename(file_path_org)
            ggname=os.path.basename(os.path.dirname(file_path_org))
            file_path_data=os.getcwd()
            path_diroi=os.path.join(file_path_data, 'files',ggname,ttname,'data') 
            os.makedirs(path_diroi,exist_ok=True) 
            path_diroi=os.path.dirname(os.path.dirname(path_diroi))
        nam_gen_show=[mm for mm in os.listdir(path_diroi) if mm not in ['.DS_Store',]]
    #     print('[[[[[---------file_path_data---------]]]]]',file_path_data,nam_gen_show)
    #     nam_gen_show=['model_0',] if len(nam_gen_show)==0 else nam_gen_show 
    # print('[[[[[[[[[[[-------]]]]]]]]]]result]',nam_gen_show)
  
    smooth_tf=True
    smooth_tf=False
    # data_studied='train'  
    
    # data_studied='test' 
    skip_first_n=1
    end_thre=15  
    subdivision_thre=2
    zoom_thre=50
    skip_end_n=14
    skip_mid_n=5 
    thre_target_number_of_triangles=10
    voxel_resolution=128
    tres=['myles','p21-models',nam]


    line_num_points_hmod=300
    line_num_points_inter_hmod=400 if nam in tres else 80 

    line_num_points_hmod=200
    line_num_points_inter_hmod=150 if nam in tres else 80 

    spline_smooth_hmod=1 
    # size_threshold=400 if nam in tres else 60
 

    dict_mesh_to_skeleton_finder_mesh=dict( 
                interval_voxel_resolution=[100 ], 
                interval_target_number_of_triangles=[200000], 
                tf_largest=False,
                disp_infos=True, 
                min_voxel_resolution=200,
                min_target_number_of_triangles=1000,
                tf_division=True, 
                alpha_fraction=.8,
                offset_fraction=0.800,
                wrap_method=wrap_method,
                    )

    param_get_segments=dict(
                            thre_distance_min=1.75 ,
                            thre_distance_max=1.5,
                            thre_explained_variance_ratio=0.97,
                            size_threshold=size_threshold, 
                            tf_merge=False, 

    )
 
    tf_skl_hmod_distance=True  

    dict_wrap=dict(  
                number_of_points=5000,
                radius=0.1, 
                max_nn=30,
                )

 
    a,b,n=3,10,6 
    weights=[np.array([nn,.81]) for nn,mm in zip(np.arange(a,b,(b-a)/n),np.arange(a,b,(b-a)/n))] 
    weights=[[3,.81]] 
    
#  [ 'result',]+
    param_clean=path_heads_show
 
    param_dropdown=['nam_gen_show','path_heads_show','actions', 'dnn_modes','path_dirs_show', ]
    param_dropdowni=['nam_gen','path_head','action','dnn_mode','path_dir',]
    # param_input=['Smooth','Resizing','model-pred', 'Morphologic Param','model_shap','clean_path_dir', ]
    param_input=['dyna param','get_training']
    param_inputbv=['neld_names_all', 'intensity_rhs','intensity_all','iou','roc','graph_center','dash_pages', 'model-pred', 'clean_path_dir', 
                   'cylinder_heatmap' ,'annotations', 'model_pred' , 'rhs','skl','Skeleton','Smooth','Resizing',  'Morphologic Param','model_shap', ]
    param_input_save=[ 'smods_segss_old','smods_segss_old_2', 'clean_path_dir','skl_hmod_pred']
    param_list=['tname','file_path_org','param_dropdown','param_input','param_all', 'path_dirs_weig', 'param_list','param_dropdowni','file_path_data','param_inputbv','path_list','head_navbar','path_heads_show_nami','file_path_data',]
    param_all=[]
    param_all.extend(param_dropdown)
    param_all.extend(param_dropdowni)
    param_all.extend(param_input)
    param_all.extend(param_inputbv)
    param_all.extend(param_input_save)
    param_all.extend(param_list)
    param_all.extend(param_clean) 

    param={key:
                {
                    'tf':{},
                    'param':{},
                    'param_fix':{},
                    'option':[],
                }
            for key in param_all
            }  
    param['tname']['param']=tname
    param['file_path_org']['param']=file_path_org
    param['file_path_data']['param']=file_path_data
    param['param_list']['param']=param_list
    param['path_heads_show']['param']=path_heads_show 
    param['nam_gen_show']['param']= nam_gen_show 
    param['nam_gen']['param']= nam_gen_show[0]
    param['actions']['param']=['test','train'] 
    param['path_dirs_show']['param']=path_dirs_show 
    param['path_list']['param']=path_list
    param['param_dropdown']['param']=param_dropdown
    param['param_dropdowni']['param']=param_dropdowni
    param['param_input']['param']=param_input
    param['param_inputbv']['param']=param_inputbv
    param['param_all']['param']=param_all 
    param['dnn_modes']['param']=dnn_modes
    param['path_head']['param']=path_heads_show[0]
    param['dnn_mode']['param']=dnn_modes[0] 
    param['path_dir']['param']=path_dirs_show[0]
    param['action']['param']=param['actions']['param'][0]
    param['path_dirs_weig']['param']=path_dirs_weig
    param['head_navbar']['param']= head_navbar if head_navbar is not None else get_navs_bar(path_heads_show=path_heads_show,**navbar)

    param['path_heads_show_nami']['param']= dict(zip([ 
                                                    'dnn_GINN_SM00000_LOC_AUG',
                                                    f'cnn_3UNet3D3_5000_hpcc_crop',
                                                    f'cnn_VGG16_FCN3D_5000_hpcc_crop',
                                                    f'cnn_VoxNetSeg_5000_hpcc_crop', 
                                                    'gcn_UNet_SM10000_LOC',
                                                    'cml_cML',
                                        ],[ 
                                                        'GINN',
                                                        f'UNet',
                                                        f'VGG16_FCN3D',
                                                        f'VoxNetSeg', 
                                                        'GCN',
                                                        'cML',
                                            ] ))


    param['dyna param']['param']=  dict(  
                                        # **params['param_flows'][0],
                                        **kward.get('param_flows'),
                                        )
    param['Smooth']['param']=dict(
                                get_data=True, 
                                n_step = n_step, 
                                method=['willmore','taubin',],
                                dt=1e-6, )
    param['Smooth']['param_fix']=dict(  
                                disp_time=500,  )
    param['Smooth']['tf']=tf_smooth=False#True 

    param['annotations']['tf']= False#True



    param['clean_path_dir']['param']=  dict(  
                                        path_clean=param_clean, 
                                        )

    param['intensity_all']['param']=dict(  
                                        thr_gauss=45,
                                        thr_mean=15,
                                        )
    param['intensity_all']['tf']= False


    param['intensity_rhs']['param_fix']=dict(   
						radius_threshold=None, 
                                        )
    param['intensity_rhs']['tf']= False

    param['model_shap']['param_fix']=dict(  
                                    neld_names_ls=[0,1],
                                    n_shap=25, 
                                    ) 


    rl_par= 0.7
    dnn_par= 1-rl_par
    mj=np.arange(1,1000,200)/1000
    weightt=np.array([ mj,mj[::-1]]).T
    param['get_training']['param']=dict(  
                                        get_training=True,
                                        # train_neld_param=train_neld_param,
                                        # line_num_points_hmod=line_num_points_hmod,
                                        # line_num_points_inter_hmod=line_num_points_inter_hmod,  
                                        itime = 1500,
                                        itime_div=100, 
                                        ls=[0.85],#[0,1,2,3],
                                        # dest_path='data_hmod_path',
                                        # weight=weightt, 
                                        # num_sub_nodes=None,
                                        # rl_par= rl_par, 
                                        # dnn_par= dnn_par,
                                        l1_values = [0,   1e-6,   1e-4, 1e-2],
                                        l2_values = [0,   1e-6,   1e-4, 1e-2], 
                                        )
    param['model-pred']['param']=dict( 
                                      size_threshold=size_threshold,
                                    )   
    param['Skeleton']['param']=dict( 
                                        # weight=weight,
                                        # size_threshold=size_threshold,
                                        # path='entry',
                                        # wrap_part='hmod_wrap' , 
                                        # weights=weights, 
                                        # hmod_thre=0.9, 
                                        # smooth_tf=tf_smooth,
                                        # neck_lim=None,
                                        # dict_wrap=dict_wrap,
                                        # dict_mesh_to_skeleton_finder=dict_mesh_to_skeleton_finder_mesh, 
                                        # tf_skl_hmod_distance=tf_skl_hmod_distance,
                                        # tf_restart=tf_restart,

                                    )   
    param['model-pred']['param_fix']=dict(
                                        weight=weight,
                                        weight2=weight2,
                                        # train_smods=train_smods,
                                        # model_type=model_type,
                                        # # # 
                                        weights=weights, 
                                        hmod_thre=0.9, 
                                        smooth_tf=tf_smooth,
                                        neck_lim=None,
                                        dict_wrap=dict_wrap,
                                        dict_mesh_to_skeleton_finder=dict_mesh_to_skeleton_finder_mesh, 
                                        tf_skl_hmod_distance=tf_skl_hmod_distance,
                                    )   




    param['Resizing']['param']=dict(  
                                        # min_target_number_of_triangles_faction=600000,
                                        target_number_of_triangles_faction=200000,
                                                )
    param['Resizing']['param_fix']=dict( 
                                        thre_target_number_of_triangles=thre_target_number_of_triangles,
                                        voxel_resolution=voxel_resolution,
                                        annotation_resized_train_tf=False, 
                                        nam=nam,
                                                )

    param['skl_hmod_pred']['tf']= True

    param['smods_segss_old']['param_fix']=dict( 
                                        seg_neld='nfull', 
                                        get_refine_=False,
                                        f=3.,
                                        zoom_thre=zoom_thre ,
                                        skip_first_n=skip_first_n,
                                        skip_mid_n=skip_mid_n,
                                        skip_end_n=skip_end_n,
                                        subdivision_thre=subdivision_thre,
                                        end_thre=end_thre,
                                        size_threshold=size_threshold, 
                                        smod_fraction=0.20,
                                        ctl_run_thre=0,
                                        spline_smooth_hmod=spline_smooth_hmod ,
                                        smooth_tf=smooth_tf, 
                                        )
    param['smods_segss_old']['tf']= True


    param['smods_segss_old_2']['param_fix']=dict( 
                                        seg_neld='nfull', 
                                        get_refine_=False,
                                        f=3.,
                                        zoom_thre=zoom_thre ,
                                        skip_first_n=skip_first_n,
                                        skip_mid_n=skip_mid_n,
                                        skip_end_n=skip_end_n,
                                        subdivision_thre=subdivision_thre,
                                        end_thre=end_thre,
                                        size_threshold=size_threshold, 
                                        smod_fraction=0.20,
                                        ctl_run_thre=0,
                                        spline_smooth_hmod=spline_smooth_hmod ,
                                        smooth_tf=smooth_tf, 
                                        param_get_segments=param_get_segments,
                                        )

    param['Morphologic Param']['param_fix']=dict(  
                                get_refine_=False,
                                f=3.,
                                zoom_thre=zoom_thre,
                                skip_first_n=skip_first_n,
                                skip_end_n=skip_end_n,
                                subdivision_thre=subdivision_thre,
                                end_thre=45,
                                size_threshold=size_threshold,
                                # head_neck_path='dest_smod_path_pre',
                                ctl_run_thre=0,
                                spline_smooth=spline_smooth_hmod , 
                                seg_neld='nfull',
                        ) 

    param['model_pred']['param_fix']=dict(
                                        file_path_data=file_path_data,
                                        neld_names_all=neld_names_all, 
                                        file_path_org=file_path_org,
                                        data_studied=data_studied, 
                                        radius_threshold=0.9,
                                        size_threshold=size_threshold, 
                                        disp_infos=True,
                                        spline_smooth_hmod=spline_smooth_hmod,
                                        line_num_points_hmod=line_num_points_hmod,
                                        line_num_points_inter_hmod=line_num_points_inter_hmod, 
                                        thre_target_number_of_triangles=thre_target_number_of_triangles,
                                        voxel_resolution=voxel_resolution, 
                                        path_display=path_display,
                                        dict_mesh_to_skeleton_finder_mesh=dict_mesh_to_skeleton_finder_mesh,
                                        kmean_n_run=kmean_n_run,
                                        kmean_max_iter=kmean_max_iter,
                                        param_dic=param_dic,
                            )
    param['model_pred']['tf']=True 
    param['model_shap']['tf']=False

    param['model-pred']['tf']=True 
    param['smods_segss_old']['tf']=False
    param['skl_hmod_pred']['tf']=False
    param['smods_segss_old_2']['tf']= False
    param['Morphologic Param']['tf']= True
    param['Resizing']['tf']=False
    param['skl']['tf']=True
    param['Skeleton']['tf']=True
    param['rhs']['tf']=True
    param['iou']['tf']=True
    param['roc']['tf']=True
    param['graph_center']['tf']=True
    param['cylinder_heatmap']['tf']=True
    param['dash_pages']['tf']=True 
    param['clean_path_dir']['tf']= False


    return param
 




class app_run_param:
    def __init__(self,param):
        self.param=param
        self.Input=[]
        self.Input_id=[]
        self.Input_id_coll_head=[]
        self.Input_id_coll=[] 
        self.Output=[]
        self.graph=[]
        self.path_head=param['path_head']['param']
        self.dnn_mode=param['dnn_mode']['param']
        self.path_dir=param['path_dir']['param'] 
        self.path_heads=param['path_heads_show']['param'] 
        vvf=param['dnn_modes']['param'] 
        self.dnn_modes=param['dnn_modes']['param']
        self.path_dirs_show=param['path_dirs_show']['param'] 
        self.path_list=param['path_list']['param']
        self.file_path_data=param['file_path_data']['param']
        self.file_path_org=param['file_path_org']['param']
        self.nam_gen=param['nam_gen']['param']
        
        self.tname=param['tname']['param']
        self.mdrp=dropdown_callback(nam=self.tname)
        # self.Input_id.append('upload-data')
        typi='prediction'
        par = self.mdrp.callback_file[typi]['par']# 'param_mapp-store'
        loa = self.mdrp.callback_file[typi]['loa']# 'page_mapp-load'
        type= self.mdrp.callback_file[typi]['type']# 'd000'
        lst = self.mdrp.callback_file[typi]['lst']# 'param_mapp-store'
        mal = self.mdrp.callback_file[typi]['mal']# 'page_mapp-load'
        share=self.mdrp.callback_file['figure']['share']

        print('[[[[[[[[[[[lang]]]]]]]]]]]',self.file_path_org,self.mdrp.callback_file)
        pag='lang'
        pag=self.pag=os.path.basename(self.file_path_org)
        paramk=param['path_heads_show_nami']['param']
        for gval,gvali,mb in zip(param['param_dropdown']['param'],param['param_dropdowni']['param'],[0,0,0,0,0]):
            # print(gval,param[gval]['param'] ) 
            # if gval=='path_heads':
            #     mnm=[param['path_heads_show_nami']['param'].get(nn,nn) for nn in param[gval]['param']]

            dropdown_option = [
                {'label': paramk.get(val,val) , 'value': val, 'style': dropdown_options_style}
                for   val in param[gval]['param'] 
            ]
            idx=f'dropdown_{gvali}_{pag}_param'
            # idx=f'dropdown_{gvali}_param'
            self.Input_id.append(idx)
            # idx=f'dropdown_{gvali}_param'
            # self.Input_id.append(idx)
            # param[gvali]['option'] = dcc.Dropdown(
            #     options=dropdown_option,
            #     id=idx,
            #     value=dropdown_option[0]['value'],   # default to first value
            #     placeholder=f'Select {gvali}',
            #     style=box_style
            # )
            '''
            param[gval]['option'] = dcc.Dropdown(
                options=dropdown_option,
                id=idx,
                value=dropdown_option[mb]['value'],  
                placeholder=f'Select {gval}',
                style=box_style
            )
            # self.graph.append(param[gval]['option'])
            self.graph.append(
                dbc.Col(
                    param[gval]['option'],
                    # xs=12, sm=6, md=4, lg=3,  # full width on mobile, 2 per row on small, 3 per row on medium, 4 per row on large
                     xs=12, sm=6, md=4, lg=3, 
                    #width=3,
                    
                    style={    "width": "100%" }
                    # style={'borderRight': '1px solid #ccc', 'paddingRight': '15px', "width": "100%" }
                                )
            )
  
'''

            # mal='container_param'
            if gvali in lst:#['nam_gen','path_head']:
                # for typ in [None,'train','gen']: 
                for typi in ['prediction','training','generation']:
                    par = self.mdrp.callback_file[typi]['par']# 'param_mapp-store'
                    loa = self.mdrp.callback_file[typi]['loa']# 'page_mapp-load'
                    type= self.mdrp.callback_file[typi]['type']# 'd000'
                    # lst = self.mdrp.callback_file[typi]['lst']# 'param_mapp-store'
                    mal = self.mdrp.callback_file[typi]['mal']# 'page_mapp-load' 

                    self.graph.append(
                        # html.Div(id=f"{mal}-{gvali}")# if typi is None else f"{mal}-{gvali}-{typi}" ) 
                        html.Div(id= f"{mal}-{gvali}-{type}" )
                        ), 
            else:

                options = [
                    {
                        "label": paramk.get(val, val),
                        "value": val,
                        "style": dropdown_options_style
                    }
                    for val in param[gval]["param"]
                ] 
                param[gval]['option'] = dcc.Dropdown(
                    options=options,
                    id=idx,
                    value=options[0]["value"] if options else None,
                    placeholder=f"Select {gvali}",
                    style=box_style
                )
                print(f'im jjjjj=============================={gvali}=')

                self.graph.append(
                    dbc.Col(
                        param[gval]['option'],
                        # xs=12, sm=6, md=4, lg=3,  # full width on mobile, 2 per row on small, 3 per row on medium, 4 per row on large
                        xs=12, sm=6, md=4, lg=3,  
                        style={    "width": "100%" }
                        # style={'borderRight': '1px solid #ccc', 'paddingRight': '15px', "width": "100%" }
                        )
                )



        for gval in param['param_input']['param'] : 
            self.Input_id.append(f"param_{gval}_tf")
            self.Input_id_coll_head.append(f"{self.pag}-collapse-header-{gval}")
            self.Input_id_coll.append(f"collapse-{gval}")
            for k, v in param[gval]['param'].items(): 
                self.Input_id.append(f"param_{gval}_{k}") 
            param[gval]['option'] = dbc.Card(
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col(
                            html.H5(f"{gval}",
                                    id=f"{self.pag}-collapse-header-{gval}",
                                    className="card-title", 
                                    style={"cursor": "pointer"}), 
                            width=9,
                            style={'display': 'flex', 'alignItems': 'center'}
                        ),
                        dbc.Col(
                            dbc.Checkbox(
                                id=f"param_{gval}_tf",
                                value=param[gval]['tf'],
                                label="Enabled"
                            ),
                            width=3,
                            style={'display': 'flex', 'justifyContent': 'flex-end', 'alignItems': 'center'}
                        )
                    ], style={'marginBottom': '10px'}),
 
                    html.Hr(style={
                        "borderTop": "1px solid #ccc",
                        "marginTop": "0.5rem",
                        "marginBottom": "1rem"
                    }), 
                    dbc.Collapse(
                        html.Div([
                            dbc.Row([
                                dbc.Col(html.Label(k, style={'textAlign': 'right'}), width=3),
                                dbc.Col(
                                    # dbc.Checkbox(
                                    #     id=f"param_{gval}_{k}",
                                    #     value=v,
                                    #     label="Enabled"
                                    # ) if isinstance(v, bool) else dcc.Input(
                                    #     id=f"param_{gval}_{k}",
                                    #     type="text",
                                    #     value=v,
                                    #     style={'width': '75px', 'marginLeft': 'auto'}  
                                    # ),
                                    # comp = (
                                        dbc.Checkbox(
                                                    id=f"param_{gval}_{k}", 
                                                    value=v, 
                                                    label="Enabled",
                                        ) if isinstance(v, bool) else
                                        dcc.Input(
                                            id=f"param_{gval}_{k}",
                                            type="text",
                                            value=v,
                                            style={'width': '105px', 'marginLeft': 'auto'}
                                        ) 
                                        # if isinstance(v, int) or isinstance(v, float) else 
                                        if isinstance(v, (int, float)) or (isinstance(v, list) and all(isinstance(x, (int, float)) for x in v)) else
                                        dcc.Dropdown(
                                            options=[{'label': val, 'value': val, 'style': dropdown_options_style} for val in list(v)],
                                            id=f"param_{gval}_{k}",
                                            value=list(v)[0],
                                            placeholder=f"Select {gval} {k}",
                                            style=dropdown_style,

                                        ),
                                    # )

                                    width=9,
                                    className="text-end"  # right-align inside column
                                )
                            ], style={'marginBottom': '10px'})
                            for k, v in param[gval]['param'].items()
                        ]),
                        id=f"{self.pag}-collapse-{gval}",
                        is_open=False
                    ),
                ]),
                style={"marginBottom": "0px", "boxShadow": "0 2px 6px rgba(0,0,0,0.15)"}
            ) 
            self.graph.append(
                dbc.Col(
                    param[gval]['option'],
                    xs=12, sm=6, md=4, lg=3, 
                   #  width=3,
 
                    style={'marginBottom': '0px', "width": "100%" },
                )
            )  
            # self.graph.append( 
            #         param[gval]['option'], 
            # )  

    
        
        self.Input=[Input(idx, 'value') for idx in self.Input_id] 
        self.prevent_initial_call=True 

        typi='prediction'
        par = self.mdrp.callback_file[typi]['par']# 'param_mapp-store'
        loa = self.mdrp.callback_file[typi]['loa']# 'page_mapp-load'
        type= self.mdrp.callback_file[typi]['type']# 'd000'
        lst = self.mdrp.callback_file[typi]['lst']# 'param_mapp-store'
        mal = self.mdrp.callback_file[typi]['mal']# 'page_mapp-load' 

        self.app_layout = html.Div([
            dcc.Store(id=f"{self.pag}-shared-data", storage_type="memory", clear_data=True, data={}),
            dcc.Interval(id=f"{loa}-{type}", interval=100, max_intervals=1),
            dcc.Store(id=f"{par}-{type}", storage_type="session" ),
                # dcc.Interval(
                #     id= 'page_mapp-load',
                #     interval=100,
                #     max_intervals=1
                # ),

                # dcc.Store(
                #     id='param_mapp-store',
                #     storage_type="session"
                # ),

            # Wrap the whole thing in a Card
            dbc.Card(
                dbc.CardBody([
                    dbc.Row([
                        # Left column
                        dbc.Col(
                            [
                                # Buttons row
                                dbc.Row([
                                    dbc.Col(
                                        # dbc.Card(
                                            dbc.CardBody([
                                                dbc.Button("Run", 
                                                id="run-button", 
                                                n_clicks=0, 
                                                color="primary",
                                               #  color="success",
                                                className="w-1",)
                                            ]),
                                        #     style={"margin": "0px", "padding": "0px"}
                                        # ),
                                        width="3", 
                                        className="d-flex justify-content-center" 
                                    ),
                                    dbc.Col(
                                        # dbc.Card(
                                            dbc.CardBody([
                                                dbc.Button("Reset", 
                                                id="reset-button", 
                                                n_clicks=0,  
                                                color="primary",
                                               #  color="warning",
                                                className="w-1",)
                                            ]),
                                        #     style={"margin": "0px", "padding": "0px"}
                                        # ),
                                        width="3", 
                                        className="d-flex justify-content-center" 
                                    ),
                                    dbc.Col(
                                        # dbc.Card(
                                            dbc.CardBody([
                                                dbc.Button("Restart", 
                                                id="restart-button", 
                                                n_clicks=0,  
                                                color="primary",
                                               #  color="warning",
                                                className="w-1",)
                                            ]),
                                            # style={"margin": "0px", "padding": "0px"}
                                        # ),
                                        width="3", 
                                        className="d-flex justify-content-center" 
                                    ),
                                    dbc.Col(
                                        # dbc.Card(
                                            dbc.CardBody([
                                                dbc.Button("Off", 
                                                id="shutdown-button", 
                                                n_clicks=0,
                                                color="primary",
                                               #   color="danger",
                                                className="w-1",)
                                            ]),
                                        #     style={"margin": "0px", "padding": "0px" }
                                        # ),
                                        width="3", 
                                        className="d-flex justify-content-center" 
                                    ),
                                ], 
                                justify="center",   # centers the columns horizontally
                                align="center",     # centers vertically inside the row
                                # className="g-0",    # removes gutter spacing between columns
                                style={"marginTop": "20px"}

                                ), 

                                
                                # # # # Destination input card 
                                # dbc.Col(
                                #     dbc.Card(
                                #         dbc.CardBody([
                                #             dbc.Row([ 
                                #                     dcc.Markdown('Destination Name', style={'textAlign': 'center', 'color': 'white'}),
                                #                     html.Hr(),
                                #                     dcc.Markdown(f'{file_path_datab}/', style={'textAlign': 'center', 'color': 'white'}),
                                #                     dcc.Input(
                                #                         id="id-destination",
                                #                         type="text",
                                #                         value=self.nam_gen,
                                #                         style={'width': '100%' }
                                #                     )
                                #             ], style={'marginBottom': '10px'}),
                        
                                #             # html.Hr(style={
                                #             #     "borderTop": "1px solid #ccc",
                                #             #     "marginTop": "0.5rem",
                                #             #     "marginBottom": "1rem"
                                #             # }),  
                                #         ]),
                                #         style={"marginBottom": "20px", "boxShadow": "0 2px 6px rgba(0,0,0,0.15)"}
                                #     ) ,
                                # xs=12, sm=6, md=4, lg=3,  
                                # style={'marginBottom': '0px', "width": "100%" },
                                # ),  
 
                                # # Parameter cards
                                
                                *self.graph
                            ],
        className="g-0"  ,
                            width=4,
                            style={"padding": "10px", "borderRight": "1px solid #ccc"}
                        ),

                        # Right column
                        dbc.Col(
                            [
                                dbc.Row([
                                    dbc.Card(
                                        dbc.CardBody([
                                            html.Div(
                                                dcc.Markdown(f'{general_nam_main}\nPrediction',
                                                            style={'textAlign': 'center', 'color': 'white'})
                                            )
                                        ]),
                                        style={"margin": "10px", "padding": "10px"}
                                    )
                                ], justify="start"),
                                
                                # Upload card
                                dbc.Row([
                                    dbc.Col(
                                        dbc.Card(
                                            dbc.CardBody([
                                                # dcc.Markdown('### Upload File', style={'textAlign': 'center', 'color': 'white'}),
                                                # html.Hr(),
                                                # dcc.Upload(
                                                #     id='upload-data',
                                                #     children=html.Div([
                                                #         'Drag and Drop or ',
                                                #         html.A('Select Files')
                                                #     ]),
                                                #     style={
                                                #         'width': '100%',
                                                #         'height': '60px',
                                                #         'lineHeight': '60px',
                                                #         'borderWidth': '1px',
                                                #         'borderStyle': 'dashed',
                                                #         'borderRadius': '5px',
                                                #         'textAlign': 'center',
                                                #         'margin': '10px',
                                                #         'color': 'white'
                                                #     },
                                                #     multiple=True
                                                # ),
                                                # html.Div(id='output-file-upload', style={'color': 'white', 'marginTop': 20}),
                                                html.Hr(style={'borderTop': '2px dashed white', 'margin': '20px 0'}),
                                                dbc.Card([
                                                    html.H3("Press Run to Start",
                                                            id=f"{self.pag}-run-status",
                                                            style={'color': 'lightgreen', 'textAlign': 'center'})
                                                ]),
                                            ]),
                                            style={
                                                "margin": "10px",
                                                "padding": "10px",
                                                "backgroundColor": "#2c2c2c",
                                                "width": "100%"
                                            }
                                        ),
                                        width='11'
                                    )
                                ], justify="center", align="center"),





                                # Parameters & Results cards
                                dbc.Row([
                                    dbc.Card(
                                        dbc.CardBody([
                                            html.Div(
                                                dcc.Markdown('### Parameters & Options',
                                                            style={'textAlign': 'center', 'color': 'white'}),
                                                id="toggle-text-starting",
                                                n_clicks=0,
                                                style={"cursor": "pointer"}
                                            ),
                                            html.Hr(),
                                            dbc.Collapse(
                                                dcc.Markdown(dnn_page()['starting']),
                                                id=f"{self.pag}-collapse-starting",
                                                is_open=False
                                            ),
                                        ]),
                                        style={"margin": "10px", "padding": "10px"}
                                    ),
                                    dbc.Card(
                                        dbc.CardBody([
                                            html.Div(
                                                dcc.Markdown('### Checking Results',
                                                            style={'textAlign': 'center', 'color': 'white'}),
                                                id="toggle-text-result",
                                                n_clicks=0,
                                                style={"cursor": "pointer"}
                                            ),
                                            html.Hr(),
                                            dbc.Collapse(
                                                html.Div([
                                                    dcc.Markdown(dnn_page()['restart']),
                                                    dcc.Markdown(dnn_page()['results']),
                                                ]),
                                                id=f"{self.pag}-collapse-result",
                                                is_open=False
                                            ),
                                        ]),
                                        style={"margin": "10px", "padding": "10px"}
                                    ),
                                ], justify="start"),

                                # GitHub link
                                dbc.Row([
                                    dbc.Col(
                                        dcc.Markdown(
                                            f"[View the full repository on GitHub]({github_link})",
                                            style={"textAlign": "center", "color": "white"}
                                        )
                                    )
                                ], justify="center")
                            ],
                            width=8,
                            style={"padding": "10px"}
                        )
                    ])
                ]),
                style={
                    "margin": "20px",
                    "padding": "20px",
                    "boxShadow": "0 4px 12px rgba(0,0,0,0.2)",
                    "backgroundColor": "#1e1e1e",
                    "maxWidth": "95%",   # scale control
                    "marginLeft": "auto",
                    "marginRight": "auto"
                }
            )
        ])



        type='train'
        typi='training'
        par = self.mdrp.callback_file[typi]['par']# 'param_mapp-store'
        loa = self.mdrp.callback_file[typi]['loa']# 'page_mapp-load'
        type= self.mdrp.callback_file[typi]['type']# 'd000'
        lst = self.mdrp.callback_file[typi]['lst']# 'param_mapp-store'
        mal = self.mdrp.callback_file[typi]['mal']# 'page_mapp-load'  
        self.app_layout_train = html.Div([
            dcc.Store(id=f"{self.pag}-shared-data-{type}", storage_type="memory", clear_data=True, data={}),

            dcc.Interval(id=f"{loa}-{type}", interval=100, max_intervals=1),
            dcc.Store(id=f"{par}-{type}", storage_type="session" ),

            # Wrap the whole thing in a Card
            dbc.Card(
                dbc.CardBody([
                    dbc.Row([
                        # Left column
                        dbc.Col(
                            [
                                # Buttons row
                                dbc.Row([
                                    dbc.Col(
                                        # dbc.Card(
                                            dbc.CardBody([
                                                dbc.Button("Run", 
                                                id=f"run-button-{type}", 
                                                n_clicks=0, 
                                                color="primary",
                                               #  color="success",
                                                className="w-1",)
                                            ]),
                                        #     style={"margin": "0px", "padding": "0px"}
                                        # ),
                                        width="3", 
                                        className="d-flex justify-content-center" 
                                    ),
                                    dbc.Col(
                                        # dbc.Card(
                                            dbc.CardBody([
                                                dbc.Button("Reset", 
                                                id=f"reset-button-{type}", 
                                                n_clicks=0,  
                                                color="primary",
                                               #  color="warning",
                                                className="w-1",)
                                            ]),
                                        #     style={"margin": "0px", "padding": "0px"}
                                        # ),
                                        width="3", 
                                        className="d-flex justify-content-center" 
                                    ),
                                    dbc.Col(
                                        # dbc.Card(
                                            dbc.CardBody([
                                                dbc.Button("Restart", 
                                                id=f"restart-button-{type}", 
                                                n_clicks=0,  
                                                color="primary",
                                               #  color="warning",
                                                className="w-1",)
                                            ]),
                                            # style={"margin": "0px", "padding": "0px"}
                                        # ),
                                        width="3", 
                                        className="d-flex justify-content-center" 
                                    ),
                                    dbc.Col(
                                        # dbc.Card(
                                            dbc.CardBody([
                                                dbc.Button("Off", 
                                                id=f"shutdown-button-{type}", 
                                                n_clicks=0,
                                                color="primary",
                                               #   color="danger",
                                                className="w-1",)
                                            ]),
                                        #     style={"margin": "0px", "padding": "0px" }
                                        # ),
                                        width="3", 
                                        className="d-flex justify-content-center" 
                                    ),
                                ], 
                                justify="center",   # centers the columns horizontally
                                align="center",     # centers vertically inside the row
                                # className="g-0",    # removes gutter spacing between columns
                                style={"marginTop": "20px"}

                                ),

                                # Destination input card 
                                dbc.Col(
                                    dbc.Card(
                                        dbc.CardBody([
                                            dbc.Row([ 
                                                    dcc.Markdown('Destination Name', style={'textAlign': 'center', 'color': 'white'}),
                                                    html.Hr(),
                                                    dcc.Markdown(f'{self.file_path_org}', style={'textAlign': 'center', 'color': 'white'}),
                                                    dcc.Input(
                                                        id=f"id-destination-{type}",
                                                        type="text",
                                                        value=self.nam_gen,
                                                        style={'width': '100%' }
                                                    )
                                            ], style={'marginBottom': '10px'}),
                        
                                            # html.Hr(style={
                                            #     "borderTop": "1px solid #ccc",
                                            #     "marginTop": "0.5rem",
                                            #     "marginBottom": "1rem"
                                            # }),  
                                        ]),
                                        style={"marginBottom": "20px", "boxShadow": "0 2px 6px rgba(0,0,0,0.15)"}
                                    ) ,
                                xs=12, sm=6, md=4, lg=3,  
                                style={'marginBottom': '0px', "width": "100%" },
                                ),
                                
                                # # Destination input card 
                                # dbc.Col(
                                #     dbc.Card(
                                #         dbc.CardBody([
                                #             dbc.Row([ 
                                #                     dcc.Markdown('Add Destination', style={'textAlign': 'center', 'color': 'white'}),
                                #                     html.Hr(),
                                #                     dcc.Input(
                                #                         id="id-destination",
                                #                         type="text",
                                #                         value=file_path_parent,
                                #                         style={'width': '100%' }
                                #                     )
                                #             ], style={'marginBottom': '10px'}),
                        
                                #             # html.Hr(style={
                                #             #     "borderTop": "1px solid #ccc",
                                #             #     "marginTop": "0.5rem",
                                #             #     "marginBottom": "1rem"
                                #             # }),  
                                #         ]),
                                #         style={"marginBottom": "20px", "boxShadow": "0 2px 6px rgba(0,0,0,0.15)"}
                                #     ) ,
                                # xs=12, sm=6, md=4, lg=3,  
                                # style={'marginBottom': '0px', "width": "100%" },
                                # ),
                                # Parameter cards
                                *self.graph
                            ],
        className="g-0"  ,
                            width=4,
                            style={"padding": "10px", "borderRight": "1px solid #ccc"}
                        ),

                        # Right column
                        dbc.Col(
                            [
                                dbc.Row([
                                    dbc.Card(
                                        dbc.CardBody([
                                            html.Div(
                                                dcc.Markdown(f'{general_nam_main}\nTraining',
                                                            style={'textAlign': 'center', 'color': 'white'})
                                            )
                                        ]),
                                        style={"margin": "10px", "padding": "10px"}
                                    )
                                ], justify="start"),
                                
                                # Upload card
                                dbc.Row([
                                    dbc.Col(
                                        dbc.Card(
                                            dbc.CardBody([
                                                # dcc.Markdown('### Upload File', style={'textAlign': 'center', 'color': 'white'}),
                                                # html.Hr(),
                                                # dcc.Upload(
                                                #     id='upload-data',
                                                #     children=html.Div([
                                                #         'Drag and Drop or ',
                                                #         html.A('Select Files')
                                                #     ]),
                                                #     style={
                                                #         'width': '100%',
                                                #         'height': '60px',
                                                #         'lineHeight': '60px',
                                                #         'borderWidth': '1px',
                                                #         'borderStyle': 'dashed',
                                                #         'borderRadius': '5px',
                                                #         'textAlign': 'center',
                                                #         'margin': '10px',
                                                #         'color': 'white'
                                                #     },
                                                #     multiple=True
                                                # ),
                                                # html.Div(id='output-file-upload', style={'color': 'white', 'marginTop': 20}),
                                                html.Hr(style={'borderTop': '2px dashed white', 'margin': '20px 0'}),
                                                dbc.Card([
                                                    html.H3("Press Run to Start",
                                                            id=f"{self.pag}-run-status-{type}",
                                                            style={'color': 'lightgreen', 'textAlign': 'center'})
                                                ]),
                                            ]),
                                            style={
                                                "margin": "10px",
                                                "padding": "10px",
                                                "backgroundColor": "#2c2c2c",
                                                "width": "100%"
                                            }
                                        ),
                                        width='11'
                                    )
                                ], justify="center", align="center"),





                                # Parameters & Results cards
                                dbc.Row([
                                    dbc.Card(
                                        dbc.CardBody([
                                            html.Div(
                                                dcc.Markdown('### Parameters & Options',
                                                            style={'textAlign': 'center', 'color': 'white'}),
                                                id=f"toggle-text-starting-{type}",
                                                n_clicks=0,
                                                style={"cursor": "pointer"}
                                            ),
                                            html.Hr(),
                                            dbc.Collapse(
                                                dcc.Markdown(dnn_page()['starting']),
                                                id=f"{self.pag}-collapse-starting-{type}",
                                                is_open=False
                                            ),
                                        ]),
                                        style={"margin": "10px", "padding": "10px"}
                                    ),
                                    dbc.Card(
                                        dbc.CardBody([
                                            html.Div(
                                                dcc.Markdown('### Checking Results',
                                                            style={'textAlign': 'center', 'color': 'white'}),
                                                id=f"toggle-text-result-{type}",
                                                n_clicks=0,
                                                style={"cursor": "pointer"}
                                            ),
                                            html.Hr(),
                                            dbc.Collapse(
                                                html.Div([
                                                    dcc.Markdown(dnn_page()['restart']),
                                                    dcc.Markdown(dnn_page()['results']),
                                                ]),
                                                id=f"{self.pag}-collapse-result-{type}",
                                                is_open=False
                                            ),
                                        ]),
                                        style={"margin": "10px", "padding": "10px"}
                                    ),
                                ], justify="start"),

                                # GitHub link
                                dbc.Row([
                                    dbc.Col(
                                        dcc.Markdown(
                                            f"[View the full repository on GitHub]({github_link})",
                                            style={"textAlign": "center", "color": "white"}
                                        )
                                    )
                                ], justify="center")
                            ],
                            width=8,
                            style={"padding": "10px"}
                        )
                    ])
                ]),
                style={
                    "margin": "20px",
                    "padding": "20px",
                    "boxShadow": "0 4px 12px rgba(0,0,0,0.2)",
                    "backgroundColor": "#1e1e1e",
                    "maxWidth": "95%",   # scale control
                    "marginLeft": "auto",
                    "marginRight": "auto"
                }
            )
        ])



        type='gen'
        typi='generation'
        par = self.mdrp.callback_file[typi]['par']# 'param_mapp-store'
        loa = self.mdrp.callback_file[typi]['loa']# 'page_mapp-load'
        type= self.mdrp.callback_file[typi]['type']# 'd000'
        lst = self.mdrp.callback_file[typi]['lst']# 'param_mapp-store'
        mal = self.mdrp.callback_file[typi]['mal']# 'page_mapp-load'  
        self.app_layout_gen = html.Div([
            dcc.Store(id=f"{self.pag}-shared-data-{type}", storage_type="memory", clear_data=True, data={}),

            dcc.Interval(id=f"{loa}-{type}", interval=100, max_intervals=1),
            dcc.Store(id=f"{par}-{type}", storage_type="session" ),
            # Wrap the whole thing in a Card
            dbc.Card(
                dbc.CardBody([
                    dbc.Row([
                        # Left column
                        dbc.Col(
                            [
                                # Buttons row
                                dbc.Row([
                                    dbc.Col(
                                        # dbc.Card(
                                            dbc.CardBody([
                                                dbc.Button("Run", 
                                                id=f"run-button-{type}", 
                                                n_clicks=0, 
                                                color="primary",
                                               #  color="success",
                                                className="w-1",)
                                            ]),
                                        #     style={"margin": "0px", "padding": "0px"}
                                        # ),
                                        width="3", 
                                        className="d-flex justify-content-center" 
                                    ),
                                    dbc.Col(
                                        # dbc.Card(
                                            dbc.CardBody([
                                                dbc.Button("Reset", 
                                                id=f"reset-button-{type}", 
                                                n_clicks=0,  
                                                color="primary",
                                               #  color="warning",
                                                className="w-1",)
                                            ]),
                                        #     style={"margin": "0px", "padding": "0px"}
                                        # ),
                                        width="3", 
                                        className="d-flex justify-content-center" 
                                    ),
                                    dbc.Col(
                                        # dbc.Card(
                                            dbc.CardBody([
                                                dbc.Button("Restart", 
                                                id=f"restart-button-{type}", 
                                                n_clicks=0,  
                                                color="primary",
                                               #  color="warning",
                                                className="w-1",)
                                            ]),
                                            # style={"margin": "0px", "padding": "0px"}
                                        # ),
                                        width="3", 
                                        className="d-flex justify-content-center" 
                                    ),
                                    dbc.Col(
                                        # dbc.Card(
                                            dbc.CardBody([
                                                dbc.Button("Off", 
                                                id=f"shutdown-button-{type}", 
                                                n_clicks=0,
                                                color="primary",
                                               #   color="danger",
                                                className="w-1",)
                                            ]),
                                        #     style={"margin": "0px", "padding": "0px" }
                                        # ),
                                        width="3", 
                                        className="d-flex justify-content-center" 
                                    ),
                                ], 
                                justify="center",   # centers the columns horizontally
                                align="center",     # centers vertically inside the row
                                # className="g-0",    # removes gutter spacing between columns
                                style={"marginTop": "20px"}

                                ),

                                # Destination input card 
                                dbc.Col(
                                    dbc.Card(
                                        dbc.CardBody([
                                            dbc.Row([ 
                                                    dcc.Markdown('Destination Name', style={'textAlign': 'center', 'color': 'white'}),
                                                    html.Hr(),
                                                    dcc.Markdown(f'{self.file_path_org}', style={'textAlign': 'center', 'color': 'white'}),
                                                    dcc.Input(
                                                        id=f"id-destination-{type}",
                                                        type="text",
                                                        value=self.nam_gen,
                                                        style={'width': '100%' }
                                                    )
                                            ], style={'marginBottom': '10px'}),
                        
                                            # html.Hr(style={
                                            #     "borderTop": "1px solid #ccc",
                                            #     "marginTop": "0.5rem",
                                            #     "marginBottom": "1rem"
                                            # }),  
                                        ]),
                                        style={"marginBottom": "20px", "boxShadow": "0 2px 6px rgba(0,0,0,0.15)"}
                                    ) ,
                                xs=12, sm=6, md=4, lg=3,  
                                style={'marginBottom': '0px', "width": "100%" },
                                ),
                                
                                # # Destination input card 
                                # dbc.Col(
                                #     dbc.Card(
                                #         dbc.CardBody([
                                #             dbc.Row([ 
                                #                     dcc.Markdown('Add Destination', style={'textAlign': 'center', 'color': 'white'}),
                                #                     html.Hr(),
                                #                     dcc.Input(
                                #                         id="id-destination",
                                #                         type="text",
                                #                         value=file_path_parent,
                                #                         style={'width': '100%' }
                                #                     )
                                #             ], style={'marginBottom': '10px'}),
                        
                                #             # html.Hr(style={
                                #             #     "borderTop": "1px solid #ccc",
                                #             #     "marginTop": "0.5rem",
                                #             #     "marginBottom": "1rem"
                                #             # }),  
                                #         ]),
                                #         style={"marginBottom": "20px", "boxShadow": "0 2px 6px rgba(0,0,0,0.15)"}
                                #     ) ,
                                # xs=12, sm=6, md=4, lg=3,  
                                # style={'marginBottom': '0px', "width": "100%" },
                                # ),
                                # Parameter cards
                                *self.graph
                            ],
        className="g-0"  ,
                            width=4,
                            style={"padding": "10px", "borderRight": "1px solid #ccc"}
                        ),

                        # Right column
                        dbc.Col(
                            [
                                dbc.Row([
                                    dbc.Card(
                                        dbc.CardBody([
                                            html.Div(
                                                dcc.Markdown(f'{general_nam_main}\nGeneration',
                                                            style={'textAlign': 'center', 'color': 'white'})
                                            )
                                        ]),
                                        style={"margin": "10px", "padding": "10px"}
                                    )
                                ], justify="start"),
                                
                                # Upload card
                                dbc.Row([
                                    dbc.Col(
                                        dbc.Card(
                                            dbc.CardBody([
                                                # dcc.Markdown('### Upload File', style={'textAlign': 'center', 'color': 'white'}),
                                                # html.Hr(),
                                                # dcc.Upload(
                                                #     id='upload-data',
                                                #     children=html.Div([
                                                #         'Drag and Drop or ',
                                                #         html.A('Select Files')
                                                #     ]),
                                                #     style={
                                                #         'width': '100%',
                                                #         'height': '60px',
                                                #         'lineHeight': '60px',
                                                #         'borderWidth': '1px',
                                                #         'borderStyle': 'dashed',
                                                #         'borderRadius': '5px',
                                                #         'textAlign': 'center',
                                                #         'margin': '10px',
                                                #         'color': 'white'
                                                #     },
                                                #     multiple=True
                                                # ),
                                                # html.Div(id='output-file-upload', style={'color': 'white', 'marginTop': 20}),
                                                html.Hr(style={'borderTop': '2px dashed white', 'margin': '20px 0'}),
                                                dbc.Card([
                                                    html.H3("Press Run to Start",
                                                            id=f"{self.pag}-run-status-{type}",
                                                            style={'color': 'lightgreen', 'textAlign': 'center'})
                                                ]),
                                            ]),
                                            style={
                                                "margin": "10px",
                                                "padding": "10px",
                                                "backgroundColor": "#2c2c2c",
                                                "width": "100%"
                                            }
                                        ),
                                        width='11'
                                    )
                                ], justify="center", align="center"),





                                # Parameters & Results cards
                                dbc.Row([
                                    dbc.Card(
                                        dbc.CardBody([
                                            html.Div(
                                                dcc.Markdown('### Parameters & Options',
                                                            style={'textAlign': 'center', 'color': 'white'}),
                                                id=f"toggle-text-starting-{type}",
                                                n_clicks=0,
                                                style={"cursor": "pointer"}
                                            ),
                                            html.Hr(),
                                            dbc.Collapse(
                                                dcc.Markdown(dnn_page()['starting']),
                                                id=f"{self.pag}-collapse-starting-{type}",
                                                is_open=False
                                            ),
                                        ]),
                                        style={"margin": "10px", "padding": "10px"}
                                    ),
                                    dbc.Card(
                                        dbc.CardBody([
                                            html.Div(
                                                dcc.Markdown('### Checking Results',
                                                            style={'textAlign': 'center', 'color': 'white'}),
                                                id=f"toggle-text-result-{type}",
                                                n_clicks=0,
                                                style={"cursor": "pointer"}
                                            ),
                                            html.Hr(),
                                            dbc.Collapse(
                                                html.Div([
                                                    dcc.Markdown(dnn_page()['restart']),
                                                    dcc.Markdown(dnn_page()['results']),
                                                ]),
                                                id=f"{self.pag}-collapse-result-{type}",
                                                is_open=False
                                            ),
                                        ]),
                                        style={"margin": "10px", "padding": "10px"}
                                    ),
                                ], justify="start"),

                                # GitHub link
                                dbc.Row([
                                    dbc.Col(
                                        dcc.Markdown(
                                            f"[View the full repository on GitHub]({github_link})",
                                            style={"textAlign": "center", "color": "white"}
                                        )
                                    )
                                ], justify="center")
                            ],
                            width=8,
                            style={"padding": "10px"}
                        )
                    ])
                ]),
                style={
                    "margin": "20px",
                    "padding": "20px",
                    "boxShadow": "0 4px 12px rgba(0,0,0,0.2)",
                    "backgroundColor": "#1e1e1e",
                    "maxWidth": "95%",   # scale control
                    "marginLeft": "auto",
                    "marginRight": "auto"
                }
            )
        ])





    def rebuild_param(self,*values):
        param=self.param 
        updated = dict(zip(self.Input_id, values[0])) 
        new_param={key:
                    {
                        'tf':{},
                        'param':{},
                        'option':[],
                    }
                for key in param['param_all']['param']
                } 
        print('[[[[[[[[[[[lang]]]]]]]]]]]',self.file_path_org)
        pag='lang'
        pag=os.path.basename(self.file_path_org)
        for gval in param['param_input']['param']:  
            new_param[gval]['tf'] = updated[f"param_{gval}_tf"] 
            for k in param[gval]['param_fix'].keys():
                new_param[gval]['param'][k] = param[gval]['param_fix'][k] 

            for k in param[gval]['param'].keys():
                val = updated[f"param_{gval}_{k}"]  
                orig_val = param[gval]['param'][k] 
                if isinstance(orig_val, bool):
                    new_param[gval]['param'][k] = bool(val)
                elif isinstance(orig_val, int):   
                    new_param[gval]['param'][k] = int(val)
                elif isinstance(orig_val, float): 
                    new_param[gval]['param'][k] = float(val)
                elif  isinstance(orig_val, list):
                    clean_val = str(val).strip().lstrip('[').rstrip(']')
                    list_items = [item.strip() for item in clean_val.split(',') if item.strip()]
                    converted_list = []                    
                    for item in list_items:
                        try: 
                            num = float(item)
                            converted_list.append(int(num) if num.is_integer() else num)
                        except (ValueError, TypeError): 
                            converted_list.append(item)                            
                    new_param[gval]['param'][k] = converted_list

                else:
                    try: 
                        new_param[gval]['param'][k] = float(val)
                    except (ValueError, TypeError):
                        try:
                            new_param[gval]['param'][k] = float(val)
                        except (ValueError, TypeError):
                            new_param[gval]['param'][k] = val  
        for gval in param['param_dropdowni']['param']: 
            new_param[gval]['param'] = updated[f"dropdown_{gval}_{pag}_param"] 
            # new_param[gval]['param'] = updated[f"dropdown_{gval}_param"] 

        for gval in param['param_dropdown']['param']:  
            new_param[gval]['param'] = param[gval]['param']

        for gval in param['param_list']['param']:  
            new_param[gval]['param'] = param[gval]['param']

        for gval in param['param_inputbv']['param']:  
            new_param[gval]['tf'] = param[gval]['tf']
            for k in param[gval]['param_fix'].keys():
                new_param[gval]['param'][k] = param[gval]['param_fix'][k] 

            for k in param[gval]['param'].keys():
                val = param[gval]['param'][k]  

                if isinstance(param[gval]['param'][k], bool):
                    new_param[gval]['param'][k] = bool(val)
                else:
                    try: 
                        new_param[gval]['param'][k] = float(val)
                    except (ValueError, TypeError):
                        try:
                            new_param[gval]['param'][k] = float(val)
                        except (ValueError, TypeError):
                            new_param[gval]['param'][k] = val  
                             
        return new_param



 

    def emerge_param(self, ):
        param=self.param  
        new_param={key:
                    {
                        'tf':{},
                        'param':{},
                        'option':[],
                    }
                for key in param['param_all']['param']
                } 
         

        for gval in param['param_input']['param']:  
            new_param[gval]['tf'] = param[gval]['tf']
            for k in param[gval]['param_fix'].keys():
                new_param[gval]['param'][k] = param[gval]['param_fix'][k] 

            for k in param[gval]['param'].keys():
                val = param[gval]['param'][k]  

                # if isinstance(param[gval]['param'][k], bool):
                #     new_param[gval]['param'][k] = bool(val)
                orig_val=param[gval]['param'][k] 
                if isinstance(orig_val, bool):
                    new_param[gval]['param'][k] = bool(val)
                elif  isinstance(orig_val, list): 
                    clean_val = str(val).strip().lstrip('[').rstrip(']')
                    list_items = [item.strip() for item in clean_val.split(',') if item.strip()]
                    converted_list = []                    
                    for item in list_items:
                        try: 
                            num = float(item)
                            converted_list.append(int(num) if num.is_integer() else num)
                        except (ValueError, TypeError): 
                            converted_list.append(item)                            
                    new_param[gval]['param'][k] = converted_list 
                else:
                    try: 
                        new_param[gval]['param'][k] = float(val)
                    except (ValueError, TypeError):
                        try:
                            new_param[gval]['param'][k] = float(val)
                        except (ValueError, TypeError):
                            new_param[gval]['param'][k] = val  

        for gval in param['param_inputbv']['param']:  
            new_param[gval]['tf'] = param[gval]['tf']
            for k in param[gval]['param_fix'].keys():
                new_param[gval]['param'][k] = param[gval]['param_fix'][k] 

            for k in param[gval]['param'].keys():
                val = param[gval]['param'][k]  

                # if isinstance(param[gval]['param'][k], bool):
                #     new_param[gval]['param'][k] = bool(val)
                orig_val=param[gval]['param'][k] 
                if isinstance(orig_val, bool):
                    new_param[gval]['param'][k] = bool(val)
                elif  isinstance(orig_val, list): 
                    clean_val = str(val).strip().lstrip('[').rstrip(']')
                    list_items = [item.strip() for item in clean_val.split(',') if item.strip()]
                    converted_list = []                    
                    for item in list_items:
                        try: 
                            num = float(item)
                            converted_list.append(int(num) if num.is_integer() else num)
                        except (ValueError, TypeError): 
                            converted_list.append(item)                            
                    new_param[gval]['param'][k] = converted_list 
                else:
                    try: 
                        new_param[gval]['param'][k] = float(val)
                    except (ValueError, TypeError):
                        try:
                            new_param[gval]['param'][k] = float(val)
                        except (ValueError, TypeError):
                            new_param[gval]['param'][k] = val  

        for gval in param['param_dropdowni']['param']: 
            new_param[gval]['param'] = param[gval]['param'] 

        for gval in param['param_dropdown']['param']:  
            new_param[gval]['param'] = param[gval]['param']

        for gval in param['param_list']['param']:  
            new_param[gval]['param'] = param[gval]['param']
 
  
        return new_param


import copy



class algorithm:
    def __init__(self, 
                param,  
                true_name='true_0',
                pre_portion='smod', 
                path_heads_true=['true',]
                ): 
        
        # print('[[[[[[[[[[[parammmm--------------------------------]]]]]]]]]]]',param)
        param['Resizing']['param']['target_number_of_triangles_faction']=None  #150000
        for val in ['get_pinn_features','get_skeleton','get_wrap','get_smooth']:
            mmn=param['model_pred']['param']
            # print('[[[[[iiii]]]]]',mmn)
            param['model_pred']['param']['param_dic']['tf_restart'][val]=False   #True   # 
        param['model_pred']['param']['param_dic']['tf_restart']['get_hmod_pred']=False   #True   #  
        param['Smooth']['param']['n_step']=20


        # param['Smooth']['param']['dt']=1e-7
        # param['Smooth']['param']['method']='taubin' #'willmore'   #
        # param['Skeleton']['param']['path']="entry"     #'old'   #  
        # param['Skeleton']['param']['dict_mesh_to_skeleton_finder']['interval_target_number_of_triangles']=[200000]

        param['clean_path_dir']['tf'] = False          # True/False: delete computed data
        param['intensity_rhs']['tf']=False #  True  #  
        param['Resizing']['tf']= False # True  #  
        param['Smooth']['tf'] =False   # True   # False   # True/False: choose whether to smooth the data
        param['Skeleton']['tf']=False # True  #  
        param['rhs']['tf']=False #  True  #  
        param['model_shap']['tf']=False # True  #  

        param['Morphologic Param']['tf'] = False   # True   #  False   # True/False: choose whether to perform head/neck segmentation
        param['iou']['tf'] = False   # True       #False   #  True/False: choose whether to compute IoU
        param['roc']['tf'] = False # True       # False   # True/False: choose whether to compute IoU
        param['graph_center']['tf'] =False  # True   # True   #False   #   True/False: compute central axis
        param['cylinder_heatmap']['tf'] =False   #True   #  False   # False   # True/False: generate cylindrical heatmap

        param['annotations']['tf'] =False   #True   #  True/False: choose whether to generate annotations for training, accuracy, and recall





        self.path_heads=param['path_heads_show']['param'] 
        self.path_dirs_show=param['path_dirs_show']['param'] 
        self.dnn_modes=param['dnn_modes']['param']
        self.path_list=param['path_list']['param'] 
        self.param=param  
        self.true_keys=[true_name]
        self.pre_portion=pre_portion
        pre_portion=self.pre_portion
        path_list=self.path_list 
        self.path_heads_true=path_heads_true
        self.head_navbar=param['head_navbar']['param'] 
        self.file_path_data=self.param['model_pred']['param']['file_path_data'] 
        self.neld_names_all=self.param['model_pred']['param']['neld_names_all']











    def pre_test_train( self, 
            path_heads_show,
            model_sufix_show, 
            path_dirs_show,
            neld_data=None,   
            train_test='test',
            true_name='true_0', 
            dnn_mode='mode0', 
            model_type='pinn' , 
            path_dir='save',
            path_display=None, 
            size_threshold=None,
            path_display_dic=None, 
            path_hmod_dir=None,
            model_type_data=None,
            data_dir=None,
            head_navbar=None, 
            pre_portion=None,
            configs=None,
            **kward,

            ): 
        pre_portion=self.pre_portion if pre_portion is None else pre_portion
        path_list=self.path_list
        model_type_data=model_type_data if model_type_data is not None else model_type
        path_heads_show_data=[model_type_data]
        modes={
                val:{  
                        name:{
                                dirs:{} 
                                for dirs in path_dirs_show
                            } 
                            for name in model_sufix_show+['mode0']+["DNN-4"] 
                        }  
                    for val in path_heads_show
                }
        # configs=get_configs() 
        configss=copy.deepcopy(configs)
        dmode = get_data_mode(pre_portion=pre_portion,path_list=path_list) 
        for val in path_heads_show:   
                ids,name,dirs=0,'mode0',path_dirs_show[0]
                data_dir_tmp=data_dir if data_dir is not None else dirs
                data_mode=dmode.test_pre(pre_portion=pre_portion,
                                            data_head=model_type_data,
                                            dest_head=val,
                                            seg_neld=dirs,
                                            data_dir=data_dir_tmp,
                                            dest_dir=dirs,
                                            )
                modes[val][name][dirs]=[dmode.mode_id] 
                for dirs in path_dirs_show:
                    data_dir_tmp=data_dir if data_dir is not None else dirs
                    for ids, name in enumerate(model_sufix_show):  
                        cfg=configss[name]
                        cfg["data_sufix"]=cfg["dest_sufix"]=None
                        data_mode = dmode.test_opt( 
                                                pre_portion=pre_portion, 
                                                train_test=train_test,
                                                data_head=model_type_data,
                                                dest_head=val,
                                                seg_neld=dirs,
                                                data_dir=data_dir_tmp,
                                                dest_dir=dirs,
                                                **cfg
                                            ) 
                        modes[val][name][dirs]=[dmode.mode_id]  
        model_sufix_dic={mm:{} for mm in ['model_sufix_dic','path','path_heads_show','path_heads_dic','model_sufix_dic_inverse','model_sufix_show','path_dirs_show','drop_dic','path_heads_dic_sec']}
        # model_sufix_dic['model_sufix_show']=[data_mode[modes[model_type][dnn_mode][path_dir][0]]['model_sufix'][0] for dnn_mode in  model_sufix_show]
        model_sufix_dic['model_sufix_show']=[dnn_mode for dnn_mode in  model_sufix_show]

        model_sufix_dic['model_sufix_dic'] ={dnn_mode :dnn_mode for dnn_mode in  model_sufix_show}
        # model_sufix_dic['model_sufix_dic'] ={data_mode[modes[model_type][dnn_mode][path_dir][0]]['model_sufix'][0] :dnn_mode for dnn_mode in  model_sufix_show}
        model_sufix_dic['model_sufix_dic']['save']='save'
        model_sufix_dic['model_sufix_dic_inverse'] = {v: k for k, v in model_sufix_dic['model_sufix_dic'].items()}  
 
        # configs=get_configs()
        
        fcgs=copy.deepcopy(configs)
        '''
        fcgs={}
        for ids, name in enumerate(model_sufix_show):
            # print(f"Finished {name}, dnn_modes={model_sufix_show}")
            fcgs[name]=configss[name]
            # print('[[[]]]',name,cfg,path_dirs_show)
            # cfg["data_sufix"],cfg["dest_sufix"]=model_sufix_dic['model_sufix_dic_inverse'][cfg["data_sufix"]],model_sufix_dic['model_sufix_dic_inverse'][cfg["dest_sufix"]]
            # cfg["data_dir"]=path_dir #if path_hmod_dir is None else path_hmod_dir.get('data_dir',path_dir)
            # cfg["dest_dir"]=path_dir #if path_hmod_dir is None else path_hmod_dir.get('data_dir',path_dir)
            # fcgs[name]=cfg
 '''

        dmode = get_data_mode(pre_portion=pre_portion,path_list=path_list) 
        for val in path_heads_show:
            # for valdata in path_heads_show_data: 
                ids,name,dirs=0,'mode0',path_dirs_show[0]
                # data_dir_tmp=data_dir if data_dir is not None else dirs
                data_mode=dmode.test_pre(pre_portion=pre_portion,
                                            data_head=model_type_data,
                                            dest_head=val,
                                            seg_neld=dirs, 
                                            data_dir=data_dir_tmp,)
                modes[val][name][dirs]=[dmode.mode_id] 
                # print(f"Finished {name}, mode_id={dmode.mode_id}") 

                for dirs in path_dirs_show: 
                    for ids, name in enumerate(model_sufix_show):
                        # fcgs[name]['data_dir']=data_dir
                        fcgs[name]['data_dir']=data_dir if data_dir is not None else dirs 
                        fcgs[name]['dest_dir']= dirs 
                        # cfg=configs[name]
                        data_mode = dmode.test_opt(
                                                data_mode=data_mode,
                                                pre_portion=pre_portion,
                                                train_test='test',
                                                data_head=model_type_data,
                                                dest_head=val, 
                                                seg_neld=dirs, 
                                                **fcgs[name]
                                            )
                        # print('[[[[[]]]]]]]]]]]]]',data_mode)
                        modes[val][name][dirs]=[dmode.mode_id]  
        self.modes,self.data_mode,self.dmode=modes,data_mode,dmode 
        # print('[[[[[[[[[[[[[[kk]]]]]]]]]]]]]]',model_sufix_show)
        path_heads=self.path_heads 
        param=self.param  
        path_display = path_display if path_display is not None else param['model_pred']['param']['path_display']
 

        # model_sufix_dic={data_mode[modes[model_type][dnn_mode]]['model_sufix'][0]:dnn_mode for dnn_mode in  model_sufix_show}
        model_sufix_dic['path_dirs_dic']={ke:va for ke,va in zip(path_dirs_show,path_dirs_show)}  
        model_sufix_dic['path_dir']=path_dir 
        result =[part if len(part.split('_'))<=1 else '_'.join(part.split('_')[1:]) for part in path_heads]
        path_headss=path_heads+['save']
        result=result+['default'] 
        model_sufix_dic['path_heads_show']=path_heads_show  
        model_sufix_dic['path_dirs_show']=path_dirs_show
        model_sufix_dic['path_heads_dic_sec']=param['path_heads_show_nami']['param']
        model_sufix_dic['path_heads_dic']={ke:va for ke,va in zip(path_headss,result)}  
        head_navbar=head_navbar if head_navbar is not None else get_navs_bar(path_heads_show) 
        print('[[[[[[[[[[[head=]]]]]]]]]]]',head_navbar,path_heads_show)
        model_sufix_dic['head_navbar']=head_navbar[0]
        model_sufix_dic['categories']=head_navbar[1]
        model_sufix_dic['path_display']=path_display 
        self.model_sufix_dic=model_sufix_dic
        self.obj_org_path_dict = neld_data['obj_org_path_dict']
        self.obj_org_path=neld_data['obj_org_path']



 

 





    def test( self,   
            neld_data=None,  
            true_name='true_0', 
            dnn_mode='mode0', 
            model_type='pinn' , 
            model_type_data=None,
            path_dir='save',
            path_display=None, 
            size_threshold=None,
            path_display_dic=None,
            path_hmod_dir=None, 
            path_heads_show=None, 
            model_sufix_show=None, 
            path_dirs_show=None,
            train_smods=False,
            data_dir=None,
            drop_dic=None,
            n_step = None,
            weight=None,
            weight2=None,
            dnn_modes=None,
            path_dirs=None, 
            param_dic=None,
            kmean_n_run=None,
            path_list=None,
            path_heads=None,
            nam=None,
            path_dirs_weig=None,
            head_navbar=None,
			wrap_method=None,
            pre_portion=None,
            configs=None,
            path_dict=None,
            file_path_org=None,
            file_path_data=None,
            neld_names_all=None,
            **kwargs,
            ): 
        self.param['model_pred']['param']['file_path_org']=self.param['model_pred']['param']['file_path_org'] or file_path_org
        self.param['model_pred']['param']['file_path_data']=self.param['model_pred']['param']['file_path_data'] or file_path_data
        self.param['model_pred']['param']['neld_names_all']=self.param['model_pred']['param']['neld_names_all'] \
                            if self.param['model_pred']['param']['neld_names_all'] is not None else neld_names_all
        param=self.param
        path_heads_show=path_heads_show if path_heads_show is not None else self.path_heads
        model_sufix_show=model_sufix_show if model_sufix_show is not None else self.dnn_modes 
        path_dirs_show=path_dirs_show if path_dirs_show is not None else self.path_dirs_show
        model_type_data=model_type_data if model_type_data is not None else model_type
        head_navbar=head_navbar if head_navbar is not None else self.head_navbar
        pre_portion=self.pre_portion if pre_portion is None else pre_portion 
        self.data(   
                exit_name=None,
                entry_name=None, 
                train_test='test',
                neld_data=neld_data,  
                true_name=true_name, 
                dnn_mode=dnn_mode, 
                model_type=model_type, 
                model_type_data=model_type_data, 
                path_dir=path_dir,
                path_display=path_display, 
                size_threshold=size_threshold,
                path_display_dic=path_display_dic, 
                path_hmod_dir=path_hmod_dir, 
                path_heads_show=path_heads_show,
                model_sufix_show=model_sufix_show, 
                path_dirs_show=path_dirs_show, 
                data_dir=data_dir,
                drop_dic=drop_dic, 
                head_navbar=head_navbar,
                pre_portion=pre_portion,
                configs=configs, 
                ) 
        neld_data_tmp=self.neld_data_tmp
        path_list=self.path_list
        data_mode=self.data_mode
        modes,data_mode,dmode=self.modes,self.data_mode,self.dmode
        self.model_sufix_dic['drop_dic']=drop_dic
        model_sufix_dic=self.model_sufix_dic
        mode_ids=modes[model_type][dnn_mode][path_dir] 
        # print('[[[[[[[dnn_mode     ]]]]]]]',dnn_mode,data_mode)
        true_keys=self.true_keys 
        path_heads=self.path_heads  
        obj_org_path_dict =self.obj_org_path_dict  
    
        time_start = time.time()
 
 
        for ixi,mode_id in enumerate(mode_ids):#data_mode.keys(): # 
            model_sufix=data_mode[mode_id]['model_sufix'][0] 


            mode_dnn1 =data_mode[mode_id]['model_init'] 
            mode_dnn11=model_sufix_dic['model_sufix_dic_inverse'].get(mode_dnn1,None)
            path_hmod_dir=None if mode_dnn11 is None else f'{model_type}_{mode_dnn11}_{path_dir}'


            neld_cla=model_pred(   
                                neld_data=neld_data_tmp, 
                                model_sufix=model_sufix,
                                data_mode=data_mode[mode_id],   
                                pinn_dir_data_all=dmode.pinn_dir_data_all,
                                model_sufix_all=dmode.model_sufix_all, 
                                path_heads= path_heads,
                                model_type=model_type, 
                                obj_org_path_dict=obj_org_path_dict ,
                                true_keys=true_keys,
                                model_sufix_dic=model_sufix_dic,
                                path_display_dic=path_display_dic,
                                # path_display=path_display,
                                **param['model_pred']['param']
                                ) 
 
            neld_cla.get_model_opt_name(model_sufix=model_sufix,
                                        model_type=model_type)  
            
 
            if param['clean_path_dir']['tf']:
                path_head_clean=param['clean_path_dir']['param']['path_clean'] 
                print('[[[[[[path_head_clean]]]]]]',path_head_clean) 
    
            size_threshold=param['model_pred']['param']['size_threshold'] 

            if param['rhs']['tf']: 
                neld_cla.get_rhs(
                                        param_neld_name=param['model_pred']['param']['param_dic']['data']['get_neld_name'],
                                        **param['Skeleton']['param'])
         

            if param['model-pred']['tf']:   
                # if (dnn_mode== 'DNN-2'):
                #         neld_cla.get_skl_hmod_pred(
                #                                 # weights=weights,
                #                                 path_hmod_dir=path_hmod_dir,
                #                                 train_smods=train_smods,
                #                                 model_type=model_type,
                #                             **param['model-pred']['param']
                #                             ) 
                if model_type.startswith(('vol','cnn')):
                    neld_cla.get_hmod_pred_cnn(
                                            # weights=weights,
                                            train_smods=train_smods,
                                            model_type=model_type,
                                        **param['model-pred']['param']
                                        )   
                elif model_type.startswith(('gcn',)): 
                    neld_cla.get_hmod_pred_gcn(
                                            # weights=weights, 
                                            train_smods=train_smods,
                                            model_type=model_type,
                                        **param['model-pred']['param']
                                        )   
                elif model_type.startswith(('rnn','tfm')): 
                    neld_cla.get_hmod_pred_kal(
                                            # weights=weights,
                                            path_dict=path_dict,
                                            train_smods=train_smods,
                                            model_type=model_type,
                                        **param['model-pred']['param']
                                        )   
                elif model_type.startswith(('ekf','jos')): 
                    neld_cla.get_hmod_pred_linear(
                                            # weights=weights,
                                            path_dict=path_dict,
                                            train_smods=train_smods,
                                            model_type=model_type,
                                        **param['model-pred']['param']
                                        )   
                elif model_type.startswith(('pnet',)): 
                    neld_cla.get_hmod_pred_pnet(
                                            # weights=weights,
                                            train_smods=train_smods,
                                            model_type=model_type,
                                        **param['model-pred']['param']
                                        )   
                elif model_type.startswith(('cml','cML', 'ML')): 
                    neld_cla.get_hmod_pred_cml(
                                            # weights=weights,
                                            train_smods=train_smods,
                                            model_type=model_type,
                                        **param['model-pred']['param']
                                        )   
                elif model_type.startswith(('pinn',)): 
                    neld_cla.get_hmod_pred_PINN(
                                            # weights=weights,
                                            train_smods=train_smods,
                                            model_type=model_type,
                                        **param['model-pred']['param']
                                        )   
                else:
                    neld_cla.get_hmod_pred(
                                            # weights=weights,
                                            train_smods=train_smods,
                                            model_type=model_type,
                                            **param['model-pred']['param']
                                        )    
                # neld_cla.get_hmod_process(
                #                         # weights=weights,
                #                         train_smods=train_smods,
                #                         model_type=model_type,
                #                         **param['model-pred']['param']
                #                     ) 

            if param['skl_hmod_pred']['tf']: 
                neld_cla.get_skl_hmod_pred(
                                        # weights=weights,
                                        path_hmod_dir=path_hmod_dir,
                                        train_smods=train_smods,
                                        model_type=model_type,
                                        **param['model-pred']['param']
                                    ) 
            
            if param['model_shap']['tf']:
                neld_cla.model_shap( 
                                        train_smods=train_smods,
                                        model_type=model_type,
                                    **param['model-pred']['param'],
                                    # **param['model_shap']['param']
                                    )  
     
         
            if param['Morphologic Param']['tf']: 
                for pdisplay in path_display:
                    neld_cla.get_head_neck_segss(  
                                        head_neck_path=pdisplay,
                                        **param['Morphologic Param']['param']
                                )  

            if param['iou']['tf']:
                neld_cla.get_iou()  

            if param['roc']['tf']:
                neld_cla.get_roc()  




            if param['graph_center']['tf']:
                neld_cla.get_graph_center() 

            if param['cylinder_heatmap']['tf']:
                # True
                neld_cla.get_cylinder_heatmap()

            if param['dash_pages']['tf']:
                neld_cla.get_dash_pages()





            mytime0 = time.time() - time_start 
            hours, rem = divmod(mytime0, 3600)
            minutes, seconds = divmod(rem, 60) 
            print(f'Prediction completed in {int(hours)}h {int(minutes)}m {seconds:.2f}s')  

        return neld_cla
                
 




    def train( self,  
            nam=None,
            neld_data=None,  
            true_name='true_0', 
            path_dir='save',
            dnn_mode='mode0',
            model_type='pinn' , 
            entry_names=[],
            path_display=None,  
            path_list=None,
            path_display_dic=None,
            path_hmod_dir=None, 
            path_heads_show=None,
            model_sufix_show=None, 
            path_dirs_show=None,
            size_threshold=None,
            model_type_data=None, 
            data_dir=None,
            head_navbar=None,
            pre_portion=None,
            configs=None,
            path_dict=None,
            file_path_org=None,
            file_path_data=None,
            neld_names_all=None,
            **kward,
            ): 
        pre_portion=self.pre_portion if pre_portion is None else pre_portion
        self.param['model_pred']['param']['file_path_org']=self.param['model_pred']['param']['file_path_org'] or file_path_org
        self.param['model_pred']['param']['file_path_data']=self.param['model_pred']['param']['file_path_data'] or file_path_data
        self.param['model_pred']['param']['neld_names_all']=self.param['model_pred']['param']['neld_names_all'] \
                            if self.param['model_pred']['param']['neld_names_all'] is not None else neld_names_all
        param=self.param
        path_heads_show=path_heads_show if path_heads_show is not None else self.path_heads_true
        model_sufix_show=model_sufix_show if model_sufix_show is not None else self.dnn_modes 
        path_dirs_show=path_dirs_show if path_dirs_show is not None else self.path_dirs_show
        model_type_data=model_type_data if model_type_data is not None else model_type
        head_navbar=head_navbar if head_navbar is not None else self.head_navbar
        # model_type='true'
        self.pre_test_train(   
            train_test='train',
            neld_data=neld_data,  
            true_name=true_name, 
            path_heads_show=path_heads_show,
            model_sufix_show=model_sufix_show, 
            path_dirs_show=path_dirs_show,
            dnn_mode=dnn_mode, 
            model_type=model_type, 
            model_type_data=model_type_data, 
            path_dir=path_dir,
            path_display=path_display, 
            size_threshold=size_threshold,
            path_display_dic=path_display_dic, 
            path_hmod_dir=path_hmod_dir, 
            data_dir=data_dir,
            head_navbar=head_navbar,
            pre_portion=pre_portion,
            configs=configs,
            )
        path_list=self.path_list
        param=self.param
        data_mode=self.data_mode
        modes,data_mode,dmode=self.modes,self.data_mode,self.dmode
        model_sufix_dic=self.model_sufix_dic
        mode_ids=modes[model_type][dnn_mode][path_dir]
        true_keys=self.true_keys 
        path_heads=self.path_heads  
        obj_org_path_dict =self.obj_org_path_dict

 
        time_start = time.time()
  
        for ixi,mode_id in enumerate(mode_ids):#data_mode.keys(): #
            nhh=data_mode[mode_id]['model_sufix'] 
            model_sufix=data_mode[mode_id]['model_sufix'][0] 
            neld_cla=model_pred(   
                                neld_data=neld_data, 
                                model_sufix=model_sufix,
                                data_mode=data_mode[mode_id],   
                                pinn_dir_data_all=dmode.pinn_dir_data_all,
                                model_sufix_all=dmode.model_sufix_all, 
                                path_heads= path_heads,
                                model_type=model_type,
                                obj_org_path_dict=obj_org_path_dict ,
                                true_keys=true_keys,
                                model_sufix_dic=model_sufix_dic,
                                **param['model_pred']['param']
                                ) 
 
            neld_cla.get_model_opt_name(model_sufix=model_sufix,
                                        model_type=model_type)   

   
            size_threshold=param['model_pred']['param']['size_threshold'] 
            train_smods=True #data_mode[mode_id]['train_smods']
            if param['get_training']['tf']:
                if model_type.startswith(('cml','ML','cML')): 
                    neld_cla.train_model_smod_cml(     
                                            model_sufix=model_sufix, 
                                            train_smods=train_smods, 
                                            model_type=model_type,  
                                            num_sub_nodes=20000,
                                            entry_names=entry_names,
                                            **param['get_training']['param']
                                            )
                elif model_type.startswith(('gcn',)):
                    neld_cla.train_model_smod_gcn(
                                            train_smods=train_smods, 
                                            model_type=model_type, 
                                            model_sufix=model_sufix, 
                                            entry_names=entry_names,
                                            **param['get_training']['param'],
                                            )  
                elif model_type.startswith(('rnn','tfm')):
                    neld_cla.train_model_smod_kal(
                                            path_dict=path_dict,
                                            train_smods=train_smods, 
                                            model_type=model_type, 
                                            model_sufix=model_sufix, 
                                            entry_names=entry_names,
                                            **param['get_training']['param'],
                                            ) 
                elif model_type.startswith(('pinn',)):
                    neld_cla.train_model_smod_PINN(
                                            train_smods=train_smods, 
                                            model_type=model_type, 
                                            model_sufix=model_sufix, 
                                            entry_names=entry_names,
                                            **param['get_training']['param'],
                                            )  
                elif model_type.startswith(('pnet',)): 
                    neld_cla.train_model_smod_pnet(
                                            train_smods=train_smods, 
                                            model_type=model_type, 
                                            model_sufix=model_sufix, 
                                            entry_names=entry_names,
                                            **param['get_training']['param'],
                                            )  
                elif model_type.startswith(('vol',"fastfcn3d","unet3d",'cnn')):
                    neld_cla.train_model_smod_cnn(
                                            train_smods=train_smods, 
                                            model_type=model_type, 
                                            model_sufix=model_sufix, 
                                            entry_names=entry_names,
                                            **param['get_training']['param'],
                                            )  
                else:
                    neld_cla.train_model_smod(
                                            train_smods=train_smods, 
                                            model_type=model_type, 
                                            model_sufix=model_sufix, 
                                            entry_names=entry_names,
                                            **param['get_training']['param'],
                                            ) 

            mytime0 = time.time() - time_start 
            hours, rem = divmod(mytime0, 3600)
            minutes, seconds = divmod(rem, 60) 
            print(f'Training completed in {int(hours)}h {int(minutes)}m {seconds:.2f}s')  

        return [f'j ']
                





    def data( self,  
            neld_data=None,  
            true_name='true_0', 
            path_dir='save',
            dnn_mode='mode0',
            model_type='pinn' , 
            entry_names=[],
            path_display=None,  
            path_list=None,
            path_display_dic=None,
            path_hmod_dir=None, 
            path_heads_show=None,
            model_sufix_show=None, 
            path_dirs_show=None,
            size_threshold=None,
            exit_name=None,
            entry_name=None,
            train_test='test',
            model_type_data=None,
            data_dir=None,
            drop_dic=None,
            head_navbar=None,
            pre_portion=None,
            configs=None,
            file_path_data=None,
            ): 
        pre_portion=self.pre_portion if pre_portion is None else pre_portion
        param=self.param
        path_heads_show=path_heads_show if path_heads_show is not None else self.path_heads_true
        model_sufix_show=model_sufix_show if model_sufix_show is not None else self.dnn_modes 
        path_dirs_show=path_dirs_show if path_dirs_show is not None else self.path_dirs_show
        file_path_data=file_path_data if file_path_data is not None else self.file_path_data
        # model_type='true'
        self.pre_test_train(   
            train_test=train_test,
            neld_data=neld_data,  
            true_name=true_name, 
            path_heads_show=path_heads_show,
            model_sufix_show=model_sufix_show, 
            path_dirs_show=path_dirs_show,
            dnn_mode=dnn_mode, 
            model_type=model_type, 
            path_dir=path_dir,
            path_display=path_display, 
            size_threshold=size_threshold,
            path_display_dic=path_display_dic, 
            path_hmod_dir=path_hmod_dir,  
            data_dir=data_dir,
            head_navbar=head_navbar,
            pre_portion=pre_portion,
            configs=configs,
            file_path_data=file_path_data,
            ) 
        data_mode=self.data_mode
        modes,data_mode,dmode=self.modes,self.data_mode,self.dmode
        self.model_sufix_dic['drop_dic']=drop_dic
        model_sufix_dic=self.model_sufix_dic
        mnm=modes.keys()#[model_type]
        print('[[[[[[[[[[[[[[-----]]]]]]]]]]]]]]',dnn_mode,mnm)
        mode_ids=modes[model_type][dnn_mode][path_dir] 
        true_keys=self.true_keys 
        path_heads=self.path_heads  
        obj_org_path_dict =self.obj_org_path_dict
        from copy import deepcopy
 
        for ixi,mode_id in enumerate(mode_ids):#data_mode.keys(): # 
            model_sufix=data_mode[mode_id]['model_sufix'][0] 
            neld_cla=model_pred(   
                                neld_data=neld_data, 
                                model_sufix=model_sufix,
                                data_mode=data_mode[mode_id],   
                                pinn_dir_data_all=dmode.pinn_dir_data_all,
                                model_sufix_all=dmode.model_sufix_all, 
                                path_heads= path_heads,
                                model_type=model_type,
                                obj_org_path_dict=obj_org_path_dict ,
                                true_keys=true_keys,
                                model_sufix_dic=model_sufix_dic,
                                **param['model_pred']['param']
                                ) 
 
            neld_cla.get_model_opt_name(model_sufix=model_sufix,
                                        model_type=model_type)   


            time_start = time.time()
            # if param['rhs']['tf']: 
            #     neld_cla.get_rhs( 
            #                     entry_name=exit_name,
            #                     exit_name =exit_name, 
            #                     **param['Skeleton']['param'],
            #         )
            """
            if param['Resizing']['tf']: 
                parr=dict(
                        get_data=True,
                        n_error = 1,
                        n_step = 0,
                        dt=1e-6,
                        disp_time=500, )
            else:
                parr=param['Smooth']['param']
            save_entry_exit={nn:[] for nn in ['entry','exit','old_path']}
            save_entry_exit['entry'].append(entry_name)
            save_entry_exit['exit'].append(exit_name)
            save_entry_exit['old_path'].append(entry_name)
            action=None
            if param['Smooth']['tf']:
                exit_name='smooth'  if entry_name is None else f'{entry_name}_smooth'
                neld_cla.get_smooth( 
                                entry_name=entry_name,
                                exit_name=exit_name,
                                # dt=1e-6, 
                                # n_step = 100,
                                **param['Smooth']['param'] 
                                )
                entry_name=exit_name
                save_entry_exit['entry'].append(entry_name)
                save_entry_exit['exit'].append(exit_name)
                save_entry_exit['old_path'].append(entry_name,)
                action='smooth'
            resize =None
            if param['Resizing']['tf']: 
                sze=param['Resizing']['param']['target_number_of_triangles_faction']
                exit_name=f'resize_{sze}' if entry_name is None else f'{entry_name}_resize_{sze}'
                neld_cla.get_resize(entry_names=[entry_name],
                                    exit_names=[exit_name],
                                    target_number_of_triangles_faction=sze,  
                                    )
                entry_name=exit_name
                save_entry_exit['entry'].append(entry_name)
                save_entry_exit['exit'].append(exit_name)
                save_entry_exit['old_path'].append(entry_name)
                resize=f'resize_{sze}'
            for nn in ['entry','exit']:
                if len(save_entry_exit[nn])==0:
                    save_entry_exit[nn].append(None) 
                    param['model_pred']['param']['param_dic']['data']['get_neld_name']['old_path']='current'
                save_entry_exit['old_path'].append('entry')

 
            if param['Resizing']['tf']: 
                # param['model_pred']['param']['param_dic']['data']['get_neld_name']['dict_neld_path']=dict(dict_neld_path='old',
                #                                 drop_dic_name=save_entry_exit['exit'][-1])
                param['model_pred']['param']['param_dic']['data']['get_neld_name']['dict_neld_path']= 'old'
                param['model_pred']['param']['param_dic']['data']['get_neld_name']['drop_dic_name']= save_entry_exit['exit'][-1]
                param['model_pred']['param']['param_dic']['data']['get_neld_name']['old_path']='current'
            # print('[[[[[[[[[[[[[[[[[[]]]]]]]]]]]]]]]]]]',self.obj_org_path)
            # nam_gen,nam_loc=self.obj_org_path.split('/')[-2:] 
            nam_gen = os.path.basename(os.path.dirname(self.obj_org_path))
            nam_loc = os.path.basename(self.obj_org_path)
            nam = "_".join(x for x in [nam_loc, action, resize] if x is not None)
            drop_dic_name='_'.join(x for x in [nam_loc,action] if x is not None) 
            if resize is not None:
                param['model_pred']['param']['param_dic']['data']['get_neld_name']=dict(dict_neld_path='old',
                                                                    old_path=None,
                                                                    drop_dic_name=drop_dic_name,
                                                                    nam_gen=nam_gen,
                                                                    )  
"""
            '''
            mmnn=param['Skeleton']['param']['path']
            pathhh=f'{mmnn}'
            for entry_name,exit_name,old_path in zip(save_entry_exit['entry'],save_entry_exit['exit'],save_entry_exit['old_path']):
                for paath in ['entry','old']:
                    param['Skeleton']['param']['path']=paath
                    # if entry_name == 'smooth':'old',
                    if param['Skeleton']['tf']: 
                        # print('[[[[[[[--------------skl------------]]]]]]]',paath,entry_name,exit_name )
                        # print('[[[[[[[--------------skl------------]]]]]]]',param['model_pred']['param']['param_dic']['data']['get_neld_name'] )
                        param_neld_name=deepcopy(param['model_pred']['param']['param_dic']['data']['get_neld_name'])
                        # param_neld_name['drop_dic_name']=None
                        # param_neld_name['nam_gen']=None
                        # print('[[[[[[[--------------skl------------]]]]]]]',param_neld_name )

                        dict_wrap=param['Skeleton']['param']['dict_wrap']
                        neld_cla.get_wrap(wrap_part='hmod', 
                                        alpha_fraction=1,
                                        offset_fraction=.7,
                                        entry_name=entry_name,
                                        exit_name=exit_name,
                                        dict_wrap=dict_wrap,
                                        old_path=old_path,
                                        param_neld_name=param_neld_name,
                                        ) 
                        param['Skeleton']['param']['wrap_part']=None
                        param['Skeleton']['param']['tf_restart']=True
                        neld_cla.get_skeleton(
                                        entry_name=entry_name,
                                        exit_name =exit_name,
                                        old_path=old_path,
                                        # wrap_part='hmod_wrap' , 
                                        **param['Skeleton']['param'], 
                        )
                        param['Skeleton']['param']['wrap_part']='hmod_wrap'
                        param['Skeleton']['param']['tf_restart']=True
                        neld_cla.get_skeleton(
                                        entry_name=entry_name,
                                        exit_name =exit_name,
                                        old_path=old_path,
                                        # wrap_part='hmod_wrap' , 
                                        **param['Skeleton']['param'], 
                        )
                    if param['rhs']['tf']: 
                        neld_cla.get_rhs( 
                                        entry_name=exit_name,
                                        exit_name =exit_name,
                                        old_path=old_path,
                                        param_neld_name=param_neld_name,
                                        **param['Skeleton']['param'],
                            )
            param['Skeleton']['param']['path']=pathhh
'''
 
                
            neld_data_tmp=deepcopy(neld_data)     
            mmm=neld_data['obj_org_path']
            neld_data_tmp['obj_org_path']=neld_data['obj_org_path'] if exit_name is None else f'{mmm}_{exit_name}'  
            neld_data_tmp['neld_path_inits']=neld_data['neld_path_inits'] if exit_name is None else [f'{nam}_{exit_name}'  for nam in neld_data['neld_path_inits']] 


            mytime0 = time.time() - time_start 
            hours, rem = divmod(mytime0, 3600)
            minutes, seconds = divmod(rem, 60) 
            print(f'Training completed in {int(hours)}h {int(minutes)}m {seconds:.2f}s')  
        self.neld_data_tmp=neld_data_tmp
    
  