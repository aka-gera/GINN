

import sys
import os
import numpy as np

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import numpy as np
import time 
import tensorflow as tf  
tf.config.run_functions_eagerly(True)  
DTYPE='float32' 
import pickle 
import pandas as pd 

import shap   
DTYPE='float32' 
DTYPE = tf.float32

from pathlib import Path
import sys
pa_dir = Path(__file__).resolve().parent.parent  
sys.path.insert(0, str(pa_dir))

  
from neld_fun_0.get_path import assign_if_none,get_name,get_param,get_files,remove_directory

device = "/GPU:0" if tf.config.list_physical_devices('GPU') else "/CPU:0"  
from tqdm import tqdm 
 
# import torch
import mod.dsa.neld_fun_0.help_neld_train_test as hntt
from mod.dsa.dend_fun_0.help_dendrite_manipulation import dendrite_manipulate
from neld_fun_0.help_neld_train_test_2 import pinn_data_adj
import random
np.random.seed(42)
tf.random.set_seed(42)
random.seed(42)

#get_files,get_name,

class model_pred( hntt.train_test_tf,dendrite_manipulate,pinn_data_adj):
    def __init__(self, file_path_org,
                        data_studied, 
                        model_sufix,
                        neld_data, 
                        file_path_data=None,
                        neld_names=None,
                        neld_namess=None,   
                        neld_path_inits=None,
                        name_spine_id=None,
                        name_head_id=None,
                        name_neck_id=None,
                        name_shaft_id=None,
                        path_train=None,
                        txt_true_file=None,
                        txt_save=False,
                        txt_save_pred=False,
                        size_threshold=100,
                        gauss_threshold=10, 
                        name_path_fin='save',
                        cts=6,
                        stoppage=4,
                        zoom_threshold=1000,
                        radius_threshold=0.05,
                        name_path_fin_save_index=20,
                        spine_filter=True,
                        numNeighbours=5,
                        zoom_threshold_min=1,
                        zoom_threshold_max=4, 
                        line_num_points_shaft=200,
                        line_num_points_inter_shaft=300, 
                        spline_smooth_shaft=1, 
                        DTYPE='float32', 
                        disp_infos=False,
                        pre_portion=None,
                        pinn_dir_data=None, 
                        pinn_dir_data_all=None,
                        model_sufix_all=None,
                        neld_names_all=None,
                        path_heads =None,
                        true_keys=None,
                        list_features=None,
                        base_features_list=None,
                        metrics={},
                        model_type=None,
                        data_mode=None,
                        path_dir=None, 
                        thre_target_number_of_triangles=None,
                        voxel_resolution=None,
                        obj_org_path_dict=None,
                        path_display=None,
                        dict_mesh_to_skeleton_finder_mesh=None,
                        model_sufix_dic=None,
                        path_display_dic=None,
                        kmean_n_run=100,
                        kmean_max_iter=600,
                        param_dic=None, 

        ): 


        self.common_args = dict(
            file_path_org=file_path_org,
            neld_data=neld_data,
            neld_names=neld_names,
            neld_namess=neld_namess,
            data_studied=data_studied,
            name_spine_id=name_spine_id,
            name_head_id=name_head_id,
            name_neck_id=name_neck_id,
            name_shaft_id=name_shaft_id,
            txt_true_file=txt_true_file,
            txt_save=txt_save,
            txt_save_pred=txt_save_pred,
            size_threshold=size_threshold,
            gauss_threshold=gauss_threshold,
            name_path_fin=name_path_fin,
            cts=cts,
            stoppage=stoppage,
            zoom_threshold=zoom_threshold,
            radius_threshold=radius_threshold,
            name_path_fin_save_index=name_path_fin_save_index,
            spine_filter=spine_filter,
            numNeighbours=numNeighbours,
            zoom_threshold_min=zoom_threshold_min,
            zoom_threshold_max=zoom_threshold_max,
            line_num_points_shaft=line_num_points_shaft,
            line_num_points_inter_shaft=line_num_points_inter_shaft,
            spline_smooth_shaft=spline_smooth_shaft,
            DTYPE=DTYPE,
            model_sufix=model_sufix,
            neld_path_inits=neld_path_inits,
            disp_infos=disp_infos,
            path_train=path_train,
            pre_portion=pre_portion,
            pinn_dir_data=pinn_dir_data,
            pinn_dir_data_all=pinn_dir_data_all,
            model_sufix_all=model_sufix_all,
            neld_names_all=neld_names_all,
            path_heads=path_heads,
            true_keys=true_keys,
            list_features=list_features,
            base_features_list=base_features_list,
            metrics=metrics,
            model_type=model_type,
            data_mode=data_mode, 
            thre_target_number_of_triangles=thre_target_number_of_triangles,
            voxel_resolution=voxel_resolution,
            obj_org_path_dict=obj_org_path_dict,
            model_sufix_dic=model_sufix_dic,
            path_display_dic=path_display_dic,
            kmean_n_run=kmean_n_run,
            kmean_max_iter=kmean_max_iter,
            param_dic=param_dic, 
            file_path_data=file_path_data,
        )


        print('[[[[[[[[[[neld_data-----neld_data]]]]]]]]]]',name_head_id,data_mode.keys())

        self.common_argsman = dict(
            file_path_org=file_path_org,
            neld_data=neld_data,
            neld_names=neld_names,
            neld_namess=neld_namess,
            data_studied=data_studied,
            name_spine_id=name_spine_id,
            name_head_id=name_head_id,
            name_neck_id=name_neck_id,
            name_shaft_id=name_shaft_id,
            txt_true_file=txt_true_file,
            txt_save=txt_save,
            txt_save_pred=txt_save_pred,
            size_threshold=size_threshold,
            gauss_threshold=gauss_threshold,
            name_path_fin=name_path_fin,
            cts=cts,
            stoppage=stoppage,
            zoom_threshold=zoom_threshold,
            radius_threshold=radius_threshold,
            name_path_fin_save_index=name_path_fin_save_index,
            spine_filter=spine_filter,
            numNeighbours=numNeighbours,
            zoom_threshold_min=zoom_threshold_min,
            zoom_threshold_max=zoom_threshold_max,
            line_num_points_shaft=line_num_points_shaft,
            line_num_points_inter_shaft=line_num_points_inter_shaft,
            spline_smooth_shaft=spline_smooth_shaft,
            DTYPE=DTYPE,
            model_sufix=model_sufix,
            neld_path_inits=neld_path_inits,
            disp_infos=disp_infos,
            path_train=path_train,
            pre_portion=pre_portion,
            pinn_dir_data=pinn_dir_data,
            pinn_dir_data_all=pinn_dir_data_all,
            model_sufix_all=model_sufix_all,
            path_heads=path_heads,
            neld_names_all=neld_names_all,
            true_keys=true_keys,
            list_features=list_features,
            base_features_list=base_features_list,
            metrics=metrics,
            model_type=model_type,
            data_mode=data_mode, 
            thre_target_number_of_triangles=thre_target_number_of_triangles,
            voxel_resolution=voxel_resolution,
            obj_org_path_dict=obj_org_path_dict,
            model_sufix_dic=model_sufix_dic,
            path_display_dic=path_display_dic,
            kmean_n_run=kmean_n_run,
            kmean_max_iter=kmean_max_iter,
            param_dic=param_dic, 
            file_path_data=file_path_data,
        )



        get_name.__init__(self)  
        print('[[[[[[[self.common_args]]]]]]]',self.common_args['file_path_org'])
        get_files.__init__(self,**self.common_args) 
        # train_test_tf.__init__(self,**self.common_args)
        pinn_data_adj.__init__(self,**self.common_args) 
        hntt.train_test_tf.__init__(self,**self.common_args)
        dendrite_manipulate.__init__(self,**self.common_argsman)
        
        
        
        self.path_display_dic=path_display_dic
        self.dict_mesh_to_skeleton_finder_mesh=dict_mesh_to_skeleton_finder_mesh
        if data_mode is not None:
            path_train=data_mode['path_train']
            # pre_portion=data_mode['pre_portion']
            # pinn_dir_data=data_mode['pinn_dir_data']
            # list_features=data_mode['list_features']
            # base_features_list=data_mode['base_features_list']
        self.path_display=path_display if path_display is not None else ['dest_spine_path','dest_shaft_path']
        path_dir=os.path.join(file_path_org, 'data') 
        np.savetxt(os.path.join(path_dir, 'model_sufix_all.txt'), np.array(list(model_sufix_all)), fmt='%s') 
        np.savetxt(os.path.join(path_dir, 'pinn_dir_data_all.txt'), np.array(list(pinn_dir_data_all)), fmt='%s')
        np.savetxt(os.path.join(path_dir, 'path_heads.txt'), np.array(path_heads), fmt='%s')
        np.savetxt(os.path.join(path_dir, 'true_keys.txt'), np.array(true_keys), fmt='%s') 
        np.savetxt(os.path.join(path_dir, 'neld_names_all.txt'), np.array(neld_names_all), fmt='%s') 
 








