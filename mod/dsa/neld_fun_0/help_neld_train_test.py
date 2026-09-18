











import sys
import os
import numpy as np

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import numpy as np
import time 
import tensorflow as tf  
tf.config.run_functions_eagerly(True) 
from tensorflow.keras.models import load_model
DTYPE='float32' 
import pickle 
import pandas as pd 

import importlib
import random
np.random.seed(42)
tf.random.set_seed(42)
random.seed(42)

device = "/GPU:0" if tf.config.list_physical_devices('GPU') else "/CPU:0"  
from tqdm import tqdm 
 
import mod.dsa.neld_fun_0.help_funn as hff   


from mod.dsa.neld_fun_0.get_path import assign_if_none,get_name,get_param,get_files
 
from mod.dsa.dend_fun_0.help_dendrite_pred import train_test_tf as train_test_tfold
def loss_fn(output, target):
    return np.mean(np.square(output - target))

class train_test_tf(get_files,get_name,train_test_tfold): 
    def __init__(self, **kwargs):
        get_name.__init__(self) 
        get_files.__init__(self,**kwargs)

    def get_train_input_ML(self,  
                        path_train, 
                        pre_portion,  
                        DTYPE=None, 
                        file_path_model_data=None,
                        data_studied=None, 
                        line_num_points_shaft=None,
                        line_num_points_inter_shaft=None,
                        spline_smooth_shaft=None,   
                        model_sufix=None,
                        disp_infos=None,
                        txt_save_file=None,
                        neld_names=None,
                        weight_positive=.5,  
                        list_features=None,
                        base_features_list=None,
                        model_type=None,
                        num_sub_nodes=None,
                        thre_target_number_of_triangles=None,
                        voxel_resolution=None,
                        dict_mesh_to_skeleton_finder_mesh=None,
                        tf_train=True,
                        entry_names=[None],
                        kmean_n_run=None,
                        kmean_max_iter=None,
                        param_dic=None,
                        ): 
        param_dic=param_dic if param_dic is not None else self.param_dic
        kmean_n_run=kmean_n_run or self.kmean_n_run
        kmean_max_iter=kmean_max_iter or self.kmean_max_iter
        dict_mesh_to_skeleton_finder_mesh=dict_mesh_to_skeleton_finder_mesh or self.dict_mesh_to_skeleton_finder_mesh
        thre_target_number_of_triangles=thre_target_number_of_triangles or self.thre_target_number_of_triangles
        voxel_resolution=voxel_resolution or self.voxel_resolution 

        base_features_list=base_features_list if base_features_list is not None else self.base_features_list
        list_features = list_features if list_features is not None else self.list_features
        disp_infos = disp_infos or self.disp_infos  
        model_sufix = model_sufix or self.model_sufix  
        file_path_model_data = file_path_model_data or self.file_path_model_data
        data_studied = data_studied or self.data_studied  
        line_num_points_shaft = line_num_points_shaft or self.line_num_points_shaft
        line_num_points_inter_shaft = line_num_points_inter_shaft or self.line_num_points_inter_shaft
        spline_smooth_shaft = spline_smooth_shaft or self.spline_smooth_shaft
        neld_names = neld_names if neld_names is not None else self.neld_names
        DTYPE=DTYPE or self.DTYPE 

        curv, rhs, weight, indices,adj,neld = [], [], [], [],[],[]

        for index, neld_name in enumerate(neld_names): 
            self.get_neld_name(data_studied=data_studied, index=index,model_type=model_type) 
            spine_portion_path=self.path_file[path_train['data_spine_path']] 
            shaft_portion_path=self.path_file[path_train['data_shaft_path']] 
            pid = pinn_data(file_path=self.file_path,
                            file_path_feat=self.file_path_feat,
                            path_file=self.path_file,
                            spine_path = spine_portion_path,
                            shaft_path =  shaft_portion_path , 
                            spine_path_pre=spine_portion_path,
                            shaft_path_pre=  shaft_portion_path ,  
                            neld_path_original_mm=self.neld_path_original_mm,
                            neld_first_name=self.neld_names[index],
                            model_sufix=model_sufix,
                            path_train=path_train,
                            line_num_points_shaft=line_num_points_shaft,
                            line_num_points_inter_shaft=line_num_points_inter_shaft,
                            spline_smooth_shaft=spline_smooth_shaft,
                            thre_target_number_of_triangles=thre_target_number_of_triangles,
                            voxel_resolution=voxel_resolution,
                            dict_mesh_to_skeleton_finder_mesh=dict_mesh_to_skeleton_finder_mesh,
                            kmean_n_run=kmean_n_run,
                            kmean_max_iter=kmean_max_iter,
                            param_dic=param_dic,
                            neld_path_true_final=self.neld_path_true_final,
                                )
            pid.save_pinn_data()    
            pid.get_neld_data()
            feat_paths=[]  
            for path_inten,name_inten in list_features:  
                pathh=os.path.join( self.file_path_feat ,name_inten)
                if os.path.exists(pathh):
                    feat_paths.append(pathh)
                    print('train data path ---->>',self.file_path_feat,pathh) 
                    print(np.loadtxt(pathh))
                else:
                    print('path doesnt exists ===----->>>',pathh)

            curv.append(np.hstack(pid.get_pinn_features(feat_paths=feat_paths,base_features_list=base_features_list,))) 
            neld.append(pid.neld)     
            if tf_train:
                rhs.append(pid.get_pinn_rhs(pre_portion=pre_portion) )

        return curv, rhs ,adj,neld





 


 

    '''


    def get_train_input_dnn(self,  
                        path_train, 
                        pre_portion,  
                        DTYPE=None, 
                        file_path_model_data=None,
                        data_studied=None, 
                        line_num_points_shaft=None,
                        line_num_points_inter_shaft=None,
                        spline_smooth_shaft=None,   
                        model_sufix=None,
                        disp_infos=None,
                        txt_save_file=None,
                        neld_names=None,
                        weight_positive=.5,  
                        list_features=None,
                        base_features_list=None,
                        model_type=None,
                        model_init=None,
                        num_sub_nodes=None,
                        thre_target_number_of_triangles=None,
                        voxel_resolution=None,
                        dict_mesh_to_skeleton_finder_mesh=None,
                        tf_train=True,
                        entry_names=[None],
                        kmean_n_run=None,
                        kmean_max_iter=None,
                        param_dic=None,
                        ): 
        param_dic=param_dic if param_dic is not None else self.param_dic
        kmean_n_run=kmean_n_run or self.kmean_n_run
        kmean_max_iter=kmean_max_iter or self.kmean_max_iter
        dict_mesh_to_skeleton_finder_mesh=dict_mesh_to_skeleton_finder_mesh or self.dict_mesh_to_skeleton_finder_mesh
        thre_target_number_of_triangles=thre_target_number_of_triangles or self.thre_target_number_of_triangles
        voxel_resolution=voxel_resolution or self.voxel_resolution 

        base_features_list=base_features_list if base_features_list is not None else self.base_features_list
        list_features = list_features if list_features is not None else self.list_features
        model_init = model_init if model_init is not None else self.model_init
        disp_infos = disp_infos or self.disp_infos  
        model_sufix = model_sufix or self.model_sufix  
        file_path_model_data = file_path_model_data or self.file_path_model_data
        data_studied = data_studied or self.data_studied  
        line_num_points_shaft = line_num_points_shaft or self.line_num_points_shaft
        line_num_points_inter_shaft = line_num_points_inter_shaft or self.line_num_points_inter_shaft
        spline_smooth_shaft = spline_smooth_shaft or self.spline_smooth_shaft
        neld_names = neld_names if neld_names is not None else self.neld_names
        DTYPE=DTYPE or self.DTYPE 

        curv, rhs, weight, indices,adj,neld = [], [], [], [],[],[]

        indexx={neld_name:index for index, neld_name  in enumerate(self.neld_names)} 
        for entry_name in entry_names:
            for neld_name in neld_names: 
                index=indexx[neld_name]
                self.get_neld_name(data_studied=data_studied, index=index,model_type=model_type,entry_name=entry_name) 
                spine_portion_path=self.path_file[path_train['data_spine_path']] 
                shaft_portion_path=self.path_file[path_train['data_shaft_path']] 
                pid = pinn_data(file_path=self.file_path,
                                file_path_feat=self.file_path_feat,
                                path_file=self.path_file,
                                spine_path = spine_portion_path,
                                shaft_path =  shaft_portion_path , 
                                spine_path_pre=spine_portion_path,
                                shaft_path_pre=  shaft_portion_path ,  
                                neld_path_original_mm=self.neld_path_original_mm,
                                neld_first_name=self.neld_namess[index][1],
                                model_sufix=model_sufix,
                                path_train=path_train,
                                line_num_points_shaft=line_num_points_shaft,
                                line_num_points_inter_shaft=line_num_points_inter_shaft,
                                spline_smooth_shaft=spline_smooth_shaft,
                                thre_target_number_of_triangles=thre_target_number_of_triangles,
                                voxel_resolution=voxel_resolution,
                                dict_mesh_to_skeleton_finder_mesh=dict_mesh_to_skeleton_finder_mesh,
                                kmean_n_run=kmean_n_run,
                                kmean_max_iter=kmean_max_iter,
                                param_dic=param_dic, 
                                        neld_path_true_final=self.neld_path_true_final,
                                    ) 
                file_path_feat = self.file_path_feat if entry_name is None else self.file_path_feat_entry
                file_path = self.file_path if entry_name is None else self.file_path_entry
                pid.save_pinn_data(file_path_feat=file_path_feat,
                                file_path=file_path,
                                )    
                pid.get_neld_data(file_path_feat=file_path_feat,
                                file_path=file_path,)
                feat_paths=[]  
                for path_inten,name_inten in list_features: 
                    if model_init is not None: 
                        mode_dnn_1=self.model_sufix_dic['model_sufix_inverse'][model_init]
                        path_dir=self.model_sufix_dic['path_dir']
                        path_shaft_dir=f'{self.model_type}_{mode_dnn_1}_{path_dir}'
                        pathini=self.path_file[path_shaft_dir]
                    else:
                        pathini=file_path_feat

                    pathini=  self.path_file[path_train['data_shaft_path']]   

                    pathh=os.path.join( pathini ,name_inten)
                    if os.path.exists(pathh):
                        feat_paths.append(pathh)
                        print('train data path ---->>',pathini,pathh) 
                        print(np.loadtxt(pathh))
                    else:
                        print('path doesnt exists ===----->>>',pathh)
                neld.append(pid.neld)     
                if tf_train:
                    # if pre_portion=='head': 
                    #     pathh=os.path.join( self.neld_path_true_final ,'intensity_shaft.neck_head.txt')
                    # elif pre_portion=='neck_head': 
                    #     pathh=os.path.join( self.neld_path_true_final ,'intensity_shaft_neck_head.txt')
                    # else:
                    #     pathh=os.path.join( self.neld_path_true_final ,'intensity_shaft_spine.txt')
                    # rhs.append(tf.cast(
                    #     pid.get_pinn_rhs(pre_portion=pre_portion,
                    #                     file_path_feat=file_path_feat,
                    #                     file_path=file_path,
                    #                     neld_path_true_final=self.neld_path_true_final,
                    #                     ), 
                    #     dtype=DTYPE)
                    #         ) 
                    rhs.append(tf.cast(
                        np.loadtxt(os.path.join( self.neld_path_true_final ,f'intensity_1hot_shaft_{pre_portion}.txt'),dtype=int), 
                        dtype=DTYPE)
                            ) 
                    pathh=os.path.join( self.neld_path_true_final ,f'intensity_shaft_{pre_portion}.txt')
                    mask=np.loadtxt(pathh,dtype=int)
                    unique=np.sort(np.unique(mask))
                    labels = {v: np.argwhere(mask == v) for v in unique}
                    counts = {v: labels[v].shape[0] for v in labels}
                    total = sum(counts.values()) 

                    w_prime = np.array([max(np.log(2 * total / counts[k]), 1) for k in unique ])
        
                    weight.append(w_prime / np.sum(w_prime))






                if len(list(list_features)+list(base_features_list))==0:
                    curv.append(pid.neld.vertices)  
                    continue
                    
                curv.append(np.hstack(pid.get_pinn_features(feat_paths=feat_paths,
                                                            base_features_list=base_features_list,
                                                            file_path_feat=file_path_feat,
                                                            file_path=file_path,))) 

        return curv, rhs ,adj,neld,weight
 
 '''
    def train_model_spine_kal(self,
                            path_train=None,
                            get_training=True, 
                            pre_portion=None,
                            full_neld=True, 
                            hidden_layers=None, 
                            neurons_per_layer=None, 
                            activation_init=None,
                            activation_hidden=None,
                            activation_last=None,  
                            DTYPE=None, 
                            model_sufix=None, 
                            data_studied=None,
                            vv_cts=None,
                            new_model=True, 
                            itime = 10000, 
                            itime_div=1,
                            loss_save_dir=None,
                            iou_save_dir=None,  
                            auc_save_dir=None,   
                            dice_save_dir=None,
                            index_save_dir=None,  
                            model_dir=None, 
                            loss_mode="bce",
                            ls=[2,3,4,5],
                            disp_infos=None,  
                            dest_path='dest_spine_path',
                            model_type=None,
                            num_sub_nodes=None,
                            rl_par= 0.5, 
                            dnn_par= 0.5,
                            thre_target_number_of_triangles=None,
                            voxel_resolution=None,
                            l1_values = [0,   1e-6,   1e-4, 1e-2],
                            l2_values = [0,   1e-6,   1e-4, 1e-2],
                            dict_mesh_to_skeleton_finder_mesh=None,
                        entry_names=[],
                        kmean_n_run=None,
                        kmean_max_iter=None,
                        param_dic=None,
                        train_neld_param=None,
                        path_dict=None,
                        **kward,

                        ): 
        ls=ls if ls is not None else [0,1,2,4]
        param_dic=param_dic if param_dic is not None else self.param_dic 
        disp_infos = disp_infos or self.disp_infos 
        pre_portion=pre_portion or self.pre_portion
        path_train=path_train or self.path_train
        DTYPE = DTYPE or self.DTYPE
        vv_cts = vv_cts or self.vv_cts 
        model_sufix = model_sufix or self.model_sufix 
        self.get_model_opt_name(model_sufix=model_sufix,model_type=model_type) 
 
 
        hidden_layers=hidden_layers or self.hidden_layers 
        neurons_per_layer=neurons_per_layer or self.neurons_per_layer 
        activation_init=activation_init or self.activation_init
        activation_hidden=activation_hidden or self.activation_hidden
        activation_last=activation_last or self.activation_last
 
        model_dirs = self.model_dir_path.get(pre_portion, self.model_dir_path['default']) 
        loss_save_dir = loss_save_dir or model_dirs['loss'] 
        oloss_save_dir =  model_dirs['oloss']   
        iou_save_dir = iou_save_dir or  model_dirs['iou'] 
        auc_save_dir = auc_save_dir or model_dirs['auc']
        dice_save_dir = dice_save_dir or model_dirs['dice']
        index_save_dir = index_save_dir or  model_dirs['index_save'] 
        # model_dir = model_dir or  model_dirs['model']   

        # from neld_pinn_0.run_1 import aka_grad ,lang_loss, neld_coef,get_neld_data_train

        train_neld_param=self.data_mode['pinn']
        param=self.data_mode['pinn']['param']
        train_neld_param['tf_train']=True 

        # print('[[[[[[[[[]]]]]]]]]',train_neld_param) 

        run_module = importlib.import_module(path_dict['run']) 
        model_choice = run_module.model_choice 
        
 
        with tf.device(device): 
            mchoice = model_choice(model_type=model_type,
                                obs_dim=param['par_0']['obs_dim'],
                            state_dim=param['par_0']['state_dim'])
            param['train_fun']['model'] = mchoice.get_model() 
            param['train_fun']['get_data']=mchoice.get_data()
            # param['train_fun']['model'] = mchoice.get_model() 
            # param['train_fun']['get_data']=mchoice.get_data()
            model = mchoice.get_model() 
            custom = mchoice.get_custom_objects(model_type) 

        aka_train = run_module.aka_train(param)
        lr = tf.keras.optimizers.schedules.PiecewiseConstantDecay([1000, 3000], [1e-2, 1e-3, 5e-4])
        optimizer = tf.optimizers.Adam(learning_rate=lr) 
        optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
        optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
        if new_model:
            print(f"I'm starting a new model")
            print('--------------------------------')
            print(f"Model Type Gen : {model_type}")
            print(f"Model Type     : {model_sufix}")
            print(f"Model Dir.     : {model_dir}")
            print(f"list train  : {ls}")
            print(f"Data Dir.      : {path_train['data_shaft_path']}")   
            print(f"Hidden Layers        : {hidden_layers}")
            print(f"neurons_per_layer    : {neurons_per_layer}")
            print('--------------------------------')
            loss_save = []
            iou_save = {0: [], 1: [], 2: []}
            auc_save = {0: [], 1: [], 2: []}
            dice_save = {0: [], 1: [], 2: []}
            loss_tmp=10**10
            head_tmp=0
        else:
            print("This is a continuation of the previous model") 
            loss_save = np.loadtxt(loss_save_dir, dtype=float).tolist()
            iou_save = np.loadtxt(iou_save_dir, dtype=float).tolist() 
            auc_save = np.loadtxt(auc_save_dir, dtype=float).tolist() 
            dice_save = np.loadtxt(dice_save_dir, dtype=float).tolist() 
            
        print(f"Using device: {device}------------------{ls}----------------------------------------------")
        lst_test=list(set(range(len(self.neld_names)))-set(ls))
        loss_tmp_all=mml=1e10
        modlist=['loss','iou','auc','dice'] 
        metr={mm:{f'{nn}_tmp':[mml,mml] for nn in modlist } for mm in modlist}
        metr['loss']['loss_tmp']=1e6 
        tf.random.set_seed(0)
        iou_tmp= {0: 0, 1: 0, 2: 0};auc_tmp= {0: 0, 1: 0, 2: 0};dice_tmp= {0: 0, 1: 0, 2: 0}
        index_save=[]
        lossi=0 
        pbar = tqdm(range(itime), desc=f"Loss: {loss_tmp:.6f}__ IoU: N/A")  
        for i in pbar: 
            # loss = aka_train.train_PINN(optimizer,model)
            _=aka_train.train_mod(optimizer, model,self,data_studied,model_type,ls)
            loss=aka_train.test_mod(model,self,data_studied,model_type,lst_test) 

            loss_save.append(loss)
            np.savetxt(loss_save_dir, np.array(loss_save), fmt='%f') 

 
 
            hh=mmjl='loss'
            # model.save(model_dirs[f'model_{hh}'])
            if loss < metr[hh][f'{hh}_tmp']: 
                model.save(model_dirs[f'model_{hh}'])  
                metr[hh][f'{hh}_tmp']=loss
                loss_tmp=loss
                index_save.append(i)
                np.savetxt(index_save_dir, np.array(index_save), fmt='%d') 



            hh='oloss'
            # if loss <loss_tmp_all:
            #     loss_tmp_all=loss
                # model.save(model_dir)
            model.save(model_dirs[f'model_{hh}']) 


            pbar.set_description(f"Loss save: {loss_tmp:.6f} __ Loss: {loss:.6f} ")






 
    def get_shaft_pred_dnn(self, 
                        path_train=None, 
                        pre_portion='spine', 
                        n_col=None,
                        hidden_layers=None, 
                        neurons_per_layer=None, 
                        activation_init=None,
                        activation_hidden=None,
                        activation_last=None, 
                        model= None,
                        curv=None,
                        size_threshold=None,
                        neld_names=None, 
                        model_dir=None, 
                        loss_save_dir=None,
                        iou_save_dir=None,
                        auc_save_dir=None,
                        model_sufix=None, 
                        data_studied=None,
                        file_path_org=None,  
                        line_num_points_shaft=None,
                        line_num_points_inter_shaft=None,
                        spline_smooth_shaft=None ,  
                        disp_infos=None, 
                        DTYPE=None,
                        list_features=None,
                        base_features_list=None,
                        shaft_thre=None,
                        train_spines=False,
                        weight=None,
                        weight2=None,
                        weights=None,
                        neck_lim=None,
                        n_clusters=3,  
                        smooth_tf=False,
                        model_type=None,
                        num_sub_nodes=None,
                        thre_target_number_of_triangles=None,
                        voxel_resolution=None,
                        reconstruction_tf=False, 
                        dict_mesh_to_skeleton_finder=None,
                        dict_wrap=None,
                        tf_skl_shaft_distance=False,
                        dict_mesh_to_skeleton_finder_mesh=None,
                        kmean_n_run=None,
                        kmean_max_iter=None,
                        param_dic=None,
                        train_neld_param=None,
                        path_dict=None,
                        **kward,
                        ): 
        from mod.dsa.dend_fun_2.help_pinn_data_fun import pinn_data 






        line_num_points_shaft=kward.get('line_num_points_shaft')
        line_num_points_inter_shaft=kward.get('line_num_points_inter_shaft')
        spline_smooth_shaft=kward.get('spline_smooth_shaft')



        param_dic=param_dic if param_dic is not None else self.param_dic   
        pre_portion=pre_portion or self.pre_portion
        path_train=path_train or self.path_train
        disp_infos = disp_infos or self.disp_infos  
        file_path_org = file_path_org or self.file_path_org
        model_sufix=model_sufix or self.model_sufix
        self.get_model_opt_name(model_sufix=model_sufix,
                                model_type=model_type,)  
        line_num_points_shaft = line_num_points_shaft or self.line_num_points_shaft
        line_num_points_inter_shaft = line_num_points_inter_shaft or self.line_num_points_inter_shaft
        spline_smooth_shaft = spline_smooth_shaft or self.spline_smooth_shaft
        DTYPE=DTYPE or self.DTYPE 
        data_studied = data_studied or self.data_studied 

        hidden_layers=hidden_layers or self.hidden_layers 
        neurons_per_layer=neurons_per_layer or self.neurons_per_layer 
        activation_init=activation_init or self.activation_init
        activation_hidden=activation_hidden or self.activation_hidden
        activation_last=activation_last or self.activation_last

        model_dirs = self.model_dir_path.get(pre_portion, self.model_dir_path['default'])
  
        loss_save_dir = loss_save_dir or model_dirs['loss']   
        iou_save_dir = iou_save_dir or  model_dirs['iou']    
        auc_save_dir = auc_save_dir or  model_dirs['auc'] 
        # model_dir = model_dir or  model_dirs['model'] 
        model_type_split=self.path_dir.lower().split('_') 
        for mo in ['dice','iou','auc','loss','oloss']:
            if mo in model_type_split:
                model_dir =  model_dirs[f'model_{mo}']  
                break 
        # mo='oloss'
        model_dir =  model_dirs[f'model_{mo}']
        # model_dir =  model_dirs[f'model']
        rhs_name='rhs_name'
        n_col=n_col or len(self.model_dir_path[pre_portion][rhs_name])  
 
        train_neld_param=self.data_mode['pinn']
        param=self.data_mode['pinn']['param']
        train_neld_param['tf_train']=True 

        # print('[[[[[[[[[]]]]]]]]]',train_neld_param) 
 
        run_module = importlib.import_module(path_dict['run']) 
        model_choice = run_module.model_choice 
        
 
        with tf.device(device): 
            mchoice = model_choice(model_type=model_type, ) 
            # param['train_fun']['model'] = mchoice.get_model() 
            # param['train_fun']['get_data']=mchoice.get_data()
            # model = mchoice.get_model() 
            custom = mchoice.get_custom_objects(model_type) 
            model = load_model(model_dir, custom_objects=custom)

        print(f'get_shaft_pred started')
        print('--------------------------------')
        print(f"Model Type Gen : {model_type}")
        print(f"Model Type     : {model_sufix}")
        print(f"Model Dir.     : {model_dir}")
        print(f"Model Portion  : {pre_portion}") 
        print(f"Data Dir.      : {path_train['data_shaft_path']}") 
        print(f"Destin. Dir.   : {path_train['dest_shaft_path']}")  
        print('--------------------------------')
        time_start = time.time()
        rhs_name='rhs_name'
        loss,losst=0,0
        neld_names = kward.get('neld_names',) or self.neld_names 
        for  index,neld_name in enumerate(neld_names):
            self.get_neld_name(data_studied=data_studied,
                               index=index,
                                model_type=model_type, )  
            neld_name=self.neld_name  
            spine_portion_path=self.path_file[path_train['data_shaft_path']] 
            shaft_portion_path=self.path_file[path_train['data_shaft_path']] 
            pid = pinn_data(file_path=self.file_path,
                            file_path_feat=self.file_path_feat,
                            path_file=self.path_file,
                            spine_path = spine_portion_path,
                            shaft_path =  shaft_portion_path , 
                            spine_path_pre=spine_portion_path,
                            shaft_path_pre=  shaft_portion_path ,  
                            neld_path_original_mm=self.neld_path_original_mm,
                            neld_first_name=self.neld_names[index],
                            model_sufix=model_sufix,
                            path_train=path_train,
                            line_num_points_shaft=line_num_points_shaft,
                            line_num_points_inter_shaft=line_num_points_inter_shaft,
                            spline_smooth_shaft=spline_smooth_shaft,
                            thre_target_number_of_triangles=thre_target_number_of_triangles,
                            voxel_resolution=voxel_resolution,
                            dict_mesh_to_skeleton_finder_mesh=dict_mesh_to_skeleton_finder_mesh,
                            kmean_n_run=kmean_n_run,
                            kmean_max_iter=kmean_max_iter,
                            param_dic=param_dic,
                            neld_path_true_final=self.neld_path_true_final,
                        )  
            spines_logit=  self.intensity_logit_dict[pre_portion]
            key,tyy=f'{self.model_type}_{self.model_sufix}_{self.path_dir}', spines_logit[0]  
            path_ex = os.path.join(self.path_file_sub[tyy][key]) 
            if os.path.exists(path_ex) and not param_dic['tf_restart']['get_shaft_pred']: 
                continue
            pid.save_pinn_data() 
            pid.get_neld_data()
            con=0

            curv,_,_,_,_=self.get_train_input_dnn(   
                                    neld_names=[neld_name],
                                    path_train=path_train,  
                                    pre_portion=pre_portion, 
                                    line_num_points_shaft=line_num_points_shaft,
                                    line_num_points_inter_shaft=line_num_points_inter_shaft, 
                                    model_sufix=model_sufix, 
                                    list_features=list_features,
                                    base_features_list=base_features_list, 
                                    model_type=model_type,
                                    num_sub_nodes=num_sub_nodes, 
                                    thre_target_number_of_triangles=thre_target_number_of_triangles,
                                    voxel_resolution=voxel_resolution,
                                    dict_mesh_to_skeleton_finder_mesh=dict_mesh_to_skeleton_finder_mesh,
                                    tf_train=False,
                                    kmean_n_run=kmean_n_run,
                                    kmean_max_iter=kmean_max_iter,
                                    param_dic=param_dic,
                                    )   
            rhs0=model(curv[0]).numpy()   
            key=f'{self.model_type}_{self.model_sufix}_{self.path_dir}'  
            for kk,tyy in enumerate(spines_logit):
                np.savetxt(self.path_file_sub[tyy][key], rhs0[:,kk], fmt='%f') 
 


 
            vert=np.loadtxt(os.path.join(self.file_path,self.txt_vertices_0)) 
            vert_old=np.loadtxt(os.path.join(self.file_path,self.txt_vertices_old)) 
            from scipy.spatial import KDTree 
            idx = KDTree(vert).query(vert_old)[1].flatten() 


            self.get_neld_name(data_studied=data_studied,
                               index=index,
                                model_type=model_type,
                                ** param_dic['data']['get_neld_name'])    
 
 
            spines_logit=self.intensity_logit_dict[pre_portion]
            print('[[[[[[[[[[[[[[[[[intensity_spines_logit]]]]]]]]]]]]]]]]]',spines_logit)
            for kk,tyy in enumerate(self.intensity_logit_dict[pre_portion]):
                np.savetxt(self.path_file_sub[tyy][key], rhs0[:,kk][idx], fmt='%f') 
        
 
            mytime0 = time.time() - time_start
 
            hours, rem = divmod(mytime0, 3600)
            minutes, seconds = divmod(rem, 60)
            if disp_infos:
                print(f'Shaft Prediction completed on {neld_name} in {int(hours)}h {int(minutes)}m {seconds:.2f}s')
                print(f'data stores in: {spine_portion_path}')  
        print('Shaft Prediction completed') 











 
    def get_shaft_pred_kal(self, 
                        path_train=None, 
                        pre_portion=None, 
                        n_col=None,
                        hidden_layers=None, 
                        neurons_per_layer=None, 
                        activation_init=None,
                        activation_hidden=None,
                        activation_last=None, 
                        model= None,
                        curv=None,
                        size_threshold=None,
                        neld_names=None, 
                        model_dir=None, 
                        loss_save_dir=None,
                        iou_save_dir=None,
                        auc_save_dir=None,
                        model_sufix=None, 
                        data_studied=None,
                        file_path_org=None,  
                        line_num_points_shaft=None,
                        line_num_points_inter_shaft=None,
                        spline_smooth_shaft=None ,  
                        disp_infos=None, 
                        DTYPE=None,
                        list_features=None,
                        base_features_list=None,
                        shaft_thre=None,
                        train_spines=False,
                        weight=None,
                        weight2=None,
                        weights=None,
                        neck_lim=None,
                        n_clusters=3,  
                        smooth_tf=False,
                        model_type=None,
                        num_sub_nodes=None,
                        thre_target_number_of_triangles=None,
                        voxel_resolution=None,
                        reconstruction_tf=False, 
                        dict_mesh_to_skeleton_finder=None,
                        dict_wrap=None,
                        tf_skl_shaft_distance=False,
                        dict_mesh_to_skeleton_finder_mesh=None,
                        kmean_n_run=None,
                        kmean_max_iter=None,
                        param_dic=None,
                        train_neld_param=None,
                        path_dict=None,
                        ): 
        param_dic=param_dic if param_dic is not None else self.param_dic   
        pre_portion=pre_portion or self.pre_portion
        path_train=path_train or self.path_train
        disp_infos = disp_infos or self.disp_infos  
        file_path_org = file_path_org or self.file_path_org
        model_sufix=model_sufix or self.model_sufix
        self.get_model_opt_name(model_sufix=model_sufix,
                                model_type=model_type,)  
        line_num_points_shaft = line_num_points_shaft or self.line_num_points_shaft
        line_num_points_inter_shaft = line_num_points_inter_shaft or self.line_num_points_inter_shaft
        spline_smooth_shaft = spline_smooth_shaft or self.spline_smooth_shaft
        DTYPE=DTYPE or self.DTYPE 
        data_studied = data_studied or self.data_studied 

        hidden_layers=hidden_layers or self.hidden_layers 
        neurons_per_layer=neurons_per_layer or self.neurons_per_layer 
        activation_init=activation_init or self.activation_init
        activation_hidden=activation_hidden or self.activation_hidden
        activation_last=activation_last or self.activation_last

        model_dirs = self.model_dir_path.get(pre_portion, self.model_dir_path['default'])
  
        loss_save_dir = loss_save_dir or model_dirs['loss']   
        iou_save_dir = iou_save_dir or  model_dirs['iou']    
        auc_save_dir = auc_save_dir or  model_dirs['auc'] 
        # model_dir = model_dir or  model_dirs['model'] 
        model_type_split=self.path_dir.lower().split('_') 
        for mo in ['dice','iou','auc','loss','oloss']:
            if mo in model_type_split:
                model_dir =  model_dirs[f'model_{mo}']  
                break 
        # mo='oloss'
        model_dir =  model_dirs[f'model_{mo}']
        # model_dir =  model_dirs[f'model']
        rhs_name='rhs_name'
        n_col=n_col or len(self.model_dir_path[pre_portion][rhs_name])  
 
        train_neld_param=self.data_mode['pinn']
        param=self.data_mode['pinn']['param']
        train_neld_param['tf_train']=True 

        # print('[[[[[[[[[]]]]]]]]]',train_neld_param) 

        fun_module = importlib.import_module(path_dict['fun']) 
        observation=fun_module.observation
        run_module = importlib.import_module(path_dict['run']) 
        model_choice = run_module.model_choice 
        
 
        with tf.device(device): 
            mchoice = model_choice(model_type=model_type,
                                obs_dim=param['par_0']['obs_dim'],
                            state_dim=param['par_0']['state_dim'])
            param['train_fun']['model'] = mchoice.get_model() 
            param['train_fun']['get_data']=get_data=mchoice.get_data()
            # param['train_fun']['model'] = mchoice.get_model() 
            # param['train_fun']['get_data']=mchoice.get_data()
            # model = mchoice.get_model() 
            custom = mchoice.get_custom_objects(model_type) 
            model = load_model(model_dir, custom_objects=custom)

        print(f'get_shaft_pred started')
        print('--------------------------------')
        print(f"Model Type Gen : {model_type}")
        print(f"Model Type     : {model_sufix}")
        print(f"Model Dir.     : {model_dir}")
        print(f"Model Portion  : {pre_portion}") 
        print(f"Data Dir.      : {path_train['data_shaft_path']}") 
        print(f"Destin. Dir.   : {path_train['dest_shaft_path']}")  
        print('--------------------------------')
        time_start = time.time()
        rhs_name='rhs_name'
        loss,losst=0,0
        nam='q'
        neld_names = neld_names or self.neld_names 
        le=len(neld_names) 
        for  index,neld_name in enumerate(neld_names):
            self.get_neld_name(data_studied=data_studied,
                               index=index,
                                model_type=model_type, )  
            neld_name=self.neld_name  
            spine_portion_path=self.path_file[path_train['data_shaft_path']]  
            # param['par_0']['step']=10
            # sav=get_data(param,model)
            # param['par_0']['step']=1000
            # # param['dt']=.0001
            # param['par_0']['KB']=sav['KB']
            path_1=self.neld_path_org_new
            path_2=os.path.dirname(os.path.dirname(path_1))
            print('[[[[[[[[[[[[[---------path_1]]]]]]]]]]]]]',path_1,path_2) 
            path_true=os.path.join(path_1,f"true_{nam}.txt")
            path_obs=os.path.join(path_1,f"obs.txt")
            if os.path.exists(path_true):
                x_trues,obs=np.loadtxt(path_true,dtype=float),np.loadtxt(path_obs,dtype=float)
            else:
                x_trues=None;obs=None
            sav=get_data(param,model,x_trues=x_trues,obs=obs)
            save={}
            for key in sav['par']:
                save[key]=sav['par'][key].numpy()
            key=f'{self.model_type}_{self.model_sufix}_{self.path_dir}'
            # ku='inference'
            # spines_logit=  self.intensity_logit_dict[ku]
            # for kk,tyy in zip(['initial'],spines_logit[:1]):  
            #     rhs00=save['approx']
            #     np.savetxt(self.path_file_sub[tyy][key], rhs00, fmt='%f') 
            #     z_approx=np.array([observation(mm) for mm in rhs00])
            # loss+=loss_fn(x_trues,rhs00)

            ku='inference'
            spines_logit=  self.intensity_logit_dict[ku]
            # for kk,tyy in zip(['initial'],spines_logit[:1]): 
            tyy=spines_logit[0] 
            rhs00=save['approx']
            np.savetxt(self.path_file_sub[tyy][key], rhs00, fmt='%f') 
            z_approx=np.array([observation(mm) for mm in rhs00])
            loss+=loss_fn(x_trues,rhs00)

            tyy=spines_logit[1]
            np.savetxt(self.path_file_sub[tyy][key], z_approx, fmt='%f')   
            losst+=loss_fn(obs,z_approx)

            '''
            ku='true'
            spines_logit=  self.intensity_logit_dict[ku]
            for kk,tyy in zip(['z_pred','x_pred'],spines_logit): 
                rhs0=save[kk]   
                np.savetxt(self.path_file_sub[tyy][key], rhs0, fmt='%f') 
                # if kk =='x_pred':
                #     losst+=loss_fn(x_trues,rhs0)    
 '''

            
            mytime0 = time.time() - time_start
 
            hours, rem = divmod(mytime0, 3600)
            minutes, seconds = divmod(rem, 60)
            if disp_infos:
                print(f'shaft Prediction completed on {neld_name} in {int(hours)}h {int(minutes)}m {seconds:.2f}s')
                print(f'data stores in: {spine_portion_path}     {path_2}')  

        np.savetxt(os.path.join(path_2,f'{self.model_type}_loss.txt'), [losst/le,loss/le], fmt='%f') 
 
        print('shaft Prediction completed') 




 
    def get_shaft_pred_linear(self, 
                        path_train=None, 
                        pre_portion=None, 
                        n_col=None, 
                        model= None,
                        curv=None,
                        size_threshold=None,
                        neld_names=None, 
                        model_dir=None, 
                        loss_save_dir=None,
                        iou_save_dir=None,
                        auc_save_dir=None,
                        model_sufix=None, 
                        data_studied=None,
                        file_path_org=None,   
                        disp_infos=None, 
                        DTYPE=None, 
                        model_type=None, 
                        param_dic=None,
                        train_neld_param=None,
                        path_dict=None,
                        **kward,
                        ): 
        param_dic=param_dic if param_dic is not None else self.param_dic   
        pre_portion=pre_portion or self.pre_portion
        path_train=path_train or self.path_train
        disp_infos = disp_infos or self.disp_infos  
        file_path_org = file_path_org or self.file_path_org
        model_sufix=model_sufix or self.model_sufix
        self.get_model_opt_name(model_sufix=model_sufix,
                                model_type=model_type,)   
        DTYPE=DTYPE or self.DTYPE 
        data_studied = data_studied or self.data_studied 
 
        model_dirs = self.model_dir_path.get(pre_portion, self.model_dir_path['default'])
  
        loss_save_dir = loss_save_dir or model_dirs['loss']   
        iou_save_dir = iou_save_dir or  model_dirs['iou']    
        auc_save_dir = auc_save_dir or  model_dirs['auc'] 
        # model_dir = model_dir or  model_dirs['model'] 
        model_type_split=self.path_dir.lower().split('_') 
        for mo in ['dice','iou','auc','loss','oloss']:
            if mo in model_type_split:
                model_dir =  model_dirs[f'model_{mo}']  
                break 
        # mo='oloss'
        model_dir =  model_dirs[f'model_{mo}']
        # model_dir =  model_dirs[f'model']
        rhs_name='rhs_name'
        n_col=n_col or len(self.model_dir_path[pre_portion][rhs_name])  
 
        train_neld_param=self.data_mode['pinn']
        param=self.data_mode['pinn']['param']
        train_neld_param['tf_train']=True 












        print('[[[[[[[[[]]]]]]]]]',path_dict) 

        fun_module = importlib.import_module(path_dict['fun']) 
        observation=fun_module.observation

        run_module = importlib.import_module(path_dict['run']) 
        model_choice = run_module.model_choice
        get_txt=run_module.get_txt 
        pred_linear=run_module.pred_linear
        
  
        mchoice = model_choice(model_type=model_type,
                            obs_dim=param['par_0']['obs_dim'],
                        state_dim=param['par_0']['state_dim'])
        param['train_fun']['model'] = mchoice.get_model() 
        param['train_fun']['get_data']=get_data=mchoice.get_data() 
        model = mchoice.get_model() 

        print(f'get_shaft_pred started')
        print('--------------------------------')
        print(f"Model Type Gen : {model_type}")
        print(f"Model Type     : {model_sufix}")
        print(f"Model Dir.     : {model_dir}")
        print(f"Model Portion  : {pre_portion}") 
        print(f"Data Dir.      : {path_train['data_shaft_path']}") 
        print(f"Destin. Dir.   : {path_train['dest_shaft_path']}")  
        print('--------------------------------')

        pred_linear(self,param,path_train,model_type,data_studied,disp_infos,neld_names)

        """
        time_start = time.time()
        rhs_name='rhs_name'
        loss,losst=0,0
        neld_names = neld_names or self.neld_names
        le=len(neld_names) 
        for  index,neld_name in enumerate(neld_names):
            self.get_neld_name(data_studied=data_studied,
                               index=index,
                                model_type=model_type, )  
            neld_name=self.neld_name  
            spine_portion_path=self.path_file[path_train['data_shaft_path']]  
            # param['par_0']['step']=10
            # sav=get_data(param,model)
            # param['par_0']['step']=1000
            # # param['dt']=.0001
            # param['par_0']['KB']=sav['KB']
            path_1=self.neld_path_org_new
            nam='q'
            path_2=os.path.dirname(os.path.dirname(path_1))
            print('[[[[[[[[[[[[[---------path_1]]]]]]]]]]]]]',path_1,path_2)
            path_true=os.path.join(path_1,f"true_{nam}.txt")
            path_obs=os.path.join(path_1,f"obs.txt")
            # if os.path.exists(path_true):
            #     x_trues,obs=np.loadtxt(path_true,dtype=float),np.loadtxt(path_obs,dtype=float)
            # else:
            #     x_trues=None;obs=None
            x_trues,obs=get_txt(path_1)
            sav=get_data(param,model,x_trues=x_trues,obs=obs)
            save={}
            for key in sav['par']:
                save[key]=sav['par'][key].numpy()
            key=f'{self.model_type}_{self.model_sufix}_{self.path_dir}'
            # ku='inference'
            # spines_logit=  self.intensity_logit_dict[ku]
            # for kk,tyy in zip(['initial'],spines_logit[:1]):  
            #         rhs00=save['approx']
            #         np.savetxt(self.path_file_sub[tyy][key], rhs00, fmt='%f')  
            # loss+=loss_fn(x_trues,rhs00)


            ku='inference'
            spines_logit=  self.intensity_logit_dict[ku]
            # for kk,tyy in zip(['initial'],spines_logit[:1]): 
            tyy=spines_logit[0] 
            rhs00=save['approx']
            np.savetxt(self.path_file_sub[tyy][key], rhs00, fmt='%f') 
            z_approx=np.array([observation(mm) for mm in rhs00])
            loss+=loss_fn(x_trues['p'],rhs00)


            tyy=spines_logit[1]
            np.savetxt(self.path_file_sub[tyy][key], z_approx, fmt='%f')   
            losst+=loss_fn(obs,z_approx)
            '''
            ku='true'
            spines_logit=  self.intensity_logit_dict[ku]
            for kk,tyy in zip(['z_pred','x_pred'],spines_logit): 
                rhs0=save[kk]   
                np.savetxt(self.path_file_sub[tyy][key], rhs0, fmt='%f') 
                # if kk =='x_pred':
                #     losst+=loss_fn(x_trues,rhs0)    
'''

            mytime0 = time.time() - time_start
 
            hours, rem = divmod(mytime0, 3600)
            minutes, seconds = divmod(rem, 60)
            if disp_infos:
                print(f'shaft Prediction completed on {neld_name} in {int(hours)}h {int(minutes)}m {seconds:.2f}s')
                print(f'data stores in: {spine_portion_path}     {path_2}')  

        np.savetxt(os.path.join(path_2,f'{self.model_type}_loss.txt'), [losst/le,loss/le], fmt='%f') 

        print('shaft Prediction completed') 

"""


 
 

