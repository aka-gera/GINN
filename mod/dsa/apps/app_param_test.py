 
import sys
import os
 
import pickle
# import dash
from dash import dcc, html, dash_table, Input, Output, State, callback 
import dash_bootstrap_components as dbc
import numpy as np  
from mod.dsa.neld_fun_0.get_path import get_files ,safe_id  


from  mod.dsa.neld_fun_0.main_0 import box_style,dropdown_style,dropdown_options_style

 
from mod.dsa.neld_fun_0.main_0_help import dropdown_callback
 







import importlib

from mod.dsa.neld_fun_0.main_0 import algorithm_param,app_run_param,algorithm
from mod.dsa.neld_fun_0.help_model_pred import model_pred 
 
from neld_data_all import paraws,params,dnn_mode,path_dict,tname 


# from mod.dsa.neld_fun_0.app_get_pinn_neld import class_data

run_module = importlib.import_module(path_dict['app']) 
class_data=run_module.class_data
get_app_param=run_module.get_app_param
get_layout=run_module.get_layout

import random
from pathlib import Path 
 
dtype = float
class app_param(get_app_param,class_data,get_layout,dropdown_callback,algorithm,model_pred): 
    def __init__(self, file_path_org, 
                model_sufix,
                path_train,
                index=0, 
                neld_names=None,
                neld_namess=None, 
                data_studied=None,
                neld_path_inits=None,
                path_file=None,
                path_file_sub=None,
                prevent_initial_call=True,  
                pinn_dir_data=None,
                neld_data=None,
                path_display_dic=None, 
                path_display= ['dest_shaft_path', 'dest_spine_path',],
                model_type=None,
                obj_org_path_dict=None,
                model_sufix_dic=None, 
                path_file_dir=None,
                param_dic=None,
                nam_gen_show=None,
                file_path_data=None,
                path_heads_show=None,
                path_dict=None,
                tname=None,
                params=None,
                paraws=None,
                nam_gen=None,
                
        num = random.randint(100000000, 9999999999) ,
                 ):
        self.num=num
        self.tname=tname
        self.path_dict=path_dict 
        doc_module = importlib.import_module(self.path_dict['doc'])  
        dict_param=doc_module.dict_param
        gdas=doc_module.gdas
        nam_gen=doc_module.nam_gen
        dnn_mode=doc_module.dnn_mode  
        self.data_studied=action=doc_module.action
        self.file_path_org=file_path_org=file_path_org or doc_module.file_path_org
        self.file_path_data=file_path_data=file_path_data or doc_module.file_path_data

        
        model_sufix_show=doc_module.dnn_modes
        path_heads_show=doc_module.path_heads_show
        path_dirs_show=doc_module.path_dirs_show  
        self.obj_org_path=obj_org_path=doc_module.obj_org_path
        print('[[[[[[[[[[callback_file]]nam_gen        ]]]]]]]]',nam_gen,file_path_data,file_path_org,dict_param)  
        path_diroi=os.path.join(file_path_org, 'data')  
        configs=dict_param['configs']  
        nam=f'{nam_gen}_{action}'


        path_heads_show=dict_param['path_heads_show']
        model_type=path_heads_show[0]

        path_dirs_show=dict_param['path_dirs_show']
        path_dir=path_dirs_show[0]

        param = algorithm_param(**dict_param)

        dnn_mode=model_sufix
        neld_data = gdas.part(nam)   
        mapp = app_run_param(param)
        self.param=mapp.emerge_param()  
        # alg = algorithm(self.param)
        algorithm.__init__(self,self.param)
        self.pre_test_train(   
            neld_data = neld_data,  
            true_name = 'true_0', 
            dnn_mode = dnn_mode,
            model_type = model_type, 
            path_dir=path_dir,
            data_dir=path_dir,
            path_dict=path_dict, 
            model_sufix_show=model_sufix_show, 
            **dict_param 
        )

        param=self.param
        data_mode=self.data_mode
        modes,data_mode,dmode=self.modes,self.data_mode,self.dmode
        model_sufix_dic=self.model_sufix_dic
        mode_ids=modes[model_type][dnn_mode][path_dir]
        true_keys=self.true_keys 
        path_heads=self.path_heads  
        obj_org_path_dict =self.obj_org_path_dict
 

        path_dir=os.path.join(self.file_path_org, 'data','pinn_dir_data_all.txt') 
        pinn_dir_data_all=dmode.pinn_dir_data_all
        if os.path.exists(path_dir):
            pinn_dir_data_all=[mm for mm in list(np.loadtxt(path_dir,dtype=str)) ]

        for ixi,mode_id in enumerate(mode_ids):#data_mode.keys(): #
            data_mode[mode_id].update(dict(
                            list_features=configs[dnn_mode]['inten_pinn_index'],
                            base_features_list=configs[dnn_mode]['base_features_index'],     
                            # data_studied=data_studied,
                            )
                    )
            nhh=data_mode[mode_id]['model_sufix'] 
            model_sufix=data_mode[mode_id]['model_sufix'][0] 
            model_pred.__init__(self,   
                                neld_data=neld_data, 
                                model_sufix=model_sufix,
                                data_mode=data_mode[mode_id],   
                                pinn_dir_data_all=pinn_dir_data_all,
                                model_sufix_all=dmode.model_sufix_all, 
                                path_heads= path_heads,
                                model_type=model_type,
                                obj_org_path_dict=obj_org_path_dict ,
                                true_keys=true_keys,
                                model_sufix_dic=model_sufix_dic,
                                **param['model_pred']['param']
                                ) 
 
            self.get_model_opt_name(model_sufix=model_sufix,
                                        model_type=model_type)    


            self.get_neld_name(index=index)

 
        print('[[[[[[[[[[[[[[[[[[[99999999999999999]]]]]]]]]]]]]]]]]]]',self.path_train) 
        path_train=self.path_train 
        self.param_dic = {
            hh: {
                kk: True
                for kk in [
                    'get_pinn_features',
                    'get_wrap',
                    'get_scale',
                    'get_smooth',
                    'get_shaft_pred',
                    'get_neld_name',
                ]
            }
            for hh in ['tf_restart', 'get_info', 'data']
        }

        # Load saved param_dic if available
        # self.param_dic = loaded_dict.get('param_dic', self.param_dic)

        # Ensure data/get_neld_name has the correct structure
        if self.param_dic['data']['get_neld_name'] ==True:
            self.param_dic['data']['get_neld_name'] =  {
                                                        'dict_neld_path': 'current',
                                                        'drop_dic_name': None,
                                                    } 



        # if neld_data is not None: 
        #     neld_names=neld_names if not None else neld_data['neld_names']
        #     neld_namess=neld_namess if not None else neld_data['neld_namess']
        #     neld_path_inits=neld_path_inits if not None else neld_data['neld_path_inits'] 
        self.pari=dict(
                            neld_data=neld_data,
                            file_path_org=file_path_org,
                            neld_names=neld_names,
                            neld_namess=neld_namess, 
                            neld_path_inits=neld_path_inits,
                            data_studied=data_studied,   
                            model_sufix=model_sufix,
                            path_file=path_file,
                            path_file_sub=path_file_sub,
                            pinn_dir_data=pinn_dir_data,
                                pinn_dir_data_all=pinn_dir_data_all,
                                model_sufix_all=dmode.model_sufix_all, 
                            true_keys=true_keys,
                            model_type=model_type,
                            path_heads=path_heads, 
                            obj_org_path_dict=obj_org_path_dict,
                            model_sufix_dic=model_sufix_dic,
                            path_display_dic=path_display_dic, 
                            param_dic=self.param_dic,
                            file_path_data=file_path_data,
                            # path_file_dir=path_file_dir,
                 ) 
        self.index=index
        get_app_param.__init__(self,
                               dropdown_path_head_option=self.dropdown_path_head_option,
                               dropdown_model_suf_option=self.dropdown_model_suf_option,
                               dropdown_path_option=self.dropdown_path_option,
                               dropdown_true_keys_option=self.dropdown_true_keys_option,
                               )  

        self.get_model_opt_name(model_sufix=model_sufix,model_type=model_type ) 
        self.get_neld_name(data_studied=data_studied,index=0, 
                            **self.param_dic['data']['get_neld_name'],) 
        cxc=self.param_dic['data']['get_neld_name']['dict_neld_path'] 
        file_path=self.file_path=self.dict_neld['path'][cxc]['file_path'] 
        path_dir=self.model_sufix_dic['path_dir'] 


         
        neld_name=self.neld_name or neld_names[index]
        neld_namess=self.neld_namess    
        self.prevent_initial_call=prevent_initial_call
        # num = random.randint(100000000, 9999999999) 
        id_name_end=f'{model_type}_{neld_name}_{index}_{file_path}_{model_sufix}_{self.num}'
        id_name_end=safe_id(id_name_end)    
            
        spine_path = self.path_file[path_train['data_shaft_path']]   
        # spine_path = self.path_file[path_train['dest_spine_path']] 
        iou_path=os.path.join(spine_path , self.txt_spine_iou) 
        if os.path.exists(iou_path):
            iou_count = np.loadtxt(iou_path, dtype=float)
        else:
            iou_count=np.array([[-1,-1,0,0]])
        # print('iou_count',len(iou_count),id_name_end,neld_name) 
        iou_count=iou_count if iou_count.ndim==2 else np.array([[-1,-1,0,0]])
        
        self.model_test()
        self.more_param(id_name_end=id_name_end,
                        model_type=model_type,
                        model_sufix=model_sufix,
                        path_dir=path_dir,
                        neld_name=neld_name,)
        self.get_dropdown_cluster(id_name_end,iou_count[:,:2].astype(int))
        self.get_dropdown_index(id_name_end=id_name_end,
                                neld_names=neld_data['neld_names'],
                                index=index,) 
        # self.get_data(
        #             neld_data=self.neld_data,
        #             model_sufix=model_sufix,
        #             data_studied=self.data_studied,
        #             index=index,
        #             )  

        print('[[[[[[[[[[[[[[[[[[[self.tname,,,,]]]]]]]]]]]]]]]]]]]',self.tname) 
        self.prevent_initial_call=True
        get_layout.__init__(self,tname=self.tname) 
        class_data.__init__(self,)
        dropdown_callback.__init__(self,nam=self.tname)
    
    def get_gen_path(self, ):
        get_app_param.__init__(self,
                               dropdown_path_head_option=self.dropdown_path_head_option,
                               dropdown_model_suf_option=self.dropdown_model_suf_option,
                               dropdown_path_option=self.dropdown_path_option,
                               dropdown_true_keys_option=self.dropdown_true_keys_option,
                               ) 
        class_data.__init__(self)
        get_layout.__init__(self,tname=self.tname) 
