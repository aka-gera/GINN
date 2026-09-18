

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

  
from neld_fun_0.get_path import assign_if_none,get_name,get_param,get_files,remove_directory,safe_id
from mod.dsa.apps.side_bar import  (get_text_dash_test,get_text_dash_dnn,
                                      get_text_dash_app,get_text_dash_main_pred,
                                      get_text_dash_main_train,get_text_dash_main_gen)
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
 


    def clean_path_dir(self,   
                        path_head_clean,
                    data_studied=None, 
                    disp_infos=None, 
                    neld_names=None,
                    neld_namess=None,
                    path_train=None,
                    ):   
        print(' path_head_clean ---- started')
        print('==========================================================')
        time_start = time.time()
        path_train=path_train or self.path_train
        disp_infos = disp_infos if disp_infos is not None else self.disp_infos    
        data_studied = data_studied if data_studied is not None else self.data_studied   
        neld_namess = neld_namess or self.neld_namess
        neld_names = neld_names if neld_names is not None else self.neld_names 
        for  index,neld_name in enumerate(neld_names):
            self.get_neld_name(data_studied=data_studied,index=index )  
            self.clean_neld_name(path_head_clean=path_head_clean)


          

    def get_dash_pages(self,  
                        data_studied=None, 
                        file_path_org=None, 
                        disp_infos=None,
                        dash_pages_path=None,
                        path_train=None,
                        model_type=None,
                        obj_org_path_dict=None,
                        model_sufix_dic=None,
                        param_dic=None,
                        ): 
        param_dic=param_dic if param_dic is not None else self.param_dic
        disp_infos = disp_infos or self.disp_infos 
        data_studied = data_studied or self.data_studied
        file_path_org = file_path_org or self.file_path_org  
        dash_pages_path = dash_pages_path or self.dash_pages_path
        path_train=path_train or self.path_train
        model_type=model_type or self.model_type
        obj_org_path_dict=obj_org_path_dict or self.obj_org_path_dict
        model_sufix_dic=model_sufix_dic or self.model_sufix_dic


        neld_names=self.neld_names 
        neld_path_inits=self.neld_path_inits
        model_sufix=self.model_sufix
  
        from neld_data_all import tname,path_dict,params,paraws,path_file_dir_text,file_path_org_text
          
        print('get_dash_pages') 
        print('--------------------------------')
        print(f"Model Type     : {model_sufix}") 
        print(f"Model Portion  : {self.pre_portion}") 
        print(f"Destination    : {path_train['dest_spine_path']}") 
        print('--------------------------------')
        for index,neld_name in enumerate(neld_names):   
            # self.get_dash_pages_name(index,data_studied)
            self.get_neld_name(data_studied=data_studied,
                                index=index,
                            **param_dic['data']['get_neld_name']
                                ) 
            print('[[[[[[[[]]]]]]]]',self.dash_pages_path, self.file_path_data ) 
            from pathlib import Path 
            dash_path_neld=self.dash_pages_path#os.path.join(self.dash_pages_path,self.data_studied,self.model_type, self.model_sufix,self.file_diff)  
            os.makedirs(dash_path_neld, exist_ok=True)
            os.makedirs(dash_path_neld, exist_ok=True)
            self.dash_pages_name=dash_pages_path=os.path.join(dash_path_neld,f'{self.neld_names[index]}.py')

            dash_pages_name=self.dash_pages_name
            neld_name ='Figures'# neld_names[index] 
            neld_path_init=neld_path_inits[index]
            # neld_namessi=neld_namess[index]
            neld_path_init=neld_path_inits[index]
            user_inputt=f'{index}' 
            # path_file_dir=os.path.join(self.dash_pages_path ,'path_files.pkl')
            # path_file_dir=os.path.join(self.dash_pages_path,self.data_studied,self.model_type,self.model_sufix ,self.file_diff ,'path_files.pkl')
            key=f'{self.model_type}_{self.model_sufix}_{self.path_dir}'
            # patty=os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(self.neld_path_org_new ))))
            # path_file_dir=os.path.join(self.path_file[key],'path_files.pkl')nam_gen
            path_file_dir=os.path.join(self.file_path_org,'data',os.path.basename(self.obj_org_path),'path_files.pkl')
            print('[[[[[[[[[[[[[[[[[[[[[[]]]dash_pages_path]]]]]]]]]]]]]]]]]]]',self.dash_pages_path,)
            print('[[[[[[[[[[[[[[[[[[[[[[]]]dash_pages_path]]]]]]]]]]]]]]]]]]]',self.file_diff,path_file_dir)


            # paramm=dict(file_path_org=self.file_path_org,
            #             file_path_path=self.file_path_path,
            #             )




            dictr=dict(
                        path_file_dir=path_file_dir,
                        path_train=path_train,
                        path_file=self.path_file,
                        path_file_sub=self.path_file_sub,
                        pinn_dir_data=self.pinn_dir_data,
                        file_path_org=self.file_path_org,
                        obj_org_path=self.obj_org_path,
                        neld_data=self.neld_data, 
                        obj_org_path_dict=obj_org_path_dict,
                        model_sufix_dic=model_sufix_dic,
                        path_display=self.path_display,
                        path_display_dic=self.path_display_dic, 
                        path_heads=self.path_heads,
                        neld_path_original_mm=self.neld_path_original_mm,
                        param_dic=param_dic,
                        params=params,
                        paraws=paraws,
                            # train_neld_param=self.data_mode['pinn'],
                            
                        )
            with open(path_file_dir, 'wb') as f:
                pickle.dump(dictr, f)







            page_name=self.gname#os.path.basename(self.file_path_org)
            dash_pages_path=os.path.dirname(self.dash_pages_path)
            dash_pages_path=os.path.join('/',dash_pages_path, f'{page_name}_data.py') 
            forbidden_endswith = ('test','train','2','gen','pred','figs' )

            page_dir_txt=os.path.join(f'{page_name}') 
            page_dir_gen=os.path.join(page_dir_txt,f'{self.gname}-{self.tname}-data' ).replace('_','-').replace('\\','/').lower() 
            get_text_dash_dnn(  
                                page_name=f'{self.tname.upper()}',
                                page_dir_txt=page_dir_gen,
                                dash_pages_path=dash_pages_path,
                                disp_infos=False,
                                path_train=None, 
                                path_file=None, 
                                pinn_dir_data=None,
                                neld_data=None,
                                index=None,
                                page_view='results', 
                                forbidden_endswith=forbidden_endswith,
                                # forbidden='-data-',
                                )






            if index>0:
                continue
            file_path=os.getcwd()
            file_path_org=file_path_org.replace('\\','/') 
            # path_file_dir=os.path.join(filedd,'path_files.pkl')
            # path_file_dir=os.path.join(filedd,self.data_studied,self.model_type,self.model_sufix ,self.file_diff ,'path_files.pkl')
            path_file_dir=os.path.relpath(path_file_dir,os.getcwd())
            path_file_dir=path_file_dir.replace('\\','/').split('/') 
            mm=[f'{mm}' for mm in path_file_dir]
            mmm=f'os.path.join(os.getcwd(),*{mm})'
            mkj=os.path.relpath(self.file_path_org, os.getcwd())
            mkj=[f'{mm}' for mm in mkj.replace('\\','/').split('/') ]
            mjjk=f'os.path.join(os.getcwd(),*{mkj})'
            from mod.dsa.neld_fun_0.main_0_help import dropdown_callback

            mdrp=dropdown_callback(nam=tname)
            lst=mdrp.callback_file['figure']

            page_dir_txt=os.path.join(self.data_studied,self.model_type  ) 
            page_dir_txt=os.path.join(page_dir_txt,f'{self.model_type}-data-' ).replace('_','-').replace('\\','/').lower()

            



            # dash_pages_path=os.path.join('/',self.dash_pages_path,  f'{self.model_type}_data.py')

            # # dash_pages_path = (
            # #     Path(self.dash_pages_path)
            # #     / self.data_studied
            # #     / f'{self.model_type}_data.py'
            # # )
            # dash_pages_path=f'{dash_pages_path}'
            # page_dir_txt=os.path.join(self.model_type  ) 
            # page_dir_txt=os.path.join(page_dir_txt,f'{self.model_type}-data-' ).replace('_','-').replace('\\','/').lower()  
            # page_dir_txt=page_dir_txt

            page_dir_txt=f'/DSA-figs'
            page_dir_txt=os.path.join(page_dir_gen,f'{self.gname}-figs').replace('_','-').replace('\\','/').lower() 
            dash_pages_path = str(Path(self.dash_pages_path) / "d000.py").replace('\\','/') 

            drop_name='figure'

  
            get_text_dash_test(type='d000',
                                page_title="DSA-figs",
                                page_dir_txt=page_dir_txt,
                                # page_dir_txt=f'/DSA-figs',
                                page_name='Figures',
                                user_input=4,
                                file_path_org=file_path_org_text ,
                                neld_path_inits=neld_path_init, 
                                data_studied=data_studied, 
                                model_sufix=model_sufix,
                                dash_pages_name=dash_pages_path,
                                # dash_pages_name=dash_pages_name,
                                disp_infos=disp_infos,
                                # path_file=self.path_file,
                                # path_file_sub=self.path_file_sub,
                                index=index,
                                model_type=model_type,
                                path_file_dir=path_file_dir_text, 
                                pinn_dir_data=self.pinn_dir_data,   
                                lst=lst,        
                                drop_name=drop_name,    
                                tname=tname,                      
                                # params=params,
                                # paraws=paraws,           
                                   ) 
            from pathlib import Path

            # forbidden_page = [
            #     str(Path(self.data_studied) / nn)
            #     for nn in self.path_heads
            # ]
            # model_sufix_all=self.model_sufix_dic['model_sufix_show'] 
            forbidden_page=[]
            # for nn in self.path_heads:
            #     for nnn in model_sufix_all:
            #         forbidden_page.append(f'/{os.path.join(self.data_studied,nn,nnn)}/')
            # forbidden_page=[f'/{os.path.join(self.data_studied,nn)}/' for nn in self.path_heads]
            # forbidden_page.extend(['/dsa','/dsa/test',])
            forbidden_page=[hh.replace('_','-').lower().replace('\\','/')  for hh in forbidden_page]
            forbidden_page.append(f'/{page_dir_gen}/') 
            pdir=os.path.dirname(self.dash_pages_path)  
            pdir=os.path.dirname(pdir)
            pdir=os.path.dirname(pdir)
            disp_first_page_end = ('data', )
            get_text_dash_app(  
                            dash_pages_path=os.path.join(pdir,'app.py'),
                            disp_infos=False, 
                            head_navbar=model_sufix_dic['head_navbar'],
                            forbidden_page = tuple(list(set(forbidden_page))),
                            disp_first_page_end=disp_first_page_end, 
                            # forbidden_endswith='-data-', 
                            )

            # print('[[[[[[[[[[[[[[[[[]]]]]]]]]]]]]]]]]',model_sufix_dic['head_navbar'])
            # print('[[[[[[[[[[[[[[[[[]]]]]]]]]]]]]]]]]',model_sufix_dic['head_navbar'])

            page_dir_txt=os.path.join(page_dir_gen,f'{self.gname}-2').replace('_','-').replace('\\','/').lower()  
            # dash_pages_path=os.path.join('/',self.dash_pages_path, f'Main_2.py')
            dash_pages_path = str(Path(self.dash_pages_path) / "Main_pred.py").replace('\\','/')
            get_text_dash_main_pred(
                    page_name="DSA",
                    page_dir_txt=page_dir_txt,
                    name='Prediction',
                    dash_pages_path=dash_pages_path,
                    disp_infos=False, 
                path_heads_show=model_sufix_dic['path_heads_show'],
                categories=model_sufix_dic['categories'],
                path_display=model_sufix_dic['path_display'],
                dnn_modes=model_sufix_dic['model_sufix_show'],
                drop_name='prediction',
                tname=tname,
                path_dict=path_dict,
                path_file_dir=path_file_dir_text, 
                file_path_org=file_path_org_text,
                )

            page_dir_txt=os.path.join(page_dir_gen,f'{self.gname}-GEN').replace('_','-').replace('\\','/').lower()  
            dash_pages_path = str(Path(self.dash_pages_path) / "Main_gen.py").replace('\\','/')
            get_text_dash_main_gen(
                    page_name="DSA",
                    page_dir_txt=page_dir_txt,
                    name='Generation',
                    dash_pages_path=dash_pages_path,
                    disp_infos=False, 
                path_heads_show=model_sufix_dic['path_heads_show'],
                categories=model_sufix_dic['categories'],
                path_display=model_sufix_dic['path_display'],
                dnn_modes=model_sufix_dic['model_sufix_show'],
                drop_name='generation',
                tname=tname,
                path_dict=path_dict,
                path_file_dir=path_file_dir_text, 
                file_path_org=file_path_org_text,
                )
            page_dir_txt=os.path.join(page_dir_gen,f'{self.gname}-TRAIN').replace('_','-').replace('\\','/').lower()  
            dash_pages_path = str(Path(self.dash_pages_path) / "Main_train.py").replace('\\','/')
            get_text_dash_main_train(
                    page_name="DSA",
                    page_dir_txt=page_dir_txt,
                    name='Training',
                    dash_pages_path=dash_pages_path,
                    disp_infos=False, 
                path_heads_show=model_sufix_dic['path_heads_show'],
                categories=model_sufix_dic['categories'],
                path_display=model_sufix_dic['path_display'],
                dnn_modes=model_sufix_dic['model_sufix_show'],
                drop_name='training',
                tname=tname,
                path_dict=path_dict,
                path_file_dir=path_file_dir_text, 
                file_path_org=file_path_org_text,
                )
            # page_name=self.model_sufix_dic['path_heads_dic'][self.model_type] #safe_id(self.neld_path_inits[index])
            # page_name= self.model_sufix_dic['path_heads_dic_sec'].get(self.model_type,self.model_type)
# 
            '''

            # page_name=self.model_sufix_dic['path_heads_dic'][self.model_type] #safe_id(self.neld_path_inits[index])
            page_name= self.model_sufix_dic['path_heads_dic_sec'].get(self.model_type,self.model_type)
            dash_pages_path=os.path.join('/',self.dash_pages_path,self.data_studied, f'{self.model_type}_data.py')

            # dash_pages_path = (
            #     Path(self.dash_pages_path)
            #     / self.data_studied
            #     / f'{self.model_type}_data.py'
            # )
            dash_pages_path=f'{dash_pages_path}'
            page_dir_txt=os.path.join(self.data_studied,self.model_type  ) 
            page_dir_txt=os.path.join(page_dir_txt,f'{self.model_type}-data-' ).replace('_','-').replace('\\','/').lower()  
            page_dir_txt=page_dir_txt
            get_text_dash_dnn(  
                                page_name=f'{page_name}',
                                page_dir_txt=page_dir_txt,
                                dash_pages_path=dash_pages_path,
                                disp_infos=False,
                                path_train=None, 
                                path_file=None, 
                                pinn_dir_data=None,
                                neld_data=None,
                                index=None,
                                page_view='results', 
                                # forbidden='-data-',
                                )
            id_name_end=self.neld_path_inits[index]
            page_name=self.model_sufix
            page_name=model_sufix_dic['model_sufix_dic'][self.model_sufix]
            dash_pages_path=os.path.join('/',self.dash_pages_path,self.data_studied,self.model_type ,f'{self.model_type}_data_{self.model_sufix}.py')
            # dash_pages_path = (
            #     Path(self.dash_pages_path)
            #     / self.data_studied
            #     / self.model_type
            #     / f'{self.model_type}_data_{self.model_sufix}.py'
            # )
            dash_pages_path=f'{dash_pages_path}' 
            page_dir_txt=os.path.join(self.data_studied,self.model_type ,self.model_sufix ,f'{self.model_type}-data-{self.model_sufix}'  ).replace('_','-').replace('\\','/').lower() 
            get_text_dash_dnn(  
                                page_name=f'{page_name}',
                                page_dir_txt=page_dir_txt,
                                dash_pages_path=dash_pages_path,
                                disp_infos=False,
                                path_train=None, 
                                path_file=None, 
                                pinn_dir_data=None,
                                neld_data=None,
                                index=None,
                                page_view='results', 
                                # forbidden='-data-',
                                )

            id_name_end=self.neld_path_inits[index]
            vhhv = self.file_diff.replace("\\", "_").replace("/", "_")
            dash_pages_path=os.path.join('/',self.dash_pages_path,self.data_studied,self.model_type ,self.model_sufix ,   f'{self.model_type}_data_{self.model_sufix}_{vhhv}.py')
            # dash_pages_path = (
            #     Path(self.dash_pages_path)
            #     / self.data_studied
            #     / self.model_type
            #     / self.model_sufix
            #     / f'{self.model_type}_data_{self.model_sufix}_{vhhv}.py'
            # )
            dash_pages_path=f'{dash_pages_path}' 
            page_dir_txt=os.path.join(self.data_studied,self.model_type ,self.model_sufix ,self.file_diff ).replace('_','-').replace('\\','/').lower() 
            # page_dir_txt=os.path.join(self.data_studied,self.model_type ,self.model_sufix ,id_name_end -{self.file_diff}).replace('_','-').replace('\\','/').lower() 
            # os.makedirs(os.path.dirname())
            print('[[[[[[[[[[[[[[[]]]]]]]]]]]]]]]', page_dir_txt,vhhv)
            page_name=f'{self.data_studied}__{os.path.basename(self.file_diff)}'#self.file_diff.split('/')[-1]
            get_text_dash_dnn(  
                                page_name=page_name,
                                # page_name=f'{id_name_end}',
                                page_dir_txt=page_dir_txt,
                                dash_pages_path=dash_pages_path,
                                disp_infos=False,
                                path_train=None, 
                                path_file=None, 
                                pinn_dir_data=None,
                                neld_data=None,
                                index=None,
                                page_view='visualization', 
                                )



'''













