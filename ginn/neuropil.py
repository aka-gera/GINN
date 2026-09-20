import os,sys 

 


path_0=os.getcwd() 

general_nam='Geometry-Informed Neural Network'
gname='dsa'
tname='ginn'
nam_gen='neuropil' 

import numpy as np
import sys,os
DTYPE='float32'  
from mod.dsa.neld_fun_0.main_0_help import    get_data_all ,get_dict_param 
# from neld_pinn_0 import run


file_path_org=os.getcwd() 
file_path_data=os.getcwd()  #os.path.join(base_dir,'neld_analysis')#  
file_path_org=os.path.join(os.path.dirname(os.getcwd()),'apps')#
path_file_dir_path=['data',nam_gen,'path_files.pkl'] 
file_path_org_text=f'os.path.join(os.path.dirname(os.getcwd()),"apps","files","{gname}","{tname}",)'
path_file_dir_text=f'os.path.join({file_path_org_text},*{path_file_dir_path})'

file_path_orgb = os.path.basename(file_path_org)
file_path_datab=os.path.join(file_path_orgb,'files',gname,tname,'data_initial')
file_path_org=os.path.join(file_path_org ,'files',gname,tname)
   # Change this directory to the path dataset   
neld_names_chld={} 
action= 'resize'  
   
from mod.dsa.neld_fun_0.help_fun import get_configs,app_dict_param
process_std,measurement_std=0.1,0.1
d_nam,state_dim,obs_dim,window,step,weight,step_test=nam_gen,3,3,100,100,3,100
dnn_mode=f'DNN_{obs_dim}'
nam_gens={
    'name':dict(
            train=f'{d_nam}',
            test=f'{d_nam}',
        ) ,
    'step':dict(train=None,
                test=step_test,)
}
nbr_train_sample,nbr_test_sample=10,3

param_flow=dict(
                # nbr_train_sample=nbr_train_sample,
                # nbr_test_sample=nbr_test_sample,
                # step=step,
                # window=window,
                # process_std_init=2, 
                # # process_std_true=process_std_true,
                # # measurement_std_true=measurement_std_true,
                # process_std=process_std,
                # measurement_std=measurement_std,
                weight=weight,
                # dt=0.001,
                # rho=0.1,
                # sigma=0.01,
                # gamma=1,
                # beta=2, 
                )
param_flow={}
action='train'
nam_gen=nam_gens['name'][action]
nam=f'{nam_gen}_{action}' 
path_heads_show=['dnn_GINN__SM00000_LOC_AUG',]
# path_heads_show=[f'rnn_KAL__{nam_gen}',f'tfm_KAL__{nam_gen}',f'ekf_KAL__{nam_gen}',f'jos_KAL__{nam_gen}']

param={mm:{} for mm in ['action','file_path_org','file_path_data','path_heads_show','path_dirs_show','dyna param']}

 
path_dirs_show=[f'loss_{weight}',f'loss_16',]
param["action"]["param"]=action
param['file_path_org']['param']=file_path_org
param['file_path_data']['param']=file_path_data
param['path_heads_show']['param']=path_heads_show
param['path_dirs_show']['param']=path_dirs_show

param['neld_names_all']['param']=neld_namess=  [f'd{str(i).zfill(3)}' for i in range(26)]
param['dyna param']['param']=dict(
                                  model_type_index=0,
                                  riplet=2,
                                  dnn_mode_index=0,
                                  path_dir_index=1,) 
# param['dyna param']['param']=param_flow
navbar=dict(
head_navbar={
            "Kalman": "kalman",
            'DSA':'dsa'
            # 'Langevin':'lang',
            # 'Figures': 'figs',
},
category=['dsa', ]
)

 
dict_param=app_dict_param(param=param,nam_gen=d_nam,navbar=navbar)
param_linear=dict(
    
)
# configs,dyna,dnn_modes=get_configs(d_nam=d_nam,dyna=dyna, **param_flow)
configs=dict_param['configs']

# llt=dict(
# list_features=configs[dnn_mode]['inten_pinn_index'],
# base_features_list=configs[dnn_mode]['base_features_index'],     
# data_studied=action,
# )
# dict_param.update(llt)

dnn_modes=dict_param['dnn_modes']
dyna=dict_param['dyna']
 
 
path_dict={'run':'mod.dsa.neld_fun_0.help_kal'}
path_dict.update({'fun':'mod.dsa.neld_fun_0.help_fun'})
path_dict.update({'app':'mod.dsa.neld_fun_0.app_get_pinn_neld'})
path_dict.update({'doc':f'ginn.{nam_gen}'})




 
obj_list=[1,2]
# obj_list=[62,63]
# obj_list=[0, 1, 2, 3, 4, 7, 8, 9, 10, 11, 12, 14, 16, 19, 20, 21, 22, 30, 31, 32, 59, 62, 63, 40, 113, ] 

neld_names = [f'd{str(i).zfill(3)}' for i in obj_list]  
neld_namess = [f'd{str(i).zfill(3)}' for i in range(len(neld_names))] 
dend_last = [f'{de}_'  for de in neld_names]
dend_first= [ f'{de}' for de in neld_names]  

 
obj_org_path= os.path.join(file_path_org,'data_initial',nam_gen,)
neld_path_inits=[nam for _ in range(len(neld_names))]
neld_names_chld[nam]= dict(  
                neld_namess=neld_namess ,
                neld_names=neld_names , 
                obj_org_path=obj_org_path,
                neld_last=dend_last ,
                neld_first=dend_first ,
                neld_path_inits=neld_path_inits,  
                data_studied=action,
                name_spine_id='sp',
                name_head_id='hsp',
                name_neck_id='nsp',
                name_shaft_id='shsp',
                )  
# gdas=get_data_all(names_dic=neld_names_chld,
#                 #   neld_data=gdas.neld_data,
#                   file_path_data=file_path_data,
#                   cpath=[nam_gen, action],)  


 
# nam_gen=f'neld_{flow}_{nPart}_{force}'
action='test'
nam_gen=nam_gens['name'][action]
nam=f'{nam_gen}_{action}'

# path_heads_show.extend([f'rnn_KAL___{nam_gen}',f'tfm_KAL___{nam_gen}'])
 
neld_names = [f'd{str(i).zfill(3)}' for i in obj_list] 
dend_namess = [[f'{de}_',de,f'{de}'] for de in neld_names]
dend_last = [f'{de}_'  for de in neld_names]
dend_first= [ f'{de}' for de in neld_names]  

  
neld_path_inits=[nam for _ in range(len(neld_names))]
neld_names_chld[nam]= dict(  
                # neld_namess=neld_namess ,
                neld_names=neld_names , 
                obj_org_path=obj_org_path,
                neld_last=dend_last ,
                neld_first=dend_first ,
                neld_path_inits=neld_path_inits,  
                data_studied=action,
                )  
 
gdas=get_data_all(names_dic=neld_names_chld,
                #   neld_data=gdas.neld_data,
                  file_path_data=file_path_data,
                  cpath=[nam_gen,  action],)  

weight=1
path_dir_nam=f'dice'
path_dir_nam=f'loss'
path_dir=f'{path_dir_nam}_we_{weight}' 
data_dir=path_dir
'''
dict_param=get_dict_param(nam=nam,
                    n_step = 0,    
                    dnn_modes=dnn_modes, 
                    configs=configs,
                    file_path_org=file_path_org,
                    path_heads_show=path_heads_show,
                    file_path_data=file_path_data,
                    path_dirs_show=['loss','oloss'],
                    )'''




action='train'
nam=f'{nam_gen}_{action}'
namt=f'{nam_gen}_test' 



github_link='https://github.com/aka-gera/Backward-Equation-Informed-Neural-Network-BEINN-/tree/main'



