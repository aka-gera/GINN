import os,sys 

 


path_0=os.getcwd() 

 

import numpy as np
import sys,os
DTYPE='float32'  
from mod.kalman.lorenz.neld_fun_0.main_0_help import    get_data_all ,get_dict_param 
# from neld_pinn_0 import run



process_std_true,measurement_std_true=0,0
d_nam,state_dim,obs_dim,window,step,weight,step_test='lorenz_0',3,3,100,100,10,100
dnn_mode=f'sd{state_dim}_od{obs_dim}'
nam_gens={
    'name':dict(
            train=f'{d_nam}_we{weight}_wn{window}_sp{step}_pt{process_std_true}',
            test=f'{d_nam}_we{weight}_wn{window}_sp{step}_pt{process_std_true}',
        ) ,
    'step':dict(train=None,
                test=step_test,)
}

action='train'
nam_gen=nam_gens['name'][action]



 

github_link='https://github.com/aka-gera/Backward-Equation-Informed-Neural-Network-BEINN-/tree/main'



general_nam='Kalman-Informed Neural Network'
gname='kalman'
tname='lorenz'






file_path_org=os.getcwd() 
file_path_data=os.getcwd()  #os.path.join(base_dir,'neld_analysis')#  
file_path_data_text =f'os.getcwd()'
file_path_org=os.path.join(os.path.dirname(os.getcwd()),'apps')#
path_file_dir_path=['data',nam_gen,'path_files.pkl'] 
file_path_org_text=f'os.path.join(os.path.dirname(os.getcwd()),"apps","files","{gname}","{tname}",)'
path_file_dir_text=f'os.path.join({file_path_org_text},*{path_file_dir_path})'

file_path_orgb = os.path.basename(file_path_org)
file_path_datab=os.path.join(file_path_orgb,'files',gname,tname,'data_initial')
file_path_org=os.path.join(file_path_org ,'files',gname,tname)













os.makedirs(file_path_data,exist_ok=True)
# file_path_data=os.path.dirname(file_path_org)    # Change this directory to the path dataset   
neld_names_chld={} 
action= 'resize'  
   
from mod.kalman.lorenz.kal_0.help_fun import get_configs,app_lorenz_dict_param

nbr_train_sample,nbr_test_sample=10,3

param_flow=dict(
                nbr_train_sample=nbr_train_sample,
                nbr_test_sample=nbr_test_sample,
                step=step,
                window=window,
                process_std_init=2,
                process_std_true=process_std_true,
                measurement_std_true=measurement_std_true,
                process_std=0.1,
                measurement_std=0.1,
                weight=weight/100,
                dt=0.001,
                )

param_linear=dict(
    
)
configs,dyna,dnn_modes=get_configs(d_nam=d_nam, **param_flow)
path_dirs_show=['loss','oloss']
nam=f'{nam_gen}_{action}' 
path_heads_show=[f'rnn_KAL__{nam_gen}',f'tfm_KAL__{nam_gen}',f'ekf_KAL__{nam_gen}',f'jos_KAL__{nam_gen}'] 
n_obj=7
obj_list=np.arange(nbr_test_sample)
neld_names = [f'd{str(i).zfill(3)}' for i in obj_list] 
neld_namess = [[f'{de}_',de,f'{de}'] for de in neld_names]
neld_last = [f'{de}_'  for de in neld_names]
neld_first= [ f'{de}' for de in neld_names]  
 
obj_org_path= os.path.join(file_path_org,'data_initial',nam_gen)
neld_path_inits=[nam for _ in range(len(neld_names))]
neld_names_chld[nam]= dict(  
                # neld_namess=neld_namess ,
                neld_names=neld_names , 
                obj_org_path=obj_org_path,
                neld_last=neld_last ,
                neld_first=neld_first ,
                neld_path_inits=neld_path_inits,  
                data_studied=action,
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
obj_list=np.arange(nbr_test_sample)
neld_names = [f'd{str(i).zfill(3)}' for i in obj_list] 
neld_namess = [[f'{de}_',de,f'{de}'] for de in neld_names]
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
                
                )  
 
gdas=get_data_all(names_dic=neld_names_chld,
                #   neld_data=gdas.neld_data,
                  file_path_data=file_path_data,
                  cpath=[nam_gen, action],)  

weight=1
path_dir_nam=f'dice'
path_dir_nam=f'loss'
path_dir=f'{path_dir_nam}_we_{weight}' 
data_dir=path_dir

navbar=dict(
head_navbar={
            "Kalman": "kalman",
            'DSA':'dsa'
            # 'Langevin':'lang',
            # 'Figures': 'figs',
},
category=['dsa', ]
)
param_flows={**param_flow, **dyna[d_nam]['param']}

dict_param=get_dict_param(nam=nam,
                    n_step = 0,    
                    dnn_modes=dnn_modes, 
                    configs=configs,
                    file_path_org=file_path_org,
                    path_heads_show=path_heads_show,
                    file_path_data=file_path_data,
                    path_dirs_show=path_dirs_show,
                        navbar=navbar,
                        param_flows=param_flows,
                    )
 


path_dict={'run':'mod.kalman.lorenz.kal_0.help_kal'}
path_dict.update({'fun':'mod.kalman.lorenz.kal_0.help_fun'})
path_dict.update({'app':'mod.kalman.lorenz.neld_fun_0.app_get_pinn_neld'})
path_dict.update({'doc':f'kalman.lorenz.lorenz_noise_t0'})


