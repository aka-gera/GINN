import os,sys 

 


path_0=os.getcwd() 

tname='lorenz'
gname='dsa'
import numpy as np
import sys,os
DTYPE='float32'  
from mod.dsa.neld_fun_0.main_0_help import    get_data_all ,get_dict_param 
# from neld_pinn_0 import run

file_path_org=os.getcwd()  #os.path.join(base_dir,'neld_analysis')# 
file_path_orgb = os.path.basename(os.getcwd())
file_path_datab=os.path.join(file_path_orgb,'files',gname,tname,'data_initial')
file_path_org=os.path.join(file_path_org ,'files',gname,tname)

file_path_data=os.path.join(file_path_org,'data_initial')
os.makedirs(file_path_data,exist_ok=True)
# file_path_data=os.path.dirname(file_path_org)    # Change this directory to the path dataset   
neld_names_chld={} 
action= 'resize'  
   
from mod.dsa.kal_0.help_fun import get_configs,app_dict_param
process_std,measurement_std=0.1,0.1
d_nam,state_dim,obs_dim,window,step,weight,step_test='lorenz_0',3,3,100,100,10,100
dnn_mode=f'sd{state_dim}_od{obs_dim}'
nam_gens={
    'name':dict(
            train=f'{d_nam}_we{weight}_wn{window}_sp{step}_pt{process_std}',
            test=f'{d_nam}_we{weight}_wn{window}_sp{step}_pt{process_std}',
        ) ,
    'step':dict(train=None,
                test=step_test,)
}
nbr_train_sample,nbr_test_sample=10,3

param_flow=dict(
                nbr_train_sample=nbr_train_sample,
                nbr_test_sample=nbr_test_sample,
                step=step,
                window=window,
                process_std_init=2,
                # process_std_true=process_std_true,
                # measurement_std_true=measurement_std_true,
                process_std=process_std,
                measurement_std=measurement_std,
                weight=weight/100,
                dt=0.001,
                )

param_linear=dict(
    
)
configs,dyna,dnn_modes=get_configs(d_nam=d_nam, **param_flow)

path_dict={'run':f'mod.{gname}.kal_0.help_kal'}
path_dict.update({'fun':f'mod.{gname}.kal_0.help_fun'})
path_dict.update({'app':f'mod.{gname}.kal_0.app_get_pinn_neld'})
path_dict.update({'doc':f'mod.{gname}.lorenz_noise_t1'})

action='train'
nam_gen=nam_gens['name'][action]
nam=f'{nam_gen}_{action}' 
path_heads_show=[f'rnn_KAL__{nam_gen}',f'tfm_KAL__{nam_gen}',f'ekf_KAL__{nam_gen}',f'jos_KAL__{nam_gen}'] 
n_obj=7
obj_list=np.arange(nbr_train_sample)
neld_names = [f'd{str(i).zfill(3)}' for i in obj_list] 
neld_namess = [[f'{de}_',de,f'{de}'] for de in neld_names]
neld_last = [f'{de}_'  for de in neld_names]
neld_first= [ f'{de}' for de in neld_names]  
 
obj_org_path= os.path.join(file_path_data,nam_gen)
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

obj_org_path= os.path.join(file_path_data,nam_gen )
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
            'dsa':'dsa'
            # 'Langevin':'lang',
            # 'Figures': 'figs',
},
category=['dsa', ]
)



dict_param=get_dict_param(nam=nam,
                    n_step = 0,    
                    dnn_modes=dnn_modes, 
                    configs=configs,
                    file_path_org=file_path_org,
                    path_heads_show=path_heads_show,
                    file_path_data=file_path_data,
                    path_dirs_show=['loss','oloss'],
                    navbar=navbar,
                    )




action='train'
nam=f'{nam_gen}_{action}'
namt=f'{nam_gen}_test' 


paraw={mm:[] for mm in ['dict_param','gdas','path_dict','nam_gens']}
paraw['dict_param']=dict_param
paraw['gdas']=gdas
paraw['path_dict']=path_dict
paraw['nam_gens']=nam_gens 


params={mm:[] for mm in ['dnn_modes','nam_gen','dict_param','path_heads_show','path_dirs_show','nam_gen_show','nam',
                    'file_path_org',
                    'file_path_data',
                    'param_flows']}

params['dnn_modes'].extend(dnn_modes)
params['param_flows'].append({**param_flow, **dyna[d_nam]['param']}) 
params['nam'].append(nam)

