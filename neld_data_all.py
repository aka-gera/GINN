import os,sys 




nams=[]


path_heads_show=[]
paraws={}
params={mm:[] for mm in ['dnn_modes','nam_gen','dict_param','path_heads_show','path_dirs_show','nam_gen_show','nam',
                    'file_path_org',
                    'file_path_data',
                    'param_flows']}
params['gdas']={}
params['dict_param']={}
params['nam_gens']={}
paraw={mm:[] for mm in ['dict_param','gdas']}

 

# from lorenz_noise_t0 import gdas,param_flow,dict_param,dnn_modes,nam,path_dir,nam_gens,dyna,d_nam,file_path_orgb,file_path_datab,file_path_org,file_path_data,nam_gen,path_dict,tname,obj_org_path,general_nam
# from harmonic import gdas,param_flow,dict_param,dnn_modes,nam,path_dir,nam_gens,dyna,d_nam,file_path_orgb,file_path_datab,file_path_org,file_path_data,nam_gen,path_dict,tname,obj_org_path,general_nam
# from harmoniclang import gdas,param_flow,dict_param,dnn_modes,nam,path_dir,nam_gens,dyna,d_nam,file_path_orgb,file_path_datab,file_path_org,file_path_data,nam_gen,path_dict,tname,obj_org_path,general_nam
# from harmonic import gdas,param_flow,dict_param,dnn_modes,nam,path_dir,nam_gens,dyna,d_nam,file_path_orgb,file_path_datab,file_path_org,file_path_data,nam_gen,path_dict,tname,obj_org_path
# from mod.dsa.lorenz_noise_t1 import gdas,param_flow,dict_param,dnn_modes,nam,path_dir,nam_gens,dyna,d_nam,file_path_orgb,file_path_datab,file_path_org,file_path_data,nam_gen,path_dict,tname,obj_org_path
# from meshes import gdas,param_flow,dict_param,dnn_modes,nam,path_dir,nam_gens,dyna,d_nam,file_path_orgb,file_path_datab,file_path_org,file_path_data,nam_gen,path_dict,tname,obj_org_path,general_nam
 
# from loc import gdas,param_flow,dict_param,dnn_modes,nam,path_dir,nam_gens,dyna,d_nam,file_path_orgb,file_path_datab,file_path_org,file_path_data,nam_gen,path_dict,tname,obj_org_path,general_nam






# gen_name=['gdas','param_flow','dict_param','dnn_modes',
#           'nam','path_dir','nam_gens','dyna','d_nam','file_path_orgb',
#           file_path_datab,file_path_org,file_path_data,nam_gen,path_dict,tname,obj_org_path,general_nam

from ginn.meshes import (gdas,param_flow,dict_param,dnn_modes,nam,path_dir,
                 nam_gens,dyna,d_nam,file_path_orgb,file_path_datab,
                 file_path_org,file_path_data,nam_gen,path_dict,tname,
                 obj_org_path,general_nam,nam_gen,
                 path_file_dir_text,path_file_dir_path,file_path_org_text,file_path_data_text,
                 github_link,)


from kalman.lorenz.lorenz_noise_t0 import (gdas,param_flow,dict_param,dnn_modes,nam,path_dir,
                 nam_gens,dyna,d_nam,file_path_orgb,file_path_datab,
                 file_path_org,file_path_data,nam_gen,path_dict,tname,
                 obj_org_path,general_nam,nam_gen,
                 path_file_dir_text,path_file_dir_path,file_path_org_text,file_path_data_text,
                 github_link,)

navbar=dict(
head_navbar={
            "Kalman": "kalman",
            'DSA':'dsa'
            # 'Langevin':'lang',
            # 'Figures': 'figs',
},
category=['dsa', ]
)
 
dict_param['neld_names_all'] = [f'd{str(i).zfill(3)}' for i in range(20)] 
dict_param['path_dirs_show']=[f'loss_{mm}' for mm in [3,6,16]]

dict_param['navbar']=navbar

name_num=0
action='train'
nam_gen=nam_gens['name'][action]
nam=f'{nam_gen}_{action}'
namt=f'{nam_gen}_test'
# dict_param['path_heads_show']=[f'rnn_KAL___{nam}',f'tfm_KAL___{nam}'] 
for val in nam_gens['name'].values():
    if val not in params['nam_gen_show']:
        params['nam_gen_show'].append(val)

params['dnn_modes'].extend(dnn_modes)
# params['param_flows'].append({**param_flow, **dyna[d_nam]['param']})
params['path_heads_show'].extend(dict_param['path_heads_show'])
params['path_dirs_show'].extend(dict_param['path_dirs_show'])
params['nam'].append(nam)
params['gdas'][nam_gen]=gdas
params['dict_param'][nam_gen]=dict_param
params['nam_gens'][nam_gen]=nam_gens

paraw={mm:[] for mm in ['dict_param','gdas','path_dict','nam_gens']}
paraw['dict_param']=dict_param
paraw['gdas']=gdas
paraw['path_dict']=path_dict
paraw['nam_gens']=nam_gens

paraws[name_num]=paraw


d_nam,state_dim,obs_dim ='lorenz_0',3,3 
dnn_mode=f'sd{state_dim}_od{obs_dim}' 
 
'''


from lorenz_noise_0 import gdas,param_flow,dict_param,dnn_modes,nam,path_dir,nam_gens,dyna,d_nam,step_test
  
name_num=1
path_dict={'run':'mod.kalman.kal_0.help_kal'}
action='train'
nam_gen=nam_gens['name'][action]
nam=f'{nam_gen}_{action}'
namt=f'{nam_gen}_test'
# dict_param['path_heads_show']=[f'rnn_KAL___{nam}',f'tfm_KAL___{nam}'] 
for val in nam_gens['name'].values():
    if val not in params['nam_gen_show']:
        params['nam_gen_show'].append(val)
 

 
params['dnn_modes'].extend(dnn_modes) 
params['param_flows'].append({**param_flow, **dyna[d_nam]['param'],})
params['path_heads_show'].extend(dict_param['path_heads_show'])
params['nam'].append(nam)
params['gdas'][nam_gen]=gdas
params['dict_param'][nam_gen]=dict_param
params['nam_gens'][nam_gen]=nam_gens

paraw={mm:[] for mm in ['dict_param','gdas','path_dict','nam_gens',]}
paraw['dict_param']=dict_param
paraw['gdas']=gdas
paraw['path_dict']=path_dict
paraw['nam_gens']=nam_gens 

paraws[name_num]=paraw


from harmonic import dnn_mode,obj_org_path


'''
params['model_sufix']=dnn_mode
params['data_studied']='test'
params['obj_org_path']=obj_org_path

params['file_path_org']=dict_param['file_path_org']
params['file_path_data']=dict_param['file_path_data']



general_nam_main= f'## {general_nam}'
 





