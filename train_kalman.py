 
#  # 
import sys,os 
DTYPE='float32'   
import time 
import random#
import numpy as np

  
from mod.neld_fun_0.main_0 import  app_run_param,algorithm,algorithm_param 
path_dict={'run':'kalman.kal_0.help_kal'} 

from neld_data_all import paraws,params
name_num=0
dict_param,gdas,path_dict =[paraws[name_num][mm] for mm in ['dict_param','gdas','path_dict']] 
configs=dict_param['configs']
  
action='test'
action='train'
nam_gen=paraws[name_num]['nam_gens']['name'][action]
step_test=paraws[name_num]['nam_gens']['step'][action]

nam=f'{nam_gen}_{action}' 

d_nam,state_dim,obs_dim ='lorenz_0',3,3 
dnn_mode=f'sd{state_dim}_od{obs_dim}' 


path_heads_show=dict_param['path_heads_show']
model_type=path_heads_show[0]
model_type=path_heads_show[1]

path_dirs_show=dict_param['path_dirs_show']
path_dir=path_dirs_show[0]

 

  
path_display = ['dest_hmod_path', ]

  
dict_param['model_type_data']=model_type 
dict_param['path_heads_show']=params['path_heads_show']
param = algorithm_param(**dict_param)



neld_data = gdas.part(nam)  
mapp = app_run_param(param)
param=mapp.emerge_param()
param['Smooth']['tf'] = False   #  False   #False   # False   # True/False: choose whether to smooth the data
param['annotations']['tf'] =  False   # True   # True/False: choose whether to generate annotations for training, accuracy, and recall 
param['get_training']['tf'] = True  # True/False: enable training data generation
param['clean_path_dir']['tf'] = False          # True/False: delete computed data
mj = [x / 1000 for x in range(1, 1000, 200)]  
param['get_training']['param']['weight']=[[a, b] for a, b in zip(mj, mj[::-1])]
param['model_pred']['param']['param_dic']['tf_restart']['get_pinn_features']= False   #True   #

entry_names=[None,]
ls =[1,2,3,4,] 
  
 
param['get_training']['param']['ls']=ls
param['get_training']['param']['itime']=100000 
alg = algorithm(param)
alg.train(
    neld_data = neld_data,  
    true_name = 'true_0', 
    dnn_mode = dnn_mode,
    model_type = model_type,  
    path_display = path_display, 
    entry_names=entry_names,
    path_dir=path_dir,
    path_heads_show=path_heads_show, 
    configs=configs,
    path_dict=path_dict,
    path_dirs_show=path_dirs_show,
)
