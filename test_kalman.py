 
import sys,os
DTYPE='float32'   


from mod.kalman.lorenz.neld_fun_0.main_0 import  app_run_param,algorithm,algorithm_param 
path_dict={'run':'kalman.kal_0.help_kal'} 

from neld_data_all import paraws,params
# name_num=0
# dict_param,gdas,path_dict =[paraws[name_num][mm] for mm in ['dict_param','gdas','path_dict']]  
# action='test'
# nam_gen=paraws[name_num]['nam_gens']['name'][action]
# nam=f'{nam_gen}_{action}' 

# obj_org_path=gdas.part(nam)['obj_org_path']
# neld_data_old=gdas.part(nam) 

name_num=0
dict_param,gdas,path_dict =[paraws[name_num][mm] for mm in ['dict_param','gdas','path_dict']] 
configs=dict_param['configs']
  
action='train'
action='test'
nam_gen=paraws[name_num]['nam_gens']['name'][action]
step_test=paraws[name_num]['nam_gens']['step'][action]

nam=f'{nam_gen}_{action}' 

d_nam,state_dim,obs_dim ='lorenz_0',3,3 
dnn_mode=f'sd{state_dim}_od{obs_dim}' 


path_heads_show=dict_param['path_heads_show']
model_type=path_heads_show[0]
model_type=path_heads_show[1]
model_type=path_heads_show[3]
model_type=path_heads_show[2]

path_dirs_show=dict_param['path_dirs_show']
path_dir=path_dirs_show[0]



  
nam=f'{nam_gen}_{action}' 
dict_param['path_heads_show']=params['path_heads_show']
# # dict_param['obj_org_show']=params['obj_org_show']
param = algorithm_param( **dict_param,)


neld_data = gdas.part(nam)  
# neld_data=neld_data_old
mapp = app_run_param(param)
param=mapp.emerge_param()  
# param['dnn_modes']['param']=dnn_modes
print('[[[[[[]]]]]]',neld_data)
param['model-pred']['tf'] =True    #False   #  True/False: choose whether to predict hmods/smods
param['skl_hmod_pred']['tf']  =False   #True  #
param['dash_pages']['tf'] =True # False   #True          #  True/False: generate Dash pages
param['roc']['tf']=False   #True # 


# param['model-pred']['tf'] =False   # True    # True/False: choose whether to predict hmods/smods
# param['skl_hmod_pred']['tf']  =False   #True  #
# param['dash_pages']['tf'] =True # False   #True          #  True/False: generate Dash pages
 

alg = algorithm(param)
self=alg.test(   
    neld_data = neld_data,  
    true_name = 'true_0', 
    dnn_mode = dnn_mode,
    model_type = model_type, 
    path_dir=path_dir,
    data_dir=path_dir,
    path_dict=path_dict, 
    **dict_param 
)


'''
get_neld_data(**param_flow,self=self,
                  dt_r_0=3,
                  dt_r1=1,                  
                fmt='%.4f',
                Nmarkov=Nmarkov,
                # Nsample=50,
                # tf_train=True,
    )
'''