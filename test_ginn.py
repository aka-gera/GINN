 
import sys,os
DTYPE='float32'   


from mod.dsa.neld_fun_0.main_0 import  app_run_param,algorithm,algorithm_param   
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

d_nam,state_dim,obs_dim ='harmonic',3,3 
dnn_mode=dict_param['dnn_modes'][0]


path_heads_show=dict_param['path_heads_show']
model_type=path_heads_show[0] 

path_dirs_show=dict_param['path_dirs_show']
path_dir=path_dirs_show[0]
path_d=path_dir.split('_') 
print('[[[[[[]]]]]]',path_dirs_show,configs[dnn_mode])
nam=f'{nam_gen}_{action}' 
dict_param['path_heads_show']=params['path_heads_show']
# # dict_param['obj_org_show']=params['obj_org_show']
param = algorithm_param( **dict_param,)

param['model-pred']['param']['weight']=int(path_d[-1])

neld_data = gdas.part(nam)  
# neld_data=neld_data_old
mapp = app_run_param(param)
param=mapp.emerge_param()  
# param['dnn_modes']['param']=dnn_modes
print('[[[[[[]]]]]]',neld_data)
param['model-pred']['tf'] =True    #False   #  True/False: choose whether to predict shafts/smods
param['skl_shaft_pred']['tf']  =False  #
param['dash_pages']['tf'] =True # False   #True          #  True/False: generate Dash pages
param['roc']['tf']=False   #True # 


# param['model-pred']['tf'] =False   # True    # True/False: choose whether to predict shafts/smods
# param['skl_shaft_pred']['tf']  =False   #True  #
# param['dash_pages']['tf'] =True # False   #True          #  True/False: generate Dash pages



print('[[[[[[]]]]]]',param['model_pred']['param'])

# param['Spine-Shaft Segm']['tf']=True 

 
param['Resizing']['param']['target_number_of_triangles_faction']=150000
for val in ['get_pinn_features','get_skeleton','get_wrap','get_smooth']:
    param['model_pred']['param']['param_dic']['tf_restart'][val]=False   #True   # 
param['model_pred']['param']['param_dic']['tf_restart']['get_shaft_pred']=False   #True   #  
param['Smooth']['param']['n_step']=20 
 
  
param['Smooth']['param']['dt']=1e-7
param['Smooth']['param']['method']='taubin' #'willmore'   #
param['Skeleton']['param']['path']="entry"     #'old'   #  
param['Skeleton']['param']['dict_mesh_to_skeleton_finder']['interval_target_number_of_triangles']=[200000]

param['clean_path_dir']['tf'] = False          # True/False: delete computed data
param['intensity_rhs']['tf']=False #  True  #  
param['Resizing']['tf']= False # True  #  
param['Smooth']['tf'] =False   # True   # False   # True/False: choose whether to smooth the data
param['Skeleton']['tf']=False  #  
param['rhs']['tf']= False #True  #  
param['model_shap']['tf']=False # True  #  

param['Morphologic Param']['tf'] = False #True   # True   #  False   # True/False: choose whether to perform head/neck segmentation
param['iou']['tf'] = False   # True       #False   #  True/False: choose whether to compute IoU
param['roc']['tf'] = False # True       # False   # True/False: choose whether to compute IoU
param['graph_center']['tf'] =False  # True   # True   #False   #   True/False: compute central axis
param['cylinder_heatmap']['tf'] =False   #True   #  False   # False   # True/False: generate cylindrical heatmap

param['riplet']['tf']=False
# param['iou']['tf']=True
# param['roc']['tf']=True
# param['graph_center']['tf']=True
param['cylinder_heatmap']['tf']=False
param['annotations']['tf'] =False  



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