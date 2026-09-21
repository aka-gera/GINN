 

import os, sys ,dash  


path_dict={'run': 'mod.kalman.lorenz.kal_0.help_kal', 'fun': 'mod.kalman.lorenz.kal_0.help_fun', 'app': 'mod.kalman.lorenz.neld_fun_0.app_get_pinn_neld', 'doc': 'kalman.lorenz.lorenz_noise_t0'}
nam_gen='lorenz_0_we10_wn100_sp100_pt0'
file_path_data='os.getcwd()'
page_title='DSA-figs'
page_name='Figures'
type='d000'
page_dir_txt='/kalman/kalman-lorenz-data/kalman-figs'
p_name='None'
tname='lorenz'
lst={'par': 'param_app-store-lorenz', 'loa': 'page_app-load-lorenz', 'mal': 'container_app_param-lorenz', 'type': 'd000', 'share': 'fig', 'lst': ['nam_gen', 'path_head', 'action', 'dnn_mode', 'path_dir', 'root']}
params=None
paraws=None
drop_name='figure'
file_path_org=os.path.join(os.path.dirname(os.getcwd()),"apps","files","kalman","lorenz",) 
from  mod.kalman.lorenz.apps.app_param_test import app_param
from dash import callback  
 
neld_path_inits =['lorenz_0_we10_wn100_sp100_pt0_test']
data_studied = 'test'  
model_sufix = 'sd3_od3' 
path_train= None
path_file= None
path_file_sub=None
path_file_dir=os.path.join(os.path.join(os.path.dirname(os.getcwd()),"apps","files","kalman","lorenz",),*['data', 'lorenz_0_we10_wn100_sp100_pt0', 'path_files.pkl'])
pinn_dir_data='None'
neld_data=None
index=0
model_type='ekf_KAL__lorenz_0_we10_wn100_sp100_pt0'
obj_org_path_dict=None
model_sufix_dic=None
path_display=None
path_display_dic=None
mapp = app_param(
    file_path_org=file_path_org,
    file_path_data=file_path_data,
    model_sufix=model_sufix,
    path_train=path_train, 
    path_file=path_file,
    path_file_sub=path_file_sub,
    path_file_dir=path_file_dir,
    pinn_dir_data=pinn_dir_data,
    index=index, 
    data_studied=data_studied,  
    neld_data=neld_data,
    model_type=model_type,
    obj_org_path_dict=obj_org_path_dict,
    model_sufix_dic=model_sufix_dic,
    path_display=path_display,
    path_display_dic=path_display_dic,
    tname=tname,
    params=params,
    paraws=paraws,
    nam_gen=nam_gen,    
    path_dict=path_dict,
)
 
dash.register_page(
    __name__,
    title=page_title,
    name=page_name,
    path=page_dir_txt,
    order=4
)

def layout():
    return mapp.app_layout

@callback(
    mapp.Output,
    mapp.Input, 
    prevent_initial_call=mapp.prevent_initial_call
)
def update_output(*args):  
    mapp.update_params(args)
    return mapp.Get_output()
 

out, inp, st, prevent = mapp.param_upload_dropdown_all(drop_name)
@callback(*out,*inp,*st,prevent_initial_call=prevent)
def update_nam_gen_dropdown(n): 
    return mapp.update_nam_gen_dropdown(drop_name,path_file_dir=path_file_dir,
                                            file_path_org=file_path_org,)

 
out, inp, st, prevent = mapp.param_create_parameter_dropdowns_all(drop_name)
@callback(*out,*inp,*st,prevent_initial_call=prevent)
def create_parameter_dropdowns(store_data):  
    return [] if store_data is None else mapp.create_parameter_dropdowns(store_data,drop_name)



