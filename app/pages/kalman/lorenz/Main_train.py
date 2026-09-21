 
from dash import callback
import sys,os,dash
 
sys.path.append(os.path.abspath(os.getcwd()))

from mod.kalman.lorenz.apps.help_app import DSAPage

dash.register_page(
    __name__,
    title="DSA",
    name="Training",
    path="/kalman/kalman-lorenz-data/kalman-train",
    order=2
)


nam_gen='lorenz_0_we10_wn100_sp100_pt0'
file_path_data=os.getcwd() 
file_path_org=os.path.join(os.path.dirname(os.getcwd()),"apps","files","kalman","lorenz",) 
path_file_dir=os.path.join(os.path.join(os.path.dirname(os.getcwd()),"apps","files","kalman","lorenz",),*['data', 'lorenz_0_we10_wn100_sp100_pt0', 'path_files.pkl']) 
path_dict={'run': 'mod.kalman.lorenz.kal_0.help_kal', 'fun': 'mod.kalman.lorenz.kal_0.help_fun', 'app': 'mod.kalman.lorenz.neld_fun_0.app_get_pinn_neld', 'doc': 'kalman.lorenz.lorenz_noise_t0'} 
tname='lorenz'
drop_name='training'
path_heads_show= ['rnn_KAL__lorenz_0_we10_wn100_sp100_pt0', 'tfm_KAL__lorenz_0_we10_wn100_sp100_pt0', 'ekf_KAL__lorenz_0_we10_wn100_sp100_pt0', 'jos_KAL__lorenz_0_we10_wn100_sp100_pt0']
categories= ['dsa']
path_display= ['dest_hmod_path']  
dnn_modes= ['sd3_od3', 'sd3_od2']
# Instantiate page
dsa_page = DSAPage( 
    path_heads_show=path_heads_show,
    categories=categories,
    path_display=path_display,
    dnn_modes=dnn_modes,
    tname=tname,
    path_dict=path_dict,
    file_path_data=file_path_data,
    file_path_org=file_path_org,
    nam_gen=nam_gen,
)

layout = dsa_page.layout_train
   
type='train'

out, inp, st, prevent = dsa_page.param_upload_dropdown_all(drop_name)
@callback(*out,*inp,*st,prevent_initial_call=prevent)
def update_nam_gen_dropdown(n): 
    return dsa_page.update_nam_gen_dropdown(drop_name,path_file_dir=path_file_dir,
                                            file_path_org=file_path_org,)


out, inp, st, prevent = dsa_page.param_create_parameter_dropdowns_all(drop_name)
@callback(*out,*inp,*st,prevent_initial_call=prevent)
def create_parameter_dropdowns(store_data): 
    return [] if store_data is None else dsa_page.create_parameter_dropdowns(store_data,drop_name)





out, inp, prevent = dsa_page.param_upload_all(type)
@callback(
    *out,
    *inp,
    # *st,
    prevent_initial_call=prevent
) 
def callback_upload_train(*args):
    return dsa_page.upload(args,type)
 

out, inp, st, prevent = dsa_page.param_run_algorithm_all(type)
@callback(
    out,
    inp,
    st,
    prevent_initial_call=prevent
)
def callback_run_algorithm_train(n_clicks, store_data):
    if not n_clicks or not store_data:
        raise dash.exceptions.PreventUpdate
    return dsa_page.run_algorithm_train(store_data)



out, inp, st, prevent = dsa_page.toggle_parameters_collapse(type=type)
@callback(
    out,
    inp,
    st,
    prevent_initial_call=prevent
)
def toggle_parameters_collapse(n_clicks, is_open):
    if n_clicks:
        return not is_open
    return is_open

 

out, inp, st, prevent = dsa_page.toggle_result_collapse(type=type)
@callback(
    out,
    inp,
    st,
    prevent_initial_call=prevent
)
def toggle_result_collapse(n_clicks, is_open):
    if n_clicks:
        return not is_open
    return is_open




