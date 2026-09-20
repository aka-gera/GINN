 
from dash import callback
import sys,os,dash
 
sys.path.append(os.path.abspath(os.getcwd()))

from mod.dsa.apps.help_app import DSAPage

dash.register_page(
    __name__,
    title="DSA",
    name="Prediction",
    path="/dsa/dsa-ginn-data/dsa-2",
    order=1
)
nam_gen='meshes'
file_path_data=os.getcwd() 
file_path_org=os.path.join(os.path.dirname(os.getcwd()),"apps","files","dsa","ginn",)  
path_file_dir=os.path.join(os.path.join(os.path.dirname(os.getcwd()),"apps","files","dsa","ginn",),*['data', 'meshes', 'path_files.pkl']) 
path_dict={'run': 'mod.dsa.neld_fun_0.help_kal', 'fun': 'mod.dsa.neld_fun_0.help_fun', 'app': 'mod.dsa.neld_fun_0.app_get_pinn_neld', 'doc': 'ginn.meshes'} 
tname='ginn'
drop_name='prediction'
path_heads_show= ['dnn_GINN__SM00000_LOC_AUG']
categories= ['dsa']
path_display= ['dest_shaft_path']  
dnn_modes= ['DNN_3', 'DNN_2']
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

layout = dsa_page.layout
 
type=None

out, inp, st, prevent = dsa_page.param_upload_dropdown_all(drop_name)
@callback(*out,*inp,*st,prevent_initial_call=prevent)
def update_nam_gen_dropdown(n): 
    return dsa_page.update_nam_gen_dropdown(drop_name,path_file_dir=path_file_dir,
                                            file_path_org=file_path_org,)


out, inp, st, prevent = dsa_page.param_create_parameter_dropdowns_all(drop_name)
@callback(*out,*inp,*st,prevent_initial_call=prevent)
def create_parameter_dropdowns(store_data): 
    return [] if store_data is None else dsa_page.create_parameter_dropdowns(store_data,drop_name)


'''
out, inp, st, prevent = dsa_page.param_toggle_all()

@callback(
    *out,
    *inp,
    *st,
    prevent_initial_call=prevent
)
def toggle_all(*args):
    return dsa_page.toggle_all(args)
'''

for gval in list(set(dsa_page.param["param_input"]["param"])):
    out, inp, st, prevent = dsa_page.param_toggle_single(gval)

    @callback(
        out,
        inp,
        st,
        prevent_initial_call=prevent
    )
    def toggle_single(n_clicks, is_open, gval=gval):
        return dsa_page.toggle_single(n_clicks, is_open)
type='pred'

out, inp, prevent = dsa_page.param_upload()

@callback(
    *out,
    *inp,
    # *st,
    prevent_initial_call=prevent
)
def callback_upload(*args):
    return dsa_page.upload(args,type)
out, inp, st, prevent = dsa_page.param_run_algorithm()

@callback(
    out,
    inp,
    st,
    prevent_initial_call=prevent
)
def callback_run_algorithm(n_clicks, store_data):
    if not n_clicks or not store_data:
        raise dash.exceptions.PreventUpdate
    return dsa_page.run_algorithm(store_data)



out, inp, st, prevent = dsa_page.toggle_parameters_collapse()
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



type=None

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


