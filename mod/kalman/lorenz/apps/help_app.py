import dash
from dash import html, dcc, Input, Output, State, callback , ctx, ALL 
import dash_bootstrap_components as dbc 
import sys
import os
from dash import callback_context, no_update
file_path_org=os.getcwd()  
  
# from app_run import app_run_param,algorithm  
from mod.kalman.lorenz.neld_fun_0.main_0_help import get_dict_param ,get_navs_bar
from  mod.kalman.lorenz.neld_fun_0.main_0_help import box_style,dropdown_style,dropdown_options_style
from mod.kalman.lorenz.neld_fun_0.main_0_help import async_shutdown,async_restart

from mod.kalman.lorenz.neld_fun_0.main_0 import algorithm_param,app_run_param,algorithm
from mod.kalman.lorenz.neld_fun_0.help_model_pred import model_pred 
from mod.kalman.lorenz.neld_fun_0.help_funn import remove_directory
import threading 
  
import os
import dash
from dash import html, Input, Output, State, callback, ctx, ALL 
# from mod.kalman.lorenz.neld_fun_0.main_0 import (
#     app_run_param, algorithm, algorithm_param 
# )
from mod.kalman.lorenz.neld_fun_0.help_funn import remove_directory
import numpy as np



def align_head(path_heads_show):
    headd = [
        'dnn_GINN_SM00000_LOC_AUG', 
    ]

    # Add new heads at the end
    for nn in path_heads_show:
        if nn not in headd:
            headd.append(nn)

    # Map head → priority index
    ghhg = {name: idx for idx, name in enumerate(headd)}

    # Sort path_heads_show by priority
    order = np.argsort([ghhg[name] for name in path_heads_show])

    return [path_heads_show[i] for i in order]



import importlib 


from neld_data_all import file_path_org,tname,navbar
from mod.kalman.lorenz.neld_fun_0.main_0_help import dropdown_callback
 

class DSAPage(dropdown_callback,algorithm,model_pred):

    def __init__(self,path_heads_show=None,
                 categories=None,
                 path_display=None,
                 dnn_modes=None,
                 nam_gen_show=None,
                 tname=None,
                 path_dict=None,
                 file_path_org=None,
                 file_path_data=None,
                 nam_gen=None,):
        self.tname=self.pag=tname 
        self.path_dict=path_dict
        dropdown_callback.__init__(self,nam=self.tname)
        print('[[[[[[[[[[callback_file]]]]]]]]]]',self.callback_file,path_dict)



        doc_module = importlib.import_module(self.path_dict['doc'])  
        dict_param=doc_module.dict_param
        gdas=doc_module.gdas
        nam_gen=doc_module.nam_gen
        dnn_mode=doc_module.dnn_mode  
        self.data_studied=action=doc_module.action
        file_path_org=file_path_org or doc_module.file_path_org
        file_path_data=file_path_data or doc_module.file_path_data
        print('[[[[[[[[[[callback_file]]nam_gen        ]]]]]]]]',nam_gen)



        self.name = "dnn"
        self.page_path = "/DSA-2"
        self.categories = ["dsa", "dnn"]  if categories is None else categories
        self.path_display = ["dest_shaft_path"] if path_display is None else path_display
        model_sufix_show=dict_param['dnn_modes']
        path_heads_show=dict_param['path_heads_show']
        path_dirs_show=dict_param['path_dirs_show']
        file_path_org=dict_param['file_path_org']
        file_path_data=dict_param['file_path_data']
        self.model_sufix=dict_param.get('model_sufix',None)
        self.obj_org_path=dict_param.get('obj_org_path',None)
        self.file_path_org,self.file_path_data=file_path_org,file_path_data

        path_diroi=os.path.join(file_path_org, 'data') 
        spath=os.path.join(path_diroi, 'neld_names_all.txt')
        if not os.path.exists(spath):
            ttname=os.path.basename(file_path_org)
            ggname=os.path.basename(os.path.dirname(file_path_org))
            file_path_data=os.getcwd()
            path_diroi=os.path.join(file_path_data, 'files',ggname,ttname,'data') 
            spath=os.path.join(path_diroi, 'neld_names_all.txt')
        self.neld_names_all=neld_names_all=np.loadtxt(spath,dtype=str,ndmin=1)
        true_keys=np.loadtxt(os.path.join(path_diroi, 'true_keys.txt'),dtype=str,ndmin=1)


        # if nam_gen_show is None: 
        #     file_path_org=file_path_org or os.path.join(os.getcwd(),'files') 
        #     file_path_data=file_path_data if file_path_data is not None else os.path.join(file_path_org,'data_initial')
        #     nam_gen_show=[mm for mm in os.listdir(file_path_data) if mm not in ['.DS_Store',]]
        #     print('[[[[[---------file_path_data---------]]]]]',file_path_data,nam_gen_show)
        #     nam_gen_show=['model_0',] if len(nam_gen_show)==0 else nam_gen_show 


        # if path_heads_show is None: 
        #     file_path_org=file_path_org or os.path.join(os.getcwd(),'files') 
        #     file_path_model=  os.path.join(file_path_org,'model')
        #     path_heads_show=[mm for mm in os.listdir(file_path_model) if mm not in ['.DS_Store',]]
        #     print('[[[[[---------file_path_data---------]]]]]',file_path_data,path_heads_show)
        #     path_heads_show=['model_0',] if len(path_heads_show)==0 else path_heads_show 
        #     os.path.basename(self.obj_org_path.replace('\\','/'))

        # name_num=0  
        # dict_param,gdas,path_dict =[paraws[name_num][mm] for mm in ['dict_param','gdas','path_dict']] 
        # configs=dict_param['configs']
        # nam_gen=paraws[name_num]['nam_gens']['name'][action]
        # step_test=paraws[name_num]['nam_gens']['step'][action] 
        action='test'
        nam=f'{nam_gen}_{action}'

        # print('[[[[neld data]]]]',nam,params['gdas'][nam_gen].neld_data.keys())
        # neld_data = params['gdas'][nam_gen].part(nam)
  
        print('[[[[[[[[[[callback_file]]]]]]]]]]',nam)
        neld_data = gdas.part(nam)
 


        path_heads_show=dict_param['path_heads_show']
        model_type=path_heads_show[0]

        path_dirs_show=dict_param['path_dirs_show']
        path_dir=path_dirs_show[0]

        dropdown_callback.__init__(self,nam=self.tname)
 
        self.path_heads_show=path_heads_show   

        param = algorithm_param(**dict_param)


        param['tname']['param']=self.tname
        self.mapp = app_run_param(param)
        neld_data = gdas.part(nam)   
        # mapp = app_run_param(param)
        self.param=self.mapp.emerge_param()  
        # alg = algorithm(self.param)
        algorithm.__init__(self,self.param)
 
        self.pre_test_train(   
            neld_data = neld_data,  
            true_name = 'true_0', 
            dnn_mode = dnn_mode,
            model_type = model_type, 
            path_dir=path_dir,
            data_dir=path_dir,
            path_dict=path_dict, 
            model_sufix_show=model_sufix_show, 
            **dict_param 
        )

        param=self.param
        data_mode=self.data_mode
        modes,data_mode,dmode=self.modes,self.data_mode,self.dmode
        model_sufix_dic=self.model_sufix_dic
        mode_ids=modes[model_type][dnn_mode][path_dir]
        true_keys=self.true_keys 
        path_heads=self.path_heads  
        obj_org_path_dict =self.obj_org_path_dict

        configs=dict_param['configs']


        path_dir=os.path.join(self.file_path_org, 'data','pinn_dir_data_all.txt')  
        if not os.path.exists(path_dir):
            ttname=os.path.basename(file_path_org)
            ggname=os.path.basename(os.path.dirname(file_path_org))
            file_path_data=os.getcwd()
            path_dir=os.path.join(file_path_data, 'files',ggname,ttname,'data','pinn_dir_data_all.txt') 
        pinn_dir_data_all=[mm for mm in list(np.loadtxt(path_dir,dtype=str)) ] 


        for ixi,mode_id in enumerate(mode_ids):#data_mode.keys(): #
            # data_mode[mode_id].update(dict(
            #                 list_features=configs[dnn_mode]['inten_pinn_index'],
            #                 base_features_list=configs[dnn_mode]['base_features_index'],     
            #                 # data_studied=data_studied,
            #                 )
            #         )
            nhh=data_mode[mode_id]['model_sufix'] 
            model_sufix=data_mode[mode_id]['model_sufix'][0] 
            model_pred.__init__(self,   
                                neld_data=neld_data, 
                                model_sufix=model_sufix,
                                data_mode=data_mode[mode_id],   
                                pinn_dir_data_all=pinn_dir_data_all,
                                model_sufix_all=dmode.model_sufix_all, 
                                path_heads= path_heads,
                                model_type=model_type,
                                obj_org_path_dict=obj_org_path_dict ,
                                true_keys=true_keys, 
                                model_sufix_dic=model_sufix_dic,
                                **param['model_pred']['param']
                                ) 
 
            self.get_model_opt_name(model_sufix=model_sufix,
                                        model_type=model_type)    






    def layout(self):
        return self.mapp.app_layout
 

    def layout_train(self):
        return self.mapp.app_layout_train
 


    def layout_gen(self):
        return self.mapp.app_layout_gen
 
    def toggle_parameters_collapse(self, type=None):
        par={key:f'{mm}' if type is None else f'{mm}-{type}' for key,mm in zip(['output','input','state'],
                                                                               [f"{self.pag}-collapse-starting","toggle-text-starting",f"{self.pag}-collapse-starting"])}
        return (
            Output(par['output'], "is_open"),
            Input(par['input'], "n_clicks"),
            State(par['state'], "is_open"),
            False
        )

    def toggle_result_collapse(self, type=None):
        par={key:f'{mm}' if type is None else f'{mm}-{type}' for key,mm in zip(['output','input','state'],
                                                                               [f"{self.pag}-collapse-result",f"toggle-text-result",f"{self.pag}-collapse-result"])}
        return (
            Output(par['output'], "is_open"),
            Input(par['input'], "n_clicks"),
            State(par['state'], "is_open"),
            False
        )

    def param_toggle_single(self, gval):
        return (
            Output(f"{self.pag}-collapse-{gval}", "is_open"),
            Input(f"{self.pag}-collapse-header-{gval}", "n_clicks"),
            State(f"{self.pag}-collapse-{gval}", "is_open"),
            False
        )

    def param_run_algorithm(self):
        outputs = Output(f"{self.pag}-run-status", "children")
        inputs = Input("run-button", "n_clicks")
        states = State(f"{self.pag}-shared-data", "data")
        return outputs, inputs, states, True
 

    def param_upload(self):
        outputs = [
            # Output("output-file-upload", "children"),
            # Output("destination-path", "children"),
            Output(f"{self.pag}-shared-data", "data"),
        ]

        inputs = [
            # Input("upload-data", "filename"),
            
            Input("reset-button", "n_clicks"),
            Input("restart-button", "n_clicks"),
            Input("shutdown-button", "n_clicks"),
        ] + [Input(idx, "value") for idx in self.mapp.Input_id],#+[Input(f"id-destination", "value")],
        states = [
            # State("upload-data", "contents"),
            # State("upload-data", "filename"),
            # State("upload-data", "last_modified"),
            # State("id-destination", "value"),
        ]

        return outputs, inputs,   False
    



    def param_upload_all(self,type):
        outputs = [
            # Output("output-file-upload", "children"),
            # Output(f"destination-path-{type}", "children"),
            Output(f"{self.pag}-shared-data-{type}", "data"),
        ]

        inputs = [
            # Input("upload-data", "filename"),
            
            Input(f"reset-button-{type}", "n_clicks"),
            Input(f"restart-button-{type}", "n_clicks"),
            Input(f"shutdown-button-{type}", "n_clicks"),
        ] + [Input(idx, "value") for idx in self.mapp.Input_id]+[Input(f"id-destination-{type}", "value")],

        states = [
            # State(f"upload-data-{type}", "contents"),
            # State(f"upload-data-{type}", "filename"),
            # State(f"upload-data-{type}", "last_modified"),
            # State(f"id-destination-{type}", "value"),
        ]

        return outputs, inputs,  False
        # return outputs, inputs, False
         
    def param_run_algorithm_all(self,type):
        outputs = Output(f"{self.pag}-run-status-{type}", "children")
        inputs = Input(f"run-button-{type}", "n_clicks")
        states = State(f"{self.pag}-shared-data-{type}", "data")
        return outputs, inputs, states, True
  
 
    def toggle_all(self, args):
        total = len(self.categories)
        nav_clicks = args[total]
        states = args[-total:]
        triggered = ctx.triggered_id

        if isinstance(triggered, dict) and triggered.get("type") == "nav":
            return [False] * total

        if isinstance(triggered, str) and triggered.startswith("btn-"):
            clicked = triggered.replace("btn-", "")
            return [
                not state if prefix == clicked else False
                for prefix, state in zip(self.categories, states)
            ]

        return states

    def toggle_single(self, n_clicks, is_open):
        return not is_open if n_clicks else is_open


 
 
    def upload(self, args,type=None):
        reset_clicks, restart_clicks, shutdown_clicks, *rest = args
        # filenames, reset_clicks, restart_clicks, shutdown_clicks, *rest = args
        values = rest
        param = self.mapp.rebuild_param(values)

        self.model_sufix=param['dnn_mode']['param']
        self.data_studied=param['action']['param']
        self.obj_org_path=os.path.join(self.file_path_org,'data_initial',param['nam_gen']['param'])

        *other_values, contents, filenames, last_modified, nam_gen = values
        # if type in ('train','gen'):
        ctx = callback_context 
        chosen_element = values[-1] 
        if ctx.triggered: 
            trigger_id = ctx.triggered[0]['prop_id'].split('.')[0] 
            if trigger_id in [f"id-destination-{type}" for type in ['train','gen',]]:
                chosen_element = values[-1] 
            elif trigger_id in [f"id-destination-{type}" for type in ['train',]]:
                chosen_element = values[2] 
            elif trigger_id in self.mapp.Input_id: 
                trigger_index = self.mapp.Input_id.index(trigger_id) 
                if trigger_index == 0: 
                    chosen_element = values[0] 
                else: 
                    chosen_element = values[-1]
        nam_gen=chosen_element if type is not None else values[0]
        nbr_train=param['dyna param']['param'].get('nbr_train_sample',0 )
        rng = np.random.default_rng()
        ls=param['get_training']['param']['ls']
        if len(ls)==1 and ls[0]<1:
            nn=int(ls[0]*nbr_train)
            param['get_training']['param']['ls']=rng.choice(nn,size=nn, replace=False)

        print('[[[[[[[[[[[parammmm--------------------------------]]]]]]]]]]]',param['get_training']['param']['ls'])
        # *other_values, contents, filenames, last_modified, objs_path_org = values["id-destination"]+

        print('[[[[[[[[[[[[[[[[[[[[[[[[[[[nam  gen]]]]]]]]]]]]]]]]]]]]]]]]]]]',type,nam_gen,chosen_element,values)
        '''
'''
        '''
        ''' 
        if shutdown_clicks and shutdown_clicks > 0:
            threading.Thread(target=async_shutdown).start()
            return html.Div("Shut Down requested"), {}

        if reset_clicks and reset_clicks > 0:
            # threading.Thread(target=async_restart).start() 
            # remove_directory(file_path_data)
            # remove_directory(os.path.join(file_path_org,'data'))
            objs_path_app=os.path.join(self.file_path_org,'data')
            for mm in [f for f in os.listdir(objs_path_app) if os.path.isdir(os.path.join(objs_path_app, f))]:
                if mm !='lorenz_0_we10_wn100_sp100_pt0.1':
                    pmm=os.path.join(objs_path_app,mm)
                    remove_directory(pmm) 
                    

            objs_path_app=os.path.join(self.file_path_org,'data_initial')
            for mm in [f for f in os.listdir(objs_path_app) if os.path.isdir(os.path.join(objs_path_app, f))]:
                if mm !='lorenz_0_we10_wn100_sp100_pt0.1':
                    pmm=os.path.join(objs_path_app,mm)
                    remove_directory(pmm) 
            objs_path_app=os.path.join(self.file_path_org,'model')
            for mm in [f for f in os.listdir(objs_path_app) if os.path.isdir(os.path.join(objs_path_app, f))]:
                if not mm.startswith(('ekf_KAL',
                                      'jos_KAL',
                                      'tfm_KAL__lorenz_0_we10_wn100_sp100_pt0.1',
                                      'rnn_kal__lorenz_0_we10_wn100_sp100_pt0.1')):# !='lorenz_0_we10_wn100_sp100_pt0':
                    pmm=os.path.join(objs_path_app,mm)
                    remove_directory(pmm) 
            # remove_directory(os.path.join(file_path_org,'app','pages','test'))
            return html.Div(f"Reset requested"), {}


        if restart_clicks and restart_clicks > 0:
            threading.Thread(target=async_restart).start()
            return html.Div("Restart requested"), {} 
        neld_names_chld = {} 
        path_heads_show=self.path_heads_show 
        store_data = dict(
            nam_gen=nam_gen,
            param=param,  
            neld_names_chld=neld_names_chld,
            path_heads_show=path_heads_show,
        ) 
        display = html.Div([
                html.H5("", style={'color': 'white'}),])
        return store_data
        # return display, store_data






    def run_algorithm(self, store_data):
        # store_data=store_datas
        # print('[[[[[[[[[store data]]]]]]]]]',len(store_data),store_data[0],store_data[1]['param'])
        param = store_data["param"]
        # neld_names_chld = store_data["neld_names_chld"]
        # export_dir = store_data["export_dir"] 
        path_heads_show=store_data["path_heads_show"] 
        for val in ['get_pinn_features', ]:
            param['model_pred']['param']['param_dic']['tf_restart'][val]=False   #True   # 

        # gdas = get_data_all(names_dic=neld_names_chld, file_path_data=export_dir)
        model_type=param["path_head"]["param"] 





        doc_module = importlib.import_module(self.path_dict['doc']) 
        path_dict=doc_module.path_dict 
        dict_param=doc_module.dict_param
        gdas=doc_module.gdas
        nam_gen=doc_module.nam_gen



        # name_num=0
        # path_dict = paraws[name_num]['path_dict']
        # dict_param = paraws[name_num]['dict_param']
        # nam_gen =model_type.split('__')[1] 
        nam_gen=param['nam_gen']['param']

        action='test'
        action=param["action"]["param"]
        nam=f'{nam_gen}_{action}'
        # neld_data = gdas.part(nam)
 

        fun_module = importlib.import_module(path_dict['fun'])  
        app_dict_param=fun_module.app_dict_param
        app_get_data_all=fun_module.app_get_data_all

        run_module = importlib.import_module(path_dict['run']) 
        model_choice = run_module.model_choice
        mchoice=model_choice()

        nam_gen = store_data["nam_gen"]
        nam=f'{nam_gen}_{action}'
        dict_param=app_dict_param(param,nam_gen=nam_gen,navbar=navbar)
        gdas=app_get_data_all(param,nam_gen=nam_gen,gdas=gdas)
        neld_data = gdas.part(nam)
        dict_param['path_heads_show']+=[model_type] if model_type not in path_heads_show else []
        head_navbar =get_navs_bar(path_heads_show=path_heads_show,**navbar)
        param['head_navbar']['param']=head_navbar 
        weight=param['model-pred']['param']['weight']
        llos=param["path_dir"]["param"].split('_')[0]

        pathdic=param["path_dir"]["param"]=f'{llos}_{weight}'
        pinn_dir_data_all=dict_param['path_dirs_show']
        path_dir=os.path.join(self.file_path_org, 'data','pinn_dir_data_all.txt') 
        if os.path.exists(path_dir):
            pinn_dir_data_all=list(np.loadtxt(path_dir,dtype=str))


        pinn_dir_data_all+=[pathdic,] if pathdic not in pinn_dir_data_all else []
        np.savetxt(path_dir,pinn_dir_data_all,fmt='%s')

        param['path_dirs_show']['param']= dict_param['path_dirs_show']=[mm for mm in pinn_dir_data_all if mm not in (None,'None') ]

        print('[[[[[[[[[ddddd====]]]]]]]]]',pinn_dir_data_all,'[[[[[======]]]]]',param['model-pred']['param'].keys(),'[[[[[]]]]]',param['model-pred'],param['path_dirs_show'])
        param['head_navbar']['param']=head_navbar
        alg = algorithm(param=param)
        alg.test(
            neld_data=neld_data,
            true_name="true_0",
            dnn_mode=param["dnn_mode"]["param"],
            model_type=model_type,
            path_dir=param["path_dir"]["param"],
            data_dir=param["path_dir"]["param"],  
            path_dict=path_dict,
            pinn_dir_data_all=pinn_dir_data_all,
            # path_display=self.path_display,
            **dict_param 
        )

        return html.H3("Completed!", style={'color': 'lightgreen', 'textAlign': 'center'})


 
    def run_algorithm_train(self, store_data):
        # store_data=store_datas
        # print('[[[[[[[[[store data]]]]]]]]]',len(store_data),store_data[0],store_data[1]['param'])
        param = store_data["param"] 
        # neld_names_chld = store_data["neld_names_chld"]
        # export_dir = store_data["export_dir"]
        # nh = param["path_dir"]["param"]
        path_heads_show=store_data["path_heads_show"]

        '''
        nhh = param["path_dirs_weig"]["param"][nh]
        param["model-pred"]["param"]["weight"] = nhh'''

        for val in ['get_pinn_features', ]:
            param['model_pred']['param']['param_dic']['tf_restart'][val]=False   #True   # 

        doc_module = importlib.import_module(self.path_dict['doc']) 
        path_dict=doc_module.path_dict 
        dict_param=doc_module.dict_param
        gdas=doc_module.gdas
        nam_gen=doc_module.nam_gen
        # gdas = get_data_all(names_dic=neld_names_chld, file_path_data=export_dir)
        model_type=param["path_head"]["param"] 
 
        nam_gen =model_type.split('__')[1] 


        action='test'
        action=param["action"]["param"]
        nam=f'{nam_gen}_{action}' 
        neld_data = gdas.part(nam)
 


        fun_module = importlib.import_module(path_dict['fun'])  
        app_dict_param=fun_module.app_dict_param
        app_get_data_all=fun_module.app_get_data_all

        run_module = importlib.import_module(path_dict['run']) 
        model_choice = run_module.model_choice
        mchoice=model_choice()


        valid_models=tuple(mchoice.models.keys())
        model_type = store_data["nam_gen"]
        if not model_type.lower().startswith(valid_models):
            raise ValueError(f"Model name prefix '{model_type}' is invalid. It needs to start with one of: {valid_models}")
 

        nam_gen=model_type.split('__')[1]
        nam=f'{nam_gen}_{action}'
        dict_param=app_dict_param(param,nam_gen=nam_gen,navbar=navbar)
        gdas=app_get_data_all(param,nam_gen=nam_gen,gdas=gdas)
        neld_data = gdas.part(nam) 

        dict_param['path_heads_show']+=[model_type] if model_type not in path_heads_show else []  
        param['head_navbar']['param']=get_navs_bar(path_heads_show=path_heads_show,**navbar)
  

        param['get_training']['tf'] = True 
        alg = algorithm(param=param)
        alg.train(
            neld_data=neld_data,
            true_name="true_0",
            dnn_mode=param["dnn_mode"]["param"],
            model_type=model_type,
            path_dir=param["path_dir"]["param"],
            data_dir=param["path_dir"]["param"],  
            path_dict=path_dict,
            # path_display=self.path_display,
            **dict_param 
        )

        return html.H3("Completed!", style={'color': 'lightgreen', 'textAlign': 'center'})


 

 
    def run_algorithm_gen(self, store_data):
        # store_data=store_datas
        # print('[[[[[[[[[store data]]]]]]]]]',len(store_data),store_data[0],store_data[1]['param'])
        param = store_data["param"] 
        path_heads_show=store_data["path_heads_show"]

        '''
        nh = param["path_dir"]["param"]
        neld_names_chld = store_data["neld_names_chld"]
        export_dir = store_data["export_dir"]
        nhh = param["path_dirs_weig"]["param"][nh]
        param["model-pred"]["param"]["weight"] = nhh'''

        for val in ['get_pinn_features', ]:
            param['model_pred']['param']['param_dic']['tf_restart'][val]=False   #True   # 

        doc_module = importlib.import_module(self.path_dict['doc']) 
        path_dict=doc_module.path_dict 
        dict_param=doc_module.dict_param
        gdas=doc_module.gdas
        # gdas = get_data_all(names_dic=neld_names_chld, file_path_data=export_dir)
        model_type=param["path_head"]["param"] 

        name_num=0
        print('[[[[[]]]]]]]]model_py',model_type)
        nam_gen =model_type.split('__')[1] 

        action='test'
        action=param["action"]["param"]
        nam=f'{nam_gen}_{action}'  

        neld_data = gdas.part(nam)
 


        fun_module = importlib.import_module(path_dict['fun']) 
        observation=fun_module.observation
        app_dict_param=fun_module.app_dict_param
        app_get_data_all=fun_module.app_get_data_all
        get_data_fun=fun_module.get_data_fun
        

        run_module = importlib.import_module(path_dict['run']) 
        model_choice = run_module.model_choice
        mchoice=model_choice()

        nam_gen = store_data["nam_gen"]
        print('[[[[[[[[[store ===============================]]]]]]]]]',nam_gen)
        nam=f'{nam_gen}_{action}'
        dict_param=app_dict_param(param,nam_gen=nam_gen,navbar=navbar,)
        gdas=app_get_data_all(param,nam_gen=nam_gen,gdas=gdas)
        neld_data = gdas.part(nam) 
        
        dict_param['path_heads_show']+=[model_type] if model_type not in path_heads_show else [] 
        param['head_navbar']['param']=get_navs_bar(path_heads_show=path_heads_show,**navbar)



        param['model-pred']['tf'] = False   # True    #True/False: choose whether to predict shafts/smods
        param['skl_shaft_pred']['tf']  =False   #True  #
        param['dash_pages']['tf'] =False   #True # True          #  True/False: generate Dash pages
        param['roc']['tf']=False   #True # 
        dnn_mode=param["dnn_mode"]["param"]
        alg = algorithm(param=param)
        mod=alg.test(
            neld_data=neld_data,
            true_name="true_0",
            dnn_mode=dnn_mode,
            model_type=model_type,
            path_dir=param["path_dir"]["param"],
            data_dir=param["path_dir"]["param"],  
            path_dict=path_dict,
            # path_display=self.path_display,
            **dict_param 
        )
        print('[[[[neld data]]]]',dnn_mode,nam_gen)
        # param_data=params['dict_param'][nam_gen]['configs'][dnn_mode]['param'] 
        param_data=dict_param['configs'][dnn_mode]['param'] 
        # step_test=params['nam_gens'][nam_gen]['step'][action] 


        get_data_fun(self=mod,param=param_data,step_test=None)

        return html.H3("Completed!", style={'color': 'lightgreen', 'textAlign': 'center'})


 
