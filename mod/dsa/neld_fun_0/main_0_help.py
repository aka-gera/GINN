import os


from dash import html, dcc, Input, Output, State, callback , ctx, ALL 
import dash_bootstrap_components as dbc 
from mod.dsa.neld_fun_0.help_funn import safe_path_join 


def get_navs_bar(head_navbar,category,path_heads_show,head_navbars=None):  

    head_navbars=head_navbars if head_navbars is not None else {
                    # "rnn": "RNN",
                    # "tfm": "Transformer",
                   
                    # "pinn":"DNN v.0" ,
                    # "cnn":"3D CNN" ,
                    # 'gcn': "GCN",
                    # 'cml':"class ML",
                }
        
    for ipath in path_heads_show :
        for key,val in head_navbars.items():
            print('[[[[[[[[[=====key]]]]]]]]]',key,ipath.startswith(key),path_heads_show)
            if ipath.startswith(key) and (val not in head_navbar): 
                head_navbar[val]=key
                category.append(key)
    return head_navbar,category





styles = {
    'pre': {
        'border': 'thin lightgrey solid',
        'overflowX': 'scroll'
    }
}
box_style = {
    'width': '100%',
    'padding': '3px',
    'font-size': '20px',
    'text-align-last': 'center',
    'margin': 'auto', 
    'background-color': 'black',
    'color': 'black'
} 
dropdown_style = {
    'width': '85%',
    'padding': '3px',
    'font-size': '20px',
    'text-align-last': 'center',
    # 'margin': 'auto', 
    'marginLeft': 'auto',
    # 'background-color': 'black',
    'color': 'black',
    # 'height': '10px',
} 
dropdown_options_style = {'color': 'white', 'background-color': 'gray'} 










def restart_appsx():
    try:
        # Kill existing gunicorn/app processes
        subprocess.run(["pkill", "-f", "python"], check=False)

        # Relaunch the app script directly
        subprocess.Popen([
            "/usr/bin/python3", "app/app.py"
        ])
        return "Restart triggered!"
    except Exception as e:
        return f"Error restarting: {e}"

 


def free_port(port=8050):
    system = platform.system()

    if system in ["Darwin", "Linux"]: 
        result = subprocess.run(["lsof", "-ti", f":{port}"], capture_output=True, text=True)
        pids = result.stdout.strip().splitlines()
        print("PIDs using port:", pids)

        if not pids:
            print(f"Port {port} is free.")
            return
        for pid in pids:
            try:
                os.kill(int(pid), signal.SIGTERM)
                print(f"Killed process {pid} using port {port}")
            except Exception as e:
                print(f"Error killing {pid}: {e}")

    elif system == "Windows": 
        result = subprocess.run(
            ["netstat", "-ano"], capture_output=True, text=True
        )
        lines = result.stdout.splitlines()
        pids = []
        for line in lines:
            if f":{port}" in line and "LISTENING" in line:
                parts = line.split()
                pid = parts[-1]
                pids.append(pid)

        if not pids:
            print(f"Port {port} is free.")
            return

        for pid in pids:
            try:
                subprocess.run(["taskkill", "/PID", pid, "/F"], check=False)
                print(f"Killed process {pid} using port {port}")
            except Exception as e:
                print(f"Error killing {pid}: {e}")



 
def start_server():
    system = platform.system()
    print('[[[[[[[[[[]]]]]]]]]]', system)

    if system == "Windows":
        venv_python = os.path.abspath(os.path.join("..", "dsa_venv", "Scripts", "python.exe"))
        proc = subprocess.Popen([venv_python, "run_app_win.py"])
        return f"Waitress started (Windows) with PID {proc.pid}"

    else:
        # The exact command you told me to run  
        subprocess.run(["pkill", "-f", "python"], check=False)

        venv_python = os.path.abspath(os.path.join("..", "dsa_venv", "bin", "python3"))
# 
        proc = subprocess.Popen([venv_python, "app/app.py"]) 
        proc = subprocess.Popen([
            venv_python,
            "-m", "gunicorn",
            "-w", "4",
            "-b", "0.0.0.0:8050",
            "wsgi:server",
            "--timeout", "1200",
            "-c", "gunicorn.conf.py"
        ])



        return "Restart triggered!"





def async_restart():
    # start_server()
    # free_port()
   # start_server()
    #  free_port(8050)   # kills old Gunicorn workers
    start_server()    # launches new Gunicorn in a detached session
     # open_terminal_and_run()


def async_shutdown():
    free_port()

    
def restart_appsx():
    try:
        # Kill existing gunicorn/app processes
        subprocess.run(["pkill", "-f", "python"], check=False) 
        subprocess.Popen(["/usr/bin/python3", "app/app.py"])
        return "Restart triggered!"
    except Exception as e:
        return f"Error restarting: {e}"

 


import subprocess
import platform
def restart_apps(process="gunicorn"):
    try:
        system = platform.system()

        if system == "Windows":
            # Kill any running waitress processes
            subprocess.run(["taskkill", "/F", "/IM", "python.exe", "/T"], check=False)

            # Start waitress server
            subprocess.Popen([
                "python", "-m", "waitress", "--listen=0.0.0.0:8050", "wsgi:server"
            ])
            return "Restart triggered using Waitress (Windows)."

        else:
            # Kill any running gunicorn processes
            subprocess.run(["pkill", "-f", process], check=False)

            # Start gunicorn server
            subprocess.Popen([
                "gunicorn",
                "-w", "4",
                "-b", "0.0.0.0:8050",
                "wsgi:server",
                "-c", "gunicorn.conf.py"
            ])
            return "Restart triggered using Gunicorn (Linux/Unix)."

    except Exception as e:
        return f"Error restarting: {e}"





def get_dict_param(file_path_org=None,
                   nam='meshes',
                   data_studied=None,
                    n_step = 20,
                    weight=3.,
                    weight2=1,
                    size_threshold=100,
                    kmean_n_run=50,
                    path_heads_show=None,
                    path_dirs_show=None,
                    neld_names_all=None,
                    path_heads=None,
                    head_navbar=None,
				    wrap_method='alpha_wrap',
                    pre_portion=None,
                    configs=None,
                    dnn_modes=None,
                    gdas=None,
                    nam_gen_show=None,
                    file_path_data=None,
                    dyna=None,
                    navbar=None,
                    param_flows=None,
                    ):
    path_heads_show=path_heads_show if path_heads_show is not None else [ 
                'dnn_GINN_SM00000_LOC_AUG', 
                'gcn_UNet_SM10000_LOC',
                'cml_cML',
    ] 
    head_navbar=head_navbar if head_navbar is not None else get_navs_bar(path_heads_show=path_heads_show,**navbar)

    path_heads=path_heads if path_heads is not None else ['save',] 


    path_display = ['dest_shaft_path', 'dest_spine_path', ] 
    path_display = ['dest_shaft_path', ]
 


    # path_heads_show=path_heads_show if model_type in path_heads_show else path_heads_show +[model_type] 
    path_heads = list(set(path_heads+path_heads_show)) 

    path_list=['shaft_path','spine_path',  ]   
    # weig=sorted(list(set([1,2, 3]  ))) 
    weig=sorted(list(set([1,2,]  )))
    # path_dirs=[f'loss_we_{wei}' for wei in weig] +[f'oloss_we_{wei}' for wei in weig] 
    path_dirs=['loss','oloss'] 
    wei=1
    # path_dirs=[f'loss_we_{wei}',f'oloss_we_{wei}' ]
    path_dirs_weig={key:val for key,val in zip(path_dirs,weig)} 
 

    path_display_dic={
                        'path':path_dirs,
                        'model_sufix':[],
                        'path_head':[],
                    } 
    param_dic={
        hh:{
            kk:True for kk in ['get_pinn_features','get_wrap','get_scale',
            'get_smooth','get_shaft_pred','get_neld_name','get_skeleton', ]
        }
        for hh in ['tf_restart','get_info','data']
    } 

    param_dic['data']['get_neld_name']=dict(
                        dict_neld_path='current',
                        drop_dic_name=None,)





    


    print('[[[[000099888]]]]',dnn_modes)
    return dict(    
        nam=nam,
        n_step = n_step,
        weight=weight,
        weight2=weight2,
        size_threshold=size_threshold,
        path_display = path_display, 
        model_type_data=None,  
        path_display_dic=path_display_dic,
        path_heads_show=path_heads_show,
        path_dirs_show=path_dirs_show,
        nam_gen_show=nam_gen_show,
        # path_shaft_dir=path_shaft_dir,
        param_dic=param_dic,
        kmean_n_run=kmean_n_run,
        dnn_modes=dnn_modes,
        path_dirs=path_dirs,
        path_list=path_list,
        path_heads=path_heads,
        path_dirs_weig=path_dirs_weig,
        head_navbar=head_navbar, 
		wrap_method=wrap_method,
        pre_portion=pre_portion,
        configs=configs,
        gdas=gdas,
        file_path_org=file_path_org,
        file_path_data=file_path_data,
        data_studied=data_studied,
        neld_names_all=neld_names_all,
        dyna=dyna,
        param_flows=param_flows,
    )

    



class get_data_all:
    def __init__(self, neld_data=None,names_dic={},obj_org_path=None,file_path_data=None,true_keys=[f'true_0',],cpath=None,data_studied=None, ):  
        self.neld_data=neld_data   
            
        if self.neld_data is None:
            self.neld_data={}
        else: 
            for nam,names in neld_data.items() :   
                data_studied=names.get('data_studied',None)
                neld_names = names.get('neld_names', [])
                neld_first = names.get('neld_first', ['d000'] * len(neld_names))
                neld_path_inits = names.get('neld_path_inits',[nam  for _ in range(len(neld_names))])
                neld_namess = names.get('neld_namess',  [f'd{str(i).zfill(3)}' for i in range(len(neld_names))] )
                neld_last = names.get('neld_last', [f'{de}_' for de in neld_first])
                name_spine_id=names.get('name_spine_id','p_sps')
                name_head_id=names.get('name_head_id','p_sphs')
                name_neck_id=names.get('name_neck_id','p_spns')
                name_shaft_id=names.get('name_shaft_id','p_spsh')
                objs_path_org_tmp=  safe_path_join(file_path_data,nam )  
                obj_org_path=names.get('obj_org_path',objs_path_org_tmp) 
                weights=names.get('weights',[[6.,0.81]])
                obj_org_path_dict=names.get('obj_org_path_dict',{val: obj_org_path for val in true_keys}) 
                for na in [nam, ]:
                    self.neld_data[nam]=dict( 
                                            neld_namess=neld_namess ,
                                            neld_names=neld_names ,
                                            neld_last=neld_last ,
                                            neld_first=neld_first ,
                                            neld_path_inits=neld_path_inits,
                                            name_spine_id=name_spine_id,
                                            name_head_id=name_head_id,
                                            name_neck_id=name_neck_id,
                                            name_shaft_id=name_shaft_id,
                                            obj_org_path=obj_org_path,
                                            obj_org_path_dict=obj_org_path_dict,
                                            weights=weights,
                                            cpath=cpath,
                                            data_studied=data_studied,
                                            ) 

        for nam,names in names_dic.items() :   
            data_studied=names.get('data_studied',None)
            neld_names = names.get('neld_names', [])
            neld_first = names.get('neld_first', ['d000'] * len(neld_names))
            neld_path_inits = names.get('neld_path_inits',[nam  for _ in range(len(neld_names))])
            neld_namess = names.get('neld_namess', [f'd{str(i).zfill(3)}' for i in range(len(neld_names))] )
            neld_last = names.get('neld_last', [f'{de}_' for de in neld_first])
            name_spine_id=names.get('name_spine_id','p_sps')
            name_head_id=names.get('name_head_id','p_sphs')
            name_neck_id=names.get('name_neck_id','p_spns')
            name_shaft_id=names.get('name_shaft_id','p_spsh')
            objs_path_org_tmp=  safe_path_join(file_path_data,nam )  
            obj_org_path=names.get('obj_org_path',objs_path_org_tmp) 
            obj_org_path_dict=names.get('obj_org_path_dict',{f'true_0': obj_org_path for val in neld_names})  
            weights=names.get('weights',[[6.,0.81]]) 
            for na in [nam, ]:
                self.neld_data[na]=dict(  
                                        neld_namess=neld_namess ,
                                        neld_names=neld_names ,
                                        neld_last=neld_last ,
                                        neld_first=neld_first ,
                                        neld_path_inits=neld_path_inits,
                                        name_spine_id=name_spine_id,
                                        name_head_id=name_head_id,
                                        name_neck_id=name_neck_id,
                                        name_shaft_id=name_shaft_id,
                                        obj_org_path=obj_org_path,
                                        data_studied=data_studied,
                                        obj_org_path_dict=obj_org_path_dict,
                                        weights=weights,
                                        cpath=cpath,
                                        )

    def part(self,nam, istart=0, ifin=None): 
        dd = self.neld_data[nam]
        return {
            'neld_namess': dd['neld_namess'][istart:ifin],
            'neld_names': dd['neld_names'][istart:ifin],
            'neld_last': dd['neld_last'][istart:ifin],
            'neld_first': dd['neld_first'][istart:ifin],
            'neld_path_inits': dd['neld_path_inits'][istart:ifin],
            'name_spine_id':dd['name_spine_id'],
            'name_head_id':dd['name_head_id'],
            'name_neck_id':dd['name_neck_id'],
            'name_shaft_id':dd['name_shaft_id'],
            'obj_org_path':dd['obj_org_path'],
            'obj_org_path_dict':dd['obj_org_path_dict'],
            'weights':dd['weights'],
            'cpath':dd['cpath'],
            'data_studied':dd['data_studied'],
        } if ifin is not None else  {
            'neld_namess': dd['neld_namess'][istart:],
            'neld_names': dd['neld_names'][istart:],
            'neld_last': dd['neld_last'][istart:],
            'neld_first': dd['neld_first'][istart:],
            'neld_path_inits': dd['neld_path_inits'][istart:],
            'name_spine_id':dd['name_spine_id'],
            'name_head_id':dd['name_head_id'],
            'name_neck_id':dd['name_neck_id'],
            'name_shaft_id':dd['name_shaft_id'],
            'obj_org_path':dd['obj_org_path'],
            'obj_org_path_dict':dd['obj_org_path_dict'],
            'weights':dd['weights'],
            'cpath':dd['cpath'],
            'data_studied':dd['data_studied'],
        } 
    


 




class dropdown_callback:
    def __init__(self,nam='kalman',share_1='fig',share_2='app'): 
        self.callback_file=dict(
            figure=dict(        
                par = f'param_{share_2}-store',
                loa = f'page_{share_2}-load',
                mal=f'container_{share_2}_param',
                type='d000',
                share=share_1,
                lst=['nam_gen','path_head','action','dnn_mode','path_dir','root'],
            ),
            generation=dict(        
                par = f'param_{share_2}-store',
                loa = f'page_{share_2}-load',
                mal=f'container_{share_2}_param',
                type='gen',
                share=share_2,
                lst=['nam_gen','path_head'],#,'action','dnn_mode','root'],
            ),
            prediction=dict(        
                par = f'param_{share_2}-store',
                loa = f'page_{share_2}-load',
                mal=f'container_{share_2}_param',
                type='pred',
                share=share_2,
                lst=['nam_gen','path_head'],#,'action','dnn_mode','root'],
            ),
            training=dict(        
                par = f'param_{share_2}-store',
                loa = f'page_{share_2}-load',
                mal =f'container_{share_2}_param',
                type='train',
                share=share_2,
                lst=['nam_gen','path_head'],#,'action','dnn_mode','root'],
            ),
        )

        

        
        for key1,val1 in self.callback_file.items(): 
            for key2 in ['par','loa','mal']:
                self.callback_file[key1][key2]=f'{val1[key2]}-{nam}'


    def param_upload_dropdown_all(self,name):
        param=self.callback_file[name]
        par=param['par']
        loa=param['loa']
        type=param['type']
        outputs = [Output(par if type is None else f"{par}-{type}", "data"),]
        inputs  = [ Input(loa if type is None else f"{loa}-{type}", "n_intervals"),]
        # inputs = [Input("shutdown-button", "n_clicks"),]
        states = []
        return outputs, inputs, states, False




    def param_create_parameter_dropdowns_all(self,name):
        param=self.callback_file[name]
        par=param['par']
        loa=param['loa']
        type=param['type']
        mal=param['mal']
        lst=param['lst']
        outputs=[Output( f"{mal}-{gvali}-{type}" , "children") for gvali in lst],
        inputs =[Input( f"{par}-{type}", "data"),]
        states = []
        return outputs, inputs, states, False



 

    def create_parameter_dropdowns(self,store_data,name):
        param=self.callback_file[name]
        # par=param['par']
        # loa=param['loa']
        # type=param['type']
        # mal=param['mal']
        lst=param['lst'] 
        share=param['share'] 
        # print('I m store +++++++++++++++mapppppppppppppppppppppppppppppppppppp+++++++++++', store_data)
        sav=[]
    
        print('[[[[[[[[[[[lang]]]]]]]]]]]',self.file_path_org)
        pag='lang'
        pag=os.path.basename(self.file_path_org)
        for gvali in lst:
        # for gvali in ['nam_gen','path_head']:
            nam_gen_show = store_data[gvali]

            dropdown_options = [
                {
                    "label": x,
                    "value": x,
                    "style": dropdown_options_style
                }
                for x in nam_gen_show
            ] 
            idx=f'dropdown_{gvali}_{pag}_param'# if share =='app' else f'dropdown_{share}_{gvali}_param' 
            # idx=f'dropdown_{gvali}_param'# if share =='app' else f'dropdown_{share}_{gvali}_param' 
            dropdown = dcc.Dropdown(
                options=dropdown_options,
                id=idx,
                value=dropdown_options[0]["value"] if dropdown_options else None,
                placeholder=f"Select {gvali}", 
                style=box_style,
            )   

            sav.append(dbc.Col(
                                dropdown,
                                xs=12,
                                sm=6,
                                md=4,
                                lg=3,                        
                                style={"width": "100%" }
                            ))

        return sav



 