

import sys
import os
import sys 
import pickle
# import dash
from dash import dcc, html, dash_table, Input, Output, State, callback 
import dash_bootstrap_components as dbc
import numpy as np 
from mod.kalman.lorenz.neld_fun_0.help_funn import get_color
import mod.kalman.lorenz.neld_fun_0.help_fun as hf
# import geometry as geo
# import help_plotly as hpp
import mod.kalman.lorenz.apps.help_plotly as hp
from mod.kalman.lorenz.apps.help_plotly import aka_plot 
# import density as den
import plotly.graph_objects as go  
import pandas as pd
import pickle
from sklearn.metrics import roc_curve, auc
import pickle




ccoll = [
        'red', 'yellow', 'blue', 'green', 'black', 'purple',
        'orange', 'pink', 'brown', 'cyan', 'magenta', 'lime',
        'teal', 'navy', 'maroon', 'olive', 'gold', 'silver'
    ] 
     
ccol=[]
for _ in range( 1000):
    ccol=ccol+ccoll

 
def get_metric(
    akp,mode, metric, metric_name, width=500, height=500, 
    nbinsx=30, xtitle='Length', ytitle='Count'
):
    metric_map = {
        metric_name[0]: 0,
        metric_name[1]: 1,
        metric_name[2]: 2,
    }
    
    if mode not in metric_map:
        raise ValueError(f"Invalid mode: {mode}")

    index = metric_map[mode]
    data = metric[:, index]
    
    title = f'{mode} Histogram'

    scatter_comp, layout_comp = hp.Plotly_histogram_return(
        data, nbinsx=nbinsx, title=title, xtitle=xtitle, ytitle=ytitle, width=width, height=height
    )
    
    return akp.Plotly_Figure(data=scatter_comp, layout=layout_comp)

import importlib




def loss_fn(output, target):
    return np.mean(np.square(output - target))
 

class class_data():
    def __init__(self):
        pass 




    def get_neld_data(self,data_studied,nam_gen): 
        doc_module = importlib.import_module(self.path_dict['doc'])  
        # dict_param=doc_module.dict_param
        gdas=doc_module.gdas
        # nam_gen=doc_module.nam_gen 
        # self.data_studied=action=doc_module.action
        self.file_path_org=doc_module.file_path_org
        self.file_path_data= doc_module.file_path_data
        self.obj_org_path=obj_org_path=doc_module.obj_org_path

         
        nam=f'{nam_gen}_{data_studied}'
        self.neld_data = gdas.part(nam)   
        print('[[[[[[[[[[callback_file]]nam_gen        ]]]]]]]]',nam_gen,self.neld_data)  
        # print('[[[[[[[[[[callback_file]]nam_gen        ]]]]]]]]',nam_gen,file_path_data,file_path_org,dict_param,neld_data)  
        self.neld_names=self.neld_data['neld_names']

    def get_data(self,model_sufix,data_studied,index,neld_data,nam_gen=None,):




        '''

        doc_module = importlib.import_module(f'ginn.{nam_gen}')  
        dict_param=doc_module.dict_param
        gdas=doc_module.gdas
        nam_gen=doc_module.nam_gen 
        # self.data_studied=action=doc_module.action
        file_path_org=doc_module.file_path_org
        file_path_data= doc_module.file_path_data

        
        model_sufix_show=doc_module.dnn_modes
        path_heads_show=doc_module.path_heads_show
        path_dirs_show=doc_module.path_dirs_show 
        # self.model_sufix=doc_module.model_sufix
        # self.data_studied=action=doc_module.data_studied
        self.obj_org_path=obj_org_path=doc_module.obj_org_path
        nam=f'{nam_gen}_{data_studied}'
        neld_data = gdas.part(nam)   
        print('[[[[[[[[[[callback_file]]nam_gen        ]]]]]]]]',nam_gen,file_path_data,file_path_org,dict_param,neld_data)  
       

'''


        obj_org_path=self.obj_org_path



        # obj_org_path=None if nam_gen is None else os.path.join(self.file_path_data,nam_gen) 
        path_train=self.path_train
        # if neld_data is not None:
        self.neld_path_inits= neld_data['neld_path_inits']
        self.neld_names= neld_data['neld_names']
        neld_data['cpath'][0]=nam_gen
        # self.neld_namess= neld_data['neld_namess']
        self.pari.update(dict(neld_data=neld_data,
                              obj_org_path=obj_org_path,
                              data_studied=data_studied,
                              neld_path_inits=self.neld_path_inits)) 
        print('9999999999999999999999999999999999 starting',neld_data,path_train) 
        self.get_names_path(neld_data=neld_data,
                            obj_org_path=obj_org_path,
                            data_studied=data_studied,
                            neld_path_inits=self.neld_path_inits)
        self.get_model_opt_name(model_sufix=model_sufix,model_type=self.model_type )
        self.get_neld_name(data_studied=data_studied,
                           index=index,
                            # neld_names= self.neld_names,
                            neld_namess=self.neld_namess,  
                            obj_org_path=obj_org_path,
                            # neld_path_inits=self.neld_path_inits,
                            # **self.param_dic['data']['get_neld_name'],
                            ) 
        cxc=self.param_dic['data']['get_neld_name']['dict_neld_path'] 
        file_path_feat=self.dict_neld['path'][cxc]['file_path_feat']
        file_path=self.file_path=self.dict_neld['path'][cxc]['file_path'] 
        smod_path= self.path_file[path_train['dest_hmod_path']] 
        # smod_path= self.path_file[path_train['dest_smod_path']]  
        neld_name=self.neld_name 
        neld_namess=self.neld_namess 
        self.plot_data_iou=None 
        self.model_shap_dic={}
        self.plot_data_iou_dic={}
        self.plot_data_center_curv={} 
        self.plot_data_cylinder_heatmap={} 
        self.annot_intensity={ke:{} for ke in self.inten_file_train} 
        # self.logit_intensity_head_neck={ke:{} for ke in self.intensity_head_neck_logit} 
        # self.logit_intensity={ke:{} for ke in self.intensity_smods_logit} 
        self.logit_intensity={ke:{} for ke in self.intensity_logit} 
        self.iou_count={}
        self.scatter_loss_dic={}
        self.scatter_iou_dic={}
        self.scatter_loss_smod_dic={}
        self.scatter_iou_smod_dic={}
        self.scatter_auc_smod_dic={}
        self.scatter_dice_smod_dic={}
        self.metric_total_dic={}
       #   self.metric_total_dic['union']['single']={}
        self.metric_total_dic={}
        self.metric_total_dic['single']={}
        self.metric_total_dic['union']={}
        self.metric_total_dic['single_dice']={}
        self.metric_total_dic['union_dice']={}
        self.iou_tr={}
 

        # self.metric_total_dic = {
        #     key: {}
        #     for iii in self.neld_path_original_mm['keys'].values() 
        #     for key in (f'single_{iii}', f'union_{iii}')
        # } 
        # print('[[[[[[[[[=============================]]]]]]]]]',self.path_heads,self.path_heads_show)
        for path_head in self.model_sufix_dic['path_heads_show']:
            for model_suf in self.model_sufix_dic['model_sufix_show']:
                for path    in self.model_sufix_dic['path_dirs_show']: 
                    id_path=f'{path_head}_{model_suf}_{path}' 
                    # if not path in self.path_display_dic['path']:
                    #     continue 
                    if id_path not in self.path_file:
                        continue
                    fgff=os.path.join(self.path_file[id_path], self.txt_smod_iou)
                    if os.path.exists(fgff):
                        self.iou_count[id_path]=  np.loadtxt(fgff, dtype=float)    

                    path_grap_center_curv=os.path.join(self.path_file[id_path] ,'plot_data_center_curv.pkl')
                    if os.path.exists(path_grap_center_curv):
                        with open(os.path.join(path_grap_center_curv), "rb") as file:
                            self.plot_data_center_curv[id_path] = pickle.load(file) 

                    path_grap_center_curv=os.path.join(self.path_file[id_path] ,'cylinder_heatmap.pkl')
                    if os.path.exists(path_grap_center_curv):
                        with open(os.path.join(path_grap_center_curv), "rb") as file:
                            self.plot_data_cylinder_heatmap[id_path] = pickle.load(file) 
                            
                    for intensity_type in self.inten_file_train :
                        path_grap_center_curv=os.path.join(self.path_file[id_path] ,f'{intensity_type}.txt')
                        if os.path.exists(path_grap_center_curv):
                            self.annot_intensity[intensity_type][id_path] = path_grap_center_curv


                    ''' DONT DELETE
                    for intensity_type in self.intensity_smods_logit:
                        path_grap_center_curv=self.path_file_sub[intensity_type][id_path] 
                        if os.path.exists(path_grap_center_curv):
                            self.logit_intensity[intensity_type][id_path] = path_grap_center_curv





                    self.scatter_loss_data=[] 
                    tyy =self.inten_file_model_head_neck_loss[0]
                    loss_path=self.path_file_sub[tyy][id_path]
                    # print('ooop------===================================',loss_path)

                    if os.path.exists(loss_path):
                        self.loss_data = np.loadtxt(loss_path, dtype=float)  
                        self.loss_data= np.vstack((np.arange(len(self.loss_data)),self.loss_data)).T
                        # print('pp----------',self.loss_data.shape)
                        self.scatter_loss_data.append(
                                hf.plotly_scatter(points=self.loss_data ,
                                                color='red',
                                                size=4.07,
                                                opacity=.8,
                                                name='Approx')
                                                ) 
                    self.scatter_loss_dic[id_path]=  self.scatter_loss_data
 

                    tyy =self.inten_file_model_head_neck_iou[0] 
                    self.scatter_iou_data=[]   
                    for ii,(tyyy,nam,couleur) in enumerate(zip(self.inten_file_model_train_iou[:-1],['head','neck','hmod'],['red','green','blue'])):
                        # iou_path=self.model_dir_path['head_neck']['iou'][ii]
                        iou_path=self.path_file_sub[tyy][id_path][tyyy]
                        if os.path.exists(iou_path):
                            _data=np.loadtxt(iou_path,dtype=float)
                            print('pp----------',_data.shape) 
                            for ii in range(_data.shape[1]):
                                _dataa= np.hstack((np.arange(_data.shape[0]).reshape(-1,1),_data[:,ii:ii+1]))
                                self.scatter_iou_data.append(hf.plotly_scatter(points=_dataa ,
                                                        color=couleur,
                                                        size=4.07,
                                                        opacity=.8,
                                                        name=f'{nam}_{ii}'))

                    self.scatter_iou_dic[id_path]=self.scatter_iou_data

'''



                    scatter_loss_data=[] 
                    tyy =self.inten_file_model_smod_loss[0]
                    loss_path=self.path_file_sub[tyy].get(id_path,None)
                    # print('ooop------',loss_path,self.path_file_sub[tyy].keys())
                    # print('ooop------',id_path,id_path in list(self.path_file_sub[tyy].keys()))
                    # if loss_path is None:
                        # continueloss_path is not None and 
                    if loss_path is not None and os.path.exists(loss_path):
                        self.loss_data = np.loadtxt(loss_path, dtype=float,ndmin=1)  
                        # print('pp----------',self.loss_data )
                        self.loss_data= np.vstack((np.arange(len(self.loss_data)),self.loss_data)).T
                        scatter_loss_data.append(
                                hf.plotly_scatter(points=self.loss_data ,
                                                color='red',
                                                size=4.07,
                                                opacity=.8,
                                                name='Approx',)
                                                ) 

                    self.scatter_loss_smod_dic[id_path]= scatter_loss_data

                                      
 



                    clor=['red','blue' ]
                    vds=[1,-1]
                    valss=['hmod','smod']
                    figg=go.Figure()
                    # tyy =self.inten_file_model_shap[0]  
                    # iou_path=self.path_file_sub[tyy][id_path] 
                    self.model_shap=[]   
                    head_neck_path = 'dest_hmod_path'
                    pathh=path_train[head_neck_path]
                    smod_path_save=     self.path_file[f'result_{pathh}']
                    iou_path=os.path.join(smod_path_save,'shap.csv') 
                     
                    if os.path.exists(iou_path):
                        # print('---------------->>>>>>>>><<<<<<<<<<<<<<<,,',iou_path)

                        df = pd.read_csv(iou_path)   
                        for nam,cl,vd,va in zip(df.columns[1:],clor,vds,valss):
                            figg.add_trace(
                                go.Bar(
                                    x=vd*df[nam][::-1],
                                    y=df['Feature'][::-1],
                                    orientation='h',
                                    marker_color=cl,
                                    name=f'{va}'
                                )
                            )  
                        figg.update_layout(
                            title='Diverging SHAP Summary',
                            barmode='relative',
                            xaxis_title='SHAP value',
                            yaxis_title='Feature',
                            xaxis=dict(zeroline=True),
                            bargap=.2,
                        )

                    self.model_shap_dic[id_path]=figg






        self.metric_total_dic['roc_curve']={}
        self.metric_total_dic['roc_curve']['curve']={'smod':[],'hmod':[]}
        self.metric_total_dic['roc_curve']['score']={'smod':[],'hmod':[]}
        icoci=0
        for iii,(true_path,true_key) in enumerate( self.neld_path_original_mm['keys'].items() ):
            for path_head in self.model_sufix_dic['path_heads_show']:
                for model_suf in self.model_sufix_dic['model_sufix_show']:
                    for path in self.model_sufix_dic['path_dirs_show']: 
                        id_path=f'{path_head}_{model_suf}_{path}'
                        
                        if id_path not in self.path_file:
                            continue
                        # if not path in self.path_display_dic['path']:
                        #     continue
                        # if not id_path in ppatt:
                        #     continue
                        nhh=self.model_sufix_dic['model_sufix_dic'][model_suf],self.model_sufix_dic['path_dirs_dic'][path]
                        path_head_sh=self.model_sufix_dic['path_heads_dic'][path_head]
                        id_path_nam=f'{path_head_sh}_{nhh[0]}_{nhh[1]}'
                        id_pathss=f'{path_head}_{model_suf}_{path}_{true_key}'
                        '''
                        path_grap_iou=os.path.join(self.path_file[id_path] ,self.pkl_mp)
                        if os.path.exists(path_grap_iou):
                            with open(path_grap_iou, "rb") as file:
                                mp = pickle.load(file) 
                            self.iou_tr[id_pathss]=iou_train(
                                                path_true=self.path_file[true_path],
                                                path_appr=self.path_file[id_path],
                                                mp=mp,
                                                ) 
                            
                        path_grap_iou=os.path.join(self.path_file[id_path] ,f'plot_iou_graph_{true_key}.pkl')
                        if os.path.exists(path_grap_iou): 
                                with open(os.path.join(path_grap_iou), "rb") as file: 
                                    self.plot_data_iou_dic[id_pathss]=pickle.load(file) 
                        # true_path=self.neld_path_original_mm['keys']['true_0']
                        path_grap_iou=os.path.join(self.path_file[id_path] ,'intensity_smods_logit.txt')
                        path_grap_true=os.path.join(self.path_file[true_path] ,'intensity_smods_logit.txt') 
                        print(']]]]]]]][[[[[[[[[[[[[]]]]]]]]]]]]]',path_grap_iou)
                        if os.path.exists(path_grap_iou) and os.path.exists(path_grap_true): 
                            print(']]]]]]]][[[[[[[[[[[[[]]]]]]]]]]]]]',path_grap_iou)
                            self.metric_total_dic['roc_curve']['true']+=np.loadtxt(path_grap_true,dtype=float)
                            self.metric_total_dic['roc_curve']['score']+=np.loadtxt(path_grap_iou,dtype=float)
'''

                        if model_suf !='save':
                            smod_path_save=     self.path_file[f'result_{id_path}']
                            for nm,nmm in zip(['hmod','smod'],['hmod','smod']):
                                metric_path=os.path.join( smod_path_save,f'roc_{nm}_{true_key}.txt') 
                                if os.path.exists(metric_path):
                                    roc=np.loadtxt(metric_path,dtype=float)
                                    scr=auc(roc[:,0], roc[:,1])
                                    # print('======[[[[[[]]]]]]',roc)
                                    self.metric_total_dic['roc_curve']['score'][nm]=scr
                                    self.metric_total_dic['roc_curve']['curve'][nm].append(go.Scatter( 
                                                                            x=roc[:,0], 
                                                                            y=roc[:,1], 
                                                                            mode="lines", 
                                                                            name=f"{id_path_nam} (AUC {nmm} = {scr:.3f})", 
                                                                            line=dict(width=3),
                                                                            marker=dict(color=ccol[icoci]),
                                                                            ) ,
                                                                            )
                            icoci+=1

 



            if len(self.metric_total_dic['single'])>0:
                df_union=pd.DataFrame(self.metric_total_dic['union']).T.sort_values(by='accuracy', ascending=False)
                df_union.to_csv(os.path.join(self.path_file[f'result_appr'],f'metric_union_{true_key}.csv'))
                df_single=pd.DataFrame(self.metric_total_dic['single']).T.sort_values(by='accuracy', ascending=False)
                df_single.to_csv(os.path.join(self.path_file[f'result_appr'],f'metric_single_{true_key}.csv'))
            if len(self.metric_total_dic['single_dice'])>0:
                df_union=pd.DataFrame(self.metric_total_dic['union_dice']).T.sort_values(by='accuracy', ascending=False)
                df_union.to_csv(os.path.join(self.path_file[f'result_appr'],f'metric_union_dice_{true_key}.csv'))
                df_single=pd.DataFrame(self.metric_total_dic['single_dice']).T.sort_values(by='accuracy', ascending=False)
                df_single.to_csv(os.path.join(self.path_file[f'result_appr'],f'metric_single_dice_{true_key}.csv'))





        # self.file_path =file_path
        self.neld_name=neld_name
        self.itera=index  
        self.vertices_0=self.vertices_00=self.faces=np.array([[0,0,0]]) 

    def update_params(self, args):
        self.param_inputii={} 
        for inp, value in zip(self.Input, args):
            key =inp.component_id
            self.param_inputii[key] = value
        self.param_inputii['index']=None
        self.param_inputii['get_return']=True
        self.param_inputii['hide_button']=True
    
    # def Get_output(self, path_head,model_suf,path,mode,  width, height ,templ,inbin,ndex=None,get_return=True,hide_button_tf=True): 
    def Get_output(self): 
        (nam_gen,path_head,action,model_suf,path,root1,mode,  width, height,templ )=(self.param_inputii[mm] for mm in self.Input_ids+self.Input_idsST)
    

        self.get_neld_data(
                    nam_gen=nam_gen,  
                    data_studied=action, 
                    ) 
 
        neld_names=self.neld_names

        if not 'neld_namess' in self.neld_data:
            self.neld_data['neld_namess']= [f'd{str(i).zfill(3)}' for i in range(len(neld_names))]         
        du={mm:nn for mm,nn in zip(self.neld_data['neld_namess'],self.neld_data['neld_names'])}
 
        root=du[root1]  
        get_return = self.param_inputii['get_return']
        hide_button_tf = self.param_inputii['hide_button']
        kik={ke:va for va,ke in enumerate(self.neld_names)}
        index=kik[root]
    # def Get_output(self,nam_gen,path_head,action,model_suf,path_dir,path,mode,mnm,  width, height ,templ,index=None,get_return=True,hide_button_tf=True): 
        print('[[[[[[[[[[[[[[[[[[[[-------------===]]]]]]]]]]]]]]]]]]]]',index,self.neld_names,root,nam_gen,path_head,action,model_suf,path,mode,  width, height ,templ,index,get_return,hide_button_tf)
        neldd=root

        neldd, clusts,    intensity_type=None,None,None
        path_train=self.path_train
        self.data_studied=action
        self.model_type,self.model_sufix,self.path_dir=path_head,model_suf,path
        true_keys='true_0'
        self.obj_org_path=obj_org_path=os.path.join(self.file_path_org,'data',nam_gen)
        # path_file_dir=os.path.join(obj_org_path,self.data_studied,'data',path_dir,self.model_type,self.model_sufix ,self.path_dir,'path_files.pkl')
        path_file_dir=os.path.join(self.file_path_org,'data',nam_gen ,'path_files.pkl')
        
        # model_suf=self.model_sufix if path_head == 'pinn' else 'save'  
        id_path=f'{true_keys}_save_save' if path_head=='true' else f'{path_head}_{model_suf}_{path}'
        id_pathss=f'{path_head}_{model_suf}_{path}_{true_keys}'
        print('[[[[[[[[[[[[[nam gen]]]]]]]]]]]]]',nam_gen)
        self.pari.update(dict( 
            path_file_dir=path_file_dir,  
            obj_org_path=obj_org_path,
            model_sufix=model_suf,
            model_type=self.model_type,
            data_studied=self.data_studied,
                # path_train,
                # index=0,
                  )
                )
        self.get_gen_path(  

                # path_train, 
                )
        self.get_data(
                    nam_gen=nam_gen,
                    neld_data=self.neld_data,
                    model_sufix=model_suf,
                    data_studied=self.data_studied,
                    index=index, 
                    ) 
        print('---------',true_keys,path,mode,neldd,path_head,'---inte',intensity_type,'===',model_suf,self.model_sufix) 
        neld_name=self.neld_name 
        bcouleur=self.bcouleur
        fsize=self.fsize 
        ppp= path.split('_')
        
        path_init=f'{ppp[0]}'
        ppp=ppp[1:]
        for pp in ppp[:-1]:
            path_init=f'{path_init}_{pp}'
        print('0000000',path_init,ppp)
        if templ=="plotly_dark":
            fcouleur='white'
            ffcouleur='black'#'black'
        else:
            fcouleur='black'
            ffcouleur='white'
        akp=aka_plot(tcouleur=templ,
                    bcouleur=bcouleur,
                    fcouleur=fcouleur,
                    fsize=fsize)
 
        vertices_pl= self.vertices_0 if neldd=='smooth' else self.vertices_00 

        intensity=None;intensity_path=''
        if intensity_type in self.inten_file_sub:  
            intensity_path =self.path_file_sub[intensity_type][id_path]
            # intensity_path = intensity_path if os.path.exists(intensity_path) else self.path_file_sub[intensity_type][f'{path_head}_{model_suf}_{path_init}']
            # intensity=np.loadtxt(intensity_path, dtype=float)
            # print('smod--------',intensity_path)
            if not os.path.exists(intensity_path):
                print('[[[[[Fail intensity_path]]]]]--------',intensity_path)

        elif intensity_type in self.inten_file:
            intensity_path =self.path_file_sub[intensity_type][id_path] 
            intensity_path = intensity_path if os.path.exists(intensity_path) else self.path_file_sub[intensity_type][f'{path_head}_{model_suf}_{path_init}']
            if not os.path.exists(intensity_path):
                print('[[[[[Fail intensity_path]]]]]--------',intensity_path)
        elif intensity_type in self.inten_pca:
            intensity_path =self.path_file_sub[intensity_type][id_path]
            intensity_path = intensity_path if os.path.exists(intensity_path) else self.path_file_sub[intensity_type][f'{path_head}_{model_suf}_{path_init}']
            # intensity=np.loadtxt(intensity_path, dtype=float)
            # print('smod--------',intensity_path) 
        elif intensity_type in self.inten_file_train:
            intensity_path =self.path_file_sub[intensity_type][id_path]#self.annot_intensity[intensity_type][id_path]#
            if not os.path.exists(intensity_path):
                print('[[[[[Fail intensity_path]]]]]--------',intensity_path) 
            # intensity=np.loadtxt(intensity_path, dtype=float)
        elif intensity_type in self.intensity_logit:
            intensity_path =self.path_file_sub[intensity_type][id_path]#self.annot_intensity[intensity_type][id_path]#
            if not os.path.exists(intensity_path):
                print('[[[[[Fail intensity_path]]]]]--------',intensity_path)
        elif intensity_type in self.intensity_head_neck_logit:
            intensity_path =self.path_file_sub[intensity_type][id_path]#self.annot_intensity[intensity_type][id_path]#
            if not os.path.exists(intensity_path):
                print('[[[[[Fail intensity_path]]]]]--------',intensity_path)

        elif intensity_type in self.inten_file_model_head_neck:
            intensity_path =self.path_file_sub[intensity_type][id_path]
            intensity_path = intensity_path if os.path.exists(intensity_path) else self.path_file_sub[intensity_type][f'{path_head}_{model_suf}_{path_init}']
            # print('smod--------',intensity_path) 
            # intensity=np.loadtxt(intensity_path, dtype=float)


        elif intensity_type in self.base_features_dict.keys():
            intensity_path =self.path_file_sub[intensity_type][id_path]
            # intensity_path = intensity_path if os.path.exists(intensity_path) else self.path_file_sub[intensity_type][f'{path_head}_{model_suf}_{path_init}']
            # intensity=np.loadtxt(intensity_path, dtype=float)
 
        intensity= None 
        if os.path.exists(intensity_path):
            intensity=np.loadtxt(intensity_path, dtype=float)
            print('smod--------',intensity_path)
        else:
            print('Intensity DOESNT EXIST --------',intensity_path)
                # intensity_path =self.path_file_sub[intensity_type]['true_save_save'] 
            # elif intensity_type in ['smod_body','head_neck_body']: 
            #     intensity_path=self.inten[intensity_type] 
            # else:
            #     intensity_path = self.neld_mapping[neldd]["intensity"][intensity_type] 

        if intensity_type in ["gauss_curv_init","mean_curv_init"]:
            vertices_pl=self.vertices_00
        elif intensity_type in ["gauss_curv_smooth","mean_curv_smooth"]:
            vertices_pl=self.vertices_0


        figure=go.Figure()
        # self.get_figure(true_keys,path_head,model_suf, path,vertices_pl,intensity, clusts, width, height,ffcouleur=ffcouleur  )
 
    
        self.layout = go.Layout(width=width, 
                        height=height,
                        title=f'Dynamics', 
                        ) 

        self.scene=dict(
            xaxis=dict(showgrid=False, zeroline=False, showline=False, showticklabels=False, title='',backgroundcolor=ffcouleur),
            yaxis=dict(showgrid=False, zeroline=False, showline=False, showticklabels=False, title='',backgroundcolor=ffcouleur),
            zaxis=dict(showgrid=False, zeroline=False, showline=False, showticklabels=False, title='',backgroundcolor=ffcouleur),
            bgcolor=ffcouleur 
        )

        if mode=='algorithm':
            figure=  self.figure_3d 


        elif mode=='comparison':
            figure=akp.Plotly_Figure(data=self.scatter, layout=self.layout)
            figure.update_layout(scene=self.scene)  
        elif mode=='skeleton':
            if self.plot_data_center_curv is not None:  
                scatterr=self.plot_data_center_curv[id_path][clusts][0:1]
                skl_path=os.path.join(self.file_path_feat, self.txt_skl_vertices)
                if os.path.exists(skl_path): 
                    scatterr.append(hf.plotly_scatter(points=np.loadtxt(skl_path,dtype=float), color='yellow', size=5.3, name='skeleton smooth.',opacity=0.5))
                skl_path=os.path.join(self.file_path_feat, self.txt_skl_vertices_org)
                if os.path.exists(skl_path): 
                    scatterr.append(hf.plotly_scatter(points=np.loadtxt(skl_path,dtype=float), color='green', size=5.3, name='skeleton init.',opacity=0.5))
                for val in self.plot_data_center_curv[id_path][clusts][1:]:
                    scatterr.append(val)
                figure=akp.Plotly_Figure(data= scatterr, layout=self.layout)
                figure.update_layout(scene=self.scene)   
        elif mode in ['heatmap_cylinder','heatmap_cylinder_surface']:
            if self.plot_data_cylinder_heatmap is not None: 
                *pathc, last = path.split('_') 
                if mode =='heatmap_cylinder':
                    data=[self.plot_data_cylinder_heatmap[id_path].density_heatmap,self.plot_data_cylinder_heatmap[id_path].density_heatmap_points,self.plot_data_cylinder_heatmap[id_path].density_heatmap_points_org]
                    figure=akp.Plotly_Figure(data= data, layout=self.layout) 
                    figure.update_layout(
                        xaxis=dict(showgrid=False),  
                        yaxis=dict(showgrid=False)  
                    ) 
                elif mode =='heatmap_cylinder_surface':
                    data=[self.plot_data_cylinder_heatmap[id_path].density_heatmap_surface]
                    data.extend(self.plot_data_cylinder_heatmap[id_path].density_org_points)
                    figure=akp.Plotly_Figure(data= data, layout=self.layout)
                figure.update_layout(scene=self.scene)
        elif mode=='IOU':
            # if self.plot_data_iou  is not None: 
            #     figure=akp.Plotly_Figure(data= self.plot_data_iou, layout=self.layout)
            #     figure.update_layout(scene=self.scene)  
            if  len(self.plot_data_iou_dic)>0:  

                figure=akp.Plotly_Figure(data= self.plot_data_iou_dic[id_pathss] , layout=self.layout)
                figure.update_layout(scene=self.scene) 




        elif mode =='INV_MEASURE':
            ii=nbin
            import pickle
            from mod.kalman.lorenz.neld_pinn_0.help_fun import aka_fun
            from mod.kalman.lorenz.neld_pinn_0.Graph import mygraph

            path_gen=self.obj_org_path.split('\\')
            path_gen=os.path.join(*path_gen)
            path_init_param=os.path.join(path_gen,f"param.pkl")
            with open(path_init_param, "rb") as f:
                pm = pickle.load(f)
            path_1=os.path.dirname(os.path.join(self.neld_path_org_new).replace('\\','/'))
            print('[[[[[pathhhh]]]]]',path_1)
            time=np.loadtxt(os.path.join(path_1,f"time.txt").replace('\\','/'),dtype=float)
            qq=np.loadtxt(os.path.join(path_1,f"qq_{ii}.txt").replace('\\','/'),dtype=float)

            Ntime= pm.N
            Ndim = pm.dim
            Nperiod = pm.Nperiod-1
            xint = 0
            yint = xint+pm.nPart
            Q1,Q2 =[ aka_fun().vec_to_mat_(qq,x,Ntime,Nperiod) for x in [xint,yint]]




            # tcouleur = 'plotly_dark'
            # bcouleur = 'navy'
            # fcouleur = 'white'
            # fsize = 20


            bheight = 600
            bwidth = 600 

            mygraph_ = mygraph(tcouleur=templ,
                            bcouleur=bcouleur,
                            fcouleur=fcouleur,
                            fsize=fsize)

            figure=mygraph_.plot_history_matrixxy(Q1,Q2,time.reshape(-1,1),bheight,bwidth)


        elif mode =='Metrics':
            cm=[]
            path_1=self.neld_path_org_new
            labelx,labely=['MSE (noise)','MSE '],[]
            # ,self.model_sufix_dic['path_heads_show']
            # sav={nn:{mm:[] for mm in ['MSE (noise)','MSE ']} for nn in self.model_sufix_dic['path_heads_show']}
            for model_type in self.model_sufix_dic['path_heads_show']: 
                pa=os.path.join(os.path.dirname(os.path.dirname(path_1)),f'{model_type}_loss.txt')
                print('==============================exist',os.path.exists(pa),pa)
                if os.path.exists(pa):
                    jk=np.loadtxt(pa, float)
                    cm.append(jk)
                    labely.append(model_type)
                # for nn,mm in zip(jk,['MSE (noise)','MSE ']):
                #     mt.append()
 
            if len(cm) > 0:
                cm = np.array(cm) 
                if cm.ndim >= 2:
                    ine = np.argsort(cm[:, 0])
                    cm = cm[ine]
                    labely = [labely[ii] for ii in ine]
                else: 
                    ine = np.argsort(cm)
                    cm = cm[ine]
                    labely = [labely[ii] for ii in ine]
            else:
                print("Warning: No valid data found in any loss files. ")
                
            data=go.Heatmap(
                z=cm,
                x=labelx,  # Predicted labels
                y=labely,  # Actual labels
                colorscale='Blues',
                text=cm,  # Show numbers in cells
                texttemplate="%{text}",  # Format as numbers
                # hovertemplate="Actual: %{y}<br>Predicted: %{x}<br>Count: %{z}<extra></extra>",
            )

            layout = go.Layout(width=width, 
                            height=height,
                            title=mode, 
                            ) 

            figure=akp.Plotly_Figure(data= data , layout=layout)
            figure.update_layout(scene=self.scene) 
            figure.update_layout(
                xaxis=dict(side="top"),       
                yaxis=dict(autorange="reversed")  
            )

        elif mode=='logit': 
            self.hist_slider['min']=-1
            self.hist_slider['max']=10
            self.hist_slider['step']=1
            self.hist_slider['value']=10
            self.hist_slider['marks']={i: f'{i}' for i in range(-1, 10, 2)} 
            path_1=self.neld_path_org_new
            path_gen=self.obj_org_path.split('\\')
            path_gen=os.path.join(*path_gen)
            path_init_param=os.path.join(path_gen,f"param.pkl") 
            scatterr=[]  
            path_true=os.path.join(path_1,f"true.txt")
            path_obs=os.path.join(path_1,f"obs.txt")

            ii=0 

            ku,kuu='inference','true' 

            for mn,tyy,tyyy in zip(['initial',],self.intensity_logit_dict[ku][:1],self.intensity_logit_dict[kuu][:1]):
                patth,patthy=self.path_file_sub[tyy][id_path].replace('\\','/'),self.path_file_sub[tyyy][id_path].replace('\\','/')

                print('[[[exist]]]',os.path.exists(patth), os.path.exists(patthy),patth,patthy)
                if os.path.exists(patth) and os.path.exists(patthy): 

                    yy,tru=np.loadtxt(patth, dtype=float),np.loadtxt(patthy, dtype=float)   
                    err=loss_fn(yy,tru) 
                    print('[[[]]]',[yy.shape   ]) 
                    scatterr.append(
                        hf.plotly_scatter(
                            points=yy,
                            color=ccoll[ii] ,
                            size=4.3,
                            mode='lines+markers', 
                            symbol='square' ,
                            name=f'Inference',
                            opacity=0.9
                        ) 
                    )
                    scatterr.append(
                        hf.plotly_scatter(
                            points=tru,
                            color=ccoll[ii+1] ,
                            size=4.3,
                            mode='lines+markers', 
                            symbol="cross"  ,
                            name=f'True+noise<br>MSE:{err:.2e}',
                            opacity=0.8
                        ) 
                    )

            if os.path.exists(path_true):# and os.path.exists(path_obs):
                tru_init=np.loadtxt(path_true, dtype=float)    
                err=None if not os.path.exists(patth) else loss_fn(yy,tru_init)
 
                scatterr.append(
                    hf.plotly_scatter(
                        points=tru_init,
                        color=ccoll[ii+2] ,
                        size=4.3,
                        mode='lines+markers', 
                        symbol="cross"  ,
                        name=f'True<br>MSE:{err:.2e}',
                        opacity=0.8
                    ) 
                )

            jkk=self.intensity_logit_dict[kuu][1]
            patthh=self.path_file_sub[jkk][id_path].replace('\\','/')
            if os.path.exists(patthh):
                obs=np.loadtxt(patthh, dtype=float)

                scatterr.append(
                    hf.plotly_scatter(
                        points=obs,
                        color=ccoll[ii+3] ,
                        size=8.3, 
                        symbol="circle"  , 
                            name=f'Observable',
                        opacity=0.3
                    ) 
                )
            
            # [print('[im he=========------]',mm) for mm in [path_1,path_true,patth,patthy]]
            figure=akp.Plotly_Figure(data= scatterr , layout=self.layout)
            figure.update_layout(scene=self.scene) 
            # figure.update_layout(title=) 

            # figure.update_yaxes(title_text="Dynamics" )
            # figure.update_xaxes(title_text="Time", ) 
   

 
 



        elif mode in self.inten_file_model_head_neck:  
            self.layout = go.Layout(width=width, 
                            height=height,  )
            # ['model_hn_loss','model_hn_iou']
            if (mode == 'model_hn_loss' ) and id_path in self.scatter_loss_dic and len( self.scatter_loss_dic[id_path])>0: 
                figure=akp.Plotly_Figure(data=self.scatter_loss_dic[id_path], layout=self.layout)
                figure.update_layout(scene=self.scene) 
            elif (mode == 'model_hn_iou' ) and id_path in self.scatter_iou_dic and len(self.scatter_iou_dic[id_path])>0: 
                figure=akp.Plotly_Figure(data= self.scatter_iou_dic[id_path], layout=self.layout)
                figure.update_layout(scene=self.scene) 

            if (mode == 'model_sp_loss' ) and id_path in self.scatter_loss_smod_dic and len( self.scatter_loss_smod_dic[id_path])>0: 
                figure=akp.Plotly_Figure(data=self.scatter_loss_smod_dic[id_path], layout=self.layout)
                figure.update_layout(yaxis_type="log")
                figure.update_layout(scene=self.scene) 
            elif (mode == 'model_sp_iou' ) and id_path in self.scatter_iou_smod_dic and len(self.scatter_iou_smod_dic[id_path])>0: 
                print('----=====----','9088==============---========================================',len(self.scatter_iou_smod_dic[id_path]))
                figure=akp.Plotly_Figure(data= self.scatter_iou_smod_dic[id_path], layout=self.layout)
                figure.update_layout(scene=self.scene)
            elif (mode == 'model_sp_auc' ) and id_path in self.scatter_auc_smod_dic and len(self.scatter_auc_smod_dic[id_path])>0: 
                print('----=====----','model_sp_auc==============---========================================',len(self.scatter_auc_smod_dic[id_path]))
                figure=akp.Plotly_Figure(data= self.scatter_auc_smod_dic[id_path], layout=self.layout)
                figure.update_layout(scene=self.scene)
            elif (mode == 'model_sp_dice' ) and id_path in self.scatter_dice_smod_dic and len(self.scatter_dice_smod_dic[id_path])>0: 
                print('----=====----','model_sp_auc==============---========================================',len(self.scatter_dice_smod_dic[id_path]))
                figure=akp.Plotly_Figure(data= self.scatter_dice_smod_dic[id_path], layout=self.layout)
                figure.update_layout(scene=self.scene)
                          
            if path_head.startswith(('cML','CML','ML')): 
                figure.update_layout(
                    xaxis=dict(
                        side="bottom",  # keep the bottom axis if you want
                        showticklabels=False  # hide bottom labels if you only want top
                    ),
                    xaxis2=dict(
                        side="top",
                        overlaying="x",      
                        ticks="outside",
                        showticklabels=True,
                        tickmode="array",
                        tickvals=[0, 1, 2, 3],  
                        ticktext=['Accuracy','Precision','Recall','F1 Score'],
                    )
                )

                # Make sure your trace uses the top axis
                for trace in figure.data:
                    trace.update(xaxis="x2")

                                

        elif mode == 'model_shap':
            # figure=self.model_shap_dic[id_path]
            # print('figure',figure)
            figure=akp.Plotly_Figure(data= self.model_shap_dic[id_path]['data'], layout=self.model_shap_dic[id_path]['layout'])
            figure.update_layout(scene=self.scene) 
 
        else:
            figure=go.Figure() 
        self.figure=figure
        if mode not in ['INV_MEASURE',]:
            hp.hide_button(figure,hide_button_tf) 
        if get_return:
            return dcc.Graph(figure=figure) ,f'Sample : {neld_name}'
        else:
            self.figure=figure
            self.vertices_pl,self.intensity, self.clusts=vertices_pl,intensity, clusts
            self.akp=akp


 

from mod.kalman.lorenz.neld_fun_0.get_path import get_name

class get_app_param(get_name ):
    def __init__(self,
                 dropdown_path_head_option=None,
                 dropdown_model_suf_option=None,
                 dropdown_path_option=None,
                 dropdown_true_keys_option=None,):
        self.dropdown_path_head_option=dropdown_path_head_option
        self.dropdown_model_suf_option=dropdown_model_suf_option
        self.dropdown_path_option=dropdown_path_option
        self.dropdown_true_keys_option=dropdown_true_keys_option
        pass

        get_name.__init__(self) 
        self.dropdown_options_style=dropdown_options_style = {'color': 'white', 'background-color': 'gray'} 
        self.styles = {
            'pre': {
                'border': 'thin lightgrey solid',
                'overflowX': 'scroll'
            }
        }
        self.box_style = {
            'width': '100%',
            'padding': '3px',
            'font-size': '20px',
            'text-align-last': 'center',
            'margin': 'auto',  # Center-align the dropdown horizontally
            'background-color': 'black',
            'color': 'black'
        }



        self.template=template= [ "seaborn", "plotly_dark","plotly", "plotly_white", "ggplot2", "simple_white", "none"]
        # Color and style settings
        self.tcouleur = 'plotly_dark' 
        self.bcouleur = 'navy'
        self.fcouleur = ['white','black','grey']
        self.fsize = 16
        self.dropdown_template_option = []  
        for temp in template:
            self.dropdown_template_option.append({'label': f'{temp}', 
                                                    'value': f'{temp}', 
                                                    'style': dropdown_options_style})  


    '''

    def model_train(self):
        dropdown_options_style=self.dropdown_options_style
        self.dropdown_mode_option=dropdown_mode_option= [
                            {'label': 'Algorithm',      'value': 'algorithm', 'style': dropdown_options_style},
                            {'label': 'Comparison',     'value': 'comparison','style': dropdown_options_style}, 
                            {'label': 'Skeleton',      'value': 'skeleton', 'style': dropdown_options_style},
                            {'label': 'Results',        'value': 'result',    'style': dropdown_options_style}, 
                            {'label': 'Results Optimum','value': 'result_opt','style': dropdown_options_style}, 
                    ] 

        action_name='mode'
        self.dropdown_mode=dropdown_mode ={
                        'option':dropdown_mode_option,
                        'id'         :f'dropdown_{action_name}',
                        'value'      :'result',
                        'placeholder':f'Select {action_name}', 
        }


        ################################ METRIC #########################################


        self.dropdown_plot_option=[
                            # {'label': 'Distribution', 'value': 'dist',     'style': dropdown_options_style} ,
                            # {'label': 'Density',      'value': 'den',      'style': dropdown_options_style},
                            {'label': 'Accuracy',              'value': 'accuracy','style': dropdown_options_style},
                            {'label': 'IOU',                   'value': 'iou',     'style': dropdown_options_style},
                            {'label': 'Confusion Matrix',      'value': 'conf',    'style': dropdown_options_style},
                            {'label': 'Classification Report', 'value': 'report',  'style': dropdown_options_style},
                            {'label': 'Compare',               'value': 'compare', 'style': dropdown_options_style},
                            {'label': 'Logit',                 'value': 'logit',   'style': dropdown_options_style},
        ]
        ################################ INTENSITY #########################################
        self.dropdown_intensity_option= [
                            {'label': 'Clusterization',  'value': 'kmean',       'style': dropdown_options_style},
                            {'label': 'Segmentation',    'value': 'cluu_rad',    'style': dropdown_options_style},
                            {'label': 'smods',          'value': 'smods',      'style': dropdown_options_style},
                            {'label': 'smods True',     'value': 'smods true', 'style': dropdown_options_style},
                            {'label': 'Gauss Curvature', 'value': 'gauss',       'style': dropdown_options_style},
                            {'label': 'Mean Curvature',  'value': 'mean',        'style': dropdown_options_style},
                    ]

'''
    def model_test(self):
        dropdown_options_style=self.dropdown_options_style
        self.dropdown_mode_option= [
                            # {'label': 'Image',         'value': 'algorithm', 'style': dropdown_options_style},
                            # {'label': 'Comparison',        'value': 'comparison',    'style': dropdown_options_style},  
                            # {'label': 'Skeleton',      'value': 'skeleton', 'style': dropdown_options_style},
                            # {'label': 'Accuracy',              'value': 'accuracy','style': dropdown_options_style}, 
                            # {'label': 'ROC Curve',              'value': 'roc_curve','style': dropdown_options_style}, 
                            {'label': 'Metrics',                 'value': 'Metrics',   'style': dropdown_options_style},
                            {'label': 'Dynamics',                 'value': 'logit',   'style': dropdown_options_style},
                            # {'label': 'Invar_Measure',                 'value': 'INV_MEASURE',   'style': dropdown_options_style},
                    ]

        for mmnn in  self.inten_file_model_smod_loss:
            self.dropdown_mode_option.append({'label': mmnn,      'value': mmnn, 'style': dropdown_options_style},)
 
        '''
        # dropdown_mode_option.append({'label': 'IOU',      'value': 'IOU', 'style': dropdown_options_style},)INV_MEASURE
        for mmnn in self.inten_file_model_head_neck:
            dropdown_mode_option.append({'label': mmnn,      'value': mmnn, 'style': dropdown_options_style},)
 

        # metric_name=self.metrics_keys
        metric_name=[]


        metric_name.extend(['heatmap_cylinder','heatmap_cylinder_surface'])
        metric_name.extend(self.metrics_keys)
        # metric_name.extend(['heatmap_iou','heatmap_iou_union','histogram_iou'])
        self.metric_mapping = {
            "name": metric_name,
            "title": [f"{mm} Histogram" for mm in metric_name],
            "index": {mm: ii for ii, mm in enumerate(metric_name)},
            "xtitle": "Length",
            "ytitle": "Count",
        }
        self.metrics_combine={'diam_head_neck_length'  :{'key':  ['head_diameter','neck_diameter','smod_length'],
                                                       'label':['Head Diameter','Neck Diameter','smod Length']},
                              'vol_area_length_smod':{'key':['smod_vol','smod_area','smod_length'],
                                                        'label':['smod Volume','smod Area','smod Length']},
                              'vol_head_neck_smod'    :{'key':['head_vol','neck_vol','smod_vol'],
                                                     'label':['Head Volume','Neck Volume','smod Volume'],
                                                     },
                              'area_head_neck_smod'  :{'key':['head_area','neck_area','smod_area'],
                                                      'label':['Head Area','Neck Area','smod Area']},
                              'length_head_neck_smod':{'key':['head_length','neck_length','smod_length'],
                                                        'label':['Head Length','Neck Length','smod Length']},
                              }
        keys=list(self.metrics_combine.keys())
        self.metric_mapping_combine = {
            "name": keys,
            "title": [f"{self.metrics_combine[mm]['label'][0]} vs {self.metrics_combine[mm]['label'][1]} vs {self.metrics_combine[mm]['label'][2]}" for mm in keys],
            "index": {mm: ii for ii, mm in enumerate(keys)},
            "xtitle": [f"{self.metrics_combine[mm]['label'][0]}" for mm in keys],
            "ytitle":[f"{self.metrics_combine[mm]['label'][1]}" for mm in keys],
        }
        for mo in self.metric_mapping_combine['name']:
            dropdown_mode_option.append({'label': mo,        'value': mo,    'style': dropdown_options_style}) 
        for mo in self.metric_mapping['name']:
            dropdown_mode_option.append({'label': mo,        'value': mo,    'style': dropdown_options_style})
            self.dropdown_plot_option=
 '''
    
        action_name='mode'
        self.dropdown_mode ={
                        'option':self.dropdown_mode_option,
                        'id'         :f'dropdown_{action_name}',
                        'value'      :self.dropdown_mode_option[0]['value'],
                        'placeholder':f'Select {action_name}', 
        }
        ################################ METRIC #########################################


        # self.dropdown_plot_option=[ 
        #                     {'label': 'Accuracy',              'value': 'accuracy','style': dropdown_options_style},
        #                     {'label': 'IOU',                   'value': 'iou',     'style': dropdown_options_style},
        #                     {'label': 'Confusion Matrix',      'value': 'conf',    'style': dropdown_options_style},
        #                     {'label': 'Classification Report', 'value': 'report',  'style': dropdown_options_style},
        #                     {'label': 'Compare',               'value': 'compare', 'style': dropdown_options_style},
        #                     {'label': 'Logit',                 'value': 'logit',   'style': dropdown_options_style},
        # ] 

        self.dropdown_intensity_option=[]
        for intt,name in zip(self.inten_file_sub[1:],self.inten_file_sub_name[1:]):
            self.dropdown_intensity_option.append({'label': name,   'value': intt,       'style': dropdown_options_style})
        for intt in self.inten_file_train:
            self.dropdown_intensity_option.append({'label': intt,   'value': intt,       'style': dropdown_options_style}) 
        # for intt in self.inten_pca:
        #     self.dropdown_intensity_option.append({'label': intt,   'value': intt,       'style': dropdown_options_style})  
        for intt in self.inten_file:
            self.dropdown_intensity_option.append({'label': intt,   'value': intt,       'style': dropdown_options_style})  
        for intt in self.base_features_dict.keys():
            self.dropdown_intensity_option.append({'label': intt,   'value': intt,       'style': dropdown_options_style}) 
        for intt in self.intensity_logit:
            self.dropdown_intensity_option.append({'label': intt,   'value': intt,       'style': dropdown_options_style}) 
        for intt in self.intensity_head_neck_logit:
            self.dropdown_intensity_option.append({'label': intt,   'value': intt,       'style': dropdown_options_style}) 



    def more_param(self,id_name_end,model_type,model_sufix,path_dir,neld_name): 
        dropdown_options_style=self.dropdown_options_style 
        self.neld_name=neld_name 

        action_name='template' 
        id_name=f'dropdown_{action_name}_{id_name_end}'
        self.dropdown_template={
            'option'     :self.dropdown_template_option,
            'id'         :id_name,
            'value'      :self.dropdown_template_option[1]['value'],
            'placeholder':f'Select {action_name}', 
        }
        action_name='intensity'
        id_name=f'dropdown_{action_name}_{id_name_end}'
        self.dropdown_intensity ={
        'option':self.dropdown_intensity_option,
        'id'         :id_name,
        'value'      :self.dropdown_intensity_option[0]['value'],
        'placeholder':f'Select {action_name}', 
        }

        action_name='metric'
        id_name=f'dropdown_{action_name}_{id_name_end}'
        self.dropdown_plot={
        'option':self.dropdown_mode_option,
        'id'          :id_name,
        'value'       :self.dropdown_mode_option[0]['value'],
        'placeholder' :f'Select {action_name}', 
        } 

        self.output_graph_1={
            'id':f'output-graph1_{action_name}_{id_name_end}', 
            'style':{'display': 'flex', 'justify-content': 'center'}
        }

        self.output_text_1={
            'id':f'output-text1_{action_name}_{id_name_end}', 
            # 'style':{'display': 'flex', 'justify-content': 'center'}
        } 

        self.height_slider={
                        'id':f'height-slider_{action_name}_{id_name_end}',
                        'min':300,
                        'max':900,
                        'step':50,
                        'value':750, 
                        'marks':{i: f'{i}px' for i in range(300, 1000, 200)}, 
        }

        self.width_slider={
                        'id':f'width-slider_{action_name}_{id_name_end}',
                        'min':400,
                        'max':1200,
                        'step':50,
                        'value':1000, 
                        'marks':{i: f'{i}px' for i in range(400, 1200, 200)}, 
        }

        self.iou_slider={
                        'id':f'iou-slider_{action_name}_{id_name_end}',
                        'min':0,
                        'max':100,
                        'step':1,
                        'value':70, 
                        'marks':{i: f'{i}%' for i in range(0, 100, 20)}, 
        } 
 

        self.hist_slider={
                        'id':f'iou-slider_{action_name}_{id_name_end}',
                        'min':-1,
                        'max':10,
                        'step':1,
                        'value':0, 
                        'marks':{i: f'{i}' for i in range(0, 10, 2)}, 
        } 
 
   
        action_name='true_keys'
        id_name=f'dropdown_{action_name}_{id_name_end}'
        self.dropdown_true_keys={
            'option'     :self.dropdown_true_keys_option,
            'id'         :id_name,
            'value'      :self.dropdown_true_keys_option[0]['value'],
            'placeholder':f'Select {action_name}', 
        }

        action_name='path_head'
        id_name=f'dropdowns_{action_name}_{id_name_end}'
        self.dropdowns_path_head={
            'option'     :self.dropdown_path_head_option,
            'id'         :id_name,
            # 'value'      :self.dropdown_path_head_option[0]['value'],
            'value' : model_type,
            'placeholder':f'Select Type of Model', 
        }

        action_name='path_head'
        id_name=f'dropdown_{action_name}_{id_name_end}'
        self.dropdown_path_head={
            'option'     :self.dropdown_path_head_option,
            'id'         :id_name,
            # 'value'      :self.dropdown_path_head_option[0]['value'],
            'value' : model_type,
            'placeholder':f'Select Type of Model', 
        }

        action_name='model_suf'
        id_name=f'dropdown_{action_name}_{id_name_end}'
        self.dropdown_model_suf={
            'option'     :self.dropdown_model_suf_option,
            'id'         :id_name,
            'value'      :f'{model_sufix}',
            'placeholder':f'Select Feature Class', 
        }
 
        action_name='path'
        id_name=f'dropdown_{action_name}_{id_name_end}'
        self.dropdown_path={
            'option'     :self.dropdown_path_option,
            'id'         :id_name,
            # 'value'      :self.dropdown_path_option[0]['value'],
            'value':  path_dir,
            'placeholder':f'Select Weight', 
        } 
        dropdown_neld_option = [{'label': 'Initial', 'value': 'init', 'style': dropdown_options_style} ,
                                {'label': f'Smoothed', 'value': 'smooth', 'style': dropdown_options_style}]
        action_name='model curve state'
        id_name=f'dropdown_{action_name}_{id_name_end}'
        self.dropdown_neld={
            'option'     :dropdown_neld_option,
            'id'         :id_name,
            'value'      :dropdown_neld_option[0]['value'],
            'placeholder':f'Select {action_name}', 
        }
 

    def get_dropdown_cluster_test(self,id_name_end,count):
        dropdown_options_style=self.dropdown_options_style 
        dropdown_cluster_option = [{'label': 'model', 'value': 0, 'style': dropdown_options_style}] 
        dropdown_cluster_option.append({'label': f'hmod', 'value': 1, 'style': dropdown_options_style}) 
        ik=0
        jj=0
        scatter=[]
        for iii in count:  
                dropdown_cluster_option.append({'label': f'sp({iii:02})', 'value': iii+2, 'style': dropdown_options_style}) 
        
        action_name='cluster'
        id_name=f'dropdown_{action_name}_{id_name_end}'
        self.dropdown_cluster={
            'option'     :dropdown_cluster_option,
            'id'         :id_name,
            'value'      :dropdown_cluster_option[1]['value'],
            'placeholder':f'Select {action_name}', 
        }


    def get_dropdown_cluster(self,id_name_end,count):
        dropdown_options_style=self.dropdown_options_style 
        dropdown_cluster_option =     [{'label': 'model', 'value': 0, 'style': dropdown_options_style}] 
        dropdown_cluster_option.append({'label': f'hmod'  , 'value': 1, 'style': dropdown_options_style})
        jjjj=2 
        for iii in count: 
            dropdown_cluster_option.append({
                'label': f'sp({iii:02})' if isinstance(iii, int) else f'sp_ap({iii[0]:02})~sp_an({iii[1]:02})', 
                'value':jjjj,
                'style': dropdown_options_style
            })
            jjjj+=1
        action_name='cluster'
        id_name=f'dropdown_{action_name}_{id_name_end}'
        self.dropdown_cluster={
            'option'     :dropdown_cluster_option,
            'id'         :id_name,
            'value'      :dropdown_cluster_option[0]['value'],
            'placeholder':f'Select {action_name}', 
        }



    def get_dropdown_index(self,id_name_end,neld_names,index):
        dropdown_options_style=self.dropdown_options_style  
        dropdown_index_option=[{
                'label': f'{iii}', 
                'value':ii,
                'style': dropdown_options_style
            } for ii,iii in enumerate(neld_names)] 
        action_name='index'
        id_name=f'dropdown_{action_name}_{id_name_end}'
        self.dropdown_index={
            'option'     :dropdown_index_option,
            'id'         :id_name,
            'value'      :index, 
            'placeholder':f'Select {action_name}', 
        } 




from mod.kalman.lorenz.neld_fun_0.main_0_help import dropdown_callback
 



class get_layout:
    def __init__(self,tname=None):
        self.mdrp=dropdown_callback(nam=tname)
        par = self.mdrp.callback_file['figure']['par']# 'param_mapp-store'
        loa = self.mdrp.callback_file['figure']['loa']# 'page_mapp-load'
        type= self.mdrp.callback_file['figure']['type']# 'd000'
        lst = self.mdrp.callback_file['figure']['lst']# 'param_mapp-store'
        mal = self.mdrp.callback_file['figure']['mal']# 'page_mapp-load' 
        share=self.mdrp.callback_file['figure']['share'] 
        self.app_layout = html.Div(
            children=[
                dcc.Interval(
                    id=loa if type is None else f"{loa}-{type}",
                    interval=100,
                    max_intervals=1
                ), 
                dcc.Store(
                    id=par if type is None else f"{par}-{type}",
                    storage_type="session"
                ), 
                *self.Get_children()
            ],

            style={
                'font-size': '20px'
            }
        ) 
    
        print('[[[[[[[[[[[lang]]]]]]]]]]]',self.file_path_org,lst)
        pag='lang'
        pag=os.path.basename(self.file_path_org)
        # lst=['nam_gen','path_head','action','dnn_mode','path_dir']
        # mal='container_mapp_param'
        self.Output=[
            Output(self.output_graph_1['id'], 'children'), 
            Output(self.output_text_1['id'], 'children'), 
        ] 
        self.Input_ids=[f'dropdown_{gvali}_{pag}_param'  for gvali in lst]
        self.Input_idsST=[self.dropdown_mode['id'],
                        #   self.dropdown_intensity['id'],
                        #   self.hist_slider['id'],
                        #   self.dropdown_cluster['id'],
                        self.width_slider['id'],
                        self.height_slider['id'],
                        self.dropdown_template['id'], ]
        self.Input=[Input(mm,'value') for mm in self.Input_ids+self.Input_idsST] 





    def Get_children(self):  
        svv=[
            html.Br(),
            # html.Label('Histogram Bin Count:', style=self.dropdown_options_style),
            # dcc.Slider(
            #     id=self.hist_slider['id'],
            #     min=self.hist_slider['min'],
            #     max=self.hist_slider['max'],
            #     step=self.hist_slider['step'],
            #     value=self.hist_slider['value'],
            #     marks=self.hist_slider['marks'],
            # ),  
            html.Label('Graph Width:', style=self.dropdown_options_style),
            dcc.Slider(
                id=self.width_slider['id'],
                min=self.width_slider['min'],
                max=self.width_slider['max'],
                step=self.width_slider['step'],
                value=self.width_slider['value'],
                marks=self.width_slider['marks'],
            ),
            # html.Br(),
            html.Label('Graph Height:', style=self.dropdown_options_style),
            dcc.Slider(
                id=self.height_slider['id'],
                min=self.height_slider['min'],
                max=self.height_slider['max'],
                step=self.height_slider['step'],
                value=self.height_slider['value'],
                marks=self.height_slider['marks'],
            ),
            # html.Br(),
    ] 

        par = self.mdrp.callback_file['figure']['par']# 'param_mapp-store'
        loa = self.mdrp.callback_file['figure']['loa']# 'page_mapp-load'
        type= self.mdrp.callback_file['figure']['type']# 'd000'
        lst = self.mdrp.callback_file['figure']['lst']# 'param_mapp-store'
        mal = self.mdrp.callback_file['figure']['mal']# 'page_mapp-load'  
        ggg=[[ html.Div(id= f"{mal}-{gvali}-{type}") ] for gvali in lst]
        dfc=sum(ggg,[])
        # print('[[[[[[[[[[[[[[[[[[[[[[[[-------------------------]]]]]]]]]]]]]]]]]]]]]]]]',ggg)
        # dfc = []
        dropdown_sources = [
            # self.dropdown_index,
            # self.dropdown_path_head,
            # self.dropdown_model_suf,
            # self.dropdown_path,
            # self.dropdown_cluster,
            self.dropdown_mode,
            # self.dropdown_intensity,
            # self.dropdown_neld,
            self.dropdown_template,
        ] 
        for ii,key in enumerate(dropdown_sources): 
            # if ii==0:
            #     dfc.extend(ggg)
            # else:
                kwargs = key.copy()
                if 'option' in kwargs and 'options' not in kwargs:
                    kwargs['options'] = kwargs.pop('option')
                    
                dfc.extend([dcc.Dropdown(**kwargs, style=self.box_style) ])

        dffc = dfc + svv
 
        return [ 
            # Additional Dropdown for Graph 2
            
            html.Br(),
            dbc.Row([
                dbc.Col(
                    html.Div(dffc),
                    style={'flex': '0 0 20%'}  # Set the width of the column using flex property
                ),
                # Column 2: Graph Output
                dbc.Col(
                    html.Div([
                        html.Br(),
                        # Graph 1 Output
                        html.Div(id=self.output_graph_1['id'], 
                                 style=self.output_graph_1['style']),
                        html.Br(),
                        html.Div(id=self.output_text_1['id'], 
                                 style=self.output_graph_1['style']
                                ),
                        html.Br(),
                    ]),
                    style={'flex': '0 0 70%'}  # Set the width of the column using flex property
                ),
            ], style={
                'display': 'flex',
                'justify-content': 'center',
                'align-items': 'center',
                'margin-top': '20px',
                'margin-bottom': '20px'
            }),
            # Text Output
            html.Br(),
            html.Br(), 
        ]



 


 

