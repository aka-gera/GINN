

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






def loss_fn(output, target):
    return np.mean(np.square(output - target))

from mod.kalman.lorenz.neld_fun_0.get_path import get_files
from mod.kalman.lorenz.apps.app_param_test import get_layout

class class_data():
    def __init__(self):
        pass 




    def get_neld_data(self,data_studied,nam_gen):
        doc_module = importlib.import_module(f'ginn.{nam_gen}')  
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
        (nam_gen,path_head,action,model_suf,root,path,root1,mode,  width, height,templ )=(self.param_inputii[mm] for mm in self.Input_ids+self.Input_idsST)
    

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


 




 

