

import sys
import os 
 
# import dash
from dash import dcc, html, dash_table, Input, Output, State, callback 
import dash_bootstrap_components as dbc
import numpy as np 
from mod.dsa.neld_fun_0.help_funn import get_color
import mod.dsa.neld_fun_0.help_fun as hf 
import mod.dsa.apps.help_plotly as hp
from mod.dsa.apps.help_plotly import aka_plot 
# import density as den
import plotly.graph_objects as go  
import pandas as pd 
from sklearn.metrics import roc_curve, auc
import pickle

import mod.dsa.dend_fun_0.curvature as cu  
import mod.dsa.dend_fun_0.help_funn as hff
from mod.dsa.neld_fun_0.get_path import get_files ,safe_id ,get_name
from mod.dsa.dend_fun_0.help_graph import get_iou_graph,get_cm_iou,compute_kl 
from mod.dsa.dend_fun_0.help_save_iou import iou_train
from mod.dsa.dend_fun_0.density import get_chi
import importlib

def log_ratio(values, eps=1e-12):
    values = np.asarray(values)

    # Differences
    dk1  = values[3:, :]  - values[2:-1, :]   # x_{k+1} - x_k
    dk   = values[2:-1, :] - values[1:-2, :]  # x_k     - x_{k-1}
    dk_1 = values[1:-2, :] - values[:-3, :]   # x_{k-1} - x_{k-2}

    # Logs of the ratios
    num = np.log(np.abs(dk1 / (dk + eps)) + eps)
    den = np.log(np.abs(dk  / (dk_1 + eps)) + eps)

    q = num / den

    # Clamp crazy values: q > 100 → 0
    q[q > 5] = 5
    q[q < -5] = -5

    return q


def add_unit(label, unit="µm",):
    return f"{label} ({unit})"

def add_unit(label: str, unit="µm",) -> str:
    label_lower = label.lower()
    if any(word in label_lower for word in ["length", "width", "diameter"]):
        return f"{label} ({unit})"
    elif "area" in label_lower:
        return f"{label} ({unit}<sup>2</sup>)"
    elif "volume" in label_lower:
        return f"{label} ({unit}<sup>3</sup>)"
    else:
        return label   



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
'''

class class_data(get_files):

    def __init__(self, file_path_org,
                        data_studied, 
                        model_sufix,
                        neld_data, 
                        file_path_data=None,
                            path_file=None,
                            path_file_sub=None,
                            path_file_dir=None,
                            obj_org_path=None, 
                        neld_names=None,
                        neld_namess=None,   
                        neld_path_inits=None,
                        name_spine_id=None,
                        name_head_id=None,
                        name_neck_id=None,
                        name_shaft_id=None,
                        path_train=None,
                        txt_true_file=None,
                        txt_save=False,
                        txt_save_pred=False,
                        size_threshold=100,
                        gauss_threshold=10, 
                        name_path_fin='save',
                        cts=6,
                        stoppage=4,
                        zoom_threshold=1000,
                        radius_threshold=0.05,
                        name_path_fin_save_index=20,
                        spine_filter=True,
                        numNeighbours=5,
                        zoom_threshold_min=1,
                        zoom_threshold_max=4, 
                        line_num_points_shaft=200,
                        line_num_points_inter_shaft=300, 
                        spline_smooth_shaft=1, 
                        DTYPE='float32', 
                        disp_infos=False,
                        pre_portion=None,
                        pinn_dir_data=None, 
                        pinn_dir_data_all=None,
                        model_sufix_all=None,
                        neld_names_all=None,
                        path_heads =None,
                        true_keys=None,
                        list_features=None,
                        base_features_list=None,
                        metrics={},
                        model_type=None,
                        data_mode=None,
                        path_dir=None, 
                        thre_target_number_of_triangles=None,
                        voxel_resolution=None,
                        obj_org_path_dict=None,
                        path_display=None,
                        dict_mesh_to_skeleton_finder_mesh=None,
                        model_sufix_dic=None,
                        path_display_dic=None,
                        kmean_n_run=100,
                        kmean_max_iter=600,
                        param_dic=None, 

        ): 


        self.common_args = dict(
            file_path_org=file_path_org,
            neld_data=neld_data,
            neld_names=neld_names,
            neld_namess=neld_namess,
            data_studied=data_studied,
            name_spine_id=name_spine_id,
            name_head_id=name_head_id,
            name_neck_id=name_neck_id,
            name_shaft_id=name_shaft_id,
            txt_true_file=txt_true_file,
            txt_save=txt_save,
            txt_save_pred=txt_save_pred,
            size_threshold=size_threshold,
            gauss_threshold=gauss_threshold,
            name_path_fin=name_path_fin,
            cts=cts,
            stoppage=stoppage,
            zoom_threshold=zoom_threshold,
            radius_threshold=radius_threshold,
            name_path_fin_save_index=name_path_fin_save_index,
            spine_filter=spine_filter,
            numNeighbours=numNeighbours,
            zoom_threshold_min=zoom_threshold_min,
            zoom_threshold_max=zoom_threshold_max,
            line_num_points_shaft=line_num_points_shaft,
            line_num_points_inter_shaft=line_num_points_inter_shaft,
            spline_smooth_shaft=spline_smooth_shaft,
            DTYPE=DTYPE,
            model_sufix=model_sufix,
            neld_path_inits=neld_path_inits,
            disp_infos=disp_infos,
            path_train=path_train,
            pre_portion=pre_portion,
            pinn_dir_data=pinn_dir_data,
            pinn_dir_data_all=pinn_dir_data_all,
            model_sufix_all=model_sufix_all,
            neld_names_all=neld_names_all,
            path_heads=path_heads,
            true_keys=true_keys,
            list_features=list_features,
            base_features_list=base_features_list,
            metrics=metrics,
            model_type=model_type,
            data_mode=data_mode, 
            thre_target_number_of_triangles=thre_target_number_of_triangles,
            voxel_resolution=voxel_resolution,
            obj_org_path_dict=obj_org_path_dict,
            model_sufix_dic=model_sufix_dic,
            path_display_dic=path_display_dic,
            kmean_n_run=kmean_n_run,
            kmean_max_iter=kmean_max_iter,
            param_dic=param_dic, 
            file_path_data=file_path_data,
        )


        print('[[[[[[[[[[neld_data--self.path_train---neld_data]]]]]]]]]]',self.path_train)
 

        get_name.__init__(self)  
        print('[[[[[[[self.common_args]]]]]]]',self.common_args['file_path_org'])
        get_files.__init__(self,**self.common_args,**dict(

                            path_file=path_file,
                            path_file_sub=path_file_sub,
                            path_file_dir=path_file_dir,
                            obj_org_path=obj_org_path, 
        )) 

'''

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
        spine_path= self.path_file[path_train['dest_shaft_path']] 
        # spine_path= self.path_file[path_train['dest_spine_path']]  
        neld_name=self.neld_name 
        neld_namess=self.neld_namess 
        self.plot_data_iou=None 
        self.model_shap_dic={}
        self.plot_data_iou_dic={}
        self.plot_data_center_curv={} 
        self.plot_data_cylinder_heatmap={} 
        self.plot_data_riplet={} 
        self.annot_intensity={ke:{} for ke in self.inten_file_train} 
        # self.logit_intensity_head_neck={ke:{} for ke in self.intensity_head_neck_logit} riplet
        # self.logit_intensity={ke:{} for ke in self.intensity_spines_logit} 
        self.logit_intensity={ke:{} for ke in self.intensity_logit} 
        self.iou_count={}
        self.scatter_loss_dic={}
        self.scatter_iou_dic={}
        self.scatter_loss_spine_dic={}
        self.scatter_iou_spine_dic={}
        self.scatter_auc_spine_dic={}
        self.scatter_dice_spine_dic={}
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
                    fgff=os.path.join(self.path_file[id_path], self.txt_spine_iou)
                    if os.path.exists(fgff):
                        self.iou_count[id_path]=  np.loadtxt(fgff, dtype=float)    

                    path_grap_center_curv=os.path.join(self.path_file[id_path] ,'plot_data_center_curv.pkl')
                    if os.path.exists(path_grap_center_curv):
                        with open(os.path.join(path_grap_center_curv), "rb") as file:
                            self.plot_data_center_curv[id_path] = pickle.load(file) 



                    for intensity_type in self.inten_file_train :
                        path_grap_center_curv=os.path.join(self.path_file[id_path] ,f'{intensity_type}.txt')
                        if os.path.exists(path_grap_center_curv):
                            self.annot_intensity[intensity_type][id_path] = path_grap_center_curv


                    ''' DONT DELETE
                    for intensity_type in self.intensity_spines_logit:
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
                    for ii,(tyyy,nam,couleur) in enumerate(zip(self.inten_file_model_train_iou[:-1],['head','neck','shaft'],['red','green','blue'])):
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
                    tyy =self.inten_file_model_spine_loss[0]
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

                    self.scatter_loss_spine_dic[id_path]= scatter_loss_data

                                      
 
       
                    tyy =self.inten_file_model_spine_iou[0] 
                    scatter_iou_data=[]   
                    cnt=True
                    for ii,(tyyy,nam,couleur) in enumerate(zip(self.inten_file_model_train_spine_iou,['shaft','spine'],['red','blue'])):
                        # iou_path=self.model_dir_path['head_neck']['iou'][ii]
                        # if  loss_path is None:
                        #     continue
                        if id_path in list(self.path_file_sub[tyy].keys()):
                        #     continue
                            iou_path=self.path_file_sub[tyy][id_path][tyyy]
                            if os.path.exists(iou_path) and os.path.getsize(iou_path) > 0: 
                                # print('[[[[[[[[[[[------========-------]]]]]]]]]]]',os.path.getsize(iou_path))
                                try: 
                                    _data=np.loadtxt(iou_path,dtype=float) 
                                except ValueError: 
                                    continue
                                if len(_data)<=0 or len(_data.shape)==1:
                                    continue
                                m1,m2=.2,.8
                                for ii,(symb,mdd,siz) in enumerate(zip(['star','star','star','star','square','square'],
                                                                ['train','train','train','train','test','test'],
                                                                [m1,m1,m1,m1,m2,m2])):
                                    if ii<_data.shape[1]:
                                        _dataa= np.hstack((np.arange(len(_data)).reshape(-1,1),_data[:,ii:ii+1])) 
                                        scatter_iou_data.append(hf.plotly_scatter(points=_dataa ,
                                                                color=couleur,
                                                                size=10.,
                                                                opacity=siz,
                                                                name=f'{nam}_{mdd}_{ii}',
                                                                symbol=symb))

                        elif os.path.exists(self.df_metric_algorithms_dir) and cnt: 
                            df_union=pd.read_csv(self.df_metric_algorithms_dir)#.sort_values(by=f'Accuracy', ascending=True) 
                            ylabels=df_union.columns[1:]
                            xlabels=df_union.iloc[:,0]
                            df_union = df_union.set_index("Unnamed: 0")
                            df_union=df_union.T.sort_values(by=f'Accuracy', ascending=True)

                            # df_union=df_union[df_union.columns[1:]]
                            cm=np.array(df_union.select_dtypes(include=[np.number]))
                            cm= np.round(cm*1e3)/1e3
 
                            scatter_iou_data.append(
                                                    go.Heatmap(
                                                                z=cm,
                                                                x=xlabels,  
                                                                y=ylabels,  
                                                                colorscale='Blues',
                                                                text=cm,  # Show numbers in cells
                                                                texttemplate="%{text}",  # Format as numbers
                                                                # hovertemplate="Actual: %{y}<br>Predicted: %{x}<br>Count: %{z}<extra></extra>",
                                                                showscale=False ,
                                                                textfont=dict(size=18),
                                                            ) 
                                            ) 
                            cnt=False
 

                    self.scatter_iou_spine_dic[id_path]=scatter_iou_data



                                     
                    tyy =self.inten_file_model_spine_auc[0] 
                    scatter_data=[]   
                    cnt=True
                    for ii,(tyyy,nam,couleur) in enumerate(zip(self.inten_file_model_train_spine_auc,['shaft','spine'],['red','blue'])): 

                        # if id_path not in list(self.path_file_sub[tyy].keys()):
                        #     continue
                        # iou_path=self.path_file_sub[tyy][id_path][tyyy]
                        # if os.path.exists(iou_path) and os.path.getsize(iou_path) > 0:  
                        #     try: 
                        #         _data=np.loadtxt(iou_path,dtype=float) 
                        #     except ValueError: 
                        #         continue
                        #     if len(_data)<=0 :
                        #         continue   
                        #     scatter_data.append(hf.plotly_scatter(points=_dataa ,
                        #                             color=couleur,
                        #                             size=10.,
                        #                             opacity=siz,
                        #                             name=f'{nam}'
                        #                             ))


                        # iou_path=self.model_dir_path['head_neck']['iou'][ii]
                        # if  loss_path is None:
                        #     continue
                        if id_path in list(self.path_file_sub[tyy].keys()):
                        #     continue
                            iou_path=self.path_file_sub[tyy][id_path][tyyy]
                            if os.path.exists(iou_path) and os.path.getsize(iou_path) > 0: 
                                # print('[[[[[[[[[[[------========-------]]]]]]]]]]]',os.path.getsize(iou_path))
                                try: 
                                    _data=np.loadtxt(iou_path,dtype=float) 
                                except ValueError: 
                                    continue
                                if len(_data)<=0 or len(_data.shape)==1:
                                    continue
                                m1,m2=.2,.8
                                for ii,(symb,mdd,siz) in enumerate(zip(['star','star','star','star','square','square'],
                                                                ['train','train','train','train','test','test'],
                                                                [m1,m1,m1,m1,m2,m2])):
                                    if ii<_data.shape[1]:
                                        _dataa= np.hstack((np.arange(len(_data)).reshape(-1,1),_data[:,ii:ii+1])) 
                                        scatter_data.append(hf.plotly_scatter(points=_dataa ,
                                                                color=couleur,
                                                                size=10.,
                                                                opacity=siz,
                                                                name=f'{nam}_{mdd}_{ii}',
                                                                symbol=symb))



                    self.scatter_auc_spine_dic[id_path]=scatter_data

 
 


                                     
                    tyy =self.inten_file_model_spine_dice[0] 
                    scatter_data=[]   
                    cnt=True
                    for ii,(tyyy,nam,couleur) in enumerate(zip(self.inten_file_model_train_spine_dice,['shaft','spine'],['red','blue'])): 

                        # if id_path not in list(self.path_file_sub[tyy].keys()):
                        #     continue
                        # iou_path=self.path_file_sub[tyy][id_path][tyyy]
                        # if os.path.exists(iou_path) and os.path.getsize(iou_path) > 0:  
                        #     try: 
                        #         _data=np.loadtxt(iou_path,dtype=float) 
                        #     except ValueError: 
                        #         continue
                        #     if len(_data)<=0 :
                        #         continue   
                        #     scatter_data.append(hf.plotly_scatter(points=_dataa ,
                        #                             color=couleur,
                        #                             size=10.,
                        #                             opacity=siz,
                        #                             name=f'{nam}'
                        #                             ))

                        if id_path not in list(self.path_file_sub[tyy].keys()):
                            continue
                        iou_path=self.path_file_sub[tyy][id_path][tyyy]
                        if os.path.exists(iou_path) and os.path.getsize(iou_path) > 0: 
                            # print('[[[[[[[[[[[------========-------]]]]]]]]]]]',os.path.getsize(iou_path))
                            try: 
                                _data=np.loadtxt(iou_path,dtype=float) 
                            except ValueError: 
                                continue
                            if len(_data)<=0 or len(_data.shape)==1:
                                continue
                            m1,m2=.2,.8
                            for ii,(symb,mdd,siz) in enumerate(zip(['star','star','star','star','square','square'],
                                                            ['train','train','train','train','test','test'],
                                                            [m1,m1,m1,m1,m2,m2])):
                                if ii<_data.shape[1]:
                                    _dataa= np.hstack((np.arange(len(_data)).reshape(-1,1),_data[:,ii:ii+1])) 
                                    scatter_data.append(hf.plotly_scatter(points=_dataa ,
                                                            color=couleur,
                                                            size=10.,
                                                            opacity=siz,
                                                            name=f'{nam}_{mdd}_{ii}',
                                                            symbol=symb))

                    self.scatter_dice_spine_dic[id_path]=scatter_data

 

 




                    clor=['red','blue' ]
                    vds=[1,-1]
                    valss=['shaft','spine']
                    figg=go.Figure()
                    # tyy =self.inten_file_model_shap[0]  
                    # iou_path=self.path_file_sub[tyy][id_path] 
                    self.model_shap=[]   
                    head_neck_path = 'dest_shaft_path'
                    pathh=path_train[head_neck_path]
                    spine_path_save=     self.path_file[f'result_{pathh}']
                    iou_path=os.path.join(spine_path_save,'shap.csv') 
                     
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
        self.metric_total_dic['roc_curve']['curve']={'spine':[],'shaft':[]}
        self.metric_total_dic['roc_curve']['score']={'spine':[],'shaft':[]}
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
                        '''
                        # true_path=self.neld_path_original_mm['keys']['true_0']
                        path_grap_iou=os.path.join(self.path_file[id_path] ,'intensity_spines_logit.txt')
                        path_grap_true=os.path.join(self.path_file[true_path] ,'intensity_spines_logit.txt') 
                        print(']]]]]]]][[[[[[[[[[[[[]]]]]]]]]]]]]',path_grap_iou)
                        if os.path.exists(path_grap_iou) and os.path.exists(path_grap_true): 
                            print(']]]]]]]][[[[[[[[[[[[[]]]]]]]]]]]]]',path_grap_iou)
                            self.metric_total_dic['roc_curve']['true']+=np.loadtxt(path_grap_true,dtype=float)
                            self.metric_total_dic['roc_curve']['score']+=np.loadtxt(path_grap_iou,dtype=float)
'''

                        if model_suf !='save':
                            spine_path_save=     self.path_file[f'result_{id_path}']
                            for nm,nmm in zip(['shaft','spine'],['shaft','spine']):
                                metric_path=os.path.join( spine_path_save,f'roc_{nm}_{true_key}.txt') 
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


                            
                            spine_path_save=     self.path_file[f'result_{id_path}']
                            metric_path=os.path.join( spine_path_save,f'iou_{true_key}.csv') 
                            if os.path.exists(metric_path):
                                df = pd.read_csv(metric_path)
                                # print('----=====----','im her')
                                # nhf=df['id_true']>0
                                # sze_checks_0 ,sze_check ,sze_check_un=sze_checks_0[nhf] ,sze_check[nhf] ,sze_check_un[nhf]
                                sze_checks_0 ,sze_check ,sze_check_un=df['id_true'],df['iou_single'],df['iou_union']
                                metric_name_path=df.columns
                                iou_dict={}
                                get_cm_iou(sze_checks_0 ,sze_check ,sze_check_un ,iou_dict=iou_dict, iou_per=70,labels = ['False', 'True'],nbinsx=300,) 
                                accuracy , precision, recall, f1_score=iou_dict['single']['metrics'].values() 
                                # self.metric_total_dic[id_path]={}
                                self.metric_total_dic['single'][id_path_nam]=dict(accuracy=accuracy,
                                                                    precision=precision,
                                                                    recall=recall,
                                                                    f1_score=f1_score) 
                                accuracy, precision, recall, f1_score=iou_dict['union']['metrics'].values()
                                self.metric_total_dic['union'][id_path_nam]=dict(accuracy=accuracy,
                                                                    precision=precision,
                                                                    recall=recall,
                                                                    f1_score=f1_score,
                                                                    )
                                
                                if 'dice_single' in list(df.columns):
                                    sze_checks_0 ,sze_check ,sze_check_un=df['id_true'],df['dice_single'],df['dice_union']
                                    metric_name_path=df.columns
                                    iou_dict={}
                                    get_cm_iou(sze_checks_0 ,sze_check ,sze_check_un ,iou_dict=iou_dict, iou_per=70,labels = ['False', 'True'],nbinsx=300,) 
                                    accuracy , precision, recall, f1_score=iou_dict['single']['metrics'].values() 
                                    # self.metric_total_dic[id_path]={}
                                    self.metric_total_dic['single_dice'][id_path_nam]=dict(accuracy=accuracy,
                                                                        precision=precision,
                                                                        recall=recall,
                                                                        f1_score=f1_score) 
                                    accuracy, precision, recall, f1_score=iou_dict['union']['metrics'].values()
                                    self.metric_total_dic['union_dice'][id_path_nam]=dict(accuracy=accuracy,
                                                                        precision=precision,
                                                                        recall=recall,
                                                                        f1_score=f1_score,
                                                                        )
                                


                                for uuyy in ['single','union','single_dice','union_dice',]:
                                    if (uuyy in self.metric_total_dic) and (id_path_nam in self.metric_total_dic[uuyy]):
                                        for nm in ['shaft','spine']: 
                                            self.metric_total_dic[uuyy][id_path_nam][f'AUC_{nm}']=self.metric_total_dic['roc_curve']['score'][nm]






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





        self.file_path =file_path
        self.neld_name=neld_name
        self.itera=index  
        self.vertices_0=vertices_0       = np.loadtxt(os.path.join(self.file_path, self.txt_vertices_0), dtype=float) 
        
        self.vertices_00=vertices_00      = np.loadtxt(os.path.join(self.file_path, self.txt_vertices_old), dtype=float)
        self.faces=faces = np.loadtxt(os.path.join(self.file_path, self.txt_faces), dtype=int)


        print(f'Starting analysis of {neld_name}')
        print(f"Number of vertices: {len(vertices_0)}")
        print(f"Number of vertices: {len(vertices_00)}")
        print(f"Number of faces: {len(faces)}")
        print(f"Number of vertices: {os.path.join(self.file_path, self.txt_vertices_0)}")
        print(f"Number of vertices: {os.path.join(self.file_path, self.txt_vertices_old)}")
        
        self.inten={} 
        for port in self.pre_portions:
            self.inten[f'{port}_body']=np.zeros_like(vertices_00[:,0]) 
            mmm=os.path.join(spine_path,f'intensity_{port}_segm.txt') 
            if os.path.exists(mmm):  
                self.inten[f'{port}_body']=mmm



    def get_figure(self,true_keys, path_head,model_suf,path,vertices_pl,intensity, spid, width, height,ffcouleur ):   

        id_path=f'{true_keys}_save_save' if path_head=='true' else f'{path_head}_{model_suf}_{path}'
        id_pathss=f'{path_head}_{model_suf}_{path}_{true_keys}'
        spine_path=self.path_file[id_path] 
        # shaft_path=self.path_mapping[self.name_spine][path]  
        if (spid == 0) : 
            self.figure_3d=cu.plotly_mesh(vertices=vertices_pl,
                                                    faces=self.faces ,
                                                    intensity=intensity,
                                                    width=width, 
                                                    height=height,
                                                    colorscale='purd')   
            self.scatter=[ 
                    hf.plotly_scatter(points=vertices_pl ,
                                      color='red',
                                      size=1.07,
                                      opacity=.8,
                                      name='Approx'),   
                    ]
            self.layout = go.Layout(width=width, 
                            height=height,
                            title=f'Dendrite', 
                            ) 
 
        elif spid==1: 
            self.spine_index =spine_index = np.loadtxt(os.path.join(spine_path,self.txt_shaft_index), dtype=int) 
            self.faces_index =faces_index = np.loadtxt(os.path.join(spine_path,self.txt_shaft_faces),dtype=int)
            self.figure_3d=cu.plotly_mesh(vertices=vertices_pl[spine_index],
                                                    faces=faces_index,
                                                    intensity=intensity[spine_index],
                                                    width=width, 
                                                    height=height,
                                                    colorscale='purd') 

            self.scatter=[ 
                    hf.plotly_scatter(points=vertices_pl[spine_index],
                                      color='red',
                                      size=.7,
                                      opacity=.6,
                                      name='Approx'),   
                    ]  
            self.layout = go.Layout(width=width, 
                            height=height,  
                            )

        else:
            color='red'  
            clustss = spid - 2
            clustsss = spid - 2
            if id_pathss in self.iou_tr: 
                self.clusts=clustss=self.iou_tr[id_pathss].count_appr_tmp[clustss]
                
            # scatter=[] 
            self.spine_index = spine_index = np.loadtxt(os.path.join(spine_path, f'{self.name_spine}_{self.name_index}_{clustss}.txt'),dtype=int)
            self.spine_faces =spine_faces  = np.loadtxt(os.path.join(spine_path, f'{self.name_spine}_{self.name_faces}_{clustss}.txt'),dtype=int)
            # print(os.path.join(spine_path, f'{self.name_spine}_{self.name_index}_{clustss}.txt') )

            self.figure_3d=cu.plotly_mesh(vertices=vertices_pl[spine_index],
                                                    faces=spine_faces,
                                                    intensity=intensity[spine_index],
                                                    width=width, height=height,
                                                    colorscale='purd') 
            if id_pathss in self.iou_tr:
                self.scatter= self.iou_tr[id_pathss].get_graph(vertices_0=vertices_pl,index=clustsss)
                rate,rate_un=self.iou_count[id_path][clustsss,-2:]
                # clustss=int(self.iou_count[path][clustss,0])
                self.clusts=clustss=self.iou_tr[id_pathss].count_appr_tmp[clustsss]
                
                
                self.layout = go.Layout(width=width, 
                                height=height,
                                title=f'Dendrite {self.neld_name} Spine <br>UOI         : {rate:.2f}<br>UOI union: {rate_un:.2f}',  
                                )   
            else:
                self.scatter=[ 
                        hf.plotly_scatter(points=vertices_pl[spine_index],
                                        color='red',
                                        size=.7,
                                        opacity=.6,
                                        name='Approx'),   
                        ]  
                self.layout = go.Layout(width=width, 
                                height=height,  
                                )

        self.scene=dict(
            xaxis=dict(showgrid=False, zeroline=False, showline=False, showticklabels=False, title='',backgroundcolor=ffcouleur),
            yaxis=dict(showgrid=False, zeroline=False, showline=False, showticklabels=False, title='',backgroundcolor=ffcouleur),
            zaxis=dict(showgrid=False, zeroline=False, showline=False, showticklabels=False, title='',backgroundcolor=ffcouleur),
            bgcolor=ffcouleur 
        )


        paath=os.path.join(spine_path,self.txt_spine_count) 
        count=np.loadtxt(paath,dtype=int)
        self.count_spine=count.ndim
        self.count=count if self.count_spine==2 else count.reshape(-1,1)    

    def get_metric(self,mode,metric,width=800,height=800,nbinsx=10000,opacity=1.,color='blue',name=None): 
        self.scatter_metric,self.layout_metric=hp.Plotly_histogram_return(data=metric,
                                                                        nbinsx=nbinsx,
                                                                        title=mode,
                                                                        xtitle=self.metric_mapping['xtitle'],
                                                                        ytitle='Count',
                                                                        width=width, 
                                                                        height=height,
                                                                        opacity=opacity,
                                                                        color=color,
                                                                        name=name,)





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
        (nam_gen,path_head,action,model_suf,path,root1,mode,intensity_type,nbin, clusts, width, height,templ )=(self.param_inputii[mm] for mm in self.Input_ids+self.Input_idsST)


        self.get_neld_data(
                    nam_gen=nam_gen,  
                    data_studied=action, 
                    ) 
         

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
            # **self.pari,

                # path_train, 
                )

        self.get_data(
                    nam_gen=nam_gen, 
                    model_sufix=model_suf,
                    data_studied=action,
                    index=index,
                    neld_data=self.neld_data,
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
        print('[[[[[self.path_file[id_path] ]]]]]',self.path_file[id_path] )

        intensity=None;intensity_path=''
        if intensity_type in self.inten_file_sub:  
            intensity_path =self.path_file_sub[intensity_type][id_path]
            # intensity_path = intensity_path if os.path.exists(intensity_path) else self.path_file_sub[intensity_type][f'{path_head}_{model_suf}_{path_init}']
            # intensity=np.loadtxt(intensity_path, dtype=float)
            # print('spine--------',intensity_path)
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
            # print('spine--------',intensity_path) 
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
            # print('spine--------',intensity_path) 
            # intensity=np.loadtxt(intensity_path, dtype=float)


        elif intensity_type in self.base_features_dict.keys():
            intensity_path =self.path_file_sub[intensity_type][id_path]
            # intensity_path = intensity_path if os.path.exists(intensity_path) else self.path_file_sub[intensity_type][f'{path_head}_{model_suf}_{path_init}']
            # intensity=np.loadtxt(intensity_path, dtype=float)
 
        intensity= None 
        if os.path.exists(intensity_path):
            intensity=np.loadtxt(intensity_path, dtype=float)
            print('spine--------',intensity_path)
        else:
            print('Intensity DOESNT EXIST --------',intensity_path)
                # intensity_path =self.path_file_sub[intensity_type]['true_save_save'] 
            # elif intensity_type in ['spine_body','head_neck_body']: 
            #     intensity_path=self.inten[intensity_type] 
            # else:
            #     intensity_path = self.neld_mapping[neldd]["intensity"][intensity_type] 

        if intensity_type in ["gauss_curv_init","mean_curv_init"]:
            vertices_pl=self.vertices_00
        elif intensity_type in ["gauss_curv_smooth","mean_curv_smooth"]:
            vertices_pl=self.vertices_0


        figure=go.Figure()
        self.get_figure(true_keys,path_head,model_suf, path,vertices_pl,intensity, clusts, width, height,ffcouleur=ffcouleur  )
 
        
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
                

        elif mode in ['heatmap_cylinder','heatmap_cylinder_surface',]:  
            path_grap_center_curv=os.path.join(self.path_file[id_path] ,'cylinder_heatmap.pkl')
            if os.path.exists(path_grap_center_curv):
                with open(os.path.join(path_grap_center_curv), "rb") as file:
                    self.plot_data_cylinder_heatmap[id_path] = pickle.load(file) 
                            
            if self.plot_data_cylinder_heatmap is not None: 
                *pathc, last = path.split('_') 
                if mode =='heatmap_cylinder':
                    print('[[[[[self.path_file[id_path] ]]]]]',self.path_file[id_path] )
                    data=[self.plot_data_cylinder_heatmap[id_path].density_heatmap,
                          self.plot_data_cylinder_heatmap[id_path].density_heatmap_points,
                          self.plot_data_cylinder_heatmap[id_path].density_heatmap_points_org]
                    figure=akp.Plotly_Figure(data= data, layout=self.layout) 
                    figure.update_layout(
                        xaxis=dict(showgrid=False),  
                        yaxis=dict(showgrid=False)  
                    ) 
                elif mode =='heatmap_cylinder_surface':
                    data=[self.plot_data_cylinder_heatmap[id_path].density_heatmap_surface]
                    data.extend(self.plot_data_cylinder_heatmap[id_path].density_org_points)
                    figure=akp.Plotly_Figure(data= data, layout=self.layout)



        elif mode in ['ripley','dijkstra_detail']:

                path_grap_center_curv=os.path.join(self.path_file[id_path] ,'riplet.pkl')
                if os.path.exists(path_grap_center_curv):
                    with open(os.path.join(path_grap_center_curv), "rb") as file:
                        self.plot_data_riplet[id_path] = pickle.load(file) 
 
                if mode =='dijkstra_detail':
                    mesh_shaft=self.plot_data_riplet[id_path].others['mesh_shaft']
                    dijk=self.plot_data_riplet[id_path].others['intensity']
                    distance=dijk['distance']
                    path=dijk['path']
                    scatterr=[]
                    scatterr.append(hf.plotly_scatter(points=vertices_pl, color='red', size=1.3, name='Mesh',opacity=0.5))
                    ii=0
                    for pathy,dis,col in zip(path,distance,ccol): 
                        scatterr.append(hf.plotly_scatter(points=mesh_shaft.vertices[pathy], color=col, size=5.3, name=f'dist {dis:.2f}',opacity=0.5))

                    # scatterr.extend(self.plot_data_cylinder_heatmap[id_path].density_org_points)

                    # scatterr.append(hf.plotly_scatter(points=mesh_shaft.vertices, color='yellow', size=5.3, name='skeleton smooth.',opacity=0.5))
 
                    # data=[sm3]
                    # data.extend(self.plot_data_cylinder_heatmap[id_path].density_org_points)
                    figure=akp.Plotly_Figure(data= scatterr, layout=self.layout)
                elif mode == 'ripley':
                    # ripp,Lrip,rall=self.plot_data_cylinder_heatmap[id_path].ripp,self.plot_data_cylinder_heatmap[id_path].Lrip,self.plot_data_cylinder_heatmap[id_path].rall

                    grp=self.plot_data_riplet[id_path]
                    # print('[[grp]]',grp.graph_ripp)

 
                    figure=akp.Plotly_Figure_Sub(   
                                            shared_xaxes=False,
                                            shared_yaxes=False ,
                        rows=3,
                        cols=2,
                        subplot_titles=("K-function", "L-function")
                    )




                    res_ripley=grp.res_ripley
                    res_area=grp.res_area
                    rall=grp.rall 
                    '''
                    # --- Extract last incremental values ---
                    pvals        = res_ripley["pvals"]
                    p_cluster    = res_ripley["p_cluster"]
                    p_regular    = res_ripley["p_regular"]
                    p_global     = [
                        res_ripley["p_global_2tail"],
                        res_ripley["p_global_cluster"],
                        res_ripley["p_global_regular"]
                    ]
                    rripp    = res_ripley["area_mean"]

                    pvalsa       = res_area["pvals"]
                    p_clustera   = res_area["p_cluster"]
                    p_regulara   = res_area["p_regular"]
                    p_global_area = [
                        res_area["p_global_2tail"],
                        res_area["p_global_cluster"],
                        res_area["p_global_regular"]
                    ]
                    rrippp   = res_area["area_mean"]'''







                    # --- Extract last incremental values ---
                    pvals        = res_ripley["pvals"][-1]
                    p_cluster    = res_ripley["p_cluster"][-1]
                    p_regular    = res_ripley["p_regular"][-1]
                    p_global     = [
                        res_ripley["p_global_2tail"][-1],
                        res_ripley["p_global_cluster"][-1],
                        res_ripley["p_global_regular"][-1]
                    ]
                    rripp    = res_ripley["area_mean"][-1]

                    pvalsa       = res_area["pvals"][-1]
                    p_clustera   = res_area["p_cluster"][-1]
                    p_regulara   = res_area["p_regular"][-1]
                    p_global_area = [
                        res_area["p_global_2tail"][-1],
                        res_area["p_global_cluster"][-1],
                        res_area["p_global_regular"][-1]
                    ]
                    rrippp   = res_area["area_mean"][-1]








                    rat = log_ratio(res_ripley["pvals"])

                    heat = go.Heatmap(
                        z=rat,
                        x=rall,
                        colorscale="Viridis",
                        colorbar=dict(
                            orientation="h",   # horizontal colorbar
                            y=.23+0.25*0,            # move above the subplot
                            x=0.2,             # center it
                            xanchor="center",
                            thickness=15,
                            len=0.4 ,           # shorten the bar                            
                            title=dict(
                                text="con. rate p-2 tails (K-fun)",
                                side="top"       
                            )
                        )
                    )

                    figure.add_trace(heat, row=3, col=1)

                    rat = log_ratio(res_area["pvals"])

                    heat = go.Heatmap(
                        z=rat,
                        x=rall,
                        colorscale="Viridis",
                        colorbar=dict(
                            orientation="h",   # horizontal colorbar
                            y=.23+.25*0,            # move above the subplot
                            x=0.8,             # center it
                            xanchor="center",
                            thickness=15,
                            len=0.4 ,           # shorten the bar
                            title=dict(
                                text="con. rate p-2 tails (Area)",
                                side="top"       
                            )
                        )
                    )

                    figure.add_trace(heat, row=3, col=2)
 


                    grp.graph_ripp=go.Scatter(
                            x=rall,
                            y=grp.ripp,
                            mode='lines+markers',
                            name=f'K(r)  #spines= {grp.count_spine}',
                            
                        )
                    # self.graph_ripp=go.Scatter(
                    #         x=rall,
                    #         y=self.ripp,
                    #         mode='lines+markers',
                    #         name=f'K(r)  #spines= {len(centroid_proj_cylder)}',
                            
                    #     )

                    grp.graph_ripp_norm=go.Scatter(
                            x=rall,
                            y=rripp,
                            mode='lines',
                            name='csr K-fun',
                        )


                    grp.graph_Lrip=go.Scatter(
                            x=rall,
                            y=grp.ripp/(rripp)-1,
                            mode='lines+markers',
                            name='L(r)-1,  K-fun'
                        )




                    grp.graph_ripp_norm_area=go.Scatter(
                            x=rall,
                            y=rrippp,
                            mode='lines',
                            name='csr Area'
                        )


                    grp.graph_Lrip_area=go.Scatter(
                            x=rall,
                            y=grp.ripp/(rrippp)-1,
                            mode='lines+markers',
                            name='L(r)-1,   Area'
                        )
                    




                    grp.graph_monte_ratio=go.Scatter(
                            x=rall,
                            y=rripp/(rrippp)-1,
                            mode='lines+markers',
                            name='csr K-fun/csr Area -1,'
                        )








                    # pg=get_chi(pvals)
                    pg=p_global[0]
                    grp.graph_pvals_2=go.Scatter(
                            x=rall,
                            y=pvals,
                            mode='lines+markers',
                            name=f'p-2 tails,  K-fun p(glo):{pg:.3f}',

                            marker=dict(
                                symbol='square',
                                size=8,
                                color='green'
                            )
                        )
                    # pg=get_chi(pvalsa)
                    pg=p_global_area[0]
                    grp.graph_pvals_2_area=go.Scatter(
                            x=rall,
                            y=pvalsa,
                            mode='lines+markers',
                            name=f'p-2 tails,   Area p(glo):{pg:.3f}',
                            marker=dict(
                                symbol='square',
                                size=8,
                                color='green'
                            )
                        )






                    # pg=get_chi(p_cluster)
                    pg=p_global[1]
                    grp.graph_pvals_cluster=go.Scatter(
                            x=rall,
                            y=p_cluster,
                            mode='lines+markers',
                            name=f'p-cluster,  K-fun p(chi):{pg:.3f}',
                        
                            marker=dict(
                                symbol='x',
                                size=8,
                                color='red'
                            )
                        )

                    pg=get_chi(p_clustera)
                    # pg=p_global_area[1]
                    grp.graph_pvals_cluster_area=go.Scatter(
                            x=rall,
                            y=p_clustera,
                            mode='lines+markers',
                            name=f'p-cluster,   Area p(chi):{pg:.3f}',
                            
                            marker=dict(
                                symbol='x',
                                size=8,
                                color='red'
                            )
                        )





                    pg=get_chi(p_regular)
                    # pg=p_global[2]
                    grp.graph_pvals_regular=go.Scatter(
                            x=rall,
                            y=p_regular,
                            mode='lines+markers',
                            name=f'p-regular,  K-fun p(chi):{pg:.3f}',
                            marker=dict(
                                symbol='circle',
                                size=8,
                                color='purple'
                            )
                        )

                    pg=get_chi(p_regulara)
                    # pg=p_global_area[2]
                    grp.graph_pvals_regular_area=go.Scatter(
                            x=rall,
                            y=p_regulara,
                            mode='lines+markers',
                            name=f'p-regular,   Area p(chi):{pg:.3f}',
                            marker=dict(
                                symbol='circle',
                                size=8,
                                color='purple'
                            )
                        )







                    if grp.graph_ripp is not None:
                        grp.graph_ripp['marker']=dict(color="red")
                        figure.add_trace(
                        grp.graph_ripp,
                        row=1,
                        col=1
                    )

                    if grp.graph_ripp_norm is not None:
                        grp.graph_ripp_norm['marker']=dict(color="green")
                        figure.add_trace(
                        grp.graph_ripp_norm,
                        row=1,
                        col=1
                    )
                    if grp.graph_ripp_norm_area is not None:
                        grp.graph_ripp_norm_area['marker']=dict(color="blue")
                        figure.add_trace(
                            grp.graph_ripp_norm_area,
                            row=1,
                            col=1
                        )

                    if grp.graph_Lrip is not None:
                        grp.graph_Lrip['marker']=dict(color="green")
                        figure.add_trace(
                        grp.graph_Lrip,
                        row=1,
                        col=2
                    )


                    if grp.graph_Lrip_area is not None:
                        grp.graph_Lrip_area['marker']=dict(color="blue")
                        figure.add_trace(
                            grp.graph_Lrip_area,
                            row=1,
                            col=2
                        )

                    # if grp.graph_monte_ratio is not None:
                    #     grp.graph_monte_ratio['marker']=dict(color="purple")
                    #     grp.graph_monte_ratio['name']='Bench. Ripley/Bench. Area -1,'
                    #     figure.add_trace(
                    #         grp.graph_monte_ratio,
                    #         row=1,
                    #         col=2
                    #     )

                    


                    figure.add_hline(
                        y=0,
                        row=1,
                        col=2,
                        line_dash='dash'
                    )









                    
                    if grp.graph_pvals_2 is not None: 
                        figure.add_trace(
                            grp.graph_pvals_2,
                            row=2,
                            col=1
                        )
                        figure.add_trace(
                            grp.graph_pvals_cluster,
                            row=2,
                            col=1
                        )
                        figure.add_trace(
                            grp.graph_pvals_regular,
                            row=2,
                            col=1
                        )
                    figure.add_hline(
                        y=0.05,
                        row=2,
                        col=1,
                        line_dash='dash'
                    )



                    
                    if grp.graph_pvals_2_area is not None: 
                        figure.add_trace(
                            grp.graph_pvals_2_area,
                            row=2,
                            col=2
                        )
                        figure.add_trace(
                            grp.graph_pvals_cluster_area,
                            row=2,
                            col=2
                        )
                        figure.add_trace(
                            grp.graph_pvals_regular_area,
                            row=2,
                            col=2
                        )
                    figure.add_hline(
                        y=0.05,
                        row=2,
                        col=2,
                        line_dash='dash'
                    )

                    figure.add_annotation(
                        text="P-values (K-fun)",
                        x=0.1,
                        y=0.23+0.25*2-0.1,
                        xref="paper",
                        yref="paper",
                        showarrow=False,
                        # textangle=-90,
                        font=dict(size=16)
                    )


                    figure.add_annotation(
                        text="P-values (Area)",
                        x=0.8,             
                        y=0.23+0.25*2-0.1,             
                        xref="paper",
                        yref="paper",
                        showarrow=False,
                        font=dict(size=16)
                    )

                    # figure.update_layout(
                    #     height=500,
                    #     width=1000,
                    #     # template='plotly_white'
                    # )

                figure.update_layout(scene=self.scene)

 
                figure.update_layout(height=height*1.5,
                                     width=width)

















        elif mode=='IOU':
            # if self.plot_data_iou  is not None: 
            #     figure=akp.Plotly_Figure(data= self.plot_data_iou, layout=self.layout)
            #     figure.update_layout(scene=self.scene)  
            if  len(self.plot_data_iou_dic)>0:  

                figure=akp.Plotly_Figure(data= self.plot_data_iou_dic[id_pathss] , layout=self.layout)
                figure.update_layout(scene=self.scene) 


 
        elif mode =='Metrics':
            cm=[]
            path_1=self.neld_path_org_new
            labelx,labely=['Dynamics Estimate','Meas. Estimate'],[]
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
            nam='p'
            path_1=self.neld_path_org_new
            path_gen=self.obj_org_path.split('\\')
            path_gen=os.path.join(*path_gen)
            path_init_param=os.path.join(path_gen,f"param.pkl") 
            path_true=os.path.join(path_1,f"true_{nam}.txt")
            path_obs=os.path.join(path_1,f"obs.txt")

            ii=-1

            ku,kuu='inference','true' 
            scatterr={mm:[] for mm in [ku,kuu]}  

            ku='inference'
            tyy=  self.intensity_logit_dict[ku][1]
            patth=self.path_file_sub[tyy][id_path]

            print('[[[exist]]]',os.path.exists(patth), os.path.exists(path_obs),patth,path_obs)
            if os.path.exists(patth) and os.path.exists(path_obs): 

                yy,tru=np.loadtxt(patth, dtype=float),np.loadtxt(path_obs, dtype=float)   
                err=loss_fn(yy,tru) 
                print('[[[]]]',[yy.shape   ]) 
                scatterr[kuu].append(
                    hf.plotly_scatter(
                        points=yy,
                        color=ccoll[ii*2] ,
                        size=9.3,
                        mode='markers', 
                        symbol='circle' ,
                        name='Meas. Estimate',
                        opacity=0.5
                    ) 
                )
                scatterr[kuu].append(
                    hf.plotly_scatter(
                        points=tru,
                        color=ccoll[ii*2+1] ,
                        size=4.3,
                        mode='lines+markers', 
                        symbol="circle"  ,
                        name=f'Meas. True <br>MSE:{err:.2e}',
                        opacity=0.8
                    ) 
                )




            '''
            # for mn,tyy,gh in zip(['Obs. prio','Mes. prio'],self.intensity_logit_dict[kuu],[path_true,path_obs]):
            for mn,tyy,gh in zip(['Obs. Estim',],self.intensity_logit_dict[kuu][:1],[path_obs,]):
                ii+=1
                patth=self.path_file_sub[tyy][id_path].replace('\\','/') 
                print('[[[exist]]]',os.path.exists(patth), os.path.exists(path_true),patth,path_true)
                if os.path.exists(patth) and os.path.exists(gh): 

                    yy,tru=np.loadtxt(patth, dtype=float),np.loadtxt(gh, dtype=float)   
                    err=loss_fn(yy,tru) 
                    print('[[[]]]',[yy.shape   ]) 
                    scatterr[kuu].append(
                        hf.plotly_scatter(
                            points=yy,
                            color=ccoll[ii*2] ,
                            size=9.3,
                            mode='markers', 
                            symbol='circle' ,
                            name='Meas. Estimate',
                            opacity=0.5
                        ) 
                    )
                    scatterr[kuu].append(
                        hf.plotly_scatter(
                            points=tru,
                            color=ccoll[ii*2+1] ,
                            size=4.3,
                            mode='lines+markers', 
                            symbol="circle"  ,
                            name=f'Meas. True <br>MSE:{err:.2e}',
                            opacity=0.8
                        ) 
                    )

'''
            ku='inference'
            tyy=  self.intensity_logit_dict[ku][0]
            patth=self.path_file_sub[tyy][id_path]
            print('[[[exist]===]]',os.path.exists(patth), os.path.exists(path_true),patth,path_true)

            if os.path.exists(patth): 
                est=np.loadtxt(patth, dtype=float)
 
                scatterr[ku].append(
                    hf.plotly_scatter(
                        points=est,
                        color=ccoll[ii*2+3] ,
                        size=9.3,
                        mode='lines+markers', 
                        symbol="cross"  ,
                        name='Dyn. Estimate',
                        opacity=0.8
                    ) 
                ) 

                if  os.path.exists(path_true):
                    tru_init=np.loadtxt(path_true, dtype=float)  
                    err=None if not os.path.exists(patth) else loss_fn(est,tru_init)
                    scatterr[ku].append(
                        hf.plotly_scatter(
                            points=tru_init,
                            color=ccoll[ii*2+2] ,
                            size=4.3,
                            mode='lines+markers', 
                            symbol='square' ,
                            name=f'Dyn. True <br>MSE' if err is None else f'Dyn. True<br>MSE:{err:.2e}',
                            opacity=0.5
                        ) 
                    )




            rows = 2
            cols = 1   
            figure = akp.Plotly_Figure_Sub(
                rows=rows, 
                cols=cols,
                subplot_titles=( "Dynamics","Measurement",),
                shared_xaxes=True,  
                specs=[[{"type": "scene"}], [{"type": "scene"}]],
                vertical_spacing=0.04, 
            )

            for trs,row in zip( [ku,kuu],[1,2]): 
                for tr in scatterr[trs]:
                    figure.add_trace(tr, row=row, col=1) 

            scene_updates = {
                "scene": self.scene,   
                "scene2": self.scene   
            }
            
            figure.update_layout(**scene_updates)
            figure.update_layout(height=height * 1.5, )

            if hasattr(figure, 'layout') and figure.layout.annotations:
                for idx, annotation in enumerate(figure.layout.annotations):
                    if idx == 0:
                        annotation.y = 1.02
                    elif idx == 1:
                        annotation.y = 0.5 

            # [print('[im he=========------]',mm) for mm in [path_1,path_true,patth,patthy]]
            # figure=akp.Plotly_Figure(data= scatterr , layout=self.layout)
            # figure.update_layout(scene=self.scene) 
            # figure.update_layout(title=) 

            # figure.update_yaxes(title_text="Dynamics" )
            # figure.update_xaxes(title_text="Time", ) 
   

 
 


        elif mode=='accuracy':
            title='Accuracy'
            width=width
            height=height
            subplot_titles=('IOU Single','IOU Union')
            rows=2
            listt=['single','union',]
            if 'single_dice' in self.metric_total_dic:
                subplot_titles=('IOU Single','IOU Union','DICE Single','DICE Union')
                rows=4
                listt=['single','union','single_dice','union_dice']

            figure=akp.Plotly_Figure_Sub( subplot_titles,rows=rows, cols=1, 
                                            shared_xaxes=False,
                                            shared_yaxes=False)
            if len(self.metric_total_dic )>0: 
                for ii,typ in enumerate(listt):
                    # df_union=pd.read_csv(os.path.join(self.path_file[f'result_appr'],f'metric_{typ}.csv')).sort_values(by=f'single_{true_keys}', ascending=True)
                    df_union=pd.read_csv(os.path.join(self.path_file[f'result_appr'],f'metric_{typ}_{true_keys}.csv')).sort_values(by=f'accuracy', ascending=True)
                    xlabels=df_union.columns[1:]
                    ylabels=df_union.iloc[:,0]

                    df_union=df_union[df_union.columns[1:]]
                    cm=np.array(df_union.select_dtypes(include=[np.number]))
                    cm= np.round(cm*1e3)/1e3

                    scatter=go.Heatmap(
                                        z=cm,
                                        x=xlabels,  
                                        y=ylabels,  
                                        colorscale='Blues',
                                        text=cm,  # Show numbers in cells
                                        texttemplate="%{text}",  # Format as numbers
                                        # hovertemplate="Actual: %{y}<br>Predicted: %{x}<br>Count: %{z}<extra></extra>",
                                        showscale=False ,
                                        textfont=dict(size=18),
                                    )  
                    figure.add_trace( scatter, row=ii+1, col=1) 
            figure.update_layout(height=2.5*height, width=width)

            figure.update_layout(
                xaxis=dict(
                    side="top",          
                    ticks="outside",     # Optional: ticks outside the plot
                    showticklabels=True, # Ensure tick labels are visible 
                    tickmode="array",
                    tickvals=['accuracy','precision','recall','f1_score','AUC_spine','AUC_shaft'],
                    ticktext=['Accuracy','Precision','Recall','F1 Score','AUC Spine','AUC Shaft'],
                )
            )

            figure.update_layout(
                xaxis2=dict(
                    side="top",          
                    ticks="outside",     # Optional: ticks outside the plot
                    showticklabels=True, # Ensure tick labels are visible 
                    tickmode="array",
                    tickvals=['accuracy','precision','recall','f1_score','AUC_spine','AUC_shaft'],
                    ticktext=['Accuracy','Precision','Recall','F1 Score','AUC Spine','AUC Shaft'],
                )
            )
            figure.update_layout(
                xaxis3=dict(
                    side="top",          
                    ticks="outside",     # Optional: ticks outside the plot
                    showticklabels=True, # Ensure tick labels are visible 
                    tickmode="array",
                    tickvals=['accuracy','precision','recall','f1_score','AUC_spine','AUC_shaft'],
                    ticktext=['Accuracy','Precision','Recall','F1 Score','AUC Spine','AUC Shaft'],
                )
            )
            figure.update_layout(
                xaxis4=dict(
                    side="top",          
                    ticks="outside",    
                    showticklabels=True, 
                    tickmode="array",
                    tickvals=['accuracy','precision','recall','f1_score','AUC_spine','AUC_shaft'],
                    ticktext=['Accuracy','Precision','Recall','F1 Score','AUC Spine','AUC Shaft'],
                )
            )
            for annotation in figure['layout']['annotations']:
                annotation['y'] += 0.045

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

            if (mode == 'model_sp_loss' ) and id_path in self.scatter_loss_spine_dic and len( self.scatter_loss_spine_dic[id_path])>0: 
                figure=akp.Plotly_Figure(data=self.scatter_loss_spine_dic[id_path], layout=self.layout)
                figure.update_layout(scene=self.scene) 
            elif (mode == 'model_sp_iou' ) and id_path in self.scatter_iou_spine_dic and len(self.scatter_iou_spine_dic[id_path])>0: 
                print('----=====----','9088==============---========================================',len(self.scatter_iou_spine_dic[id_path]))
                figure=akp.Plotly_Figure(data= self.scatter_iou_spine_dic[id_path], layout=self.layout)
                figure.update_layout(scene=self.scene)
            elif (mode == 'model_sp_auc' ) and id_path in self.scatter_auc_spine_dic and len(self.scatter_auc_spine_dic[id_path])>0: 
                print('----=====----','model_sp_auc==============---========================================',len(self.scatter_auc_spine_dic[id_path]))
                figure=akp.Plotly_Figure(data= self.scatter_auc_spine_dic[id_path], layout=self.layout)
                figure.update_layout(scene=self.scene)
            elif (mode == 'model_sp_dice' ) and id_path in self.scatter_dice_spine_dic and len(self.scatter_dice_spine_dic[id_path])>0: 
                print('----=====----','model_sp_auc==============---========================================',len(self.scatter_dice_spine_dic[id_path]))
                figure=akp.Plotly_Figure(data= self.scatter_dice_spine_dic[id_path], layout=self.layout)
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


        elif mode =='roc_curve':  
            subplot_titles=('ROC Spine','ROC Shaft')
            figure = akp.Plotly_Figure_Sub(
                subplot_titles,
                rows=2, cols=1,
                shared_xaxes=False,
                shared_yaxes=False
            )

            if len(self.metric_total_dic['roc_curve']) > 0:
                for ii, typ in enumerate(['spine','shaft']):
                    traces = self.metric_total_dic['roc_curve']['curve'][typ]

                    figure.add_traces(
                        traces,
                        rows=[ii+1] * len(traces),
                        cols=[1] * len(traces)
                    )

                    figure.add_trace(
                        go.Scatter(
                            x=[0, 1], y=[0, 1],
                            mode="lines",
                            name="",
                            line=dict(color="gray", dash="dash")
                        ),
                        row=ii+1, col=1
                    )

                figure.update_layout(scene=self.scene)

                figure.update_layout(
                    title="ROC Curves",
                    xaxis=dict(title="False Positive Rate"),
                    xaxis2=dict(title="False Positive Rate"),
                    yaxis=dict(title="True Positive Rate"),
                    yaxis2=dict(title="True Positive Rate"),
                    barmode="overlay"
                )



        elif mode in ['heatmap_iou','heatmap_iou_union','histogram_iou','roc_curve']: 
            spine_path_save=     self.path_file[f'result_{id_path}']
            metric_path=os.path.join( spine_path_save,f'iou_{true_keys}.csv') 
            if os.path.exists(metric_path):
                df = pd.read_csv(metric_path) 
                sze_checks_0 ,sze_check ,sze_check_un=df['id'],df['iou_single'],df['iou_union']
                # nhf=df['id_true']>0
                # sze_checks_0 ,sze_check ,sze_check_un=sze_checks_0[nhf] ,sze_check[nhf] ,sze_check_un[nhf]
                metric_name_path=df.columns
                iou_dict={}
                get_cm_iou(sze_checks_0 ,sze_check ,sze_check_un ,iou_dict=iou_dict, iou_per=70,labels = ['False', 'True'],nbinsx=nbin,)
                if mode =='heatmap_iou':
                    accuracy, precision, recall, f1_score=iou_dict['single']['metrics'].values()
                    metrics_text = (f'Accuracy: {accuracy:.3f}   '
                                    f'Precision: {precision:.3f}<br>'
                                    f'Recall    : {recall:.3f}   '
                                    f'F1 Score: {f1_score:.3f}')
                    figure=akp.Plotly_Figure(data=iou_dict['single']['heatmap_cm'], layout=self.layout)
                    figure.update_layout(scene=self.scene)

                    figure.update_layout(
                        title={
                            'text': metrics_text,
                            'x': 0.5,
                            'y': 0.92,
                            'xanchor': 'center',
                            'yanchor': 'top'
                        }
                    )
                    figure.update_layout(
                        xaxis=dict(
                            # title=dict(text="Iterations", font=dict(size=16)),
                            tickmode="array",
                            tickvals=['accuracy','precision','recall','f1_score'],
                            ticktext=['Accuracy','Precision','Recall','F1 Score'],
                            # tickfont=dict(size=14, family="Arial", color="black")
                        ),
                        # yaxis=dict(
                        #     title=dict(text="IoU", font=dict(size=16)),
                        #     tickmode="linear",
                        #     dtick=0.1,  # step size
                        #     tickfont=dict(size=14, family="Arial", color="black")
                        # )
                    )
                elif mode =='heatmap_iou_union':
                    accuracy, precision, recall, f1_score=iou_dict['union']['metrics'].values()
                    metrics_text = (f'Accuracy: {accuracy:.3f}   '
                                    f'Precision: {precision:.3f}<br>'
                                    f'Recall    : {recall:.3f}   '
                                    f'F1 Score: {f1_score:.3f}')
                    figure=akp.Plotly_Figure(data=iou_dict['union']['heatmap_cm'], layout=self.layout)
                    figure.update_layout(scene=self.scene)

                    figure.update_layout(
                        title={
                            'text': metrics_text,
                            'x': 0.5,
                            'y': 0.92,
                            'xanchor': 'center',
                            'yanchor': 'top'
                        } 
                    )
                    fig.update_layout(
                        xaxis=dict(
                            # title=dict(text="Iterations", font=dict(size=16)),
                            tickmode="array",
                            tickvals=['accuracy','precision','recall','f1_score'],
                            ticktext=['Accuracy','Precision','Recall','F1 Score'],
                            # tickfont=dict(size=14, family="Arial", color="black")
                        ), 
                    )
                elif mode =='histogram_iou':

                    figure=akp.Plotly_Figure(data= iou_dict['single']['histogram'] , layout=self.layout)
                    figure.add_annotation(
                        text=f'Total Count: {len(sze_checks_0)}',
                        xref='paper',
                        yref='paper',
                        x=0.98,
                        y=0.95,
                        showarrow=False,
                        font=dict(size=22)
                    )
                    figure.update_layout(scene=self.scene)  

                    figure.add_trace(iou_dict['union']['histogram'] )
                    figure.add_annotation(
                        text=f'Total Count Union.: {len(sze_checks_0)}',
                        xref='paper',
                        yref='paper',    
                        x=0.98,
                        y=0.88,
                        showarrow=False,
                        font=dict(size=22)
                    )

                    figure.update_layout(
                        barmode='overlay',  
                        title=mode,
                        xaxis_title='Values',
                        yaxis_title='Iou'
                    )



                    # figure.add_annotation(
                    #     text=f'Total Count Union.: {len(sze_checks_0)}',
                    #     xref='paper',
                    #     yref='paper',    
                    #     x=0.98,
                    #     y=0.88,
                    #     showarrow=False,
                    #     font=dict(size=22)
                    # )

                    # figure.update_layout(
                    #     barmode='overlay',  
                    #     title=mode,
                    #     xaxis_title='Values',
                    #     yaxis_title='Iou'
                    # )



        elif mode in self.metric_mapping_combine['name']:  
            neld_names=self.neld_names 
            spine_path_save=     self.path_file[f'result_{id_path}']
            metric_path=os.path.join( spine_path_save,'metrics.csv') 
            if os.path.exists(metric_path):  
                df = pd.read_csv(metric_path)   
                
                headd,neckk,lengthh=df[self.metrics_combine[mode]['key'][0]],df[self.metrics_combine[mode]['key'][1]],df[self.metrics_combine[mode]['key'][2]]
                fig=hp.plotly_metric(headd,neckk,lengthh,height=height,width=width,title_size=22,colorscale='Blues',
                                         xtitle=add_unit(self.metrics_combine[mode]['label'][0]),
                                         ytitle=add_unit(self.metrics_combine[mode]['label'][1]),
                                         ztitle=add_unit(self.metrics_combine[mode]['label'][2]),
                                         marginal='box')
                figure=akp.Plotly_Figure(data= fig.data, layout=fig.layout) 
                # if mode=='vol_area_length_spine':
                m, b = np.polyfit(headd, neckk, 1) 
                y_pred = m * headd + b 
                y_mean = np.mean(neckk)
                ss_res = np.sum((neckk - y_pred)**2)
                ss_tot = np.sum((neckk - y_mean)**2)
                r2 = 1 - (ss_res / ss_tot)
                n=len(neckk) 
                reg_x = np.linspace(headd.min(), headd.max()*1.2, 100)
                reg_y = m * reg_x + b 
                # reg_y = np.clip(reg_y, neckk.min(), neckk.max()*1.2) 
                figure.add_trace(go.Scatter(x=reg_x, y=reg_y, mode="lines", name="Linear fit"))
                equation_text = f"y = {m:.3f}x + {b:.3f}<br>R<sup>2</sup> = {r2:.3f}<br>n = {n}" 
                # equation_text = ( 
                #     f"y   = {m:.3f}x + {b:.3f}\n"
                #     f"R<sup>2</sup> = {r2:.3f}\n"
                #     f"n   = {n}" 
                # ) 
                figure.add_annotation(
                    x=headd.min() + 0.01*(headd.max() - headd.min()),  
                    y=neckk.min() + 0.9*(neckk.max() - neckk.min()),   
                    text=equation_text,
                    showarrow=False,
                    font=dict(  size=20, color="white"),
                    bgcolor="black",
                    bordercolor="black", 
                    xanchor="left",    
                    align="left"   ,
                )
  
            #     figure.update_layout(
            #     # title=title,
            #     xaxis2_title='Counts',#self.metrics_combine[mode]['label'][0],
            #     yaxis2_title='Counts',self.metrics_combine[mode]['label'][1], 
            # )                
                figure.update_layout(
                                xaxis1=dict(
                                    title=dict(
                                        text=add_unit(self.metrics_combine[mode]['label'][0]),
                                        font=dict(size=20)
                                    ),
                                    # range=[-3, 3],
                                    tickfont=dict(size=20),
                                    showticklabels=True,
                                ),
                                yaxis1=dict(
                                    title=dict(
                                        text=add_unit(self.metrics_combine[mode]['label'][1]),
                                        font=dict(size=20),
                                    ),
                                    # range=[-5, 5],
                                    tickfont=dict(size=20),
                                    showticklabels=True,
                                ),  

                                
                                xaxis2=dict(
                                    title=dict(
                                        text="Counts",
                                        font=dict(size=20)
                                    ),
                                    # range=[-3, 3],
                                    tickfont=dict(size=20),
                                    showticklabels=True,
                                ),
                                yaxis2=dict(
                                    title=dict(
                                        text="Counts",
                                        font=dict(size=20),
                                    ),
                                    # range=[-5, 5],
                                    tickfont=dict(size=20),
                                    showticklabels=True,
                                ),  



                                
                                xaxis3=dict(
                                    title=dict(
                                        text="x3",
                                        font=dict(size=20)
                                    ),
                                    range=[-3, 3],
                                    tickfont=dict(size=20),
                                    showticklabels=True,
                                ),
                                yaxis3=dict(
                                    title=dict(
                                        text="y3",
                                        font=dict(size=20),
                                    ),
                                    range=[-5, 5],
                                    tickfont=dict(size=20),
                                    showticklabels=True,
                                ),  
            )

 
        elif mode in self.metric_mapping['name']:  
            neld_names=self.neld_names 
            spine_path_save=     self.path_file[f'result_{id_path}']
            metric_path=os.path.join( spine_path_save,'metrics.csv') 
            # figure=akp.Plotly_Figure(data=[],layout=None)
            # figure.update_layout(scene=self.scene)  

            scolor=['blue','red','yellow','purple','green']
            data_path=self.path_file['result_true']

            df_save={}
            name=name_approx='Approx'
            df_save[name]={}
            xmax=0
            xmin=-np.inf
            df_save[name]['ann']=['blue',.6,0.98,0.95]
            df_save[name]['path']=metric_path
            if os.path.exists(df_save[name]['path']):  
                df_save[name]['df']=df = pd.read_csv(df_save[name]['path'])   
                # df=df[df[df.columns[0]].astype(str).str.startswith(tuple(neld_names))] 
                df_save[name]['df']=  df
                if mode in df.columns:  
                    xmax=max(xmax,max(df[mode]))
                    xmin=min(xmin,min(df[mode]))
            namesan=[]
            names=[]
            for ii in range(1,5):
                data_name=f'spine_head_analysis.trial_{ii}.dat' 
                df_save[name]['ann']=[scolor[ii],.5,0.98,0.95-(ii*0.07)/1.3] 
                if os.path.exists( os.path.join(data_path,data_name)):
                    name=f'Annot_{ii}'
                    names.append(name)
                    namesan.append(name)
                    df_save[name]={}
                    df_save[name]['path']=hff.get_conversion_file(data_path=data_path,data_name=data_name)
                    df = pd.read_csv(df_save[name]['path'])  
                    # df=df[df[df.columns[0]].astype(str).str.startswith(tuple(neld_names))] 
                    df_save[name]['df']=  df
                    if mode in df.columns:  
                        xmax=max(xmax,max(df[mode]))
                        xmin=min(xmin,min(df[mode]))
            names.append(name_approx)
            title='KL Divergence D_KL(P||Q)'
            xaxis_title,yaxis_title='P','Q'


            if len(namesan)>0 :
                subplot_titles=('Histogram','KL Divergence D_KL(P||Q)')
                figure=akp.Plotly_Figure_Sub( subplot_titles,rows=2, cols=1, )
            else:
                figure= akp.Plotly_Figure(data=[],layout=None)

            for name in names:
                df = df_save[name]['df']
                color,opacity,x_ann,y_ann=df_save[name]['ann']  
                if os.path.exists(df_save[name]['path']):  
                    if mode in df.columns:  
                        scatter_metric,layout_metric=hp.Plotly_histogram_return(data=np.abs(df[mode]),
                                                                                nbinsx=nbin,
                                                                                title=mode,
                                                                                xtitle='Length',
                                                                                ytitle='Count',
                                                                                xrange=[xmin,xmax],
                                                                                yrange=None,
                                                                                width=width, 
                                                                                height=height,
                                                                                opacity=opacity,
                                                                                color=color,
                                                                                name=name,)
                        if len(namesan)>0 :
                            figure.add_trace( scatter_metric, row=1, col=1)
                        else:
                            figure.add_trace( scatter_metric )
                        cname=f'Count {name}  ' if name=='Approx' else f'Count {name}' 
                        figure.add_annotation(
                            text=f'{cname}: {len(df[mode])}',
                            xref='paper',
                            yref='paper',    
                            x=x_ann,
                            y=y_ann,
                            showarrow=False,
                            font=dict(size=22)
                        )  

                    figure.update_layout(layout_metric)
                        
                    if len(namesan)>0 :
                        scatter,layout=compute_kl(df_save,names,mode,width,height,bins =nbin)
                        figure.add_trace( scatter, row=2, col=1)
                        figure.update_layout(layout )
                        figure.update_layout(
                        # title=title,
                        xaxis2_title=xaxis_title,
                        yaxis2_title=yaxis_title,
                        # xaxis=dict(tickmode='array', tickvals=list(range(len(labels))), ticktext=labels),
                        # yaxis=dict(tickmode='array', tickvals=list(range(len(labels))), ticktext=labels), 
                        )
                        figure.update_layout(height=1.3*height, width=width)
                    else:
                        figure.update_layout(height=height, width=width)

 

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
        ]
        ################################ INTENSITY #########################################
        self.dropdown_intensity_option= [
                            {'label': 'Clusterization',  'value': 'kmean',       'style': dropdown_options_style},
                            {'label': 'Segmentation',    'value': 'cluu_rad',    'style': dropdown_options_style},
                            {'label': 'Spines',          'value': 'spines',      'style': dropdown_options_style},
                            {'label': 'Spines True',     'value': 'spines true', 'style': dropdown_options_style},
                            {'label': 'Gauss Curvature', 'value': 'gauss',       'style': dropdown_options_style},
                            {'label': 'Mean Curvature',  'value': 'mean',        'style': dropdown_options_style},
                    ]


    def model_test(self):
        dropdown_options_style=self.dropdown_options_style
        dropdown_mode_option= [
                            {'label': 'Image',         'value': 'algorithm', 'style': dropdown_options_style},
                            # {'label': 'Comparison',        'value': 'comparison',    'style': dropdown_options_style},  
                            {'label': 'Skeleton',      'value': 'skeleton', 'style': dropdown_options_style},
                            {'label': 'Accuracy',              'value': 'accuracy','style': dropdown_options_style}, 
                            {'label': 'ROC Curve',              'value': 'roc_curve','style': dropdown_options_style}, 
                    ]  
        # dropdown_mode_option.append({'label': 'IOU',      'value': 'IOU', 'style': dropdown_options_style},)
        for mmnn in self.inten_file_model_head_neck:
            dropdown_mode_option.append({'label': mmnn,      'value': mmnn, 'style': dropdown_options_style},)
 
        # metric_name=self.metrics_keys
        metric_name=[]


        metric_name.extend(['heatmap_cylinder','heatmap_cylinder_surface','ripley','dijkstra_detail'])
        metric_name.extend(self.metrics_keys)
        # metric_name.extend(['heatmap_iou','heatmap_iou_union','histogram_iou'])
        self.metric_mapping = {
            "name": metric_name,
            "title": [f"{mm} Histogram" for mm in metric_name],
            "index": {mm: ii for ii, mm in enumerate(metric_name)},
            "xtitle": "Length",
            "ytitle": "Count",
        }
        self.metrics_combine={'diam_head_neck_length'  :{'key':  ['head_diameter','neck_diameter','spine_length'],
                                                       'label':['Head Diameter','Neck Diameter','Spine Length']},
                              'vol_area_length_spine':{'key':['spine_vol','spine_area','spine_length'],
                                                        'label':['Spine Volume','Spine Area','Spine Length']},
                              'vol_head_neck_spine'    :{'key':['head_vol','neck_vol','spine_vol'],
                                                     'label':['Head Volume','Neck Volume','Spine Volume'],
                                                     },
                              'area_head_neck_spine'  :{'key':['head_area','neck_area','spine_area'],
                                                      'label':['Head Area','Neck Area','Spine Area']},
                              'length_head_neck_spine':{'key':['head_length','neck_length','spine_length'],
                                                        'label':['Head Length','Neck Length','Spine Length']},
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
        action_name='mode'
        self.dropdown_mode ={
                        'option':dropdown_mode_option,
                        'id'         :f'dropdown_{action_name}',
                        'value'      :'algorithm',
                        'placeholder':f'Select {action_name}', 
        }
 
        ################################ METRIC #########################################


        self.dropdown_plot_option=[ 
                            {'label': 'Accuracy',              'value': 'accuracy','style': dropdown_options_style},
                            {'label': 'IOU',                   'value': 'iou',     'style': dropdown_options_style},
                            {'label': 'Confusion Matrix',      'value': 'conf',    'style': dropdown_options_style},
                            {'label': 'Classification Report', 'value': 'report',  'style': dropdown_options_style},
                            {'label': 'Compare',               'value': 'compare', 'style': dropdown_options_style},
        ] 

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
        'option':self.dropdown_plot_option,
        'id'          :id_name,
        'value'       :self.dropdown_plot_option[0]['value'],
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
                        'min':0,
                        'max':1000,
                        'step':5,
                        'value':100, 
                        'marks':{i: f'{i}' for i in range(0, 1000, 200)}, 
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
        dropdown_dend_option = [{'label': 'Initial', 'value': 'init', 'style': dropdown_options_style} ,
                                {'label': f'Smoothed', 'value': 'smooth', 'style': dropdown_options_style}]
        action_name='dendrite curve state'
        id_name=f'dropdown_{action_name}_{id_name_end}'
        self.dropdown_dend={
            'option'     :dropdown_dend_option,
            'id'         :id_name,
            'value'      :dropdown_dend_option[0]['value'],
            'placeholder':f'Select {action_name}', 
        }
 

    def get_dropdown_cluster_test(self,id_name_end,count):
        dropdown_options_style=self.dropdown_options_style 
        dropdown_cluster_option = [{'label': 'Dendrite', 'value': 0, 'style': dropdown_options_style}] 
        dropdown_cluster_option.append({'label': f'Shaft', 'value': 1, 'style': dropdown_options_style}) 
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
        dropdown_cluster_option =     [{'label': 'Dendrite', 'value': 0, 'style': dropdown_options_style}] 
        dropdown_cluster_option.append({'label': f'Shaft'  , 'value': 1, 'style': dropdown_options_style})
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


 
from mod.dsa.neld_fun_0.main_0_help import dropdown_callback
 



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
                          self.dropdown_intensity['id'],
                          self.hist_slider['id'],
                          self.dropdown_cluster['id'],
                        self.width_slider['id'],
                        self.height_slider['id'],
                        self.dropdown_template['id'], ]
        self.Input=[Input(mm,'value') for mm in self.Input_ids+self.Input_idsST] 





    def Get_children(self):  
        svv=[
            html.Br(),
            html.Label('Histogram Bin Count:', style=self.dropdown_options_style),
            dcc.Slider(
                id=self.hist_slider['id'],
                min=self.hist_slider['min'],
                max=self.hist_slider['max'],
                step=self.hist_slider['step'],
                value=self.hist_slider['value'],
                marks=self.hist_slider['marks'],
            ),  
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
            self.dropdown_cluster,
            self.dropdown_mode,
            self.dropdown_intensity,
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



 

