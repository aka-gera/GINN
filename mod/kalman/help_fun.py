import tensorflow as tf
import numpy as np


 

def dynamics(x,dt,param): 
    return x + dt * param['fun'](x,**param['param'])

def observation(x,dim=None):
    
    dim=dim if dim is not None else x.shape[-1] 
    # print('[[[[[[[[ppppp]]]]]]]]',[x[...,i] for i in range(dim)],tf.stack([x[...,i] for i in range(dim)]))
    return  tf.stack([x[...,i] for i in range(dim)]) 





import os,time,pickle

def get_data_fun(self,param,fmt='%.4f',disp_infos=True,step_test=None): 
    x0,dt,step,process_std_init,process_std_true,process_std,measurement_std,state_dim,obs_dim,dyna,window,KB0,weight= [param['par_0'][ky] for ky in param['par_0']['KY']] 
    step=step if step_test is None else step_test
    print(f'get_neld_data started')
    print('--------------------------------')


    path_gen=self.obj_org_path
    path_geni=os.path.join(self.obj_org_path,self.model_sufix ,self.data_studied)
    os.makedirs(path_gen,exist_ok=True)
    os.makedirs(path_geni,exist_ok=True)
    path = os.path.join(path_gen,"simulation_parameters.txt")


    path_init_param=os.path.join(path_geni,f"param.pkl")
    with open(path_init_param, "wb") as f:
            pickle.dump(param, f) 




    time_start=time.time() 
    # for index in range(Nsample):            
    for  index,neld_name in enumerate(self.neld_names):
        self.get_neld_name(index=index, ) 
        # path_1=os.path.join(path_gen,f'data_{index}')
        path_1=self.neld_path_org_new
        os.makedirs(path_1,exist_ok=True)



        path_init=os.path.join(path_1,f"save.pkl")

        with open(path_init_param, "rb") as f:
            param = pickle.load(f)
        
 

        get_new_data=False 
        if not get_new_data: 
            x_true = np.random.uniform(-process_std_init, process_std_init, size=[state_dim])

            save={}
            save['par']={mm:[] for mm in ['true','obs',]}
            save['par']['true'] = np.zeros((step, state_dim))
            save['par']['obs']  = np.zeros((step, obs_dim))

            for ii in range(step):
                x_true=dynamics(x_true,dt=dt,param=dyna)+process_std_true *np.random.standard_normal([state_dim])
                z=observation(x_true,dim=obs_dim)+measurement_std * np.random.standard_normal([obs_dim])
                save['par']['true'][ii]=x_true
                save['par']['obs'][ii]=z


            for key in save['par']: 
                path_=os.path.join(path_1,f"{key}.txt")
                np.savetxt(path_,save['par'][key],fmt=fmt)
  


        mytime0 = time.time() - time_start

        hours, rem = divmod(mytime0, 3600)
        minutes, seconds = divmod(rem, 60)
        if disp_infos:
            print(f'Data generation completed on {neld_name} in {int(hours)}h {int(minutes)}m {seconds:.2f}s')





def get_data_rnn(param,model=None,x_trues=None,obs=None): 
    x0,dt,step,process_std_init,process_std_true,process_std,measurement_std,state_dim,obs_dim,dyna,window,KB0,weight= [param['par_0'][ky] for ky in param['par_0']['KY']] 
 
    if x_trues is not None:
        step=x_trues.shape[0]
        x_true=tf.cast(x_trues[0,:], tf.float32)  
    else:
        x_true = tf.cast(x0, tf.float32)
    xhat=x_true+ process_std * tf.random.normal([state_dim]) 
 
    hidden=None
    save={}
    save['par']={mm:[] for mm in ['true','obs','approx']}
    for ii in range(step):
        if x_trues is None:
            x_true=dynamics(x_true,dt=dt,param=dyna)+process_std *tf.random.normal([state_dim])
            z=observation(x_true,dim=obs_dim)+measurement_std * tf.random.normal([obs_dim])
        else:
            x_true = tf.cast(x_trues[ii, :], tf.float32) 
            z = tf.cast(obs[ii, :], tf.float32)  

        x_pred=dynamics(xhat,dt=dt,param=dyna)
        z_pred=observation(x_pred,dim=obs_dim) 
        innov=z-z_pred 
        if model is not None:
            feats=tf.reshape(tf.concat([x_pred,innov],axis=0),[1,1,state_dim+obs_dim]) 
            KB,hidden=model(feats,hidden)
            KB0=weight*KB[0] 
        corr=tf.reshape(tf.matmul(KB0,tf.reshape(innov,shape=[obs_dim,1])),[state_dim])
        xhat=x_pred+corr
        save['par']['true'].append(x_true)
        save['par']['obs'].append(z)
        save['par']['approx'].append(xhat)
    for key in save['par']:
        save['par'][key]=tf.stack(save['par'][key])
    save['KB']=KB0
    return save





def get_data_trans(param,model=None,x_trues=None,obs=None): 
    x0,dt,step,process_std_init,process_std_true,process_std,measurement_std,state_dim,obs_dim,dyna,window,KB0,weight= [param['par_0'][ky] for ky in param['par_0']['KY']] 
 
    if x_trues is not None:
        step=x_trues.shape[0]
        x_true=tf.cast(x_trues[0,:], tf.float32)  
    else:
        x_true = tf.cast(x0, tf.float32)
    xhat=x_true+ process_std * tf.random.normal([state_dim])

    history=[]
    hidden=None
    save={}
    save['par']={mm:[] for mm in ['true','obs','approx']}
    for ii in range(step):
        if x_trues is None:
            x_true=dynamics(x_true,dt=dt,param=dyna)+process_std *tf.random.normal([state_dim])
            z=observation(x_true,dim=obs_dim)+measurement_std * tf.random.normal([obs_dim])
        else:
            x_true = tf.cast(x_trues[ii, :], tf.float32) 
            z = tf.cast(obs[ii, :], tf.float32)  
        x_pred=dynamics(xhat,dt=dt,param=dyna)
        z_pred=observation(x_pred,dim=obs_dim) 
        print('9999999999999--------',obs_dim,x_pred,z,z_pred,obs[ii, :],obs)
        innov=z-z_pred
        if model is not None:
            feats= tf.concat([x_pred,innov],axis=0)  
            history.append(feats) 
            if len(history) > window:
                history.pop(0) 
            seq=tf.expand_dims(tf.stack(history),axis=0)
            KB=model(seq)
            KB0=weight*KB[:,-1][0]
        # print(f'{ii}=== ==={KB[0]=}',x_pred) 
        # KB0=0.1*KB[0]
        corr=tf.reshape(tf.matmul(KB0,tf.reshape(innov,shape=[obs_dim,1])),[state_dim])
        xhat=x_pred+corr
        save['par']['true'].append(x_true)
        save['par']['obs'].append(z)
        save['par']['approx'].append(xhat)
    for key in save['par']:
        save['par'][key]=tf.stack(save['par'][key])
    save['KB']=KB0
    return save

 
















def lorenz(x, 
        sigma = 10.0,
        rho = 28.0,
        beta = 8.0 / 3.0,
           ):
    """
    x shape: (..., 3)

    Returns dx/dt
    """

    x1 = x[..., 0]
    x2 = x[..., 1]
    x3 = x[..., 2]

    dx1 = sigma * (x2 - x1)
    dx2 = x1 * (rho - x3) - x2
    dx3 = x1 * x2 - beta * x3

    return tf.stack(
        [dx1, dx2, dx3],
        axis=-1
    )











def get_configs(x0=None,
                step=100,
                step_test=None,
                window=10,
                process_std_init=2.1,
                process_std_true=2,
                process_std=0.001,
                measurement_std=0.01,
                dt=0.001,
                weight=0.1):

    dyna={
        'lorenz_0':dict(param=dict( 
                                sigma = 10.0,
                                rho = 28.0,
                                beta = 8.0 / 3.0, 
                                ),
                        fun=lorenz,
                    ),
        'lorenz_1':dict(param=dict(
                                sigma = 10.0,
                                rho = 8.0,
                                beta = 3.0, 
                                ),
                        fun=lorenz,
                    )
    }
    Model_all={}
    if x0 is None:
        x0=tf.constant([1,1,1],dtype=tf.float32) 
    dnn_modes=[]
    KB=None
    for d_nam in dyna:
        # for step in steps:
        #     for window in [10,step]:
        for state_dim,obs_dim in zip([3,3  ],[3,2 ], ):
            # mo=f'{d_nam}___s{state_dim}_o{obs_dim}_w{window}_s{step}'
            mo=f'{d_nam}___sd{state_dim}_od{obs_dim}'
            
            step=step if step_test is None else step_test
            param={
                mm:{} for mm in ['par_0','train_param','train_fun']
            }
            KY =['x0','dt','step',',process_std_init','process_std_true','process_std','measurement_std','state_dim','obs_dim','dynamics','window','KB','weight']
            VAL=[x0,dt,step,process_std_init,process_std_true,process_std,measurement_std,state_dim,obs_dim,dyna[d_nam],window,KB,weight]
            param['par_0']['KY']=KY
            for key,val in zip(KY,VAL):
                param['par_0'][key]=val
            param['train_param']['epoc']=10
            param['train_param']['avg_train']=5

            dnn_modes.append(mo)
            Model_all[mo]=dict(
                            data_sufix= mo,
                            dest_sufix= mo, 
                            param=param,
                    )

    # for ii,g_name in enumerate(rhs_param['g_names']):
    #     mo=f'{g_name}'
    #     Model_all[mo]=dict(
    #                     data_sufix= mo,
    #                     dest_sufix= mo, 
    #             )
         
 

    # rhs_dim_is,rhs_dim_js=[[0,0],[1,2]] if rhs_param is None else [rhs_param['dim'][mm] for mm in  ['i','j',]]
    # for rhs_dim_i,rhs_dim_j in zip(rhs_dim_is,rhs_dim_js):
    #     for tf_mean in [True,False]:    
    #         for ii,g_name in enumerate(rhs_param['g_names']):
    #             nma=f'mean_{rhs_dim_i}{rhs_dim_j}' if tf_mean else f'{rhs_dim_i}{rhs_dim_j}'
    #             mo=f'{g_name}--{nma}' 
    #             Model_all[mo]=dict(
    #                         data_sufix= mo,
    #                         dest_sufix= mo,
    #                         Nmarkov=Nmarkov,
    #                         Nsample=None,
    #                         nSample_interval=nSample_interval,
    #                         nTest=nTest,
    #                         g_name=g_name,
    #                         tf_initial_bcs=True,
    #                         tf_final_bcs=True,
    #                         Samples_nbr=None,
    #                         tf_train=True, 
    #                         inter=None,
    #                         pm=pm,
    #                         rhs_dim=dict(zip(['i','j'],[rhs_dim_i,rhs_dim_j])),
    #                         tf_mean=tf_mean,
    #                     )
             



    return Model_all,dyna,dnn_modes








