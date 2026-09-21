import numpy as np
import time  
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense,Dropout
from keras.saving import register_keras_serializable
from tensorflow.keras import regularizers
from sklearn.metrics import roc_curve, auc
 
DTYPE = tf.float32
import random
np.random.seed(42)
tf.random.set_seed(42)
random.seed(42)
import tensorflow as tf
from tensorflow.keras import layers, Model

 
from mod.kalman.lorenz.kal_0.help_fun import get_data_trans,get_data_rnn,get_data_ekf,dynamics_jacobian,observable_jacobian
 

 

@register_keras_serializable(package="Custom")
class dnn_GINN(Model):
    def __init__(self, Dtype=DTYPE, 
                 hidden_layers=4, 
                 neurons_per_layer=50, 
                 n_classes=2,
                 activation_init="tanh",
                 activation_hidden="relu",
                 activation_last="tanh", 
                 l1=0.01,
                 l2=0.0000001,
                 **kwargs):
        super(dnn_GINN, self).__init__()
        self.hidden_layers = hidden_layers
        self.Dtype = Dtype
        self.n_classes=n_classes
        self.activation_init=activation_init
        self.activation_hidden=activation_hidden
        self.activation_last=activation_last
        k_regularizers=regularizers.l1_l2(l1=l1,l2=l2)
        
        self.ld_init = Dense(neurons_per_layer, activation=activation_init, dtype=Dtype)
        self.hidden_layers_list = [
            Dense(neurons_per_layer, 
                  activation=activation_hidden, 
                  kernel_regularizer=k_regularizers,
                  dtype=Dtype) 
            for _ in range(hidden_layers)
        ]
        self.ld_last = Dense(n_classes,  activation=activation_last, dtype=Dtype)

    def call(self, x):
        x = self.ld_init(x)
        for layer in self.hidden_layers_list:
            x = layer(x)
        x = self.ld_last(x)
        return x


    def get_config(self):
        config = super().get_config()
        config.update({
            "Dtype": self.Dtype,
            "n_classes": self.n_classes,
            "hidden_layers": self.hidden_layers,
            "activation_init": self.activation_init,
            "activation_hidden": self.activation_hidden,
            "activation_last": self.activation_last,
        })
        return config


    @classmethod
    def from_config(cls, config):
        return cls(**config)


 
 
@register_keras_serializable(package="Custom")
class rnn_KAL(tf.keras.Model): 
    def __init__(self,
                 state_dim=3,
                 obs_dim=2,
                 nbr_gru=64,
                 nbr_hidden=64,
                 activation_last='tanh',
                 activation_hidden='tanh',
                **kwargs,
            ):
        # super().__init__(self)
        super().__init__(**kwargs)
        self.state_dim=state_dim
        self.obs_dim=obs_dim
        self.nbr_gru=nbr_gru
        self.nbr_hidden=nbr_hidden
        self.config=dict(
            state_dim=state_dim,
            obs_dim=obs_dim,
            nbr_gru=nbr_gru,
            nbr_hidden=nbr_hidden,
            activation_last=activation_last,
            activation_hidden=activation_hidden,
        ) 

        self.gru = tf.keras.layers.GRU(
            self.nbr_gru,
            return_sequences=True,
            return_state=True
        )

        self.fc = tf.keras.Sequential([
            tf.keras.layers.Dense(
                self.nbr_hidden,
                activation=activation_hidden,
            ),

            tf.keras.layers.Dense(
                self.state_dim*self.obs_dim,
                activation=activation_last,
            )
        ])

    def call(self, inputs, hidden=None):
        output, hidden = self.gru(inputs,initial_state=hidden)
        gain =  self.fc(output) 
        gain = tf.reshape(gain,[-1, self.state_dim, self.obs_dim])

        return gain, hidden


    def get_config(self):
        config = super().get_config()    
        config.update(** self.config,)  
        return config


    @classmethod
    def from_config(cls, config):
        return cls(**config)




class ekf_KAL:
    def __init__(self,**kward): 
        pass
    def __call__(self,xhat,x_pred,P,param):
        dyna=param['dyna']
        R=param['R']
        Q=param['Q'] 
        Id=param['Id'] 
        dt=param['dt']
        F=dynamics_jacobian(fun=dyna['dyn_jacobian'],dt=dt,x=xhat,param=dyna)  
        H=observable_jacobian(fun=dyna['obs_jacobian'],dt=dt,x=x_pred,param=dyna) 
        P_pred=tf.matmul(F, tf.matmul(P, tf.linalg.matrix_transpose(F))) + Q 
        S=tf.matmul(H, tf.matmul(P_pred, tf.linalg.matrix_transpose(H)))+ R 
        KB0=tf.linalg.solve(S, tf.matmul(H, P_pred))
        KB0=tf.linalg.matrix_transpose(KB0)
        P=tf.matmul((Id-tf.matmul(KB0, H)), P_pred)

        return KB0,P









class jos_KAL:
    def __init__(self,**kward): 
        pass
    def __call__(self,xhat,x_pred,P,param):
        dyna=param['dyna']
        R=param['R']
        Q=param['Q'] 
        Id=param['Id'] 
        dt=param['dt']
        F=dynamics_jacobian(fun=dyna['dyn_jacobian'],dt=dt,x=xhat,param=dyna)  
        H=observable_jacobian(fun=dyna['obs_jacobian'],dt=dt,x=x_pred,param=dyna) 
        P_pred=tf.matmul(F, tf.matmul(P, tf.linalg.matrix_transpose(F))) + Q 

        P_pred = 0.5 * (
                    P_pred +
                    tf.linalg.matrix_transpose(P_pred)
                )

        S=tf.matmul(H, tf.matmul(P_pred, tf.linalg.matrix_transpose(H)))+ R 
        KB0=tf.linalg.solve(S, tf.matmul(H, P_pred))
        KB0=tf.linalg.matrix_transpose(KB0)

        I_KH = Id - KB0 @ H

        P = (I_KH @ P_pred @ tf.linalg.matrix_transpose(I_KH)
                    + KB0 @ R @ tf.linalg.matrix_transpose(KB0)
                )

        P = 0.5 *(P +tf.linalg.matrix_transpose(P) )

        
        return KB0,P













@register_keras_serializable(package="Custom")
class tfm_KAL(tf.keras.Model):

    def __init__(
        self,
        state_dim=3,
        obs_dim=2,
        d_model=64,
        num_heads=4,
        ff_dim=128,
        activation_last='tanh',
        activation_hidden='gelu',
        **kwargs,
    ):
        # super().__init__(self)
        super().__init__(**kwargs)

        self.state_dim = state_dim
        self.obs_dim = obs_dim

        self.config=dict(
            state_dim=state_dim,
            obs_dim=obs_dim,
            d_model=d_model,
            num_heads=num_heads,
            ff_dim=ff_dim,
            activation_last=activation_last,
            activation_hidden=activation_hidden,
        ) 
        
        self.input_projection = tf.keras.layers.Dense(
            d_model
        )

        self.attention = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=d_model // num_heads
        )

        self.norm1 = tf.keras.layers.LayerNormalization()

        self.ffn = tf.keras.Sequential([
            tf.keras.layers.Dense(
                ff_dim,
                activation=activation_hidden,
            ),
            tf.keras.layers.Dense(
                d_model
            )
        ])

        self.norm2 = tf.keras.layers.LayerNormalization()
        self.output_layer = tf.keras.layers.Dense(
            state_dim * obs_dim,
            # activation=activation_last
        )

    def call(self, inputs):
        # inputs:
        # [batch, time, state_dim + obs_dim]
        x = self.input_projection(inputs)
        # print("Transformer input:", inputs.shape)
        # print("Projected:", x.shape)
        attn = self.attention(
            query=x,
            key=x,
            value=x
        )
        x = self.norm1(x + attn)
        f = self.ffn(x)
        x = self.norm2(x + f)
        gain = self.output_layer(x)
        gain = tf.tanh(gain)

        gain = tf.reshape(
            gain,
            [
                tf.shape(gain)[0],
                tf.shape(gain)[1],
                self.state_dim,
                self.obs_dim
            ]
        )

        return gain
 

    def get_config(self):
        config = super().get_config()    
        config.update(** self.config,)  
        return config


    @classmethod
    def from_config(cls, config):
        return cls(**config)



class model_choice: 
    def __init__(self, model_type=None, 
        **kwargs,
    ):
        self.model_type = model_type 
        self.kwargs=kwargs
 
        self.models = {
            "tfm_KAL".lower(): {
                "class": tfm_KAL, 
                'get_data':get_data_trans,
            },
            'rnn_KAL'.lower(): {
                "class": rnn_KAL, 
                'get_data':get_data_rnn
            },  
            'ekf_KAL'.lower(): {
                "class": ekf_KAL, 
                'get_data':get_data_ekf
            },  
            'jos_KAL'.lower(): {
                "class": jos_KAL, 
                'get_data':get_data_ekf
            },  
        } 
 
    def get_custom_objects(self, model_type):
        model_type = model_type.lower()

        for key, entry in self.models.items():
            if model_type.startswith(key):
                cls = entry["class"]
                return {cls.__name__: cls}

        raise ValueError(f"Unknown model type: {model_type}")




    def get_model(
        self,
        model_type=None, 
        **kwargs,
    ):
        model_type = model_type or self.model_type 
        model_type = model_type.lower()
        for key, entry in self.models.items():
            if model_type.startswith(key):
                print(f'{model_type=}')

                cls = entry["class"]
 
                if cls in (tfm_KAL,rnn_KAL,ekf_KAL,jos_KAL):
                    return cls( 
                        **self.kwargs,
                        **kwargs,
                    )
                # elif cls in (ekf_KAL):
                #     return cls( 
                #         **self.kwargs,
                #         **kwargs,
                #     )


        raise ValueError(f"Unknown model type: {model_type}")


    def get_data(
        self,
        model_type=None, 
        **kwargs,
    ):
        model_type = model_type or self.model_type 
        model_type = model_type.lower()
        for key, entry in self.models.items():
            if model_type.startswith(key):
                return self.models[key]['get_data']
 
        raise ValueError(f"Unknown model type: {model_type}")





class fun_loss():
    def __init__(self,param): 
        pass
        self.fun=param['train_fun']['get_data']
        self.param=param
    def call(self,model,x_trues=None,obs=None):  
        sav=self.fun(self.param,model,x_trues=x_trues,obs=obs)
        # return tf.reduce_mean(tf.square(sav['par']['approx']-sav['par']['true']))
        # print('[[[[[[[[]]]]]]]]',x_trues)
        return tf.reduce_mean(tf.square(sav['par']['approx']-x_trues))
 

import os 

class aka_train(fun_loss):
    def __init__(self,param):
        self.param=param 
        fun_loss.__init__(self,self.param)
        
    @tf.function
    def train_PINN(self, optimizer, model):
        losst=0
        for _ in range(self.param['train_param']['avg_train']):
            with tf.GradientTape() as tape:
                loss = self.call(model)
            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
            losst+=loss.numpy()
        return losst/self.param['train_param']['avg_train']
 


    @tf.function
    def train_mod(self, optimizer, model,mod,data_studied,model_type,lst):
        losst=0
        tot=len(mod.kalman.lorenz.neld_names)
        for  index in lst:
            mod.kalman.lorenz.get_neld_name(data_studied=data_studied,
                               index=index,
                                model_type=model_type, ) 
            path_1=mod.kalman.lorenz.neld_path_org_new
            path_true=os.path.join(path_1,f"true.txt")
            path_obs=os.path.join(path_1,f"obs.txt")
            if os.path.exists(path_true):
                x_trues,obs=np.loadtxt(path_true,dtype=float),np.loadtxt(path_obs,dtype=float)
            else:
                x_trues=None;obs=None
            # print('[[[[[[[[[[[[path]]]]]]]]]]]]',path_1,x_trues)


            
            with tf.GradientTape() as tape:
                loss=self.call(model,x_trues=x_trues,obs=obs) 
            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

            losst+=loss 
        return losst/tot
 


    def test_mod(self, model,mod,data_studied,model_type,lst):
        losst=0
        tot=len(mod.kalman.lorenz.neld_names)
        for  index in lst:
            mod.kalman.lorenz.get_neld_name(data_studied=data_studied,
                               index=index,
                                model_type=model_type, ) 
            path_1=mod.kalman.lorenz.neld_path_org_new
            path_true=os.path.join(path_1,f"true.txt")
            path_obs=os.path.join(path_1,f"obs.txt")
            if os.path.exists(path_true):
                x_trues,obs=np.loadtxt(path_true,dtype=float),np.loadtxt(path_obs,dtype=float)
            else:
                x_trues=None;obs=None 
 
            loss=self.call(model,x_trues=x_trues,obs=obs)  

            losst+=loss 
        return losst/tot

