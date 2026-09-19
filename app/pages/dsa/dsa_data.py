

import os, sys ,dash 
sys.path.append(os.getcwd() ) 
from mod.dsa.apps.side_bar import sidebar ,get_dnn,dnn_page 


page_dir= '/dsa/dsa-ginn-data' 
page_name='GINN'  
page_name_view=dnn_page()['results']
forbidden_endswith=('test', 'train', '2', 'gen', 'pred', 'figs') 
forbidden_endswith = None if forbidden_endswith in (None, 'None') else forbidden_endswith


dash.register_page(__name__, 
                    title=page_name, 
                    name=page_name,
                    path=page_dir,
                    order=0) 

def layout():
    return get_dnn(page_dir,page_name_view,forbidden_endswith,)
