 

import dash_bootstrap_components as dbc
from dash import html, dcc
import dash,os
 

style={
        'color': 'black',
        'backgroundColor': 'grey',
        'height': '100vh'  # Full viewport height
    },
style_equation={'textAlign': 'center', 'color': 'white' }
 

from neld_data_all import general_nam


'''




def sidebar_grouped_dropdown():
    # Define categories and their prefixes
    categories = {
        "DNN": "dnn",
        "Volume": "vol",
        "Panel": "pnel",
    }

    # Prepare container for grouped links
    grouped_links = {cat: [] for cat in categories}

    # Loop through Dash pages
    for page in dash.page_registry.values():
        path = page["path"].lstrip("/")  # remove leading slash
        name = page["name"]

        # Assign page to its category
        for cat, prefix in categories.items():
            if path.startswith(prefix):
                grouped_links[cat].append(
                    dbc.NavLink(
                        name,
                        href=page["path"],
                        active="exact",
                        className="ms-3 mb-1"
                    )
                )

    # Build accordion items
    accordion_items = []
    for cat, links in grouped_links.items():
        accordion_items.append(
            dbc.AccordionItem(
                children=links if links else html.Div("No pages", className="text-muted ms-3"),
                title=cat
            )
        )

    return html.Div(
        [
            dbc.Accordion(
                accordion_items,
                start_collapsed=True,
                flush=True,
                always_open=False,
            )
        ],
        className="bg-dark p-3",
        style={"height": "100vh", "overflow-y": "auto"}
    )









def sidebar(page_dir):
    nav_links = []
    
    for page in dash.page_registry.values():
        # if page["path"].startswith(page_dir):
        if page["path"]==page_dir:
        # if page["path"].find(page_dir) == 0: 
            display_name = page["name"]
        elif page["path"] == "/projects":
            display_name = "App 1"
        else:
            continue  # Skip unrelated pages
        
        nav_links.append(
            dbc.NavLink(
                html.Div(display_name, className="text-center"),
                href=page["path"],
                active="exact",
            )
        )
    return dbc.Nav(
        children=nav_links,
        vertical=True,
        pills=True,
        className="bg-dark p-2",
    )
'''
def sidebar(page_dir,forbidden=None):
    nav_links = []

    for page in dash.page_registry.values():
        path = page["path"] 
        if not path.startswith(page_dir):
            continue
        if forbidden is not None:
            if not path.endswith(forbidden):
                continue
        display_name = page["name"] 
        diff = path[len(page_dir):] 

        if '/' in diff: 
            if not diff.startswith('/'):
                continue
 
             
        nav_links.append(
            dbc.NavLink(
                html.Div(display_name, className="text-center"),
                href=page["path"],
                active="exact",
            )
        )
    return dbc.Nav(
        children=nav_links,
        vertical=True,
        pills=True,
        className="bg-dark p-2",
    )



 
def layout_1(page_dir,forbidden=None,):
    return html.Div([
    dbc.Row(
        [
            # dbc.Col(
            #     [
            #         sidebar(page_dir=page_dir,forbidden=forbidden,) 
            #     ], xs=4, sm=4, md=2, lg=2, xl=2, xxl=2),
            dbc.Col(
                [
                    sidebar(page_dir=page_dir, forbidden=forbidden,)
                ],
                xs=12, sm=6, md=4, lg=4, xl=4, xxl=4
            ),

            dbc.Col(
                [
 
        html.Br(),
        html.H1(general_nam,
                style={'textAlign': 'center', 'color': 'white', 'font-size': 40}),
        html.Hr(),
        html.Br(),
        html.Br(),

  
              
                ], xs=8, sm=8, md=10, lg=10, xl=10, xxl=10)
        
        ]
    )
])
           

 


def get_dnn(page_dir,page_container,forbidden=None,):
    return html.Div([
    dbc.Row(
        [
            # dbc.Col(
            #     [
            #         sidebar(page_dir=page_dir)
            #     ], xs=4, sm=4, md=2, lg=2, xl=2, xxl=2)
            dbc.Col(
                [
                    sidebar(page_dir=page_dir,forbidden=forbidden,)
                ],
                xs=4, sm=4, md=2, lg=2, xl=2, xxl=2,
                style={
                    "minWidth": "220px",
                    "maxWidth": "260px",
                    "whiteSpace": "normal",
                    "wordWrap": "break-word",
                    "flexShrink": 0
                }
            )
            ,

            dbc.Col(
                [
 
        html.Br(),
        html.H1(general_nam,
                style={'textAlign': 'center', 'color': 'white', 'font-size': 40}),
        html.Hr(),
        html.Br(),
        html.Br(),
        dcc.Markdown(
            page_container,
        )
                ], xs=8, sm=8, md=10, lg=10, xl=10, xxl=10)
        
        ]
    )
])

def get_text_dash_main_pred(
        page_name,
        page_dir_txt,
        dash_pages_path,
        disp_infos=False, 
        path_heads_show=None,
        categories=None,
        path_display=None,
        dnn_modes=None,
        name=None,
        **kward,
    ):

    code_content = f""" 
from dash import callback
import sys,os,dash
 
sys.path.append(os.path.abspath(os.getcwd()))

from mod.dsa.apps.help_app import DSAPage

dash.register_page(
    __name__,
    title="{page_name}",
    name="{name}",
    path="/{page_dir_txt}",
    order=1
)

path_file_dir={kward.get('path_file_dir')} 
path_dict={kward.get('path_dict')} 
tname='{kward.get('tname')}'
drop_name='{kward.get('drop_name',)}'
path_heads_show= {path_heads_show}
categories= {categories}
path_display= {path_display}  
dnn_modes= {dnn_modes}
# Instantiate page
dsa_page = DSAPage( 
    path_heads_show=path_heads_show,
    categories=categories,
    path_display=path_display,
    dnn_modes=dnn_modes,
    tname=tname,
    path_dict=path_dict,
)

layout = dsa_page.layout
 
type=None

out, inp, st, prevent = dsa_page.param_upload_dropdown_all(drop_name)
@callback(*out,*inp,*st,prevent_initial_call=prevent)
def update_nam_gen_dropdown(n): 
    return dsa_page.update_nam_gen_dropdown(drop_name,path_file_dir=path_file_dir,)


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


"""

    with open(dash_pages_path, "w") as file:
        file.write(code_content)

    if disp_infos:
        print(f"Python file saved as {dash_pages_path}")







def get_text_dash_main_train(
        page_name,
        page_dir_txt,
        dash_pages_path,
        disp_infos=False, 
        path_heads_show=None,
        categories=None,
        path_display=None,
        dnn_modes=None,
        name=None,
        **kward,
    ):

    code_content = f""" 
from dash import callback
import sys,os,dash
 
sys.path.append(os.path.abspath(os.getcwd()))

from mod.dsa.apps.help_app import DSAPage

dash.register_page(
    __name__,
    title="{page_name}",
    name="{name}",
    path="/{page_dir_txt}",
    order=2
)
path_file_dir={kward.get('path_file_dir')} 
path_dict={kward.get('path_dict')} 
tname='{kward.get('tname')}'
drop_name='{kward.get('drop_name',)}'
path_heads_show= {path_heads_show}
categories= {categories}
path_display= {path_display}  
dnn_modes= {dnn_modes}
# Instantiate page
dsa_page = DSAPage( 
    path_heads_show=path_heads_show,
    categories=categories,
    path_display=path_display,
    dnn_modes=dnn_modes,
    tname=tname,
    path_dict=path_dict,
)

layout = dsa_page.layout_train
   
type='train'

out, inp, st, prevent = dsa_page.param_upload_dropdown_all(drop_name)
@callback(*out,*inp,*st,prevent_initial_call=prevent)
def update_nam_gen_dropdown(n): 
    return dsa_page.update_nam_gen_dropdown(drop_name,path_file_dir=path_file_dir,)


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




"""

    with open(dash_pages_path, "w") as file:
        file.write(code_content)

    if disp_infos:
        print(f"Python file saved as {dash_pages_path}")






def get_text_dash_main_gen(
        page_name,
        page_dir_txt,
        dash_pages_path,
        disp_infos=False, 
        path_heads_show=None,
        categories=None,
        path_display=None,
        dnn_modes=None,
        name=None,
        **kward,
    ):

    code_content = f""" 
from dash import callback
import sys,os,dash
 
sys.path.append(os.path.abspath(os.getcwd()))

from mod.dsa.apps.help_app import DSAPage

dash.register_page(
    __name__,
    title="{page_name}",
    name="{name}",
    path="/{page_dir_txt}",
    order=0
)

path_file_dir={kward.get('path_file_dir')} 
path_dict={kward.get('path_dict')} 
drop_name='{kward.get('drop_name',)}'
path_heads_show= {path_heads_show}
categories= {categories}
path_display= {path_display}  
dnn_modes= {dnn_modes}
tname='{kward.get('tname')}'
# Instantiate page
dsa_page = DSAPage( 
    path_heads_show=path_heads_show,
    categories=categories,
    path_display=path_display,
    dnn_modes=dnn_modes,
    tname=tname,
    path_dict=path_dict,
)

layout = dsa_page.layout_gen
   
type='gen'

out, inp, st, prevent = dsa_page.param_upload_dropdown_all(drop_name)
@callback(*out,*inp,*st,prevent_initial_call=prevent)
def update_nam_gen_dropdown(n): 
    return dsa_page.update_nam_gen_dropdown(drop_name,path_file_dir=path_file_dir,)


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
def callback_upload_gen(*args):
    return dsa_page.upload(args,type)


    
out, inp, st, prevent = dsa_page.param_run_algorithm_all(type)
@callback(
    out,
    inp,
    st,
    prevent_initial_call=prevent
)
def callback_run_algorithm_gen(n_clicks, store_data):
    if not n_clicks or not store_data:
        raise dash.exceptions.PreventUpdate
    return dsa_page.run_algorithm_gen(store_data) 

    



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




"""

    with open(dash_pages_path, "w") as file:
        file.write(code_content)

    if disp_infos:
        print(f"Python file saved as {dash_pages_path}")






def get_text_dash_dnn(page_name,page_dir_txt,dash_pages_path,
                        disp_infos=False, 
                        path_train=None, 
                        path_file=None, 
                        pinn_dir_data=None,
                        neld_data=None,
                        index=None,
                        page_view=None,
                        forbidden_endswith=None,
                        ): 
    code_content = f"""

import os, sys ,dash 
sys.path.append(os.getcwd() ) 
from mod.dsa.apps.side_bar import sidebar ,get_dnn,dnn_page 


page_dir= '/{page_dir_txt}' 
page_name='{page_name}'  
page_name_view=dnn_page()['{page_view}']
forbidden_endswith={forbidden_endswith} 
forbidden_endswith = None if forbidden_endswith in (None, 'None') else forbidden_endswith


dash.register_page(__name__, 
                    title=page_name, 
                    name=page_name,
                    path=page_dir,
                    order=0) 

def layout():
    return get_dnn(page_dir,page_name_view,forbidden_endswith,)
""" 
    with open(dash_pages_path, "w") as file:
        file.write(code_content)
    if disp_infos:
        print(f"Python file saved as {dash_pages_path}")







from dash import html, dcc, Input, Output, State, callback , ctx, ALL 
def param_toggle_all(categories):

    prefixes = list(categories.values())

    outputs = [
        Output(f"collapse-{p}", "is_open")
        for p in prefixes
    ]

    inputs = (
        [
            Input(f"btn-{p}", "n_clicks")
            for p in prefixes
        ]
        +
        [
            Input(
                {"type": "nav", "prefix": ALL, "path": ALL},
                "n_clicks"
            )
        ]
    )

    states = [
        State(f"collapse-{p}", "is_open")
        for p in prefixes
    ]

    return outputs, inputs, states, True

def fun_toggle_all(args, categories):

    prefixes = list(categories.values())

    total = len(prefixes)

    states = args[-total:]

    triggered = ctx.triggered_id

    # A NavLink was clicked → close all collapses
    if isinstance(triggered, dict) and triggered.get("type") == "nav":
        return [False] * total 
    if isinstance(triggered, str) and triggered.startswith("btn-"): 
        clicked = triggered.replace("btn-", "")

        return [
            not state if prefix == clicked else False
            for prefix, state in zip(prefixes, states)
        ]

    return states

def header_navbar(groups=None,forbidden_page=None,disp_first_page_end=None):
    if groups is None:
        groups = {
            "DSA": "dsa",
            "DNN": "dnn", 
            "RNN": "kal", 
            # "3D CNN": "cnn",
            # "GCN": 'gcn',
            # "class ML":'cml',,'gen'
        } 
    disp_first_page_end=disp_first_page_end or ('data','test','train','dsa','2')
    # Group pages by prefix}{
    grouped = {prefix: [] for prefix in groups.values()}

    for page in dash.page_registry.values():
        if page["path"].startswith(forbidden_page):
            continue
        raw_path = page["path"].lower().rstrip("/")
        nlast = raw_path.split("/")[-1].split("-") 
        last,lasti=nlast[0],nlast[-1]
        print('[[[[[[[[[[[[[[[[[[=====]]]]]]]]]]]]]]]]]]',page['path'])
        name = page["name"] 
        for label, prefix in groups.items():
            if last.startswith(prefix) and lasti.endswith(disp_first_page_end):
                grouped[prefix].append(
                        dbc.NavLink(
                            name,
                            href=page["path"],
                            className="ms-2",
                            active="exact"
                        )
                    )

    horizontal_groups = []

    for label, prefix in groups.items():
        horizontal_groups.append(
            html.Div(
                [
                    # BUTTON stays fixed in place
                    dbc.Button(
                        label,
                        id=f"btn-{prefix}",
                        color="secondary",
                        className="px-2 py-1",
                        n_clicks=0,
                    ),
                    dbc.Collapse(
                        dbc.Card(
                            dbc.CardBody(
                                dbc.Nav(
                                    grouped[prefix],
                                    vertical=True,
                                    pills=True,
                                )
                            ),
                            className="mt-1"
                        ),
                        id=f"collapse-{prefix}",
                        is_open=False,
                    ),
                ], className="d-flex flex-column", #",#
                # style={{"margin-right": "1cm"}} ,
            )
        )

    return dbc.Navbar(
        dbc.Container(
                    dbc.Stack(
            horizontal_groups,
            gap=0, 
            direction="horizontal",
        ),
 
            fluid=True,
            className="d-flex flex-row align-items-start p-0 m-0",
            style={"margin-right": "0.1cm"} ,
        ),
        dark=True,
        color="dark",
        className="p-1"
    )


 

def get_text_dash_app(dash_pages_path,
                      disp_infos=False, 
                      head_navbar=None,
                      forbidden_page = "",
                      forbidden_endswith=None,
                        disp_first_page_end=None,  ):

    code_content = f"""
import os, sys, dash  
 
sys.path.append(os.path.abspath(os.getcwd()))
import dash_bootstrap_components as dbc
from dash import Dash, html,callback
import webbrowser, threading
from mod.dsa.apps.side_bar import header_navbar,param_toggle_all,fun_toggle_all

forbidden_page={forbidden_page}
forbidden_endswith='{forbidden_endswith}'
forbidden_endswith = None if forbidden_endswith in (None, 'None') else forbidden_endswith
head_navbar={head_navbar} 
disp_first_page_end={disp_first_page_end}
 
app = Dash(__name__, use_pages=True, external_stylesheets=[dbc.themes.DARKLY])
server = app.server 

header=header_navbar(head_navbar,forbidden_page,disp_first_page_end)


out, inp, st, prevent = param_toggle_all(head_navbar)

@callback(
    *out,
    *inp,
    *st,
    prevent_initial_call=prevent
)
def toggle_all(*args):
    return fun_toggle_all(args,head_navbar)

def open_browser():
    webbrowser.open('http://127.0.0.1:8050/langevin/dsa-langevin-data/dsa-2')

app.layout = dbc.Container([header, dash.page_container], fluid=False)

if __name__ == '__main__':
    threading.Timer(1.25, open_browser).start()
    app.run(debug=False)
"""

    with open(dash_pages_path, "w") as file:
        file.write(code_content)

    if disp_infos:
        print(f"Python file saved as {dash_pages_path}")





def dnn_page():
    page={} 




    page['starting']= """
    
 ## Getting Started with KALMAN

- Click on **Kalman** or/then **Generate/Prediction/Train** in the top-left corner for their execution.  

--- 

## Parameters & Options 

- **Clean Path Directory**  
  - Enable the **“clean_path_dir”** box to clear previous output directories before running a new prediction.  

- **Run Analysis**  
  - Double-click to run the prediction after all parameters are set up.
  
- **Restart**  
  - Click the **Restart** button to kill and restart the process.

- **Model Architecture**   
  - The first dropdown selects the **basename of the dataset**.
  - The second dropdown selects/creates a **training/testing** run.
  - The third dropdown selects the **model** starting with:   
    - **rnn_KAL** — *Recurrent Neural Network*   
    - **tfm_KAL** — *Transformer*
  - The fourth dropdown selects the **feature class**.   
  - The fifth dropdown selects a **model loss**.
  
- **Lorenz Param**
  - Modify the Lorenz dataset parameters.

---
 

    """















    page['instructions']= """
    

---

# Instructions 
---

## Parameters & Options

- **Model Architecture**  
  Use the dropdowns to select the model.    

- **Clean Path Directory**  
  - Enable the **“clean_path_dir”** box to clear previous output directories before running new prediction.  

- **Run Analysis**  
  -Double Click to run after all parameters are set up.
  
- **Restart**  
  -Click the **Restart** button to kill and restart the process.
 
 
    
    
    """



    page['restart']= """
    
    
 

Once prediction has terminated, you can check the results by following these steps:  

---

## Restart the Application  

1. Kill the current running code in the terminal using **`Ctrl + C`**.  
2. Reopen the application by running:  

```bash
python -m  gunicorn -w 4 -b 0.0.0.0:8050 wsgi:server -c gunicorn.conf.py
```
3. Alternatively, use the **Restart button** in the interface and then refresh your browser page.
 
    """


    page['results']= """
     

---
 
## Navigating the Interface

- On the right side of **Kalman** (top corner), click on one of the following to visualize the results:
    - **PINN** — *Physics-Informed Neural Network*   
    - **cML** — *Classic Machine Learning*  
- Select the **architecture (DNN-nth)** that was used.  
- Click to choose the **file parent name**.  
- Click on the **sampe name**.  
- A new page will appear with visualization options.

 
    """

    page['visualization']= """


## Visualization Options

Use the dropdown buttons to explore different features:

- **Samples Selection**  
  - Switch between multiple samples if more than one was dropped.  

- **Method Selection**  
  - Choose the method used (e.g., **PINN**, **cML**) if available.  
 
- **Architecture Selection**  
  - Select the **DNN-nth** architecture.  
 
- **SHAP Coefficients**  
  - **model_shap** → visualize SHAP coefficients using [Shapley values](https://en.wikipedia.org/wiki/Shapley_value).
 
---
 
 
    
    """

    return page


 
def get_text_dash_test(user_input,file_path_org,neld_path_inits,  data_studied, 
                       model_sufix,dash_pages_name,
                       disp_infos=False,
                       path_train=None, 
                    path_file=None,
                    path_file_dir=None,
                    path_file_sub=None, 
                    pinn_dir_data=None,
                    neld_data=None,
                    index=None,
                    model_type=None,
                    obj_org_path_dict=None,
                    model_sufix_dic=None,
                    path_display=None,
                    path_display_dic=None,
                    **kward):
    # Generate the Python script content
    code_content = f""" 

import os, sys ,dash  
page_title='{kward.get('page_title',)}'
page_name='{kward.get('page_name')}'
type='{kward.get('type','page_name')}'
page_dir_txt='/{kward.get('page_dir_txt')}'
p_name='{kward.get('p_name')}'
tname='{kward.get('tname')}'
lst={kward.get('lst',)}
params={kward.get('params',)}
paraws={kward.get('paraws',)}
drop_name='{kward.get('drop_name',)}'
file_path_org={file_path_org} 
from  mod.dsa.apps.app_param_test import app_param
from dash import callback  
 
neld_path_inits =['{neld_path_inits}']
data_studied = '{data_studied}'  
model_sufix = '{model_sufix}' 
path_train= {path_train}
path_file= {path_file}
path_file_sub={path_file_sub}
path_file_dir={path_file_dir}
pinn_dir_data='{pinn_dir_data}'
neld_data={neld_data}
index={index}
model_type='{model_type}'
obj_org_path_dict={obj_org_path_dict}
model_sufix_dic={model_sufix_dic}
path_display={path_display}
path_display_dic={path_display_dic}
mapp = app_param(
    file_path_org=file_path_org,
    model_sufix=model_sufix,
    path_train=path_train, 
    path_file=path_file,
    path_file_sub=path_file_sub,
    path_file_dir=path_file_dir,
    pinn_dir_data=pinn_dir_data,
    index=index, 
    data_studied=data_studied,  
    neld_data=neld_data,
    model_type=model_type,
    obj_org_path_dict=obj_org_path_dict,
    model_sufix_dic=model_sufix_dic,
    path_display=path_display,
    path_display_dic=path_display_dic,
    tname=tname,
    params=params,
    paraws=paraws,
)
 
dash.register_page(
    __name__,
    title=page_title,
    name=page_name,
    path=page_dir_txt,
    order={user_input}
)

def layout():
    return mapp.app_layout

@callback(
    mapp.Output,
    mapp.Input, 
    prevent_initial_call=mapp.prevent_initial_call
)
def update_output(*args):  
    mapp.update_params(args)
    return mapp.Get_output()
 

out, inp, st, prevent = mapp.param_upload_dropdown_all(drop_name)
@callback(*out,*inp,*st,prevent_initial_call=prevent)
def update_nam_gen_dropdown(n): 
    return mapp.update_nam_gen_dropdown(drop_name,path_file_dir=path_file_dir,)


out, inp, st, prevent = mapp.param_create_parameter_dropdowns_all(drop_name)
@callback(*out,*inp,*st,prevent_initial_call=prevent)
def create_parameter_dropdowns(store_data):  
    return [] if store_data is None else mapp.create_parameter_dropdowns(store_data,drop_name)



""" 
    



    with open(dash_pages_name, "w") as file:
        file.write(code_content)
    if disp_infos:
        print(f"Python file saved as {dash_pages_name}")

'''
@callback(
    mapp.Output,
    mapp.Input, 
    prevent_initial_call=mapp.prevent_initial_call
)
def update_output( path_head,model_suf,path,mode,neldd, clusts,  intensity_type, width, height,  templ,nbin,index): 
    figure,txt = mapp.Get_output( path_head,model_suf,path,mode,neldd, clusts,   intensity_type, width, height, templ,nbin,index)
    return figure ,txt






'''
 


def get_text_dash_all(page_name,page_dir_txt,dash_pages_path,
                    disp_infos=False,
                    path_train=None, 
                    path_file=None, 
                    pinn_dir_data=None,
                    neld_data=None,
                    index=None,
                    ): 
    code_content = f"""

import os, sys ,dash 
sys.path.append(os.getcwd() ) 
from  mod.dsa.neld_fun_0.side_bar import layout_1


page_dir= '/{page_dir_txt}' 
page_name='{page_name}'
dash.register_page(__name__, title=page_name, name=page_name,order=0) 

def layout():
    return layout_1(page_dir)
""" 
    with open(dash_pages_path, "w") as file:
        file.write(code_content)
    if disp_infos:
        print(f"Python file saved as {dash_pages_path}")



 
