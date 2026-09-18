
import os, sys, dash  
 
sys.path.append(os.path.abspath(os.getcwd()))
import dash_bootstrap_components as dbc
from dash import Dash, html,callback
import webbrowser, threading
from mod.dsa.apps.side_bar import header_navbar,param_toggle_all,fun_toggle_all

forbidden_page=('/dsa/dsa-ginn-data/',)
forbidden_endswith='None'
forbidden_endswith = None if forbidden_endswith in (None, 'None') else forbidden_endswith
head_navbar={'Kalman': 'kalman', 'DSA': 'dsa'} 
disp_first_page_end=('data',)
 
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
