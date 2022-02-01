import pandas as pd
import numpy as np 
import seaborn as sns 
sns.set(style='ticks', palette='Set2')
sns.set_context("talk", font_scale=1.2)
import matplotlib.pyplot as plt
import random
import warnings
warnings.filterwarnings('ignore')
import matplotlib
from ipywidgets import interact, interactive, fixed, interact_manual, widgets, FloatSlider, Output, AppLayout, HBox, VBox, Layout
from ipywidgets import interactive_output
from IPython.display import display, HTML, clear_output
import os
import ipyvuetify as v
from traitlets import (Unicode, List)
from threading import Timer
import plotly.graph_objects as go
import plotly.express as px
from matplotlib import font_manager

def initial_config():
    font_path = "C:\\Users\\AniltonCardoso\\OneDrive - BITKA\\Projetos\\CustomFonts"
    font_dirs = [font_path]
    font_files = font_manager.findSystemFonts(fontpaths=font_dirs)

    for font_file in font_files:
        font_manager.fontManager.addfont(font_file)

    csfont = {'fontname':'Roboto Condensed'}
    hfont = {'fontname':'Roboto Condensed'}
    matplotlib.rc('font',family='Roboto Condensed')
    plt.rcParams.update({
        "text.usetex": False})


def default_plot_layout(ax, title, xlabel, ylabel):
    ax.set_title(title, size = 16)
    ax.set_ylabel(ylabel, size = 14)
    ax.set_xlabel(xlabel, size = 14)
    plt.xticks(size = 12)
    plt.yticks(size = 12)
    ax.spines["top"].set_visible(False)  
    ax.spines["right"].set_visible(False)  
    
def fill_percentage(ax, total):
    for p in ax.patches:
        height = p.get_height()
        ax.text(p.get_x() + p.get_width()/2.,
                height + 3,
                '{:1.2f}%'.format(height/total * 100),
                ha="center", fontsize = 12) 

def background_with_norm(s):
    cmap = matplotlib.cm.get_cmap('RdBu') 
    norm = matplotlib.colors.TwoSlopeNorm(vmin=s.values.min(), vcenter=0, vmax=s.values.max())
    return ['background-color: {:s}'.format(matplotlib.colors.to_hex(c.flatten())) for c in cmap(norm(s.values))]

def format_cols(perc_cols, numerical_cols, money_cols, date_cols):
    perc_dict = {x: '{:.2%}' for x in perc_cols}
    num_dict =  {x: '{0:,.2f}' for x in numerical_cols}
    mny_dict =  {x: 'R${0:,.2f}' for x in money_cols}
    dt_dict =  {x: '{:%d-%m-%Y}' for x in date_cols}

    format_dict = dict(**perc_dict, **num_dict)
    format_dict = dict(**format_dict, **mny_dict)
    format_dict = dict(**format_dict, **dt_dict)
    return format_dict

def set_css_properties():
    # Set CSS properties for th elements in dataframe
    th_props = [('font-size', '16px'),
                ('font-name', 'Calibri'),
                ('text-align', 'center'),
                ("border-top", "1px solid #C1C3D1;"),
                ("border-bottom-", "1px solid #C1C3D1;"),
                ("text-shadow", "0 1px 1px rgba(256, 256, 256, 0.1);"),
                ('font-weight', 'normal'),
                ('color', '#D5DDE5'),
                ('background-color', '#1b1e24')]

                
    table_props = [("background-color", "#f0ebeb;"), 
                  ('width', '100%')]

    tr_hover_props = [("background-color", '#b5e3ff'), 
            ('color', 'white'), 
            ('cursor', 'pointer'),
            ('font-weight', "bold"),
        #  ("border-top", "1px solid #22262e;")
            ]

    td_props = [('font-size', '14px'),
                # ("background", "#FFFFFF;"),
                ("text-align", "left;"),
                ('font-name', 'Calibri'),
                ("vertical-align", "middle;"),
                ("font-weight", "500;"),
                ('color', 'black'),
                # ("text-shadow", "-1px -1px 1px rgba(0, 0, 0, 0.1);"),
                ("border-right", "1px solid #C1C3D1;")]

    caption_props = [
                    ('font-size', '20px'),
                    ('font-weight', 'bold')]

    # Set table styles
    styles = [dict(selector="th", props=th_props),
                dict(selector="td", props=td_props),
                dict(selector="caption", props=caption_props),
                dict(selector="tr:hover", props=tr_hover_props),
                dict(selector="tr", props = table_props),
                ]

    return styles

def render_html():
    HTML("""\
    <style>
    .app-subtitle {
        font-size: 1.5em;
    }

    .app-subtitle a {
        color: #106ba3;
    }

    .app-subtitle a:hover {
        text-decoration: underline;
    }

    .app-sidebar p {
        margin-bottom: 1em;
        line-height: 1.7;
    }

    .app-sidebar a {
        color: #106ba3;
    }

    .app-sidebar a:hover {
        text-decoration: underline;
    }

    th {
        font-size: 16px;
        font-name: Calibri;
        text-align: left;
        border-top: 1px solid #C1C3D1;
        border-bottom: 1px solid #C1C3D1;
        text-shadow: 0 1px 1px rgba256: 256: 256: 0.1;
        font-weight: normal;
        color: #D5DDE5;
        background-color: #1b1e24;
    }

    tr {
        background-color: #f0ebeb; 
    }

    tr:hover {
        background-color: #b5e3ff;
        color: white;
        cursor: pointer;
        font-weight: bold;
    }

    td {
        font-size: 14px;
        text-align: left;
        font-name: Calibri;
        vertical-align: middle;
        font-weight: 500;
        color: black;
        border-right: 1px solid #C1C3D1;
    }

    caption {
        font-size: 20px;
        font-weight: bold;
    }

    .output_png {
        display: table-cell;
        text-align: center;
        vertical-align: middle;
    }

    </style>
    """)
