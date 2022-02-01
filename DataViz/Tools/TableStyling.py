import pandas as pd
import numpy as np 
import seaborn as sns 
import matplotlib.pyplot as plt 
from matplotlib import colors
import matplotlib   
import sparklines

### METHODS
def sparkline_str(x):
    """Works with agg method after a groupby"""

    bins=np.histogram(x)[0]
    sl = ''.join(sparklines(bins))
    return sl

sparkline_str.__name__ = "sparkline"

class Table:
    def __init__(self, df, params, title = ""):
        """
        params: Column Name => Transformation 
            Possible Transformations: Money, Numerical, Percent, Date, Ticker, Sparklines, Bar, DoubleBar, Direction
        """
        self.title = title
        self.df = df
        self.style = df.style
        self.params = params 
        self.format()

    def format(self):
        tickers_col = [x for x,t in self.params if t.lower() == "ticker"]
        ticker = None
        if len(tickers_col) > 0:
            ticker = tickers_col[0]

        self.style = self.df.style.pipe(self.hide_index)\
                .pipe(self.set_css_properties)\
                .pipe(self.make_tickers_hyperlinks, ticker_col = ticker)\
                .pipe(self.gradient_backgroud, cols = [x for x,t in self.params if t.lower() == "gradient_color"])\
                .pipe(self.double_bar, cols = [x for x,t in self.params if t.lower() == "double_bar"])\
                .pipe(self.bar, cols = [x for x,t in self.params if t.lower() == "bar"])\
                .pipe(self.format_cols, perc_cols =       [x for x,t in self.params if t.lower() == "percent"],
                                        numerical_cols =  [x for x,t in self.params if t.lower() == "numerical"],
                                        money_cols =      [x for x,t in self.params if t.lower() == "money"],
                                        date_cols  =      [x for x,t in self.params if t.lower() == "date"])\
                .set_caption(self.title)


    def make_tickers_hyperlinks(self, style, ticker_col = None):
        if ticker_col is None:
            return style
        return style.format({'TICKER': self.__make_clickable_both})

    def __make_clickable_both(self, val): 
        url = f"https://finance.yahoo.com/quote/{val}?p={val}"
        return f'<a href="{url}">{val}</a>'

    def hide_index(self, style):
        return style.hide_index()

    def format_cols(self, style, perc_cols, numerical_cols, money_cols, date_cols):
        perc_dict = {x: '{:.2%}' for x in perc_cols}
        num_dict =  {x: '{0:,.2f}' for x in numerical_cols}
        mny_dict =  {x: 'R${0:,.2f}' for x in money_cols}
        dt_dict =  {x: '{:%d-%m-%Y}' for x in date_cols}

        format_dict = dict(**perc_dict, **num_dict)
        format_dict = dict(**format_dict, **mny_dict)
        format_dict = dict(**format_dict, **dt_dict)
        return style.format(format_dict)

    def double_bar(self, style, cols):
        if cols is None or len(cols) == 0:
            return style
        return style.bar(color=['#FFA07A','skyblue'], vmin=0, subset = cols, align='zero')

    def bar(self, style, cols):
        if cols is None or len(cols) == 0:
            return style
        return style.bar(color= 'lightblue', subset = cols)

    def set_css_properties(self, style, css = None):
        if css is None:
            # Set CSS properties for th elements in dataframe
            th_props = [('font-size', '16px'),
                        ('text-align', 'center'),
                        ("border-top", "1px solid #C1C3D1;"),
                        ("border-bottom-", "1px solid #C1C3D1;"),
                        ("text-shadow", "0 1px 1px rgba(256, 256, 256, 0.1);"),
                        ('font-weight', 'normal'),
                        ('color', '#D5DDE5'),
                        ('background-color', '#1b1e24')]

                        
            table_props = [("background-color", "#f0ebeb;")]

            tr_hover_props = [("background-color", '#b5e3ff'), 
                 ('color', 'white'), 
                 ('cursor', 'pointer'),
                 ('font-weight', "bold"),
                #  ("border-top", "1px solid #22262e;")
                 ]

            td_props = [('font-size', '14px'),
                        # ("background", "#FFFFFF;"),
                        ("text-align", "left;"),
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

            return style.set_table_styles(styles)
        else:
            return style.set_table_styles(css)

    def gradient_backgroud(self, style, cols):
        if cols is None or len(cols) == 0:
            return style
        return style.apply(self.__background_with_norm, subset = cols)

    def __background_with_norm(self, s):
        cmap = matplotlib.cm.get_cmap('RdBu') 
        norm = matplotlib.colors.DivergingNorm(vmin=s.values.min(), vcenter=0, vmax=s.values.max())
        return ['background-color: {:s}'.format(matplotlib.colors.to_hex(c.flatten())) for c in cmap(norm(s.values))]

    def direction(self, cols):
        self.df = self.df['direction'].map({0: '↓', 1: '↑'})

# https://pbpython.com/styling-pandas.html
# https://medium.com/@kristina.reut96/you-think-you-know-stylish-tables-ee59beadb487
# https://pandas.pydata.org/pandas-docs/stable/user_guide/style.html
# https://codepen.io/alassetter/pen/cyrfB
# https://codepen.io/websanity/pen/fxCwg
# https://codepen.io/njessen/pen/naLCv
# https://catalin.red/practical-css3-tables-with-rounded-corners/
# https://jasonet.co/posts/scheduled-actions/