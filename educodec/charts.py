"""Readable experiment charts with explicit units and finite-value handling."""
import altair as alt
import numpy as np
import pandas as pd
import streamlit as st


def comparison_chart(data,key,curve=False):
    metric=st.selectbox('Vertical axis',['psnr','ssim','mse'],format_func=lambda x:{'psnr':'PSNR (dB) · higher is better','ssim':'Luminance SSIM · higher is better','mse':'RGB MSE (sample levels²) · lower is better'}[x],key=f'{key}_metric')
    table=data.copy()
    if 'label' not in table: table['label']=['Q = '+str(q) for q in table['q']]
    table=table[np.isfinite(pd.to_numeric(table[metric],errors='coerce')) & np.isfinite(pd.to_numeric(table['bpp'],errors='coerce'))]
    if len(table)!=len(data): st.caption('Non-finite values (including infinite PSNR for exact reconstruction) are omitted from the plot.')
    if table.empty: st.info('No finite measurements to plot.'); return
    ylabel={'psnr':'PSNR (dB)','ssim':'Luminance SSIM','mse':'RGB MSE (sample levels²)'}[metric]
    tooltip=[alt.Tooltip('label:N',title='Run'),alt.Tooltip('q:Q',title='Q scale',format='.3f'),alt.Tooltip('bpp:Q',title='Bits / pixel',format='.4f'),alt.Tooltip(f'{metric}:Q',title=ylabel,format='.4f')]
    base=alt.Chart(table).encode(x=alt.X('bpp:Q',title='Encoded rate (bits / pixel)',scale=alt.Scale(zero=False)),y=alt.Y(f'{metric}:Q',title=ylabel,scale=alt.Scale(zero=False)),tooltip=tooltip)
    if curve:
        chart=base.mark_line(point=True,color='#36dfc2').encode(order='bpp:Q')
    else:
        chart=base.mark_circle(size=140,opacity=.9).encode(color=alt.Color('label:N',title='Experiment'))
    chart=chart.properties(height=380).interactive().configure_axis(labelFontSize=13,titleFontSize=14,gridOpacity=.18).configure_legend(labelFontSize=12,titleFontSize=13,orient='bottom')
    st.altair_chart(chart,width='stretch')
    st.caption('Hover for exact measurements; drag to pan and scroll to zoom. Axes use the observed range rather than starting at zero.')
