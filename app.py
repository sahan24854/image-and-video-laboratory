"""Launch with: streamlit run app.py"""
import io
import time
from dataclasses import asdict, replace
import numpy as np
import pandas as pd
from PIL import Image
from scipy.fft import dctn
import streamlit as st
from educodec import Config, encode_image, decode_image, encode_video, decode_video, target_size, quality
from educodec.codec import _table, _unpack, _predict
from educodec.lab import panel, specimen
from educodec.charts import comparison_chart
from educodec.media import read_video
from educodec.learning import lessons, simulate, motion_simulation

@st.cache_data(max_entries=3)
def load_video(data, max_frames, max_width):
    return read_video(data, max_frames, max_width)

st.set_page_config(page_title='EduCodec · Coding laboratory',page_icon='◩',layout='wide')
st.title(':material/science: Image & video coding laboratory')
st.caption('EDUCODEC / SIGNAL ANALYSIS WORKSPACE / LOCAL PROCESSING')
st.caption('Explore prediction, transforms, quantization and the rate–distortion tradeoff using a working codec.')

@st.cache_data
def demo():
    y,x=np.indices((96,128))
    image=np.stack([(x*2)%256,(y*3)%256,((x+y)*2)%256],-1).astype('uint8')
    frames=[]
    for t in range(10):
        f=image.copy(); f[24:48,8+t*6:32+t*6]=[240,35,70]
        if t>=8: f=255-f
        frames.append(f)
    return image,np.stack(frames)

with st.sidebar:
    st.header(':material/tune: Instrument controls')
    st.caption('01 SOURCE · 02 CODE · 03 INSPECT')
    kind=st.radio('Media',['Image','Video'])
    upload=st.file_uploader('Upload an image or video',type=['png','jpg','jpeg','webp'] if kind=='Image' else ['mp4','mov','mkv','avi','webm','m4v','gif','npy'])
    sample=st.selectbox('Built-in image specimen',['Original gradient','Texture','Edges','Noise']) if kind=='Image' else 'Original gradient'
    st.caption('A synthetic example is available without uploading. MP4, MOV, MKV, AVI, WebM, GIF and NumPy videos are supported (depending on the encoded video codec).')
    clip_frames=st.slider('Maximum clip length (frames)',2,60,12) if kind=='Video' else 12
    clip_width=st.select_slider('Maximum analysis width (pixels)',[96,128,192,256,384],value=192) if kind=='Video' else 192
    q=st.slider('Quantization scale (dimensionless multiplier)',.05,20.,1.,.05,help='Larger values discard more detail. Size is not guaranteed to decrease at every step.')
    quantizer=st.selectbox('Quantization table',['jpeg','weighted','uniform'])
    block=st.selectbox('Transform block size (pixels)',[8] if quantizer=='jpeg' else [4,8,16],index=0 if quantizer=='jpeg' else 1,format_func=lambda n:f'{n} × {n} pixels',disabled=quantizer=='jpeg',key=f'block_size_{quantizer}')
    if quantizer=='jpeg':
        st.caption('JPEG quantization tables are fixed at 8 × 8. Select weighted or uniform to choose 4 × 4, 8 × 8, or 16 × 16 blocks.')
    transform=st.selectbox('Transform',['dct','identity'])
    entropy=st.selectbox('Lossless coder',['zlib','bz2','lzma'])
    chroma=st.selectbox('Chroma sampling',['444','422','420']) if kind=='Image' else '444'
    padding=st.selectbox('Boundary padding',['edge','constant'])
    cost=st.selectbox('Motion matching cost',['sad','mse']) if kind=='Video' else 'sad'
    prediction=st.selectbox('Video predictor',['hybrid','motion','previous']) if kind=='Video' else 'hybrid'
    search=st.slider('Motion search radius (± pixels)',0,12,4) if kind=='Video' else 4
    step=st.select_slider('Search step (pixels)',[1,2,4],value=2) if kind=='Video' else 2
    key=st.slider('I-frame interval (frames)',1,20,8) if kind=='Video' else 12
    fps=st.number_input('Synthetic / NumPy frame rate (frames/s)',1.,120.,24.) if kind=='Video' else 24.
    target=st.checkbox('Control total encoded size')
    budget=st.number_input('Target bytes',min_value=1,value=5000) if target else None
    mode=st.selectbox('Size objective',['closest','maximum']) if target else 'closest'
    run=st.button('Run experiment',type='primary',width="stretch")

image,frames=demo()
with st.expander('Validation bench · guided tests',expanded=False):
    panel()
try:
    if kind=='Image' and sample!='Original gradient': image=specimen(sample)
    if upload:
        if kind=='Image': image=np.asarray(Image.open(upload).convert('RGB'))
        elif upload.name.endswith('.npy'): frames=np.load(upload,allow_pickle=False)
        else:
            frames,fps=load_video(upload.getvalue(),clip_frames,clip_width)
            st.caption(f'Decoded {len(frames)} frames at {fps:.3f} FPS · analysis resolution {frames.shape[2]}×{frames.shape[1]}. The initial clip is resized only if wider than the selected limit.')
    source=image if kind=='Image' else frames
    if source.dtype!=np.uint8 or source.ndim!=(3 if kind=='Image' else 4) or source.shape[-1]!=3:
        st.error('Input must be a uint8 RGB image or video array.'); st.stop()
    if source.size>128*128*3*30 and kind=='Video':
        st.info('For interactive motion search, use a short clip with small frames (for example 128×96, 10 frames).')
    preview,explanation=st.columns([1,1])
    with preview:
        st.image(source if kind=='Image' else source[0],caption=f'Input · {source.shape}',width='stretch')
    with explanation:
        st.subheader('Follow the signal')
        st.markdown('**Pixels → prediction / level shift → transform → quantization → entropy coding → bitstream → inverse operations → reconstruction**')
        lessons()
    with st.expander('Interactive mechanism simulation',expanded=True):
        simulation_frame=int(st.number_input('Simulation frame',0,len(source)-1,0)) if kind=='Video' else 0
        simulate(source if kind=='Image' else source[simulation_frame],Config(q=q,block=block,quantizer=quantizer,transform=transform,padding=padding))
    if kind=='Video' and len(source)>1:
        with st.expander('Temporal search and mode decision simulation'):
            pair=int(st.number_input('Current frame index (frames)',1,len(source)-1,1))
            motion_simulation(source[pair-1],source[pair],Config(search=search,search_step=step,cost=cost))
    c=Config(q=q,block=block,quantizer=quantizer,transform=transform,entropy=entropy,chroma=chroma,prediction=prediction,search=search,search_step=step,key_interval=key,padding=padding,cost=cost)
    if run:
        with st.status('Signal pipeline running…',expanded=True) as pipeline:
            st.write(':material/input: Sampling input and configuring the encoder')
            encode=encode_image if kind=='Image' else lambda a,c:encode_video(a,c,fps)
            start=time.perf_counter()
            if target: stream,diagnostic=target_size(encode,source,budget,c,iterations=12,mode=mode); used=replace(c,q=diagnostic['q'])
            else: stream=encode(source,c); diagnostic={}; used=c
            seconds=time.perf_counter()-start
            st.write(':material/memory: Bitstream generated · reconstructing from encoded data')
            rec=decode_image(stream) if kind=='Image' else decode_video(stream)[0]
            scores=[quality(source,rec)] if kind=='Image' else [quality(a,b) for a,b in zip(source,rec)]
            pipeline.update(label='Experiment complete · reconstruction verified',state='complete',expanded=False)
            st.session_state.result=dict(source=source,rec=rec,stream=stream,scores=scores,c=used,kind=kind,diagnostic=diagnostic,seconds=seconds)
    result=st.session_state.get('result')
    if result:
        st.divider(); st.subheader('Last completed experiment')
        st.caption('Results retain the settings used when Run experiment was pressed. Run again after changing controls or input.')
        a,b,stream,scores,used=result['source'],result['rec'],result['stream'],result['scores'],result['c']
        cols=st.columns(4)
        cols[0].metric('Encoded size',f'{len(stream):,} bytes')
        cols[1].metric('Bits / pixel',f'{8*len(stream)/(a.size/3):.3f}')
        cols[2].metric('Mean PSNR',f'{np.mean([s["psnr_db"] for s in scores]):.2f} dB')
        cols[3].metric('Mean luminance SSIM',f'{np.mean([s["ssim_y"] for s in scores]):.4f}')
        st.caption(f'Encode/search time: {result["seconds"]:.3f} s · Raw/encoded ratio: {a.nbytes/len(stream):.2f} · Complete stream size includes metadata and video side information.')
        view,inside,compare,theory=st.tabs(['Input → output','Inside the codec','Compare experiments','Theory guide'])
        with view:
            index=st.slider('Inspect frame (frame index)',0,len(a)-1,0,key='error_frame') if result['kind']=='Video' and len(a)>1 else 0
            if result['kind']=='Video': st.caption(f'Frame {index} of {len(a)-1} · original, reconstruction and error shown below')
            orig=a[index] if result['kind']=='Video' else a; decoded=b[index] if result['kind']=='Video' else b
            columns=st.columns(3)
            for col,pixels,label in zip(columns,[orig,decoded,np.clip(np.abs(orig.astype(float)-decoded)*4,0,255).astype('uint8')],['Original','Reconstructed','Absolute error ×4']): col.image(pixels,caption=label,width="stretch")
            st.dataframe(pd.DataFrame(scores).rename_axis('frame'),width="stretch")
            st.line_chart(pd.DataFrame(scores)[['psnr_db','ssim_y']]) if result['kind']=='Video' else None
            st.download_button('Download encoded stream',stream,file_name='experiment.edc')
            if result['kind']=='Image':
                buffer=io.BytesIO(); Image.fromarray(b).save(buffer,format='PNG'); st.download_button('Download reconstruction',buffer.getvalue(),file_name='reconstructed.png')
            else:
                buffer=io.BytesIO(); np.save(buffer,b); st.download_button('Download reconstructed frames',buffer.getvalue(),file_name='reconstructed.npy')
        with inside:
            st.json(asdict(used))
            if result['diagnostic']:
                st.json(result['diagnostic'])
                st.line_chart(pd.DataFrame(result['diagnostic']['trace'])['bytes'])
            st.write('Quantization steps (luminance / RGB residual planes)'); st.dataframe(_table(used))
            if result['kind']=='Image':
                y=st.slider('Block row',0,(orig.shape[0]-1)//used.block,0); x=st.slider('Block column',0,(orig.shape[1]-1)//used.block,0)
                plane=orig.astype(float)@np.array([.299,.587,.114])-128
                plane=np.pad(plane,((0,(-plane.shape[0])%used.block),(0,(-plane.shape[1])%used.block)),mode='edge')
                patch=plane[y*used.block:(y+1)*used.block,x*used.block:(x+1)*used.block]
                coeff=dctn(patch,norm='ortho') if used.transform=='dct' else patch
                st.write('Selected luminance block: transform coefficients'); st.dataframe(coeff)
                quantized=np.rint(coeff/_table(used)); st.write('Quantized coefficients'); st.dataframe(quantized)
                st.metric('Zero coefficients',f'{100*np.mean(quantized==0):.1f}%')
            else:
                _,meta,arrays=_unpack(stream,'video'); cursor=0; details=[]; predictions=[]
                for i,t in enumerate(meta['types']):
                    if t=='I': pred=np.full(b[i].shape,128.); intra=0
                    else:
                        mv=arrays[cursor]; cursor+=1; pred=_predict(b[i-1],mv,used.macroblock); intra=int(mv[:,2].sum())
                    predictions.append(pred); details.append(dict(frame=i,type=t,dc_blocks=intra)); cursor+=3
                st.dataframe(details)
                pred=predictions[index]; residual=a[index].astype(float)-pred
                st.image(np.clip(pred,0,255).astype('uint8'),caption='Decoder-reproducible predictor',width=480)
                st.image(np.clip(residual+128,0,255).astype('uint8'),caption='Signed residual shown around gray 128 (display clipped)',width=480)
        with compare:
            if st.button('Simulate rate–distortion sweep (five Q scales)'):
                rows=[]
                with st.spinner('Measuring reconstruction at each scale…'):
                    for scale in (.25,.5,1.,4.,12.):
                        cfg=replace(used,q=scale)
                        data=encode_image(a,cfg) if result['kind']=='Image' else encode_video(a,cfg)
                        output=decode_image(data) if result['kind']=='Image' else decode_video(data)[0]
                        measured=[quality(a,output)] if result['kind']=='Image' else [quality(x,y) for x,y in zip(a,output)]
                        rows.append(dict(q=scale,bpp=8*len(data)/(a.size/3),psnr=np.mean([s['psnr_db'] for s in measured]),mse=np.mean([s['mse'] for s in measured])))
                st.session_state.sweep=rows
            if st.session_state.get('sweep'):
                sweep=pd.DataFrame(st.session_state.sweep); st.dataframe(sweep); comparison_chart(sweep,'sweep',curve=True)
            if st.button('Add completed experiment to comparison'):
                st.session_state.setdefault('history',[]).append(dict(label=f'Run {len(st.session_state.get("history",[]))+1}',include=True,notes='',media=result['kind'],shape=str(a.shape),q=used.q,method=used.quantizer,prediction=used.prediction,bytes=len(stream),bpp=8*len(stream)/(a.size/3),psnr=float(np.mean([s['psnr_db'] for s in scores])),ssim=float(np.mean([s['ssim_y'] for s in scores])),settings=str(asdict(used))))
            history=pd.DataFrame(st.session_state.get('history',[]))
            st.caption('Compare runs using the same input. Different content changes rate–distortion behavior.')
            if not history.empty:
                st.caption('Edit run labels and notes, or uncheck Include to hide a run from the chart. Measured values stay fixed; rerun the codec to change them.')
                with st.form('comparison_edits'):
                    edited=st.data_editor(history,width='stretch',hide_index=True,disabled=[col for col in history if col not in ('label','notes','include')],column_config={'label':st.column_config.TextColumn('Run label',required=True),'include':st.column_config.CheckboxColumn('Include'),'notes':st.column_config.TextColumn('Notes'),'bpp':st.column_config.NumberColumn('Rate (bits/pixel)',format='%.4f'),'psnr':st.column_config.NumberColumn('PSNR (dB)',format='%.3f'),'ssim':st.column_config.NumberColumn('SSIM',format='%.5f'),'bytes':st.column_config.NumberColumn('Encoded bytes'),'q':st.column_config.NumberColumn('Q scale',format='%.3f')},key='comparison_editor')
                    if st.form_submit_button('Apply table edits'):
                        st.session_state.history=edited.to_dict('records')
                history=pd.DataFrame(st.session_state.history)
                selected=history[history['include'].fillna(False)]
                if not selected.empty: comparison_chart(selected,'comparison')
                else: st.info('Include at least one run to display the chart.')
                st.download_button('Download comparison CSV',history.to_csv(index=False),file_name='comparison.csv')
        with theory:
            st.markdown('''**Spatial coding:** color conversion separates luminance and chroma. The DCT concentrates smooth-block energy into low frequencies. Quantization rounds coefficients to fewer values; lossless coding exploits their redundancy.

**Temporal coding:** motion compensation predicts blocks from a reconstructed reference. Hybrid mode compares luminance SAD against a DC prediction. Only signed prediction errors and side information need coding. I-frames reset prediction dependencies.

**Try these experiments:** increase quantization and watch high-frequency coefficients become zero; compare identity with DCT on smooth and noisy images; compare 4:4:4 with 4:2:0 on colored text; compare all-intra with motion on a moving scene; increase search radius and inspect encoding time; introduce a scene cut and inspect DC selections.

**Interpret carefully:** lower MSE means higher PSNR, but perceptual quality also depends on structure and color. SSIM here uses Gaussian-window luminance statistics. Video PSNR is averaged in dB. Target-size search is heuristic and may miss a budget; inspect its diagnostics. These are educational custom streams rather than standard JPEG or H.264 files.''')
except ImportError:
    st.error("Video decoding requires PyAV. Install the UI dependencies: python -m pip install -e '.[ui]'")
except (ValueError,OSError,KeyError) as exc:
    st.error(str(exc))
