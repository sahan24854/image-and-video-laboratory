"""Deterministic lab checks: no uploaded media or network required."""
import numpy as np
from .codec import Config, encode_image, decode_image, encode_video, decode_video, target_size
from .metrics import quality


def specimen(name='Texture'):
    y,x=np.indices((48,64)); rng=np.random.default_rng(42)
    if name=='Noise': return rng.integers(0,256,(48,64,3),dtype=np.uint8)
    if name=='Edges': return np.repeat((((x//8+y//8)%2)*255)[...,None],3,axis=2).astype('uint8')
    return np.stack([(x*4)%256,(y*5)%256,((x+y)*3)%256],-1).astype('uint8')


def checks():
    image=specimen(); frames=np.stack([np.roll(image,i,axis=1) for i in range(4)])
    high=quality(image,decode_image(encode_image(image,Config(q=.25))))
    low=quality(image,decode_image(encode_image(image,Config(q=12))))
    exact,fps=decode_video(encode_video(frames,Config(q=.001,quantizer='uniform',transform='identity'),30))
    stream,report=target_size(encode_image,image,1600,mode='maximum')
    return [dict(check='Increasing Q increases distortion on the fixed test image',passed=low['mse']>high['mse'],observed=f"MSE {high['mse']:.2f} → {low['mse']:.2f}"),dict(check='Independent video reconstruction preserves signed residuals',passed=bool(np.array_equal(frames,exact)),observed=f'{len(exact)} frames, {fps:g} frames/s'),dict(check='Maximum-size search fits the feasible test budget',passed=report['within_budget'],observed=f'{len(stream)} / 1600 bytes'),dict(check='Identity quality gives SSIM = 1',passed=bool(np.isclose(quality(image,image)['ssim_y'],1)),observed='SSIM = 1')]


def panel():
    import streamlit as st
    st.subheader('Lab validation bench')
    st.caption('Run deterministic checks, then follow the hands-on experiments below using the sidebar controls.')
    if st.button('Run image + video self-checks',icon=':material/fact_check:'):
        st.session_state.lab_checks=checks()
    if st.session_state.get('lab_checks'):
        for item in st.session_state.lab_checks:
            (st.success if item['passed'] else st.error)(f"{'PASS' if item['passed'] else 'FAIL'} · {item['check']} · {item['observed']}")
    st.markdown('''| Experiment | Adjust | What to check |
|---|---|---|
| Image distortion | Same input, Q = 0.25 then 12 | MSE generally rises, PSNR falls, coefficient zeros increase |
| Transform compaction | Smooth input, DCT then identity at same Q | Inspect upper-left energy and file sizes; these quantizers need not give equal distortion |
| Color loss | Colored image, 4:4:4 then 4:2:0 | Inspect color boundaries and SSIM; compare actual measured sizes |
| Boundary handling | Odd-sized image, edge then constant | Inspect the final block's AC values and edge reconstruction |
| Video prediction | Built-in moving clip, previous then hybrid | Inspect residuals, vectors and total bytes at the same Q |
| Search coverage | Radius ±8 px, step 2 then 1 | Full search tests more candidates; best inter cost cannot be higher on the same block |
| Scene change | Built-in video, frame 8 | Inspect DC selections and residuals; this content may favor DC blocks |
| Reference integrity | Run self-checks | Exact low-step identity video reconstruction must pass |
| Size feedback | Enable target size; inspect trace | Check actual bytes and within_budget; an infeasible target can fail |
''')
