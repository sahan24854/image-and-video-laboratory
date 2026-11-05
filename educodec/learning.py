"""Interactive, numeric demonstrations backed by the codec's mechanisms."""
import heapq
from collections import Counter
import numpy as np
import pandas as pd
from scipy.fft import dctn, idctn
import streamlit as st
from .codec import _color, _table

LESSONS = {
'1 · Signals and perception': '''The human visual system is generally more sensitive to luminance structure than fine chroma detail. Spatial masking means texture can hide errors that are visible in smooth areas; the fixed tables here approximate frequency sensitivity and do not implement adaptive masking.

RGB is converted to YCbCr: Y = 0.299R + 0.587G + 0.114B; Cb and Cr encode blue/red differences around 128. 4:4:4 retains every sample, 4:2:2 halves horizontal chroma, and 4:2:0 halves both dimensions. Subsampling discards color detail before quantization.

Blocks of 8×8 or 16×16 require boundary handling. Edge replication avoids introducing a jump toward black; zero padding can create artificial high-frequency coefficients. Constant padding in the codec acts on the centered signal (zero represents gray 128 for shifted pixels).''',
'2 · Transform and quantization': '''The orthonormal 2D DCT decomposes a block into cosine basis patterns. Smooth blocks concentrate energy near the upper-left DC coefficient; other coefficients are AC spatial frequencies. IDCT reverses the transform, up to numerical precision, before quantization losses.

Raw pixels are shifted by 128. Signed prediction residuals are already centered and receive no additional shift. Quantization computes round(C/Q); reconstruction computes round(C/Q) × Q. Increasing Q creates more zeros but loses detail. Separate luma/chroma JPEG tables preserve brightness structure more gently than chroma. Blocking appears at independent block boundaries; ringing appears near sharp edges after high-frequency suppression.''',
'3 · Temporal prediction': '''The encoder and decoder both reference reconstructed previous frames. Referencing the pristine original only at the encoder causes prediction drift.

16×16 block matching searches displacements within ±p (try p=8). Step 1 is exhaustive within that window; larger steps sample fewer candidates. SAD sums absolute error; MSE averages squared error and penalizes large deviations more strongly. Vectors are stored internally as (dy, dx).

Motion compensation forms the predictor, then R = current − predicted. Hybrid mode compares inter cost with a constant block-mean DC predictor. DC values are explicitly signaled here. This is an educational fallback rather than a causal neighboring-block predictor. One luminance search supplies vectors reused across RGB channels; a YCbCr video extension would reuse them with chroma coordinates scaled for subsampling. Current video residual coding uses RGB 4:4:4.''',
'4 · Entropy and signaling': '''Quantized arrays flatten to little-endian int32 streams. Long zero/repeated patterns often compress well. DEFLATE (zlib) combines LZ77 length/distance dictionary matches with Huffman coding of literals and match symbols. Explicit run-length coding is a separate technique; it is demonstrated below, not a separate stage in this codec. Huffman codes are prefix-free: frequent symbols usually receive shorter codewords.

The file also includes dimensions, tables/settings, array shapes, frame types, mode flags, motion vectors and DC values. Tiny files can grow because overhead dominates. Coefficient sparsity is useful but does not alone determine the final byte count.''',
'5 · Rate and distortion control': '''Rate R and distortion D trade off nonlinearly. A quantization sweep gives an empirical rate–distortion curve. SAD mode selection alone is not RDO; true rate-aware mode selection typically minimizes D + λR including side information.

Target-size search adjusts global Q using feedback from complete encoded size. Quantization plateaus and entropy coding make output size non-monotonic; binary/geometric bracketing is a heuristic, not a guarantee.

A fixed Q is a constant-quantizer/VBR experiment, not a strict quality guarantee. CBR requires time-aware rate allocation and buffering; this codec only targets whole-file size. Strict quality thresholds require measuring reconstruction and searching feasible candidates. The UI sweep exposes those measurements without claiming CBR or guaranteed PSNR control.''',
'6 · Measuring fidelity': '''MSE averages squared differences over all RGB samples. PSNR = 10 log10(255² / MSE). Exact equality gives MSE=0 and infinite PSNR. Values such as 28–45 dB are common examples, not universal quality thresholds.

Luminance SSIM evaluates local brightness, contrast and structure. Inspect original/reconstruction/error views as well: high PSNR can coexist with visible blocking, ringing, color bleeding or lost texture. Video tables report each frame and the summary averages frame PSNR in dB.'''
}


def simulate(image,c):
    st.subheader('Step through spatial coding')
    stage=st.select_slider('Mechanism stage',options=['Color planes','Boundary padding','Level shift','DCT / identity','Quantize','Dequantize','Reconstruct','Entropy symbols'])
    channel=st.selectbox('Inspect channel',['Y','Cb','Cr'])
    plane=_color(image)[['Y','Cb','Cr'].index(channel)]
    n=c.block
    row=int(st.number_input('Block row',0,(image.shape[0]-1)//n,0,key='simrow'))
    col=int(st.number_input('Block column',0,(image.shape[1]-1)//n,0,key='simcol'))
    padded=np.pad(plane,((0,(-plane.shape[0])%n),(0,(-plane.shape[1])%n)),mode=c.padding)
    patch=padded[row*n:(row+1)*n,col*n:(col+1)*n]
    shifted=patch-128; coefficients=dctn(shifted,norm='ortho') if c.transform=='dct' else shifted
    table=_table(c,channel!='Y'); quantized=np.rint(coefficients/table); dequantized=quantized*table
    recovered=(idctn(dequantized,norm='ortho') if c.transform=='dct' else dequantized)+128
    matrices={'Color planes':plane,'Boundary padding':patch,'Level shift':shifted,'DCT / identity':coefficients,'Quantize':quantized,'Dequantize':dequantized,'Reconstruct':recovered}
    if stage in matrices:
        value=matrices[stage]; left,right=st.columns(2)
        left.dataframe(pd.DataFrame(value if stage!='Color planes' else value[:32,:32]).round(2))
        right.image(np.clip(value if stage in ('Color planes','Boundary padding','Reconstruct') else 128+value,0,255).astype('uint8'),caption='Numeric signal preview; signed values displayed around gray',width=256)
        st.caption('Color-plane table shows the first 32×32 samples. Transform previews are display-clipped; the numeric matrix retains the actual values.')
    else:
        values=quantized.astype(int).ravel().tolist(); runs=[]
        for v in values:
            if runs and runs[-1][0]==v: runs[-1][1]+=1
            else: runs.append([v,1])
        counts=Counter(values); heap=[(count,i,{symbol:''}) for i,(symbol,count) in enumerate(counts.items())]; heapq.heapify(heap); counter=len(heap)
        while len(heap)>1:
            a,_,ac=heapq.heappop(heap); b,_,bc=heapq.heappop(heap)
            codes={**{k:'0'+v for k,v in ac.items()},**{k:'1'+v for k,v in bc.items()}}
            heapq.heappush(heap,(a+b,counter,codes)); counter+=1
        codes=heap[0][2]
        st.write('Row-major integer stream',values); st.dataframe(pd.DataFrame(runs,columns=['value','run length']))
        st.dataframe(pd.DataFrame([dict(symbol=k,count=v,code=codes[k] or '0') for k,v in counts.items()]))
        st.caption('Illustrative RLE and Huffman codes. The actual selected lossless compressor encodes the complete stream, including its own headers.')
    energy=float(np.sum(coefficients**2)); low=float(np.sum(coefficients[:max(1,n//4),:max(1,n//4)]**2))
    st.caption(f'DC coefficient: {coefficients[0,0]:.2f} · Low-frequency corner energy: {100*low/energy if energy else 0:.1f}% · Quantized zeros: {100*np.mean(quantized==0):.1f}% · Block reconstruction MSE: {np.mean((patch-recovered)**2):.3f}')


def motion_simulation(previous,current,c):
    st.subheader('Motion search simulation')
    st.caption('Uses the selected input frame pair for exploration. Completed-codec predictor views use reconstructed references and show the actual coded decisions.')
    n=c.macroblock; h,w=current.shape[:2]
    by=int(st.number_input('Macroblock row',0,(h-1)//n,0)); bx=int(st.number_input('Macroblock column',0,(w-1)//n,0))
    y,x=by*n,bx*n; bh,bw=min(n,h-y),min(n,w-x)
    ref=_color(previous)[0]; cur=_color(current)[0]; patch=cur[y:y+bh,x:x+bw]
    rows=[]
    for dy in sorted(set(range(max(-c.search,-y),min(c.search,h-y-bh)+1,c.search_step))|{0}):
        for dx in sorted(set(range(max(-c.search,-x),min(c.search,w-x-bw)+1,c.search_step))|{0}):
            error=patch-ref[y+dy:y+dy+bh,x+dx:x+dx+bw]
            rows.append(dict(dx=dx,dy=dy,cost=float(np.abs(error).sum() if c.cost=='sad' else np.mean(error**2))))
    step=int(st.number_input('Candidate traversal step (candidate index)',0,len(rows)-1,0))
    candidate=rows[step]; dy,dx=candidate['dy'],candidate['dx']
    cols=st.columns(2); cols[0].image(current[y:y+bh,x:x+bw],caption='Current block',width=192); cols[1].image(previous[y+dy:y+dy+bh,x+dx:x+dx+bw],caption=f'Candidate dx={dx} px, dy={dy} px',width=192)
    st.line_chart(pd.DataFrame(rows)['cost'])
    best=min(rows,key=lambda r:(r['cost'],abs(r['dy'])+abs(r['dx']),r['dy'],r['dx']))
    dc=np.rint(current[y:y+bh,x:x+bw].mean((0,1)))@np.array([.299,.587,.114]); err=patch-dc
    intra=float(np.abs(err).sum() if c.cost=='sad' else np.mean(err**2))
    st.write(dict(best_inter=best,dc_cost=intra,hybrid_decision='DC' if intra<best['cost'] else 'inter'))
    st.caption('SAD units: summed luma sample levels. MSE units: squared luma sample levels. Candidate traversal is an interactive discrete simulation, not real-time playback.')

# Further techniques are reference material, not selectable codec implementations.
EXTENSIONS = {
'1 · Signals and perception': '''**Further methods:** linear-light versus gamma-coded processing; BT.601/BT.709 color matrices; full versus limited signal range; anti-alias filtering before chroma decimation; bilinear/bicubic chroma interpolation; contrast sensitivity, just-noticeable-difference models and saliency-weighted distortion. Bit depth (8/10/12-bit) and transfer functions affect quantization and metric interpretation. This implementation accepts 8-bit RGB and uses one fixed color matrix; chroma decimation uses averaging and reconstruction uses nearest-neighbor repetition.''',
'2 · Transform and quantization': '''**Further methods:** separable integer transforms, discrete wavelet transforms, Hadamard transforms, transform skip, variable transform partitions, dead-zone quantization, trellis quantization, adaptive quantization, zigzag scans and differential DC coding. Deblocking reduces block discontinuities; directional intra prediction uses reconstructed neighbors instead of a flat mean. These are reference extensions: current choices are orthonormal DCT/identity and uniform/weighted/JPEG-table quantization, without in-loop filtering or zigzag/DC differencing.''',
'3 · Temporal prediction': '''**Further methods:** three-step, diamond and hierarchical motion search; half/quarter-pixel interpolation; multiple references; bidirectional B-frame prediction; skip/merge modes; variable block sizes; scene-cut-triggered I-frames and optical-flow prediction. Faster searches trade candidate coverage for computation. Fractional-pixel prediction needs identical interpolation at encoder and decoder. B-frames introduce reordering delay. Current video supports bounded sampled/full integer-pixel search, one previous reference, fixed macroblocks, periodic I-frames and DC fallback.''',
'4 · Entropy and signaling': '''**Further methods:** arithmetic/range coding, context-adaptive binary arithmetic coding (CABAC), asymmetric numeral systems (ANS), Golomb–Rice coding, signed integer mappings, explicit zero-run/end-of-block symbols and adaptive probability models. Each requires an agreed syntax and decoder state; entropy below a simple symbol histogram does not include dependencies or headers. Current stream compressors are zlib, bzip2 and LZMA. The Huffman/RLE table is a standalone teaching demonstration.''',
'5 · Rate and distortion control': '''**Further methods:** Lagrangian D + λR mode decisions, two-pass bit allocation, GOP/frame-level budgets, virtual-buffer/VBV feedback, region-of-interest Q allocation, content-adaptive quantization and constrained quality search. Buffer fullness and per-frame bits are needed to demonstrate CBR properly. The current whole-file Q search and measured Q sweep do not implement those controllers. Equal-rate comparisons should interpolate empirical curves carefully; BD-rate summarizes curves only across an overlapping quality interval.''',
'6 · Measuring fidelity': '''**Further metrics:** luma-only PSNR, per-channel errors, multiscale SSIM, VMAF, perceptual learned metrics such as LPIPS, temporal flicker and motion-compensated error. Metrics may require trained models and calibrated input conventions. Subjective double-stimulus viewing complements objective scores. Current measured metrics are RGB MSE/RMSE/MAE/max error, RGB PSNR and local luminance SSIM; the others are explained extensions. PSNR curves should state whether video aggregation averages dB or first pools MSE.'''
}

def lessons():
    topic=st.selectbox('Learning topic',list(LESSONS))
    st.markdown(LESSONS[topic])
    with st.expander('Additional methods and implementation status'):
        st.markdown(EXTENSIONS[topic])
