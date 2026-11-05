"""Reference implementation: readable loops, explicit metadata, reconstructed references."""
from dataclasses import dataclass, asdict, replace
import json
import struct
import zlib
import bz2
import lzma
import numpy as np
from scipy.fft import dctn, idctn
from .tables import LUMA_QUANT_MATRIX, CHROMA_QUANT_MATRIX


@dataclass(frozen=True)
class Config:
    q: float = 1.0
    block: int = 8
    quantizer: str = 'weighted'
    transform: str = 'dct'
    entropy: str = 'zlib'
    chroma: str = '444'
    search: int = 4
    macroblock: int = 16
    key_interval: int = 12
    prediction: str = 'hybrid'
    search_step: int = 2
    padding: str = 'edge'
    cost: str = 'sad'

    def validate(self):
        if not np.isfinite(self.q) or not 0.001 <= self.q <= 10000:
            raise ValueError('q must be finite and between 0.001 and 10000')
        if self.block not in (4, 8, 16): raise ValueError('block must be 4, 8 or 16')
        for value, allowed in [(self.padding, ('edge','constant')), (self.cost, ('sad','mse')), (self.quantizer, ('uniform','weighted','jpeg')), (self.transform, ('dct','identity')), (self.entropy, ('zlib','bz2','lzma')), (self.chroma, ('444','422','420')), (self.prediction, ('hybrid','motion','previous'))]:
            if value not in allowed: raise ValueError(f'Unsupported option: {value}')
        if self.search_step not in (1,2,4): raise ValueError('search_step must be 1, 2 or 4')
        if not 0 <= self.search <= 32 or self.macroblock not in (8,16,32) or self.key_interval < 1:
            raise ValueError('Invalid video settings')


def _rgb(a):
    a = np.asarray(a)
    if a.dtype != np.uint8 or a.ndim != 3 or a.shape[2] != 3 or min(a.shape[:2]) < 1:
        raise ValueError('Expected nonempty H×W×3 uint8 RGB array')
    return a


def _color(a):
    r,g,b = np.moveaxis(a.astype(float), -1, 0)
    return [0.299*r+0.587*g+0.114*b, 128-0.168736*r-0.331264*g+0.5*b, 128+0.5*r-0.418688*g-0.081312*b]


def _uncolor(p):
    y,cb,cr = p; cb=cb-128; cr=cr-128
    return np.clip(np.rint(np.stack([y+1.402*cr,y-0.344136*cb-0.714136*cr,y+1.772*cb],-1)),0,255).astype('uint8')


def _table(c, chroma=False):
    n=c.block
    if c.quantizer=='jpeg':
        if n!=8: raise ValueError('Original JPEG tables require block=8')
        return np.maximum(np.rint((CHROMA_QUANT_MATRIX if chroma else LUMA_QUANT_MATRIX)*c.q),1)
    y,x=np.indices((n,n))
    return c.q * (np.ones((n,n))*8 if c.quantizer=='uniform' else 4+4*(x+y)+2*np.maximum(x,y))


def _plane(a,c,decode=False,chroma=False):
    n=c.block
    if decode:
        blocks=a*_table(c,chroma)
        if c.transform=='dct': blocks=idctn(blocks,axes=(-2,-1),norm='ortho')
        return blocks.transpose(0,2,1,3).reshape(blocks.shape[0]*n,blocks.shape[1]*n)
    h,w=a.shape
    a=np.pad(a,((0,(-h)%n),(0,(-w)%n)),mode=c.padding)
    blocks=a.reshape(a.shape[0]//n,n,a.shape[1]//n,n).transpose(0,2,1,3)
    if c.transform=='dct': blocks=dctn(blocks,axes=(-2,-1),norm='ortho')
    return np.rint(blocks/_table(c,chroma)).astype('<i4')


def _pack(kind,c,meta,arrays):
    raw=b''.join(np.asarray(a,dtype='<i4').tobytes() for a in arrays)
    header=json.dumps(dict(version=1,kind=kind,config=asdict(c),meta=meta,shapes=[a.shape for a in arrays]),separators=(',',':')).encode()
    payload={'zlib':zlib,'bz2':bz2,'lzma':lzma}[c.entropy].compress(raw)
    return b'EDC1'+struct.pack('<I',len(header))+header+payload


def _unpack(data,kind):
    if data[:4]!=b'EDC1' or len(data)<8: raise ValueError('Invalid EDC stream')
    size=struct.unpack('<I',data[4:8])[0]
    try:
        h=json.loads(data[8:8+size]); c=Config(**h['config']); c.validate()
        if h['version']!=1 or h['kind']!=kind: raise ValueError('Wrong stream type/version')
        raw={'zlib':zlib,'bz2':bz2,'lzma':lzma}[c.entropy].decompress(data[8+size:])
        arrays=[]; offset=0
        for shape in h['shapes']:
            count=int(np.prod(shape)); end=offset+count*4
            arrays.append(np.frombuffer(raw[offset:end],dtype='<i4').reshape(shape)); offset=end
        if offset!=len(raw): raise ValueError('Invalid payload length')
        return c,h['meta'],arrays
    except (KeyError,TypeError,json.JSONDecodeError,zlib.error,EOFError) as e:
        raise ValueError('Malformed EDC stream') from e


def encode_image(image,config=Config()):
    config.validate(); image=_rgb(image); planes=_color(image); arrays=[]
    for i,p in enumerate(planes):
        if i and config.chroma!='444':
            sy=2 if config.chroma=='420' else 1
            p=np.pad(p,((0,(-p.shape[0])%sy),(0,p.shape[1]%2)),mode='edge')
            p=p.reshape(p.shape[0]//sy,sy,p.shape[1]//2,2).mean((1,3))
        arrays.append(_plane(p-128,config,chroma=bool(i)))
    return _pack('image',config,{'height':image.shape[0],'width':image.shape[1]},arrays)


def decode_image(data):
    c,m,arrays=_unpack(data,'image'); h,w=m['height'],m['width']; planes=[]
    for i,a in enumerate(arrays):
        p=_plane(a,c,True,chroma=bool(i))+128
        if i and c.chroma!='444': p=np.repeat(np.repeat(p,2 if c.chroma=='420' else 1,0),2,1)
        planes.append(p[:h,:w])
    return _uncolor(planes)


def _predict(ref,motions,mb):
    h,w=ref.shape[:2]; out=np.empty_like(ref,dtype=float); k=0
    for y in range(0,h,mb):
        for x in range(0,w,mb):
            entry=motions[k]; dy,dx=entry[:2]; k+=1; bh,bw=min(mb,h-y),min(mb,w-x)
            if len(entry)>2 and entry[2]: out[y:y+bh,x:x+bw]=entry[3:6]
            else: out[y:y+bh,x:x+bw]=ref[y+dy:y+dy+bh,x+dx:x+dx+bw]
    return out


def _motion(ref,frame,c):
    h,w=frame.shape[:2]; vectors=[]
    r=_color(ref)[0]; f=_color(frame)[0]
    for y in range(0,h,c.macroblock):
        for x in range(0,w,c.macroblock):
            bh,bw=min(c.macroblock,h-y),min(c.macroblock,w-x)
            best=(float('inf'),0,0)
            radius=c.search if c.prediction in ('motion','hybrid') else 0
            for dy in sorted(set(range(max(-radius,-y),min(radius,h-y-bh)+1,c.search_step))|{0}):
                for dx in sorted(set(range(max(-radius,-x),min(radius,w-x-bw)+1,c.search_step))|{0}):
                    error=f[y:y+bh,x:x+bw]-r[y+dy:y+dy+bh,x+dx:x+dx+bw]
                    cost=np.abs(error).sum() if c.cost=='sad' else np.mean(error**2)
                    candidate=(cost,abs(dy)+abs(dx),dy,dx)
                    if len(best)==3 or candidate<best: best=candidate
            dc=np.rint(frame[y:y+bh,x:x+bw].mean((0,1))).astype(int)
            dc_y=float(dc @ np.array([.299,.587,.114]))
            error=f[y:y+bh,x:x+bw]-dc_y
            intra_cost=np.abs(error).sum() if c.cost=='sad' else np.mean(error**2)
            use_intra=c.prediction=='hybrid' and intra_cost<best[0]
            vectors.append([0,0,1,*dc] if use_intra else [*best[-2:],0,0,0,0])
    return np.array(vectors,dtype='<i4')


def encode_video(frames,config=Config(),fps=24.0):
    config.validate()
    if not np.isfinite(fps) or fps<=0: raise ValueError('fps must be positive')
    frames=[_rgb(f) for f in frames]
    if not frames or any(f.shape!=frames[0].shape for f in frames): raise ValueError('Frames must share dimensions')
    if config.chroma!='444': raise ValueError('Video currently supports 444 only')
    arrays=[]; types=[]; ref=None
    for index,frame in enumerate(frames):
        intra=index%config.key_interval==0; types.append('I' if intra else 'P')
        if intra: pred=np.full(frame.shape,128.)
        else:
            mv=_motion(ref,frame,config); arrays.append(mv); pred=_predict(ref,mv,config.macroblock)
        rec=np.empty(frame.shape,float)
        for ch in range(3):
            a=_plane(frame[:,:,ch].astype(float)-pred[:,:,ch],config); arrays.append(a)
            rec[:,:,ch]=_plane(a,config,True)[:frame.shape[0],:frame.shape[1]]+pred[:,:,ch]
        ref=np.clip(np.rint(rec),0,255).astype('uint8')
    return _pack('video',config,dict(height=frames[0].shape[0],width=frames[0].shape[1],fps=float(fps),types=types),arrays)


def decode_video(data):
    c,m,arrays=_unpack(data,'video'); h,w=m['height'],m['width']; it=iter(arrays); frames=[]
    for t in m['types']:
        pred=np.full((h,w,3),128.) if t=='I' else _predict(frames[-1],next(it),c.macroblock)
        rec=np.stack([_plane(next(it),c,True)[:h,:w] for _ in range(3)],-1)+pred
        frames.append(np.clip(np.rint(rec),0,255).astype('uint8'))
    return np.stack(frames),m['fps']


def target_size(encode,source,target_bytes,config=Config(),iterations=16,mode='closest'):
    """Bracketed heuristic; includes headers. Returns stream and search diagnostics."""
    if target_bytes<=0 or iterations<2 or mode not in ('closest','maximum'): raise ValueError('Invalid size search parameters')
    low,high=0.001,10000.; candidates=[]
    def trial(q):
        stream=encode(source,replace(config,q=q)); candidates.append((q,stream)); return len(stream)
    trial(low); trial(high)
    for _ in range(iterations-2):
        mid=(low*high)**0.5
        if trial(mid)>target_bytes: low=mid
        else: high=mid
    feasible=[p for p in candidates if len(p[1])<=target_bytes]
    pool=feasible if mode=='maximum' and feasible else candidates
    q,stream=min(pool,key=lambda p:abs(len(p[1])-target_bytes))
    return stream,dict(q=q,bytes=len(stream),target_bytes=target_bytes,within_budget=len(stream)<=target_bytes,relative_error=(len(stream)-target_bytes)/target_bytes,evaluations=len(candidates),mode=mode,trace=[dict(q=q,bytes=len(s)) for q,s in candidates])
