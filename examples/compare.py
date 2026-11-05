"""Generate original assets and compare adjustable codec mechanisms without downloads."""
import csv
import json
import time
from pathlib import Path
from dataclasses import replace,asdict
import numpy as np
from PIL import Image
from educodec import *
from educodec.codec import _unpack

out=Path('results'); out.mkdir(exist_ok=True)
y,x=np.indices((64,96)); base=np.stack([(x*3)%256,(y*4)%256,((x+y)*2)%256],-1).astype('uint8')
frames=[]
for t in range(8):
    f=base.copy(); f[16:36,8+t*4:28+t*4]=[240,40,80]
    if t>=6: f=255-f
    frames.append(f)
frames=np.stack(frames); Image.fromarray(base).save(out/'input.png'); np.save(out/'input-video.npy',frames)
rows=[]
for kind,source in [('image',base),('video',frames)]:
    variants=({'original-tables':Config(quantizer='jpeg'),'weighted':Config(),'uniform':Config(quantizer='uniform'),'identity':Config(transform='identity'),'420':Config(chroma='420'),'bz2':Config(entropy='bz2'),'lzma':Config(entropy='lzma')} if kind=='image' else {'hybrid-step2':Config(),'hybrid-step1':Config(search_step=1),'motion':Config(prediction='motion'),'previous':Config(prediction='previous'),'all-intra':Config(key_interval=1)})
    for name,c in variants.items():
        for q in (.5,2,8):
            c=replace(c,q=q); start=time.perf_counter()
            stream=encode_image(source,c) if kind=='image' else encode_video(source,c)
            enc=time.perf_counter()-start; start=time.perf_counter()
            rec=decode_image(stream) if kind=='image' else decode_video(stream)[0]
            dec=time.perf_counter()-start
            scores=[quality(source,rec)] if kind=='image' else [quality(a,b) for a,b in zip(source,rec)]
            row=dict(kind=kind,method=name,q=q,bytes=len(stream),bpp=8*len(stream)/(source.size/3),encode_seconds=enc,decode_seconds=dec,**{key:float(np.mean([s[key] for s in scores])) for key in scores[0]})
            rows.append(row); stem=f'{kind}-{name}-q{q}'
            (out/f'{stem}.edc').write_bytes(stream)
            if kind=='image':
                Image.fromarray(rec).save(out/f'{stem}.png')
                Image.fromarray(np.clip(np.abs(source.astype(float)-rec)*4,0,255).astype('uint8')).save(out/f'{stem}-error-x4.png')
            else:
                np.save(out/f'{stem}.npy',rec)
                Image.fromarray(np.concatenate([source[3],rec[3],np.clip(np.abs(source[3].astype(float)-rec[3])*4,0,255).astype('uint8')],axis=1)).save(out/f'{stem}-comparison.png')
            _,meta,arrays=_unpack(stream,kind)
            stats={'config':asdict(c),'metrics_per_frame':scores,'frame_types':meta.get('types'),'intra_blocks':sum(int(a[:,2].sum()) for a in arrays if a.ndim==2 and a.shape[1]==6)}
            (out/f'{stem}.json').write_text(json.dumps(stats,indent=2))
with (out/'comparison.csv').open('w') as f:
    writer=csv.DictWriter(f,fieldnames=rows[0]); writer.writeheader(); writer.writerows(rows)
print('Saved inputs, decoded outputs, error images, per-frame metrics, settings and comparison.csv to results/')
