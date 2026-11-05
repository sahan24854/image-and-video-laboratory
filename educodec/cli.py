import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image
from .codec import Config,encode_image,decode_image,encode_video,decode_video,target_size
from .metrics import quality


def main():
    p=argparse.ArgumentParser(description='Educational image/video codec. Video input/output: THWC uint8 .npy arrays.')
    p.add_argument('action',choices=['encode-image','decode-image','encode-video','decode-video','analyze'])
    p.add_argument('input'); p.add_argument('output')
    p.add_argument('--q',type=float,default=1)
    p.add_argument('--block',type=int,choices=[4,8,16],default=8)
    p.add_argument('--quantizer',choices=['uniform','weighted','jpeg'],default='weighted')
    p.add_argument('--transform',choices=['dct','identity'],default='dct')
    p.add_argument('--entropy',choices=['zlib','bz2','lzma'],default='zlib')
    p.add_argument('--chroma',choices=['444','420'],default='444')
    p.add_argument('--prediction',choices=['hybrid','motion','previous'],default='hybrid')
    p.add_argument('--search-step',type=int,choices=[1,2,4],default=2)
    p.add_argument('--search',type=int,default=4)
    p.add_argument('--key-interval',type=int,default=12)
    p.add_argument('--fps',type=float,default=24)
    p.add_argument('--target-bytes',type=int)
    p.add_argument('--size-mode',choices=['closest','maximum'],default='closest')
    args=p.parse_args(); c=Config(**{k:getattr(args,k) for k in ['q','block','quantizer','transform','entropy','chroma','prediction','search','search_step','key_interval']})
    if args.action.startswith('decode'):
        data=Path(args.input).read_bytes()
        if args.action=='decode-image': Image.fromarray(decode_image(data)).save(args.output)
        else:
            frames,fps=decode_video(data); np.save(args.output,frames); print(json.dumps({'fps':fps,'frames':len(frames)}))
        return
    video=args.action=='encode-video' or (args.action=='analyze' and Path(args.input).suffix=='.npy')
    source=np.load(args.input,allow_pickle=False) if video else np.asarray(Image.open(args.input).convert('RGB'))
    encode=(lambda f,c:encode_video(f,c,args.fps)) if video else encode_image
    if args.target_bytes:
        stream,report=target_size(encode,source,args.target_bytes,c,mode=args.size_mode)
    else: stream=encode(source,c); report={'bytes':len(stream),'q':c.q}
    reconstructed=decode_video(stream)[0] if video else decode_image(stream)
    scores=[quality(a,b) for a,b in zip(source,reconstructed)] if video else [quality(source,reconstructed)]
    report.update(metrics=scores,bits_per_pixel=len(stream)*8/(source.size/3),compression_ratio=source.nbytes/len(stream))
    if video: report['bitrate_bps']=len(stream)*8*args.fps/len(source)
    # JSON has no infinity: null denotes a perfect-reconstruction PSNR.
    for score in scores:
        if not np.isfinite(score['psnr_db']): score['psnr_db']=None
    if args.action=='analyze': Path(args.output).write_text(json.dumps(report,indent=2))
    else: Path(args.output).write_bytes(stream)
    print(json.dumps(report,indent=2))

if __name__=='__main__': main()
