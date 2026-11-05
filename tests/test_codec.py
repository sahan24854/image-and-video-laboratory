import unittest
import numpy as np
from educodec import *

class CodecTests(unittest.TestCase):
    def setUp(self):
        self.image=np.random.default_rng(9).integers(0,256,(19,23,3),dtype=np.uint8)
    def test_image_options(self):
        for entropy in ('zlib','bz2','lzma'):
            for chroma in ('444','420'):
                for block in (4,8,16):
                    out=decode_image(encode_image(self.image,Config(entropy=entropy,chroma=chroma,block=block)))
                    self.assertEqual(out.shape,self.image.shape)
                    self.assertGreater(quality(self.image,out)['psnr_db'],10)
    def test_quality_direction(self):
        high=quality(self.image,decode_image(encode_image(self.image,Config(q=.1))))
        low=quality(self.image,decode_image(encode_image(self.image,Config(q=10))))
        self.assertGreater(high['psnr_db'],low['psnr_db'])
        self.assertAlmostEqual(quality(self.image,self.image)['ssim_y'],1)
    def test_video_references(self):
        frames=np.stack([self.image]*5)
        for prediction in ('previous','motion'):
            out,fps=decode_video(encode_video(frames,Config(q=.001,transform='identity',quantizer='uniform',prediction=prediction,key_interval=3),30))
            np.testing.assert_array_equal(out,frames)
            self.assertEqual(fps,30)
    def test_target(self):
        stream,report=target_size(encode_image,self.image,1500,mode='maximum')
        self.assertEqual(len(stream),report['bytes'])
        self.assertLessEqual(len(stream),1500)
        self.assertEqual(decode_image(stream).shape,self.image.shape)
    def test_invalid(self):
        with self.assertRaises(ValueError): decode_image(b'bad')
        with self.assertRaises(ValueError): encode_image(self.image,Config(q=-1))
        with self.assertRaises(ValueError): encode_video([],Config())

if __name__=='__main__': unittest.main()
