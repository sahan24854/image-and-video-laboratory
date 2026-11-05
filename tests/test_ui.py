"""UI tests require the optional UI dependencies."""
import importlib.util
import unittest
from pathlib import Path

@unittest.skipUnless(importlib.util.find_spec('streamlit'), 'Install the ui extra')
class BlockControlTests(unittest.TestCase):
    def test_block_size_visibility_and_encoding(self):
        from streamlit.testing.v1 import AppTest
        at=AppTest.from_file(str(Path(__file__).resolve().parents[1] / 'app.py'),default_timeout=30).run()
        control=next(s for s in at.selectbox if s.label=='Transform block size (pixels)')
        self.assertEqual(control.value,8)
        self.assertTrue(control.disabled)
        for quantizer in ('weighted','uniform'):
            next(s for s in at.selectbox if s.label=='Quantization table').select(quantizer).run()
            for block in (4,16):
                control=next(s for s in at.selectbox if s.label=='Transform block size (pixels)')
                self.assertFalse(control.disabled)
                control.select(block).run()
                next(b for b in at.button if b.label=='Run experiment').click().run()
                self.assertFalse(at.exception)
                self.assertEqual(at.session_state.result['c'].block,block)
        next(s for s in at.selectbox if s.label=='Quantization table').select('jpeg').run()
        self.assertEqual(next(s for s in at.selectbox if s.label=='Transform block size (pixels)').value,8)
        self.assertFalse(at.exception)
