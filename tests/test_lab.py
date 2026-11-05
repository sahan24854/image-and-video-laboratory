import unittest
from educodec.lab import checks

class LabTests(unittest.TestCase):
    def test_demonstration_invariants(self):
        for item in checks():
            with self.subTest(check=item['check']):
                self.assertTrue(item['passed'],item['observed'])
