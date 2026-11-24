"""Exercise the running chatbot pipeline through its Flask entry points."""
import io
import unittest

from app import app
from triage.paths import IMAGES
from triage.vision import configure_ocr


class ChatIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        configure_ocr()
        cls.images = IMAGES

    def setUp(self):
        self.client = app.test_client()

    def submit(self, text='', image=None):
        data = {'text': text}
        if image:
            data['image'] = (io.BytesIO((self.images / image).read_bytes()), image)
        response = self.client.post('/api/chat', data=data)
        self.assertEqual(response.status_code, 200)
        return response.get_json()

    def test_ocr_continues_pending_return_after_digression(self):
        start = self.submit('我想退貨')
        self.assertEqual(start['tickets'], [])
        self.assertTrue(start['bot']['ask'])
        digression = self.submit('今天天氣怎麼樣')
        self.assertEqual(digression['tickets'], [])
        self.assertIn('退', digression['bot']['text'])
        completed = self.submit(image='order_ship.png')
        self.assertEqual(len(completed['tickets']), 1)
        self.assertEqual(completed['tickets'][0]['order_id'], 'A20261104-0087')
        self.assertEqual(completed['tickets'][0]['intent'], 'request_return')

    def test_existing_protective_fixtures_preserve_pending_task(self):
        import json
        fixtures = json.loads((self.images.parent / 'eval_sets.json').read_text(encoding='utf-8'))
        self.submit('我想退貨')
        sample = fixtures['attack_text']['obfuscated'][0]
        self.assertTrue(self.submit(sample)['bot']['blocked'])
        image, caption = fixtures['attack_image'][0]
        blocked = self.submit(caption, image)
        self.assertTrue(blocked['bot']['blocked'])
        self.assertEqual(blocked['bot']['guard']['where'], 'image')
        self.assertEqual(blocked['tickets'], [])
        completed = self.submit('訂單編號是 A20261104-0087')
        self.assertEqual(completed['tickets'][0]['intent'], 'request_return')

    def test_invalid_attachment_and_new_session(self):
        bad = self.client.post('/api/chat', data={'image': (io.BytesIO(b'invalid data'), 'fake.png')})
        self.assertEqual(bad.status_code, 415)
        self.submit('我想退貨')
        self.assertEqual(self.client.post('/api/reset').status_code, 200)
        response = self.submit('需要登入帳號的協助，我的帳號登入失敗')
        self.assertTrue(all(ticket['intent'] != 'request_return' for ticket in response['tickets']))

    def test_demo_web_pages_serve(self):
        for path in ['/', '/demo/1', '/demo/2']:
            with self.subTest(path=path):
                response = self.client.get(path)
                self.assertEqual(response.status_code, 200)
                self.assertIn('EriXNet 商城客服', response.get_data(as_text=True))


if __name__ == '__main__':
    unittest.main()
