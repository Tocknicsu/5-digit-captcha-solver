import json
import uuid

import requests
from captcha_solver import CaptchaSolver

try:
    data = json.loads(open('data.json').read())
except:
    data = []

solver = CaptchaSolver('browser')
while True:
    pic = requests.get(
        'https://course.nctu.edu.tw/function/Safecode.asp').content
    ans = solver.solve_captcha(pic)
    filename = '%s.png' % str(uuid.uuid4())
    print(ans)
    with open(filename, 'wb') as f:
        f.write(pic)
    data.append((ans, filename))
    with open('data.json', 'w') as f:
        f.write(json.dumps(data))
