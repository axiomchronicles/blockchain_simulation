from transutil.chiper import AXTokenManager, Algorithm
from transutil import eid

import numpy as np

arr = np.array([])

secret_key = eid.eid.random(16)

manager = AXTokenManager(secret_key, Algorithm.HS256)
result = manager.generate_tokens("main.user", "axiomchronicles", subject = "This is a test token")

split_token = result["refresh_token"].split("=")

regen_token = manager.refresh_access_token(split_token)
print(regen_token)