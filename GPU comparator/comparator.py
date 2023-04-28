import torch
import time

t_0 = time.time()
device = "cuda" if torch.cuda.is_available() else "cpu"
N = 100000

sum = torch.ones(size = (100,100), device=device)

for _ in range(N):
    sum += torch.ones(size = (100,100), device=device)

print(time.time() -t_0)
