import torch
import time

t_0 = time.time()
device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)
N = 100000
k = 100

sum = torch.ones(size = (k,k), device=device)

for _ in range(N):
    sum = torch.add(torch.ones(size = (k,k), device=device),sum)

print(time.time() -t_0)
