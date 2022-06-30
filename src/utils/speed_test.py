import time

N = 1000
s = 0
m = 0

for i in range(N):
    start = time.perf_counter()
    # 処理
    end = time.perf_counter()
    s += end - start
    m = max(m, end-start)
print('avg:', s/N)
print('max:', m)
