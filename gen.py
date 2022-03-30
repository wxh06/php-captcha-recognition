import subprocess, cv2
import numpy as np
import pickle
import pandas as pd
import threading

ims = []
def run(num, wid):
    p = subprocess.Popen('php gen.php {}'.format(num), stdout=subprocess.PIPE, shell=True)
    for i in range(num):
        if i % 1000 == 0:
            print(f"Worker{wid}: Generated{i}")
        s = p.stdout.read(2)
        ph = str(p.stdout.read(4), encoding='utf-8')
        l = s[0] * 256 + s[1]
        d = p.stdout.read(l)
        ims.append((ph, d))
def mainrun(num, workers):
    each = num // workers
    tasks = []
    for i in range(workers - 1):
        tasks.append(threading.Thread(target=run, args=(each, i)))
    tasks.append(threading.Thread(target=run, args=(num - each * (workers - 1), workers - 1)))

    for thread in tasks:
        thread.start()
    for thread in tasks:
        thread.join()

    df = pd.DataFrame(ims, columns=['label', 'jpg'], dtype=bytes)
    pickle.dump(df, open('./dat.pkl', 'wb+'), 2)

mainrun(30000, 16)