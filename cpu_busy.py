import torch
import torch.nn as nn

import os
import psutil
import time

"""
This script occupied GPU momory to avoid others to insert jobs
"""

def main(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.g)
    p = psutil.Process(os.getpid())
    p.nice(19)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    block_num = 30 # 30 (48G)
    
    def get_block():
        a = torch.rand(500, 500, 500, requires_grad=True).to(device)
        b = torch.rand(500, 500, 500, requires_grad=True).to(device)
        c = torch.ones(500, 500, 500, requires_grad=False).to(device)
        return a, b, c

    # fake computation loop
    def block_by_hour(block_num, start_time, hours=1.0):
        block_got = 0
        block_list = list()
        while time.perf_counter()-start_time < 3600*hours:
            # get blocks one-by-one to prevent long wait in allocating memory
            if block_got < block_num:
                block_list.append(get_block())
                a, b, c = block_list[-1]
                q = 3*a**3 - b**2
                q.backward(gradient=c)
                block_got += 1
            else:
                for idx in range(block_got):
                    a, b, c = block_list[idx]
                    q = -1*a + 2*a**2 + 2*a**3 - 1*a**4 + 2*a**5 - 3*a**6 + 1*a**7 - 2*a**8 \
                        -2*b + 4*b**2 + 1*b**3 + 2*b**4 - 1*b**5 + 3*b**6 - 2*b**7 + 1*b**8
                    q.backward(gradient=c)
    
    # run
    start_time = time.perf_counter()
    block_by_hour(block_num, start_time, hours=args.hours)
            
def get_args():
    import argparse
    parser = argparse.ArgumentParser(description="occupied GPU momory to avoid others to insert jobs")
    parser.add_argument('-g', default=0, type=int, help="GPU id")
    parser.add_argument('--hours', default=1.0, type=float, help="hour for GPU to be block")
    return parser

if __name__ == '__main__':
    args = get_args().parse_args()
    main(args)