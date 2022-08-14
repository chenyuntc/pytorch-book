# Matrix.py
import mpi4py.MPI as MPI
import numpy as np

# 初始化环境
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()  # world_size

# 在rank-0初始化数据
if rank == 0:
    print(f"当前的world_size为：{size}")
    array = np.arange(8)
    splits = np.split(array, size)  # 将数据分为N份
    print(f"原始数据为：\n {array}")
else:
    splits = None

# 将数据从rank-0切片并传到其他进程
local_data = comm.scatter(splits, root=0)
print(f"rank-{rank} 拿到的数据为：\n {local_data}")

# 在每一进程求和，并将结果进行allreduce
local_sum = local_data.sum()
all_sum = comm.allreduce(local_sum, op=MPI.SUM)

# 在每个进程计算平方，并将结果allgather
local_square = local_data ** 2
result = comm.allgather(local_square)
result = np.vstack(result)

# 只在某一个进程打印结果
if rank == 1:
    print("元素和为：", all_sum)
    print("按元素平方后的结果为：\n", result)