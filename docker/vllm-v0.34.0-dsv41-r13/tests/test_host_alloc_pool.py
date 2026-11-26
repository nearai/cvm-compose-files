import ctypes, mmap, os, sys
from unittest import mock
import torch
import vllm.v1.simple_kv_offload.worker as w

class FakeCudart:
    def __init__(self): self.allocs=[]; self.frees=[]; self.keep=[]
    def cudaHostAlloc(self, ref, size, flags):
        buf=(ctypes.c_int8*size)(); self.keep.append(buf)
        ref._obj.value=ctypes.addressof(buf); self.allocs.append((size,flags)); return 0
    def cudaFreeHost(self, p): self.frees.append(p.value); return 0

def mk():
    ww=w.SimpleCPUOffloadWorker.__new__(w.SimpleCPUOffloadWorker)
    ww._hostalloc_ptr=None; ww._hostalloc_nbytes=0
    ww.num_cpu_blocks=8; ww.cpu_hugepage_size=None
    return ww
gpu={"l0":torch.zeros(4,3,16,dtype=torch.bfloat16), "l1":torch.zeros(4,3,16,dtype=torch.bfloat16)}
need=2*8*3*16*2

# flag ON
fake=FakeCudart()
with mock.patch.dict(os.environ,{"VLLM_NEAR_CPU_OFFLOAD_HOSTALLOC":"1"}), \
     mock.patch.object(w,"_get_cudart",return_value=fake), \
     mock.patch.object(w,"pin_tensor",side_effect=AssertionError("pin_tensor called")), \
     mock.patch.object(w,"alloc_host_buffer",side_effect=AssertionError("mmap path called")), \
     mock.patch.object(w.atexit,"register"):
    ww=mk(); c=ww._alloc_cpu_caches(gpu,True)
    assert len(fake.allocs)==1 and fake.allocs[0][1]==0, fake.allocs
    assert fake.allocs[0][0]>=need and fake.allocs[0][0]%mmap.PAGESIZE==0 and fake.allocs[0][0]-need<mmap.PAGESIZE
    for n,t in c.items():
        assert t.shape==(8,3,16) and t.dtype==torch.bfloat16 and not t.is_cuda
    c["l0"].fill_(1); c["l1"].fill_(2)
    assert c["l0"].float().sum()==8*48 and c["l1"][0,0,0].item()==2
    assert c["l0"].data_ptr()+8*3*16*2==c["l1"].data_ptr()
    ptr=ww._hostalloc_ptr; ww.shutdown(); ww.shutdown()
    assert fake.frees==[ptr], fake.frees
print("flag-on OK: 1 alloc size",fake.allocs[0][0],"1 free")

# flag OFF: original path only
for val in (None,"0"):
    env={k:v for k,v in os.environ.items() if k!="VLLM_NEAR_CPU_OFFLOAD_HOSTALLOC"}
    if val: env["VLLM_NEAR_CPU_OFFLOAD_HOSTALLOC"]=val
    fake=FakeCudart(); calls=[]
    def fb(n,h=None): calls.append(n); return torch.zeros((n+mmap.PAGESIZE-1)//mmap.PAGESIZE*mmap.PAGESIZE,dtype=torch.int8)
    with mock.patch.dict(os.environ,env,clear=True), \
         mock.patch.object(w,"_get_cudart",side_effect=AssertionError("cudart touched")), \
         mock.patch.object(w,"alloc_host_buffer",side_effect=fb), \
         mock.patch.object(w,"pin_tensor") as pt:
        ww=mk(); ww._alloc_cpu_caches(gpu,True)
        assert calls==[need] and pt.call_count==1 and ww._hostalloc_ptr is None
        ww.shutdown()
print("flag-off OK: original path untouched")
# alloc failure
class Bad(FakeCudart):
    def cudaHostAlloc(self,*a): return 801
with mock.patch.object(w,"_get_cudart",return_value=Bad()):
    try: w._cuda_host_alloc(4096); raise SystemExit("no raise")
    except RuntimeError as e: assert "801" in str(e)
print("alloc-error OK\nALL PASS")
