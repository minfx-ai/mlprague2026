import nvtx
import torch

DIM_M = 1024
DIM_K = 1024
DIM_N = 1024


@nvtx.annotate("main")
def main():
    with nvtx.annotate("alloc"):
        A = torch.randn(DIM_M, DIM_K, device="cuda", dtype=torch.float16)

    for i in range(10):
        with nvtx.annotate("init", domain="matmul", payload=i, color="red"):
            B = torch.randn(DIM_K, DIM_N, dtype=torch.float16)

        with nvtx.annotate("move", domain="matmul", payload=i, color="green"):
            B = B.to(device="cuda")

        with nvtx.annotate("step", domain="matmul", payload=i, color="blue"):
            _ = torch.matmul(A, B)


if __name__ == "__main__":
    main()
    print("Done")

