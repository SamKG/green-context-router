#include <iostream>
#include <set>

__global__ void myKernel(int* out_smids) {
    int sm_id;
    asm("mov.u32 %0, %%smid;" : "=r"(sm_id));
    out_smids[blockIdx.x] = sm_id;
}

int main() {
    int num_blocks = 100;
    int* d_out;
    cudaMalloc(&d_out, num_blocks * sizeof(int));
    
    myKernel<<<num_blocks, 1>>>(d_out);
    cudaDeviceSynchronize();
    
    int* h_out = new int[num_blocks];
    cudaMemcpy(h_out, d_out, num_blocks * sizeof(int), cudaMemcpyDeviceToHost);
    
    std::set<int> unique_sms;
    for (int i = 0; i < num_blocks; i++) {
        unique_sms.insert(h_out[i]);
    }
    
    std::cout << "Kernel ran on " << unique_sms.size() << " unique SMs." << std::endl;
    
    cudaFree(d_out);
    delete[] h_out;
    return 0;
}
