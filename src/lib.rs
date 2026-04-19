use cuda_interposer::{cuda_hook, install_hooks};
use cuda_interposer_sys::driver_sys::cudaError_enum::CUDA_SUCCESS;
use cuda_interposer_sys::driver_sys::*;
use std::os::raw::c_uint;
use std::sync::OnceLock;

pub type CUstream = *mut std::ffi::c_void;

static POOL: OnceLock<Vec<usize>> = OnceLock::new();

fn get_green_contexts() -> &'static [usize] {
    POOL.get_or_init(|| unsafe {
        let mut dev: CUdevice = 0;
        if cuCtxGetDevice(&mut dev) != CUDA_SUCCESS {
            return vec![];
        }

        let mut max_sms: std::os::raw::c_int = 0;
        if cuDeviceGetAttribute(
            &mut max_sms,
            CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT,
            dev,
        ) != CUDA_SUCCESS
        {
            return vec![];
        }

        let mut ctxs = Vec::new();

        let mut current_ctx: CUcontext = std::ptr::null_mut();
        let _ = cuCtxGetCurrent(&mut current_ctx);
        
        let mut dev_res: CUdevResource = std::mem::zeroed();
        let res = cuDeviceGetDevResource(dev, &mut dev_res, CUdevResourceType::CU_DEV_RESOURCE_TYPE_SM);
        if res != CUDA_SUCCESS {
            println!("GREEN_CTX router: cuDeviceGetDevResource failed with error {:?}", res);
            return vec![];
        }

        for sm_count in 1..=max_sms {
            let mut num_groups: std::os::raw::c_uint = 1;
            let mut split_res: CUdevResource = std::mem::zeroed();
            let mut remainder_res: CUdevResource = std::mem::zeroed();

            let res = cuDevSmResourceSplitByCount(
                &mut split_res,
                &mut num_groups,
                &dev_res,
                &mut remainder_res,
                0,
                sm_count as std::os::raw::c_uint,
            );

            if res == CUDA_SUCCESS {
                let mut desc: CUdevResourceDesc = std::ptr::null_mut();
                let res = cuDevResourceGenerateDesc(&mut desc, &mut split_res, 1);
                if res == CUDA_SUCCESS {
                    let mut gctx: CUgreenCtx = std::ptr::null_mut();
                    let res = cuGreenCtxCreate(&mut gctx, desc, dev, 1 /* CU_GREEN_CTX_DEFAULT_STREAM */);
                    if res == CUDA_SUCCESS {
                        let mut new_ctx: CUcontext = std::ptr::null_mut();
                        let res = cuCtxFromGreenCtx(&mut new_ctx, gctx);
                        if res == CUDA_SUCCESS {
                            ctxs.push(new_ctx as usize);
                        } else {
                            println!("GREEN_CTX router: cuCtxFromGreenCtx for SM count {} failed {:?}", sm_count, res);
                            ctxs.push(0);
                        }
                    } else {
                        println!("GREEN_CTX router: cuGreenCtxCreate for SM count {} failed {:?}", sm_count, res);
                        ctxs.push(0);
                    }
                } else {
                    println!("GREEN_CTX router: cuDevResourceGenerateDesc for SM count {} failed {:?}", sm_count, res);
                    ctxs.push(0);
                }
            } else {
                println!("GREEN_CTX router: cuDevSmResourceSplitByCount for SM count {} failed {:?}", sm_count, res);
                ctxs.push(0);
            }
        }

        if !current_ctx.is_null() {
            let _ = cuCtxSetCurrent(current_ctx);
        }

        ctxs
    })
}

install_hooks!();

cuda_hook! {
    pub unsafe extern "C" fn cuInit(Flags: c_uint) -> CUresult {
        println!("GREEN_CTX router: cuInit intercepted");
        let real_fn = *__real_cuInit;
        real_fn(Flags)
    }
}

cuda_hook! {
    pub unsafe extern "C" fn cuLaunchKernel(
        f: CUfunction,
        grid_dim_x: c_uint,
        grid_dim_y: c_uint,
        grid_dim_z: c_uint,
        block_dim_x: c_uint,
        block_dim_y: c_uint,
        block_dim_z: c_uint,
        shared_mem_bytes: c_uint,
        h_stream: CUstream,
        kernel_params: *mut *mut std::ffi::c_void,
        extra: *mut *mut std::ffi::c_void
    ) -> CUresult {
        unsafe {
            let pool = get_green_contexts();

            let mut target_ctx: usize = 0;

            if !pool.is_empty() {
                if let Ok(val) = std::env::var("GREEN_CTX") {
                    if let Ok(idx) = val.parse::<usize>() {
                        if idx < pool.len() && pool[idx] != 0 {
                            target_ctx = pool[idx];
                            println!("GREEN_CTX router: using green context {} (SMs: {})", idx, idx + 1);
                        }
                    }
                }
            }
            let mut old_ctx: CUcontext = std::ptr::null_mut();
            if target_ctx != 0 {
                let _ = cuCtxGetCurrent(&mut old_ctx);
                let _ = cuCtxSetCurrent(target_ctx as CUcontext);
            }

            let real_fn = *__real_cuLaunchKernel;
            let res = real_fn(f, grid_dim_x, grid_dim_y, grid_dim_z, block_dim_x, block_dim_y, block_dim_z, shared_mem_bytes, h_stream, kernel_params, extra);

            if target_ctx != 0 && !old_ctx.is_null() {
                let _ = cuCtxSetCurrent(old_ctx);
            }

            res
        }
    }
}

cuda_hook! {
    pub unsafe extern "C" fn cuLaunchKernel_ptsz(
        f: CUfunction,
        grid_dim_x: c_uint,
        grid_dim_y: c_uint,
        grid_dim_z: c_uint,
        block_dim_x: c_uint,
        block_dim_y: c_uint,
        block_dim_z: c_uint,
        shared_mem_bytes: c_uint,
        h_stream: CUstream,
        kernel_params: *mut *mut std::ffi::c_void,
        extra: *mut *mut std::ffi::c_void
    ) -> CUresult {
        unsafe {
            let pool = get_green_contexts();
            
            let mut target_ctx: usize = 0;
            
            if !pool.is_empty() {
                if let Ok(val) = std::env::var("GREEN_CTX") {
                    if let Ok(idx) = val.parse::<usize>() {
                        if idx < pool.len() && pool[idx] != 0 {
                            target_ctx = pool[idx];
                            println!("GREEN_CTX router: using green context {} (SMs: {}) [ptsz]", idx, idx + 1);
                        }
                    }
                }
            }
            
            let mut old_ctx: CUcontext = std::ptr::null_mut();
            if target_ctx != 0 {
                let _ = cuCtxGetCurrent(&mut old_ctx);
                let _ = cuCtxSetCurrent(target_ctx as CUcontext);
            }
            
            let real_fn = *__real_cuLaunchKernel_ptsz;
            let res = real_fn(f, grid_dim_x, grid_dim_y, grid_dim_z, block_dim_x, block_dim_y, block_dim_z, shared_mem_bytes, h_stream, kernel_params, extra);
            
            if target_ctx != 0 && !old_ctx.is_null() {
                let _ = cuCtxSetCurrent(old_ctx);
            }
            
            res
        }
    }
}

cuda_hook! {
    pub unsafe extern "C" fn cuLaunchKernelEx(
        config: *const CUlaunchConfig,
        f: CUfunction,
        kernelParams: *mut *mut std::ffi::c_void,
        extra: *mut *mut std::ffi::c_void
    ) -> CUresult {
        unsafe {
            let pool = get_green_contexts();
            let mut target_ctx: usize = 0;
            if !pool.is_empty() {
                if let Ok(val) = std::env::var("GREEN_CTX") {
                    if let Ok(idx) = val.parse::<usize>() {
                        if idx < pool.len() && pool[idx] != 0 {
                            target_ctx = pool[idx];
                            println!("GREEN_CTX router: using green context {} (SMs: {}) [Ex]", idx, idx + 1);
                        }
                    }
                }
            }
            let mut old_ctx: CUcontext = std::ptr::null_mut();
            if target_ctx != 0 { let _ = cuCtxGetCurrent(&mut old_ctx); let _ = cuCtxSetCurrent(target_ctx as CUcontext); }
            let real_fn = *__real_cuLaunchKernelEx;
            let res = real_fn(config, f, kernelParams, extra);
            if target_ctx != 0 && !old_ctx.is_null() { let _ = cuCtxSetCurrent(old_ctx); }
            res
        }
    }
}

cuda_hook! {
    pub unsafe extern "C" fn cuLaunchKernelEx_ptsz(
        config: *const CUlaunchConfig,
        f: CUfunction,
        kernelParams: *mut *mut std::ffi::c_void,
        extra: *mut *mut std::ffi::c_void
    ) -> CUresult {
        unsafe {
            let pool = get_green_contexts();
            let mut target_ctx: usize = 0;
            if !pool.is_empty() {
                if let Ok(val) = std::env::var("GREEN_CTX") {
                    if let Ok(idx) = val.parse::<usize>() {
                        if idx < pool.len() && pool[idx] != 0 {
                            target_ctx = pool[idx];
                            println!("GREEN_CTX router: using green context {} (SMs: {}) [Ex_ptsz]", idx, idx + 1);
                        }
                    }
                }
            }
            let mut old_ctx: CUcontext = std::ptr::null_mut();
            if target_ctx != 0 { let _ = cuCtxGetCurrent(&mut old_ctx); let _ = cuCtxSetCurrent(target_ctx as CUcontext); }
            let real_fn = *__real_cuLaunchKernelEx_ptsz;
            let res = real_fn(config, f, kernelParams, extra);
            if target_ctx != 0 && !old_ctx.is_null() { let _ = cuCtxSetCurrent(old_ctx); }
            res
        }
    }
}
