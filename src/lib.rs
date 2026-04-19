#![allow(unsafe_op_in_unsafe_fn)]
use cuda_interposer::{cuda_hook, install_hooks};
use cuda_interposer_sys::driver_sys::cudaError_enum::CUDA_SUCCESS;
use cuda_interposer_sys::driver_sys::*;
use std::os::raw::c_uint;
use std::sync::OnceLock;

static INIT_TRACING: std::sync::Once = std::sync::Once::new();

fn init_tracing() {
    INIT_TRACING.call_once(|| {
        let filter = tracing_subscriber::EnvFilter::builder()
            .with_env_var("GREEN_CTX_TRACE")
            .from_env_lossy();
        let _ = tracing_subscriber::fmt()
            .with_env_filter(filter)
            .with_writer(std::io::stderr)
            .try_init();
    });
}

pub type CUstream = *mut std::ffi::c_void;

static POOL: OnceLock<Vec<(usize, i32)>> = OnceLock::new();

unsafe fn create_green_context_pair(
    dev: CUdevice,
    dev_res: &CUdevResource,
    sm_count: u32,
) -> Result<(CUcontext, CUcontext), (cudaError_enum, &'static str)> {
    let dev_sm = dev_res.__bindgen_anon_1.sm;
    let align = dev_sm.smCoscheduledAlignment;

    let mut params = [
        CU_DEV_SM_RESOURCE_GROUP_PARAMS_st {
            smCount: sm_count,
            coscheduledSmCount: align,
            preferredCoscheduledSmCount: align,
            flags: 0,
            reserved: [0; 12],
        },
        CU_DEV_SM_RESOURCE_GROUP_PARAMS_st {
            smCount: 0,
            coscheduledSmCount: align,
            preferredCoscheduledSmCount: align,
            flags: CUdevSmResourceGroup_flags::CU_DEV_SM_RESOURCE_GROUP_BACKFILL as u32,
            reserved: [0; 12],
        },
    ];

    let mut split_res: [CUdevResource; 2] = std::mem::zeroed();
    let mut remainder_res: CUdevResource = std::mem::zeroed();

    let res = cuDevSmResourceSplit(
        split_res.as_mut_ptr(),
        2,
        dev_res,
        &mut remainder_res,
        0,
        params.as_mut_ptr(),
    );
    if res != CUDA_SUCCESS {
        return Err((res, "cuDevSmResourceSplit"));
    }

    let mut desc1: CUdevResourceDesc = std::ptr::null_mut();
    let res = cuDevResourceGenerateDesc(&mut desc1, &mut split_res[0], 1);
    if res != CUDA_SUCCESS {
        return Err((res, "cuDevResourceGenerateDesc (1)"));
    }

    let mut desc2: CUdevResourceDesc = std::ptr::null_mut();
    let res = cuDevResourceGenerateDesc(&mut desc2, &mut split_res[1], 1);
    if res != CUDA_SUCCESS {
        return Err((res, "cuDevResourceGenerateDesc (2)"));
    }

    let mut gctx1: CUgreenCtx = std::ptr::null_mut();
    let res = cuGreenCtxCreate(&mut gctx1, desc1, dev, 1);
    if res != CUDA_SUCCESS {
        return Err((res, "cuGreenCtxCreate (1)"));
    }

    let mut gctx2: CUgreenCtx = std::ptr::null_mut();
    let res = cuGreenCtxCreate(&mut gctx2, desc2, dev, 1);
    if res != CUDA_SUCCESS {
        return Err((res, "cuGreenCtxCreate (2)"));
    }

    let mut new_ctx1: CUcontext = std::ptr::null_mut();
    let res = cuCtxFromGreenCtx(&mut new_ctx1, gctx1);
    if res != CUDA_SUCCESS {
        return Err((res, "cuCtxFromGreenCtx (1)"));
    }

    let mut new_ctx2: CUcontext = std::ptr::null_mut();
    let res = cuCtxFromGreenCtx(&mut new_ctx2, gctx2);
    if res != CUDA_SUCCESS {
        return Err((res, "cuCtxFromGreenCtx (2)"));
    }

    Ok((new_ctx1, new_ctx2))
}

fn get_green_contexts() -> &'static [(usize, i32)] {
    POOL.get_or_init(|| unsafe {
        let mut dev: CUdevice = 0;
        if cuCtxGetDevice(&mut dev) != CUDA_SUCCESS {
            return Vec::<(usize, i32)>::new();
        }

        let mut max_sms: std::os::raw::c_int = 0;
        if cuDeviceGetAttribute(
            &mut max_sms,
            CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT,
            dev,
        ) != CUDA_SUCCESS
        {
            return Vec::<(usize, i32)>::new();
        }

        let mut ctxs: Vec<(usize, i32)> = Vec::new();

        let mut current_ctx: CUcontext = std::ptr::null_mut();
        let _ = cuCtxGetCurrent(&mut current_ctx);

        let mut dev_res: CUdevResource = std::mem::zeroed();
        let res = cuDeviceGetDevResource(
            dev,
            &mut dev_res,
            CUdevResourceType::CU_DEV_RESOURCE_TYPE_SM,
        );
        if res != CUDA_SUCCESS {
            tracing::error!(
                "GREEN_CTX router: cuDeviceGetDevResource failed with error {:?}",
                res
            );
            return Vec::<(usize, i32)>::new();
        }

        let step = 8;
        for sm_count in (step..=max_sms / 2).step_by(step as usize) {
            match create_green_context_pair(dev, &dev_res, sm_count as u32) {
                Ok((ctx1, ctx2)) => {
                    tracing::info!(
                        "GREEN_CTX router: created green context pair ({}, {}) with SM counts ({}, {})",
                        ctxs.len(),
                        ctxs.len() + 1,
                        sm_count,
                        max_sms - sm_count
                    );
                    ctxs.push((ctx1 as usize, sm_count));
                    ctxs.push((ctx2 as usize, max_sms - sm_count));
                }
                Err((err, func)) => {
                    tracing::error!(
                        "GREEN_CTX router: {} for SM count {} failed {:?}",
                        func,
                        sm_count,
                        err
                    );
                    ctxs.push((0, 0));
                    ctxs.push((0, 0));
                }
            }
        }

        if !current_ctx.is_null() {
            let _ = cuCtxSetCurrent(current_ctx);
        }

        ctxs
    })
}

unsafe fn with_green_ctx<F, R>(hook_name: &str, f: F) -> R
where
    F: FnOnce() -> R,
{
    init_tracing();
    let pool = get_green_contexts();

    let mut target_ctx: usize = 0;

    if !pool.is_empty() {
        if let Ok(val) = std::env::var("GREEN_CTX") {
            if let Ok(idx) = val.parse::<usize>() {
                if idx < pool.len() && pool[idx].0 != 0 {
                    target_ctx = pool[idx].0;
                    tracing::info!(
                        "GREEN_CTX router: using green context {} (SMs: {}) [{}]",
                        idx,
                        pool[idx].1,
                        hook_name
                    );
                }
            }
        }
    }

    let mut old_ctx: CUcontext = std::ptr::null_mut();
    if target_ctx != 0 {
        unsafe {
            let _ = cuCtxGetCurrent(&mut old_ctx);
            let _ = cuCtxSetCurrent(target_ctx as CUcontext);
        }
    }

    let res = f();

    if target_ctx != 0 && !old_ctx.is_null() {
        unsafe {
            let _ = cuCtxSetCurrent(old_ctx);
        }
    }

    res
}

install_hooks!();

cuda_hook! {
    pub unsafe extern "C" fn cuInit(flags: c_uint) -> CUresult {
        init_tracing();
        tracing::debug!("GREEN_CTX router: cuInit intercepted");
        let real_fn = *__real_cuInit;
        real_fn(flags)
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
            with_green_ctx("cuLaunchKernel", || {
                let real_fn = *__real_cuLaunchKernel;
                real_fn(
                    f,
                    grid_dim_x,
                    grid_dim_y,
                    grid_dim_z,
                    block_dim_x,
                    block_dim_y,
                    block_dim_z,
                    shared_mem_bytes,
                    h_stream,
                    kernel_params,
                    extra,
                )
            })
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
            with_green_ctx("cuLaunchKernel_ptsz", || {
                let real_fn = *__real_cuLaunchKernel_ptsz;
                real_fn(
                    f,
                    grid_dim_x,
                    grid_dim_y,
                    grid_dim_z,
                    block_dim_x,
                    block_dim_y,
                    block_dim_z,
                    shared_mem_bytes,
                    h_stream,
                    kernel_params,
                    extra,
                )
            })
        }
    }
}

cuda_hook! {
    pub unsafe extern "C" fn cuLaunchKernelEx(
        config: *const CUlaunchConfig,
        f: CUfunction,
        kernel_params: *mut *mut std::ffi::c_void,
        extra: *mut *mut std::ffi::c_void
    ) -> CUresult {
        unsafe {
            init_tracing();
            let real_fn = *__real_cuLaunchKernelEx;
            real_fn(config, f, kernel_params, extra)
        }
    }
}

cuda_hook! {
    pub unsafe extern "C" fn cuLaunchKernelEx_ptsz(
        config: *const CUlaunchConfig,
        f: CUfunction,
        kernel_params: *mut *mut std::ffi::c_void,
        extra: *mut *mut std::ffi::c_void
    ) -> CUresult {
        unsafe {
            init_tracing();
            let real_fn = *__real_cuLaunchKernelEx_ptsz;
            real_fn(config, f, kernel_params, extra)
        }
    }
}

cuda_hook! {
    pub unsafe extern "C" fn cuStreamCreate(
        phStream: *mut CUstream,
        Flags: ::std::os::raw::c_uint
    ) -> CUresult {
        unsafe {
            with_green_ctx("cuStreamCreate", || {
                let real_fn = *__real_cuStreamCreate;
                real_fn(phStream, Flags)
            })
        }
    }
}

cuda_hook! {
    pub unsafe extern "C" fn cuStreamCreateWithPriority(
        phStream: *mut CUstream,
        flags: ::std::os::raw::c_uint,
        priority: ::std::os::raw::c_int
    ) -> CUresult {
        unsafe {
            with_green_ctx("cuStreamCreateWithPriority", || {
                let real_fn = *__real_cuStreamCreateWithPriority;
                real_fn(phStream, flags, priority)
            })
        }
    }
}
