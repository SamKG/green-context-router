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

static POOL: OnceLock<Vec<usize>> = OnceLock::new();

unsafe fn create_green_context(
    dev: CUdevice,
    dev_res: &CUdevResource,
    sm_count: u32,
) -> Result<CUcontext, (cudaError_enum, &'static str)> {
    let mut num_groups: std::os::raw::c_uint = 1;
    let mut split_res: CUdevResource = std::mem::zeroed();
    let mut remainder_res: CUdevResource = std::mem::zeroed();

    let res = cuDevSmResourceSplitByCount(
        &mut split_res,
        &mut num_groups,
        dev_res,
        &mut remainder_res,
        0,
        sm_count as std::os::raw::c_uint,
    );
    if res != CUDA_SUCCESS {
        return Err((res, "cuDevSmResourceSplitByCount"));
    }

    let mut desc: CUdevResourceDesc = std::ptr::null_mut();
    let res = cuDevResourceGenerateDesc(&mut desc, &mut split_res, 1);
    if res != CUDA_SUCCESS {
        return Err((res, "cuDevResourceGenerateDesc"));
    }

    let mut gctx: CUgreenCtx = std::ptr::null_mut();
    let res = cuGreenCtxCreate(
        &mut gctx, desc, dev, 1, /* CU_GREEN_CTX_DEFAULT_STREAM */
    );
    if res != CUDA_SUCCESS {
        return Err((res, "cuGreenCtxCreate"));
    }

    let mut new_ctx: CUcontext = std::ptr::null_mut();
    let res = cuCtxFromGreenCtx(&mut new_ctx, gctx);
    if res != CUDA_SUCCESS {
        return Err((res, "cuCtxFromGreenCtx"));
    }

    Ok(new_ctx)
}

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
            return vec![];
        }

        for sm_count in 1..=max_sms {
            match create_green_context(dev, &dev_res, sm_count as u32) {
                Ok(ctx) => {
                    tracing::info!(
                        "GREEN_CTX router: created green context {} with {} SMs",
                        sm_count - 1,
                        sm_count
                    );
                    ctxs.push(ctx as usize);
                }
                Err((err, func)) => {
                    tracing::error!(
                        "GREEN_CTX router: {} for SM count {} failed {:?}",
                        func,
                        sm_count,
                        err
                    );
                    ctxs.push(0);
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
                if idx < pool.len() && pool[idx] != 0 {
                    target_ctx = pool[idx];
                    tracing::info!(
                        "GREEN_CTX router: using green context {} (SMs: {}) [{}]",
                        idx,
                        idx + 1,
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
            with_green_ctx("cuLaunchKernelEx", || {
                let real_fn = *__real_cuLaunchKernelEx;
                real_fn(config, f, kernel_params, extra)
            })
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
            with_green_ctx("cuLaunchKernelEx_ptsz", || {
                let real_fn = *__real_cuLaunchKernelEx_ptsz;
                real_fn(config, f, kernel_params, extra)
            })
        }
    }
}
