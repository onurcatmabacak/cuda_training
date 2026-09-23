// matmul_rust_cublas.rs
// Rust + CUDA Float64 DGEMM via direct FFI to the CUDA runtime and cuBLAS.
// No external crates are used (the machine has no crates.io access), so this
// links libcudart and libcublas directly.
//
// Modes (argv[1]):
//   cublas -> cublasDgemm
//   graph  -> cublasDgemm captured in a CUDA graph and replayed

use std::ffi::c_void;
use std::os::raw::{c_int, c_uint};

type CudaStream = *mut c_void;
type CudaEvent = *mut c_void;
type CudaGraph = *mut c_void;
type CudaGraphExec = *mut c_void;
type CublasHandle = *mut c_void;

const CUDA_MEMCPY_HOST_TO_DEVICE: c_int = 1;
const STREAM_CAPTURE_GLOBAL: c_int = 0;
const CUBLAS_OP_N: c_int = 0;

#[link(name = "cudart")]
extern "C" {
    fn cudaMalloc(dev: *mut *mut c_void, size: usize) -> c_int;
    fn cudaFree(dev: *mut c_void) -> c_int;
    fn cudaMemcpy(dst: *mut c_void, src: *const c_void, count: usize, kind: c_int) -> c_int;
    fn cudaStreamCreate(stream: *mut CudaStream) -> c_int;
    fn cudaStreamDestroy(stream: CudaStream) -> c_int;
    fn cudaStreamBeginCapture(stream: CudaStream, mode: c_int) -> c_int;
    fn cudaStreamEndCapture(stream: CudaStream, graph: *mut CudaGraph) -> c_int;
    fn cudaGraphInstantiateWithFlags(
        exec: *mut CudaGraphExec,
        graph: CudaGraph,
        flags: u64,
    ) -> c_int;
    fn cudaGraphLaunch(exec: CudaGraphExec, stream: CudaStream) -> c_int;
    fn cudaDeviceSynchronize() -> c_int;
    fn cudaEventCreate(ev: *mut CudaEvent) -> c_int;
    fn cudaEventRecord(ev: CudaEvent, stream: CudaStream) -> c_int;
    fn cudaEventSynchronize(ev: CudaEvent) -> c_int;
    fn cudaEventElapsedTime(ms: *mut f32, start: CudaEvent, stop: CudaEvent) -> c_int;
}

#[link(name = "cublas")]
extern "C" {
    #[link_name = "cublasCreate_v2"]
    fn cublasCreate(h: *mut CublasHandle) -> c_int;
    #[link_name = "cublasDestroy_v2"]
    fn cublasDestroy(h: CublasHandle) -> c_int;
    #[link_name = "cublasSetStream_v2"]
    fn cublasSetStream(h: CublasHandle, stream: CudaStream) -> c_int;
    #[link_name = "cublasDgemm_v2"]
    fn cublasDgemm(
        h: CublasHandle,
        transa: c_int,
        transb: c_int,
        m: c_int,
        n: c_int,
        k: c_int,
        alpha: *const f64,
        a: *const f64,
        lda: c_int,
        b: *const f64,
        ldb: c_int,
        beta: *const f64,
        c: *mut f64,
        ldc: c_int,
    ) -> c_int;
}

fn ck(rc: c_int, what: &str) {
    if rc != 0 {
        eprintln!("CUDA/cuBLAS error ({what}): rc={rc}");
        std::process::exit(1);
    }
}

fn env_usize(key: &str, default: usize) -> usize {
    std::env::var(key)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mode = args.get(1).map(String::as_str).unwrap_or("cublas");

    let n = env_usize("MATMUL_N", 1024);
    let runs = env_usize("MATMUL_RUNS", 10);
    let ne = n * n;
    let bytes = ne * std::mem::size_of::<f64>();
    let flop = 2.0 * (n as f64) * (n as f64) * (n as f64);
    let alpha: f64 = 1.0;
    let beta: f64 = 0.0;

    let mut ha = vec![0.0f64; ne];
    let mut hb = vec![0.0f64; ne];
    for i in 0..ne {
        ha[i] = ((i % 17) + 1) as f64 * 1e-3 + 1.0;
        hb[i] = ((i % 13) + 1) as f64 * 1e-3 + 2.0;
    }

    unsafe {
        let mut da: *mut c_void = std::ptr::null_mut();
        let mut db: *mut c_void = std::ptr::null_mut();
        let mut dc: *mut c_void = std::ptr::null_mut();
        ck(cudaMalloc(&mut da, bytes), "cudaMalloc A");
        ck(cudaMalloc(&mut db, bytes), "cudaMalloc B");
        ck(cudaMalloc(&mut dc, bytes), "cudaMalloc C");
        ck(
            cudaMemcpy(da, ha.as_ptr() as *const c_void, bytes, CUDA_MEMCPY_HOST_TO_DEVICE),
            "H2D A",
        );
        ck(
            cudaMemcpy(db, hb.as_ptr() as *const c_void, bytes, CUDA_MEMCPY_HOST_TO_DEVICE),
            "H2D B",
        );

        let mut stream: CudaStream = std::ptr::null_mut();
        ck(cudaStreamCreate(&mut stream), "stream");
        let mut handle: CublasHandle = std::ptr::null_mut();
        ck(cublasCreate(&mut handle), "cublasCreate");
        ck(cublasSetStream(handle, stream), "cublasSetStream");

        // cublasDgemm with N x N x N
        let gemm = |h: CublasHandle| {
            cublasDgemm(
                h, CUBLAS_OP_N, CUBLAS_OP_N, n as c_int, n as c_int, n as c_int, &alpha,
                da as *const f64, n as c_int, db as *const f64, n as c_int, &beta,
                dc as *mut f64, n as c_int,
            )
        };

        // CUDA graph capture (graph mode)
        let mut graph: CudaGraph = std::ptr::null_mut();
        let mut gexec: CudaGraphExec = std::ptr::null_mut();
        if mode == "graph" {
            ck(cudaStreamBeginCapture(stream, STREAM_CAPTURE_GLOBAL), "begin capture");
            ck(gemm(handle), "dgemm (capture)");
            ck(cudaStreamEndCapture(stream, &mut graph), "end capture");
            ck(cudaGraphInstantiateWithFlags(&mut gexec, graph, 0), "instantiate");
        }

        let launch = |stream: CudaStream| {
            if mode == "graph" {
                ck(cudaGraphLaunch(gexec, stream), "graph launch");
            } else {
                ck(gemm(handle), "dgemm");
            }
        };

        // Warm up until the GPU boosts its clocks (~2 s).
        let mut w0: CudaEvent = std::ptr::null_mut();
        let mut w1: CudaEvent = std::ptr::null_mut();
        ck(cudaEventCreate(&mut w0), "event");
        ck(cudaEventCreate(&mut w1), "event");
        ck(cudaEventRecord(w0, stream), "record");
        let mut wms: f32 = 0.0;
        loop {
            launch(stream);
            ck(cudaEventRecord(w1, stream), "record");
            ck(cudaEventSynchronize(w1), "sync");
            ck(cudaEventElapsedTime(&mut wms, w0, w1), "elapsed");
            if wms >= 2000.0 {
                break;
            }
        }

        let mut start: CudaEvent = std::ptr::null_mut();
        let mut stop: CudaEvent = std::ptr::null_mut();
        ck(cudaEventCreate(&mut start), "event");
        ck(cudaEventCreate(&mut stop), "event");

        let mut total_ms = 0.0f32;
        for _ in 0..runs {
            ck(cudaEventRecord(start, stream), "record");
            launch(stream);
            ck(cudaEventRecord(stop, stream), "record");
            ck(cudaEventSynchronize(stop), "sync");
            let mut ms: f32 = 0.0;
            ck(cudaEventElapsedTime(&mut ms, start, stop), "elapsed");
            total_ms += ms;
        }
        let avg_ms = total_ms / runs as f32;

        let label = if mode == "graph" {
            "CUDA-graph cublasDgemm"
        } else {
            "cuBLAS DGEMM"
        };
        println!("Rust + CUDA {} N={}", label, n);
        println!("Average kernel time: {:.3} ms", avg_ms);
        println!("Effective GFLOPS: {:.1}", flop / (avg_ms as f64 * 1e6));

        let _ = c_uint::default();

        if mode == "graph" {
            let _ = graph;
            let _ = gexec;
        }
        cublasDestroy(handle);
        cudaStreamDestroy(stream);
        cudaFree(da);
        cudaFree(db);
        cudaFree(dc);
    }
}
