use anyhow::{Context, Result, ensure};
use ryft::pjrt::protos::{CompilationOptions, ExecutableCompilationOptions, Precision};
use ryft::pjrt::{
    BufferType, ClientOptions, CpuClientOptions, ExecutionDeviceInputs, ExecutionInput, GpuClientOptions,
    GpuMemoryAllocator, Plugin, Program, load_cpu_plugin,
};

const PROGRAM_SOURCE: &str = r#"
module {
  func.func @main(%arg0: tensor<2x1xi32>, %arg1: tensor<2x1xi32>) -> tensor<2x1xi32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<2x1xi32>
    return %0 : tensor<2x1xi32>
  }
}
"#;

const LHS_INPUT: [i32; 2] = [7, -1];
const RHS_INPUT: [i32; 2] = [35, -41];
const EXPECTED_OUTPUT: [i32; 2] = [42, -42];

fn main() -> Result<()> {
    print_artifact_configuration();

    run_cpu_validation()?;
    run_gpu_validation()?;

    println!("All backend checks passed.");
    Ok(())
}

fn print_artifact_configuration() {
    let xla_archive = std::env::var("RYFT_XLA_SYS_ARCHIVE")
        .unwrap_or_else(|_| "/home/monday/ryft/ryft-xla-sys-linux-arm64-cpu.tar.gz".to_string());
    let plugin_archive = std::env::var("PJRT_PLUGIN_CUDA_13_LIB")
        .unwrap_or_else(|_| "/home/monday/ryft/pjrt-plugin-linux-arm64-cuda-13.tar.gz".to_string());

    println!("RYFT_XLA_SYS_ARCHIVE={xla_archive}");
    println!("PJRT_PLUGIN_CUDA_13_LIB={plugin_archive}");
}

fn run_cpu_validation() -> Result<()> {
    let plugin = load_cpu_plugin().context("failed to load CPU PJRT plugin")?;
    run_backend_validation("CPU", plugin, ClientOptions::CPU(CpuClientOptions { device_count: Some(1) }), "cpu", false)
}

#[cfg(feature = "cuda-13")]
fn run_gpu_validation() -> Result<()> {
    let plugin = ryft::pjrt::load_cuda_13_plugin().context("failed to load CUDA 13 PJRT plugin")?;
    run_backend_validation(
        "GPU",
        plugin,
        ClientOptions::GPU(GpuClientOptions {
            allocator: GpuMemoryAllocator::CudaAsync { memory_fraction_to_preallocate: None },
            ..Default::default()
        }),
        "cuda",
        true,
    )
}

#[cfg(not(feature = "cuda-13"))]
fn run_gpu_validation() -> Result<()> {
    bail!("the GPU validation step requires building this binary with `--features cuda-13`");
}

fn run_backend_validation(
    label: &str,
    plugin: Plugin,
    client_options: ClientOptions,
    expected_platform: &str,
    require_gpu_device: bool,
) -> Result<()> {
    println!("== {label} validation ==");

    let client = plugin.client(client_options).with_context(|| format!("failed to create the {label} client"))?;
    let platform_name = client.platform_name().context("failed to query platform name")?;
    let platform_version = client.platform_version().context("failed to query platform version")?;
    ensure!(
        platform_name.eq_ignore_ascii_case(expected_platform),
        "{label} client reported platform `{platform_name}` instead of `{expected_platform}`"
    );
    println!("Platform: {platform_name} ({platform_version})");

    let program = Program::Mlir { bytecode: PROGRAM_SOURCE.as_bytes().to_vec() };
    let executable = client
        .compile(&program, &compilation_options())
        .with_context(|| format!("failed to compile the embedded MLIR program on {label}"))?;
    let device = executable
        .addressable_devices()
        .context("failed to enumerate addressable devices for the executable")?
        .into_iter()
        .next()
        .with_context(|| format!("no addressable devices were returned for the {label} executable"))?;

    let device_id = device.id().context("failed to query device ID")?;
    let device_kind = device.kind().context("failed to query device kind")?;
    println!("Device: id={device_id}, kind={device_kind}");

    if require_gpu_device {
        let lowered_kind = device_kind.to_ascii_lowercase();
        ensure!(
            lowered_kind.contains("gpu") || lowered_kind.contains("cuda") || lowered_kind.contains("thor"),
            "{label} validation expected a GPU device but saw kind `{device_kind}`"
        );
    }

    let lhs_bytes = encode_i32s(&LHS_INPUT);
    let rhs_bytes = encode_i32s(&RHS_INPUT);
    let lhs_buffer = client
        .buffer(lhs_bytes.as_slice(), BufferType::I32, &[2, 1], None, device.clone(), None)
        .with_context(|| format!("failed to create the left-hand-side buffer on {label}"))?;
    let rhs_buffer = client
        .buffer(rhs_bytes.as_slice(), BufferType::I32, &[2, 1], None, device, None)
        .with_context(|| format!("failed to create the right-hand-side buffer on {label}"))?;

    let inputs = [
        ExecutionInput { buffer: lhs_buffer, donatable: false },
        ExecutionInput { buffer: rhs_buffer, donatable: false },
    ];
    let mut outputs = executable
        .execute(vec![ExecutionDeviceInputs { inputs: &inputs, ..Default::default() }], 0, None, None, None, None)
        .with_context(|| format!("failed to execute the embedded MLIR program on {label}"))?;
    let mut outputs = outputs.pop().with_context(|| format!("the {label} execution returned no device outputs"))?;
    outputs
        .done
        .r#await()
        .with_context(|| format!("the {label} execution did not complete successfully"))?;

    let output_bytes = outputs
        .outputs
        .pop()
        .with_context(|| format!("the {label} execution returned no result buffer"))?
        .copy_to_host(None)
        .context("failed to initiate copy of the result buffer back to the host")?
        .r#await()
        .with_context(|| format!("failed to copy the {label} result buffer back to the host"))?;
    let output_values = decode_i32s(&output_bytes)?;
    ensure!(
        output_values == EXPECTED_OUTPUT,
        "{label} execution produced {output_values:?}, expected {EXPECTED_OUTPUT:?}"
    );

    println!("Output: {output_values:?}");
    println!("{label} validation passed.");
    Ok(())
}

fn compilation_options() -> CompilationOptions {
    CompilationOptions {
        executable_build_options: Some(ExecutableCompilationOptions {
            device_ordinal: -1,
            replica_count: 1,
            partition_count: 1,
            ..Default::default()
        }),
        matrix_unit_operand_precision: Precision::Default as i32,
        ..Default::default()
    }
}

fn encode_i32s(values: &[i32]) -> Vec<u8> {
    values.iter().flat_map(|value| value.to_ne_bytes()).collect()
}

fn decode_i32s(bytes: &[u8]) -> Result<[i32; 2]> {
    ensure!(bytes.len() == 8, "expected 8 output bytes, received {}", bytes.len());

    let mut values = [0i32; 2];
    for (index, chunk) in bytes.chunks_exact(4).enumerate() {
        let mut value_bytes = [0u8; 4];
        value_bytes.copy_from_slice(chunk);
        values[index] = i32::from_ne_bytes(value_bytes);
    }
    Ok(values)
}
#[cfg(not(feature = "cuda-13"))]
use anyhow::bail;
