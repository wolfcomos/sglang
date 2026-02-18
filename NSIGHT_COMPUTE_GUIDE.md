# Nsight Compute Profiling Guide for MXFP8 Triton Kernel

This guide explains how to use NVIDIA Nsight Compute to profile the MXFP8 Triton kernel for detailed performance analysis.

## Prerequisites

1. **Install Nsight Compute**
   ```bash
   # Download from NVIDIA Developer website
   # Or install via package manager (if available)
   ```

2. **Verify Installation**
   ```bash
   ncu --version
   ```

3. **Ensure CUDA is available**
   ```bash
   nvidia-smi
   ```

## Quick Start

### Basic Profiling

```bash
# Full profile with all metrics
ncu --set full python profile_mxfp8_nsight.py

# Profile with default test case (M=128, N=256, K=512)
ncu --set full python profile_mxfp8_nsight.py --M 128 --N 256 --K 512
```

### Export Results

```bash
# Export to file (creates mxfp8_profile.ncu-rep)
ncu -o mxfp8_profile python profile_mxfp8_nsight.py

# View the report
ncu --import mxfp8_profile.ncu-rep
```

## Common Profiling Commands

### 1. Full Profile (Recommended for First Run)

```bash
ncu --set full python profile_mxfp8_nsight.py
```

This captures:
- Execution time
- Memory throughput
- Compute throughput
- Warp efficiency
- Occupancy
- And many more metrics

### 2. Focused Metrics

```bash
# Time and throughput
ncu --metrics gpu__time_duration.sum,sm__throughput.avg.pct_of_peak_sustained_elapsed \
    python profile_mxfp8_nsight.py

# Memory metrics
ncu --metrics \
    gpu__time_duration.sum,\
    dram__bytes_read.sum,\
    dram__bytes_write.sum,\
    l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum,\
    l1tex__t_bytes_pipe_lsu_mem_global_op_st.sum \
    python profile_mxfp8_nsight.py

# Compute metrics
ncu --metrics \
    gpu__time_duration.sum,\
    sm__throughput.avg.pct_of_peak_sustained_elapsed,\
    sm__sass_thread_inst_executed_op_fp16_pred_on.sum,\
    sm__sass_thread_inst_executed_op_fp32_pred_on.sum \
    python profile_mxfp8_nsight.py
```

### 3. Kernel-Specific Profiling

```bash
# Profile only matmul kernel (default)
ncu --set full python profile_mxfp8_nsight.py --profile-matmul

# Profile only quantization
ncu --set full python profile_mxfp8_nsight.py --profile-quantization

# Profile end-to-end function
ncu --set full python profile_mxfp8_nsight.py --profile-e2e

# Profile all components
ncu --set full python profile_mxfp8_nsight.py \
    --profile-quantization --profile-matmul --profile-e2e
```

## Understanding the Output

### Key Metrics to Look For

1. **Execution Time**
   - `gpu__time_duration.sum`: Total kernel execution time
   - Compare with baseline from profiling script

2. **Memory Throughput**
   - `dram__bytes_read.sum` / `dram__bytes_write.sum`: DRAM bandwidth
   - `l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum`: L1 cache reads
   - Look for memory-bound vs compute-bound behavior

3. **Compute Throughput**
   - `sm__throughput.avg.pct_of_peak_sustained_elapsed`: SM utilization
   - `sm__sass_thread_inst_executed_op_*.sum`: Instruction counts
   - Identify compute bottlenecks

4. **Efficiency Metrics**
   - `sm__warps_active.avg.pct_of_peak_sustained_active`: Warp efficiency
   - `launch__occupancy_limit_register`: Occupancy limits
   - `launch__occupancy_limit_shared_mem`: Shared memory limits

5. **Shared Memory**
   - `l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum`: Bank conflicts
   - `l1tex__t_sectors_pipe_lsu_mem_shared_op_ld.sum`: Shared memory usage

### Interpreting Results

**Good Performance Indicators:**
- High SM utilization (>80%)
- High warp efficiency (>90%)
- Memory throughput close to peak bandwidth
- Low shared memory bank conflicts

**Performance Issues to Look For:**
- Low SM utilization → Underutilized compute units
- Low warp efficiency → Divergent execution
- High memory latency → Memory-bound kernel
- High shared memory bank conflicts → Poor memory access patterns
- Low occupancy → Resource constraints (registers/shared memory)

## Advanced Usage

### 1. Compare Multiple Runs

```bash
# Profile baseline
ncu -o baseline_profile python profile_mxfp8_nsight.py

# Profile optimized version (after making changes)
ncu -o optimized_profile python profile_mxfp8_nsight.py

# Compare in Nsight Compute GUI
ncu --import baseline_profile.ncu-rep optimized_profile.ncu-rep
```

### 2. Kernel Filtering

```bash
# Profile only specific kernel name pattern
ncu --kernel-name-base "mxfp8" --set full python profile_mxfp8_nsight.py
```

### 3. Detailed Section Analysis

```bash
# Get detailed section breakdown
ncu --set full --section SpeedOfLight \
    python profile_mxfp8_nsight.py
```

### 4. Export to CSV

```bash
# Export metrics to CSV for analysis
ncu --csv --set full python profile_mxfp8_nsight.py > mxfp8_metrics.csv
```

## Optimization Workflow

1. **Baseline Profile**
   ```bash
   ncu -o baseline python profile_mxfp8_nsight.py --M 128 --N 256 --K 512
   ```

2. **Identify Bottlenecks**
   - Check SM utilization
   - Check memory throughput
   - Check shared memory usage
   - Check warp efficiency

3. **Make Optimizations**
   - Adjust block sizes
   - Optimize memory access patterns
   - Reduce shared memory usage
   - Improve occupancy

4. **Profile Optimized Version**
   ```bash
   ncu -o optimized python profile_mxfp8_nsight.py --M 128 --N 256 --K 512
   ```

5. **Compare Results**
   ```bash
   ncu --import baseline.ncu-rep optimized.ncu-rep
   ```

## Common Issues

### Issue: "No kernels found"
**Solution**: Ensure the kernel actually runs. Check that:
- CUDA is available
- Matrix dimensions are valid
- Block sizes are correct

### Issue: "Permission denied"
**Solution**: Nsight Compute may need elevated permissions:
```bash
sudo ncu --set full python profile_mxfp8_nsight.py
```

### Issue: "Out of memory"
**Solution**: Reduce number of iterations:
```bash
ncu --set full python profile_mxfp8_nsight.py --num-iterations 1
```

## Recommended Metrics for MXFP8 Optimization

For MXFP8 kernel optimization, focus on these metrics:

```bash
ncu --metrics \
    gpu__time_duration.sum,\
    sm__throughput.avg.pct_of_peak_sustained_elapsed,\
    sm__warps_active.avg.pct_of_peak_sustained_active,\
    dram__bytes_read.sum,\
    dram__bytes_write.sum,\
    l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum,\
    l1tex__t_bytes_pipe_lsu_mem_global_op_st.sum,\
    l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum,\
    launch__occupancy_limit_register,\
    launch__occupancy_limit_shared_mem \
    python profile_mxfp8_nsight.py
```

## Next Steps

1. Run baseline profile with full metrics
2. Analyze the report to identify bottlenecks
3. Make targeted optimizations based on findings
4. Re-profile and compare results
5. Iterate until performance targets are met

## References

- [Nsight Compute Documentation](https://docs.nvidia.com/nsight-compute/)
- [Nsight Compute Metrics Reference](https://docs.nvidia.com/nsight-compute/NsightComputeCli/index.html#metrics-reference)
