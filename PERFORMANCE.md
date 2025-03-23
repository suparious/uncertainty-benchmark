# Performance Optimization Guide

This document provides details on how to optimize the performance of the LLM Uncertainty Benchmark, particularly when using the parallel implementation.

## Parallel Implementation Overview

The parallel implementation in `examples_parallel.py` offers significant performance improvements over the sequential implementation by:

1. Processing multiple samples concurrently using batched API requests
2. Parallelizing across different tasks and prompt strategies
3. Utilizing either asynchronous I/O or thread-based concurrency
4. Providing configurable parameters to adjust to your infrastructure

## Key Performance Parameters

### Batch Size (`--batch-size`)

Controls how many samples are processed in a single batch:

- **Higher values** increase throughput but consume more memory
- **Lower values** reduce memory usage but may increase overall processing time
- **Recommended range**: 10-50 depending on your hardware
- **Default**: 10

### Worker Count (`--max-workers`)

Controls how many parallel workers (threads or async tasks) are used:

- **Higher values** allow more concurrent API requests
- **Lower values** reduce resource usage but increase processing time
- **Optimal value** depends on:
  - Your API endpoint's capacity and rate limits
  - Network bandwidth
  - CPU availability
- **Recommended range**: 4-16 depending on your server
- **Default**: 5

### Parallelization Model

Two parallelization models are available:

- **Async I/O** (default): 
  - Better for I/O-bound operations like API calls
  - More efficient for large numbers of concurrent requests
  - Preferred for most scenarios
  
- **Thread-based** (enabled with `--use-threads`):
  - May work better in some environments
  - Easier to debug
  - More compatible with some APIs

## Example Configurations

### High-Performance Server

For a powerful server with good network connection and a robust API endpoint:

```bash
python examples_parallel.py --api-base http://yourserver.example.com/v1 \
  --batch-size 50 --max-workers 16 \
  single --model your-model-name
```

### Balanced Configuration

Balances throughput and resource usage:

```bash
python examples_parallel.py --api-base http://yourserver.example.com/v1 \
  --batch-size 20 --max-workers 8 \
  single --model your-model-name
```

### Resource-Constrained Environment

For systems with limited memory or CPU resources:

```bash
python examples_parallel.py --api-base http://yourserver.example.com/v1 \
  --batch-size 10 --max-workers 4 --use-threads \
  single --model your-model-name
```

### Rate-Limited API

If your API endpoint has rate limits:

```bash
python examples_parallel.py --api-base http://yourserver.example.com/v1 \
  --batch-size 5 --max-workers 2 \
  single --model your-model-name
```

## Performance Troubleshooting

### API Rate Limiting

If you encounter rate limiting or "Too many requests" errors:

1. Reduce the number of workers (`--max-workers`)
2. Consider using thread-based parallelism (`--use-threads`)
3. Add a delay parameter in the code for more aggressive throttling

### Memory Issues

If you encounter memory-related errors:

1. Reduce the batch size (`--batch-size`)
2. Process fewer samples by reducing `--sample-size`
3. Run one task at a time by modifying the code

### Network Timeouts

If you experience connection timeouts:

1. Increase the timeout in the `ParallelProcessor` or `ThreadedProcessor` class
2. Ensure your network connection to the API endpoint is stable
3. Consider using thread-based parallelism which may be more reliable for unstable connections

### High CPU Usage

If the benchmark is consuming too much CPU:

1. Reduce the number of workers
2. Switch to async-based parallelism which is generally more CPU-efficient
3. Reduce the batch size

## Implementation Details

The parallel implementation uses the following pattern:

1. **Hierarchical Parallelism**:
   - Top level: Tasks can run in parallel
   - Middle level: Prompt strategies can run in parallel
   - Bottom level: Samples can run in parallel in batches

2. **Batch Processing**:
   - Samples are grouped into batches
   - Each batch is processed by a single worker
   - Results are collected and merged

3. **Error Handling**:
   - Each level has independent error handling
   - Failures in individual samples don't affect the entire batch
   - Exponential backoff for API request retries

4. **Progress Tracking**:
   - Progress bars at multiple levels
   - Detailed logging of progress and errors
   - Timing information for performance analysis

## Advanced Customization

For further performance optimizations, you can modify the code to:

1. **Implement caching** to avoid re-processing the same prompts
2. **Add persistent connection pools** in the async client
3. **Implement adaptive batch sizing** based on server response times
4. **Add distributed processing** across multiple machines
5. **Optimize memory usage** by streaming results rather than collecting them all

Refer to the comments in `parallel_utils.py` for more details on these advanced options.
