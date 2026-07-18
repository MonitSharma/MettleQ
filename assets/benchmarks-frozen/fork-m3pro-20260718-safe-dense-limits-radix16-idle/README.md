# Safe dense-state capacity probe

This campaign runs each engine/width in a fresh monitored process and requests only analytic Z on qubit zero. It probes internal dense-state capacity without returning another giant host state.

Hardware: `{"architecture": "arm64", "chip": "Apple M3 Pro", "cpu_cores": "proc 12:6:6:0", "gpu": "Apple M3 Pro", "gpu_cores": "18", "memory": "36 GB", "metal_support": "spdisplays_metal4", "model_identifier": "Mac15,7", "model_name": "MacBook Pro", "physical_memory_bytes": 38654705664}`

| Qubits | Aer CPU ms | MettleQ GPU ms | Aer / MettleQ | Absolute Z0 error |
| ---: | ---: | ---: | ---: | ---: |
| 27 | 2388.091 | 696.868 | **3.427x** | `6.76e-10` |
| 28 | 4696.781 | 1088.176 | **4.316x** | `6.76e-10` |
| 29 | 7994.548 | 2569.355 | **3.111x** | `2.99e-07` |

A ratio above 1.0x favors MettleQ. Widths shown as safety refusals in the plot were never launched.

The default policy caps each engine's projected peak at 45% of physical memory, requires 6 GiB available headroom before launch, terminates a worker if available memory falls below 5 GiB, disables the MLX free cache, and imposes a per-process timeout. Passing is a capacity result for this shallow structured circuit and scalar-output contract, not a guarantee for arbitrary depth or full-state return.

Protocol: `{"critical_runtime_headroom_gib": 5.0, "depth": 1, "maximum_memory_fraction": 0.45, "minimum_headroom_gib": 6.0, "output_contract": "analytic Z on qubit zero; no dense host return", "process_isolation": true, "timeout_seconds": 600.0, "widths": [27, 28, 29, 30]}`
