# MettleQ Studio

MLX Quantum Benchmarking Suite for Apple Silicon.
License: Source code and binary distributions are covered by the MIT License.
See `LICENSE`, `LICENSE.md`, and `BINARY-LICENSE.txt`.
The codebase is cross-platform, but we currently provide [macOS binaries only](https://qneura.ai/apps.html).

## Quick Start

```bash
./install.sh
./bin/appctl up
```

Backend: http://localhost:8127
API docs: http://localhost:8127/docs

Control commands:
- `./bin/appctl up`
- `./bin/appctl down`
- `./bin/appctl status`
- `./bin/appctl logs backend`
- `./bin/appctl clean`

Build commands:
- `./scripts/build_flutter_app.sh --release`
- `./scripts/build_flutter_app.sh --debug`
- `./scripts/build_dmg.sh`

## MCP Integration

MCP server script: `bin/quantumstudio_mcp_server.py`

Example Claude Code config:

```json
{
  "mcpServers": {
    "quantumstudio": {
      "command": "python3",
      "args": ["/absolute/path/to/bin/quantumstudio_mcp_server.py", "--port", "8087"],
      "env": {
        "QUANTUMSTUDIO_BACKEND_URL": "http://127.0.0.1:8127"
      }
    }
  }
}
```

## License

Source code is licensed under the [MIT License](LICENSE).
Compiled binaries use the same terms, as recorded in
[BINARY-LICENSE.txt](BINARY-LICENSE.txt). The overview is in
[LICENSE.md](LICENSE.md).

## Unsigned DMG / Gatekeeper

As of February 21, 2026, the MettleQ Studio DMG may be distributed without Apple notarization.

1. Open the DMG and drag MettleQ to `Applications`.
2. Remove the quarantine attribute by running one of these commands in Terminal:
   ```bash
   # If installed to /Applications (system-wide):
   xattr -d com.apple.quarantine /Applications/MettleQ.app

   # If installed to ~/Applications (user-only):
   xattr -d com.apple.quarantine ~/Applications/MettleQ.app
   ```
3. In `Applications`, right-click `MettleQ.app` and select `Open`.
4. If macOS still blocks launch, go to `System Settings -> Privacy & Security -> Open Anyway`, then confirm with password or Touch ID.

## Screenshots

### MettleQ Studio UI

![MettleQ Studio Circuits QASM](assets/screenshots/quantumstudio-circuits-qasm-2026-05-06.jpeg)
![MettleQ Studio Screen 1](assets/screenshots/screen001.png)
![MettleQ Studio Screen 2](assets/screenshots/screen002.png)
![MettleQ Studio Screen 3](assets/screenshots/screen003.png)
![MettleQ Studio Screen 4](assets/screenshots/screen004.png)
![MettleQ Studio Screen 5](assets/screenshots/screen005.png)

### Key Benchmark Figures (Paper)

![All Benchmarks Comparison](assets/paper-figures/all_benchmarks_comparison.png)
![QFT Scaling](assets/paper-figures/qft_scaling.png)
![QAOA Scaling](assets/paper-figures/qaoa_scaling.png)
![VQE Scaling](assets/paper-figures/vqe_scaling.png)
![QCBM Scaling](assets/paper-figures/qcbm_scaling.png)
![Grover Scaling](assets/paper-figures/grover_scaling.png)
![Hamiltonian Simulation Scaling](assets/paper-figures/hamiltonian_simulation_scaling.png)
