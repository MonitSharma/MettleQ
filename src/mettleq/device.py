from typing import List, Dict, Any, Optional

import os as _os
import time as _time
from .sim import StateVectorSimulator
from .mps_state import MPSState, MPSOptions
from .gates import I, H, X, Y, Z, S, SDG, T, TDG, SX, RX, RY, RZ, U1, U2, U3, SWAP, iSWAP, CNOT, CZ, CPHASE, CRX, CRY, CRZ, Toffoli, Fredkin, CH
import math as _math
from ._mlx_compat import mx

from .execution import (build_execution_plan, mark_graph_built,
                        mark_synchronized, metal_checkpoint_policy,
                        metal_memory_snapshot, metal_runtime_enabled,
                        record_custom_dispatch,
                        record_evaluation_checkpoint,
                        state_memory_estimate)

# Constant (parameterless) gate matrices built once at import; read-only thereafter
_CONST_GATES = {
    "I": I(), "H": H(), "X": X(), "Y": Y(), "Z": Z(), "S": S(), "SDG": SDG(),
    "T": T(), "TDG": TDG(), "SX": SX(),
    "CNOT": CNOT(), "CH": CH(), "CZ": CZ(), "SWAP": SWAP(), "ISWAP": iSWAP(),
    "TOFFOLI": Toffoli(), "FREDKIN": Fredkin(),
}

# Diagonals (1-D) for phase-type gates; applied via broadcast multiply on SV backend
_CONST_DIAGS = {
    "Z": mx.array([1+0j, -1+0j], mx.complex64),
    "S": mx.array([1+0j, 1j], mx.complex64),
    "SDG": mx.array([1+0j, -1j], mx.complex64),
    "T": mx.array([1+0j, complex(_math.cos(_math.pi/4), _math.sin(_math.pi/4))], mx.complex64),
    "TDG": mx.array([1+0j, complex(_math.cos(_math.pi/4), -_math.sin(_math.pi/4))], mx.complex64),
    "CZ": mx.array([1+0j, 1+0j, 1+0j, -1+0j], mx.complex64),
}


def _phase_diag(phi: float) -> mx.array:
    return mx.array([1+0j, complex(_math.cos(phi), _math.sin(phi))], mx.complex64)


def _rz_diag(theta: float) -> mx.array:
    ht = theta / 2.0
    return mx.array([complex(_math.cos(ht), -_math.sin(ht)),
                     complex(_math.cos(ht), _math.sin(ht))], mx.complex64)


def _pauli_pair_phase(name: str, theta: float) -> mx.array:
    """Dense 4x4 exp(-i*theta*P⊗P) fallback for backends without fast paths."""
    c = complex(_math.cos(theta), 0.0)
    ms = complex(0.0, -_math.sin(theta))
    z = 0+0j
    if name == "ZZPHASE":
        em = complex(_math.cos(theta), -_math.sin(theta))
        ep = complex(_math.cos(theta), _math.sin(theta))
        return mx.array([[em, z, z, z], [z, ep, z, z], [z, z, ep, z], [z, z, z, em]], mx.complex64)
    if name == "XXPHASE":
        return mx.array([[c, z, z, ms], [z, c, ms, z], [z, ms, c, z], [ms, z, z, c]], mx.complex64)
    if name == "YYPHASE":
        return mx.array([[c, z, z, -ms], [z, c, ms, z], [z, ms, c, z], [-ms, z, z, c]], mx.complex64)
    raise ValueError(f"Unknown Pauli-pair phase op: {name}")


class Device:
    def __init__(
        self,
        wires: int,
        shots: int = 1000,
        backend: Optional[str] = None,
        mps_opts: Optional[MPSOptions] = None,
        *,
        metal_checkpoint_budget_bytes: Optional[int] = None,
        allow_unsafe_statevector: Optional[bool] = None,
        execution_device: Optional[str] = None,
    ):
        self.wires = int(wires)
        self.shots = int(shots)
        self._allow_unsafe_statevector = allow_unsafe_statevector
        if backend is None:
            backend = _os.environ.get('METTLEQ_BACKEND', 'sv').lower()
        backend = str(backend).strip().lower()
        if backend == "mps":
            if execution_device is not None and str(execution_device).strip().lower() == "gpu":
                raise ValueError("MPS is CPU-only; use execution_device='cpu' or None")
            self._mx_device = mx.cpu
        elif execution_device is None:
            self._mx_device = mx.default_device()
        else:
            normalized_device = str(execution_device).strip().lower()
            if normalized_device not in ("cpu", "gpu"):
                raise ValueError("execution_device must be 'cpu', 'gpu', or None")
            if normalized_device == "gpu" and not mx.metal.is_available():
                raise RuntimeError(
                    "Apple GPU execution was requested but Metal is unavailable"
                )
            self._mx_device = mx.cpu if normalized_device == "cpu" else mx.gpu
        self.execution_device = "cpu" if self._mx_device == mx.cpu else "gpu"
        with mx.stream(self._mx_device):
            if backend == 'mps':
                self.backend = 'mps'
                # Read MPS options from env if not provided
                if mps_opts is None:
                    try:
                        dmax = int(_os.environ.get('METTLEQ_MPS_DMAX', '64'))
                    except Exception:
                        dmax = 64
                    try:
                        eps = float(_os.environ.get('METTLEQ_MPS_EPS', '1e-10'))
                    except Exception:
                        eps = 1e-10
                    mps_opts = MPSOptions(dmax=dmax, eps=eps)
                self.sim = MPSState(self.wires, mps_opts)
            else:
                self.backend = 'sv'
                self.sim = StateVectorSimulator(
                    self.wires,
                    allow_unsafe_statevector=allow_unsafe_statevector,
                )
        self.statevector_preflight = getattr(self.sim, "preflight", None)
        if metal_checkpoint_budget_bytes is not None:
            metal_checkpoint_policy(metal_checkpoint_budget_bytes)
        self._metal_checkpoint_budget_bytes = metal_checkpoint_budget_bytes
        self._pending_custom_passes = 0
        self._pending_custom_io_bytes = 0
        self.last_execution_plan: Optional[Dict[str, Any]] = None

    def reset(self):
        with mx.stream(self._mx_device):
            self.sim.reset()
        self._pending_custom_passes = 0
        self._pending_custom_io_bytes = 0

    def execute(self, operations: List[Dict[str, Any]], *, report: Optional[bool] = None):
        with mx.stream(self._mx_device):
            return self._execute(operations, report=report)

    def _execute(self, operations: List[Dict[str, Any]], *, report: Optional[bool] = None):
        # Optional ASCII dump for any executed circuit (controlled via env)
        try:
            import os as _os
            ascii_on = _os.environ.get('METTLEQ_PRINT_ASCII', '0') == '1'
            # Avoid flooding: cap by qubit count (default 8) unless overridden
            try:
                max_q = int(_os.environ.get('METTLEQ_PRINT_ASCII_MAX_QUBITS', '8'))
            except Exception:
                max_q = 8
            if ascii_on and self.wires <= max_q:
                from .draw import circuit_ascii as _ascii  # lazy import to avoid cycles
                from .pretty import console as _console
                try:
                    _console.print("\n[dim]ASCII circuit:[/dim]\n" + _ascii(self.wires, operations))
                except Exception:
                    pass
        except Exception:
            pass
        checkpoint_policy_data = metal_checkpoint_policy(
            self._metal_checkpoint_budget_bytes
        )
        optimized_operations = self._fuse_zz_layers(operations)
        if self.backend == "mps" and hasattr(self.sim, "prepare_routing"):
            self.sim.prepare_routing(
                [
                    tuple(operation.get("wires", []))
                    for operation in optimized_operations
                    if (
                        len(operation.get("wires", [])) == 2
                        and str(operation.get("name", "")).upper() != "SWAP"
                    )
                ]
            )
        if report is None:
            report = _os.environ.get("METTLEQ_EXECUTION_REPORT", "0") == "1"
        if report:
            self.last_execution_plan = build_execution_plan(
                n_qubits=self.wires,
                backend=self.backend,
                operations=list(operations),
                optimized_operations=optimized_operations,
                checkpoint_policy_data=checkpoint_policy_data,
                statevector_preflight_data=self.statevector_preflight,
                initial_pending_custom_passes=self._pending_custom_passes,
                initial_pending_custom_io_bytes=self._pending_custom_io_bytes,
            )
        else:
            self.last_execution_plan = None

        if self.last_execution_plan is not None:
            checkpoint_enabled = bool(
                self.last_execution_plan["checkpointing"]["enabled"]
            )
        else:
            checkpoint_enabled = bool(
                checkpoint_policy_data["configured"]
                and self.backend == "sv"
                and hasattr(self.sim, "state")
                and metal_runtime_enabled(self.wires, backend=self.backend)
            )
        if not checkpoint_enabled:
            self._pending_custom_passes = 0
            self._pending_custom_io_bytes = 0
        checkpoint_budget = int(
            checkpoint_policy_data.get("budget_bytes") or 0
        )
        checkpoint_bytes_per_pass = (
            int(state_memory_estimate(self.wires)[
                "minimum_input_plus_output_bytes"
            ] or 0)
            if checkpoint_enabled else 0
        )
        max_pending_passes = (
            max(1, checkpoint_budget // checkpoint_bytes_per_pass)
            if checkpoint_enabled and checkpoint_bytes_per_pass > 0 else None
        )

        for optimized_index, op in enumerate(optimized_operations):
            dispatch = None
            if self.last_execution_plan is not None or checkpoint_enabled:
                dispatch = record_custom_dispatch(
                    self.last_execution_plan,
                    op,
                    n_qubits=self.wires,
                    optimized_operation_index=optimized_index,
                )
            layer_passes = (
                int(dispatch["expected_metal_launches"])
                if checkpoint_enabled and dispatch is not None else 0
            )
            layer_io_bytes = checkpoint_bytes_per_pass * layer_passes
            if (checkpoint_enabled and self._pending_custom_io_bytes > 0
                    and self._pending_custom_io_bytes + layer_io_bytes
                    > checkpoint_budget):
                self._evaluate_pending_custom_graph(
                    boundary="before_optimized_operation",
                    optimized_operation_index=optimized_index,
                    reason="next_fused_layer_would_exceed_budget",
                    adjacent_dispatch=dispatch,
                )

            streamed_layer_passes = 0
            observed_layer_passes = 0
            pending_stream_chunk_passes = 0
            launch_observer = None
            if (checkpoint_enabled and dispatch is not None
                    and dispatch.get("supports_intra_layer_streaming", False)
                    and max_pending_passes is not None
                    and layer_passes > max_pending_passes):
                def _observe_launch(state):
                    nonlocal streamed_layer_passes
                    nonlocal observed_layer_passes
                    nonlocal pending_stream_chunk_passes
                    observed_layer_passes += 1
                    pending_stream_chunk_passes += 1
                    if (pending_stream_chunk_passes >= max_pending_passes
                            and observed_layer_passes < layer_passes):
                        chunk_passes = pending_stream_chunk_passes
                        self._evaluate_streamed_custom_chunk(
                            state,
                            optimized_operation_index=optimized_index,
                            dispatch=dispatch,
                            passes=chunk_passes,
                            io_bytes=(
                                chunk_passes * checkpoint_bytes_per_pass
                            ),
                            within_layer_launch_index=observed_layer_passes,
                        )
                        streamed_layer_passes += chunk_passes
                        pending_stream_chunk_passes = 0

                launch_observer = _observe_launch

            op_name = op.get("name")
            if op_name == "_ZZLAYER":
                self.sim.apply_zz_layer(op["theta"], op["bonds"])
            elif op_name == "_RXLAYER":
                from . import shaders as metal_kernels
                # The original RX-specialized path fused two qubits per
                # traversal.  The calibrated generic radix-16 kernel keeps
                # sixteen amplitudes in registers and applies RX to four
                # adjacent qubits per traversal.  On M3 Pro this is 1.6-3.1x
                # faster for 20-26q RX layers and halves the dominant launch
                # count in QAOA/TFIM schedules.
                rx_radix16 = _os.environ.get(
                    "METTLEQ_RX_RADIX16", "1"
                ).strip().lower() not in {"0", "false", "off", "no"}
                if rx_radix16:
                    gate = self._dense_gate_for("RX", [op["theta"]])
                    u = mx.reshape(gate.astype(mx.complex64), (4,))
                    self.sim.state = metal_kernels.u2_layer_all(
                        self.sim.state, self.sim.n, u,
                        on_launch=launch_observer)
                else:
                    self.sim.state = metal_kernels.rx_layer_all(
                        self.sim.state, self.sim.n, op["theta"],
                        on_launch=launch_observer)
            elif op_name == "_CHAINPHASE_RXLAYER":
                from . import shaders as metal_kernels
                gate = self._dense_gate_for("RX", [op["rx_theta"]])
                u = mx.reshape(gate.astype(mx.complex64), (4,))
                self.sim.state = metal_kernels.chain_phase_u2_layer_all(
                    self.sim.state,
                    self.sim.n,
                    u,
                    kind=op["phase_kind"],
                    theta=op["theta"],
                    n_bonds=op["n_bonds"],
                    edge_mask=op["edge_mask"],
                    wrap_mask=op["wrap_mask"],
                    on_launch=launch_observer,
                )
            elif op_name == "_U2LAYER":
                from . import shaders as metal_kernels
                gate = self._dense_gate_for(op["gate"], op["params"])
                u = mx.reshape(gate.astype(mx.complex64), (4,))
                self.sim.state = metal_kernels.u2_layer_all(
                    self.sim.state, self.sim.n, u,
                    on_launch=launch_observer)
            elif op_name == "_DIAGLAYER":
                from . import shaders as metal_kernels
                self.sim.state = metal_kernels.diag_pair_layer(
                    self.sim.state, op["theta"], self.sim.n, op["bonds"])
            elif op_name == "_DIAGWEIGHTED":
                from . import shaders as metal_kernels
                self.sim.state = metal_kernels.diag_weighted_layer(
                    self.sim.state, self.sim.n, op["bonds"], op["thetas"])
            elif op_name == "_ZZWEIGHTED":
                from . import shaders as metal_kernels
                self.sim.state = metal_kernels.zz_weighted_layer(
                    self.sim.state, self.sim.n, op["bonds"], op["thetas"])
            elif op_name == "_QFTSTAGE":
                from . import shaders as metal_kernels
                self.sim.state = metal_kernels.qft_stage_sub(
                    self.sim.state, self.sim.n, op["j"], op["m"],
                    inverse=op["inverse"])
            elif op_name == "_QFTLAYER":
                from . import shaders as metal_kernels
                self.sim.state = metal_kernels.qft_stage_all(
                    self.sim.state, self.sim.n)
            elif op_name in ("_XXLAYER", "_YYLAYER"):
                from . import shaders as metal_kernels
                fn = (metal_kernels.xx_layer if op_name == "_XXLAYER"
                      else metal_kernels.yy_layer)
                self.sim.state = fn(self.sim.state, op["theta"],
                                    self.sim.n, op["bonds"],
                                    on_launch=launch_observer)
            elif op_name == "_XORLAYER":
                from . import shaders as metal_kernels
                self.sim.state = metal_kernels.xor_affine_gather(
                    self.sim.state, self.sim.n, op["rows"], op["c"])
            elif op_name == "_U2LISTLAYER":
                from . import shaders as metal_kernels
                self.sim.state = metal_kernels.u2_list_layer_all(
                    self.sim.state, self.sim.n, op["mats"],
                    active=op.get("active"), on_launch=launch_observer)
            elif op_name == "_CULISTLAYER":
                from . import shaders as metal_kernels
                self.sim.state = metal_kernels.controlled_u_groups(
                    self.sim.state, self.sim.n, op["groups"],
                    on_launch=launch_observer)
            else:
                name = str(op_name or "").upper()
                wires = list(op.get("wires", []))
                params = list(op.get("parameters", []))
                lookahead = []
                if self.backend == "mps" and len(wires) == 2:
                    limit = int(getattr(self.sim.opts, "routing_lookahead", 0))
                    if limit > 0:
                        for future in optimized_operations[optimized_index + 1:]:
                            future_wires = list(future.get("wires", []))
                            if len(future_wires) == 2:
                                lookahead.append(tuple(future_wires))
                                if len(lookahead) >= limit:
                                    break
                self._apply(name, wires, params, mps_lookahead=lookahead)

            if checkpoint_enabled and dispatch is not None:
                if (launch_observer is not None
                        and observed_layer_passes != layer_passes):
                    raise RuntimeError(
                        "intra-layer Metal launch accounting mismatch: "
                        f"expected {layer_passes}, observed "
                        f"{observed_layer_passes}"
                    )
                remaining_layer_passes = (
                    layer_passes - streamed_layer_passes
                )
                self._pending_custom_passes += remaining_layer_passes
                self._pending_custom_io_bytes += (
                    remaining_layer_passes * checkpoint_bytes_per_pass
                )
                if self._pending_custom_io_bytes > checkpoint_budget:
                    reason = (
                        "single_custom_launch_exceeds_budget"
                        if (remaining_layer_passes == 1
                            and checkpoint_bytes_per_pass > checkpoint_budget)
                        else "single_fused_layer_exceeds_budget"
                    )
                    self._evaluate_pending_custom_graph(
                        boundary="after_optimized_operation",
                        optimized_operation_index=optimized_index,
                        reason=reason,
                        adjacent_dispatch=dispatch,
                    )
        if self.last_execution_plan is not None:
            mark_graph_built(
                self.last_execution_plan,
                pending_custom_passes=self._pending_custom_passes,
                pending_custom_io_bytes=self._pending_custom_io_bytes,
            )
        # Return state vector only for SV; MPS has no .state
        return getattr(self.sim, 'state', None)

    def _evaluate_streamed_custom_chunk(
        self,
        state,
        *,
        optimized_operation_index: int,
        dispatch: Dict[str, Any],
        passes: int,
        io_bytes: int,
        within_layer_launch_index: int,
    ) -> None:
        """Evaluate a safe chunk between launches of one fused layer."""
        if passes <= 0:
            return
        if self.last_execution_plan is None:
            mx.eval(state)
            return
        memory_before = metal_memory_snapshot()
        start = _time.perf_counter_ns()
        mx.eval(state)
        elapsed_ms = (_time.perf_counter_ns() - start) / 1e6
        memory_after = metal_memory_snapshot()
        event = {
            "boundary": "within_optimized_operation",
            "optimized_operation_index": optimized_operation_index,
            "within_layer_launch_index": within_layer_launch_index,
            "within_layer_launch_count": int(
                dispatch["expected_metal_launches"]
            ),
            "reason": "multi_launch_layer_streaming_budget",
            "adjacent_kernel_family": dispatch.get("kernel_family"),
            "adjacent_expected_metal_launches": int(
                dispatch["expected_metal_launches"]
            ),
            "estimated_passes_evaluated": passes,
            "estimated_input_output_bytes_evaluated": io_bytes,
            "evaluation_ms": elapsed_ms,
            "metal_memory_before": memory_before,
            "metal_memory_after": memory_after,
        }
        record_evaluation_checkpoint(self.last_execution_plan, event)

    def _evaluate_pending_custom_graph(
        self,
        *,
        boundary: str,
        optimized_operation_index: int,
        reason: str,
        adjacent_dispatch: Optional[Dict[str, Any]],
    ) -> None:
        """Synchronize a pending graph at an optimized-operation boundary."""
        state = getattr(self.sim, "state", None)
        if state is None or self._pending_custom_passes <= 0:
            return
        if self.last_execution_plan is None:
            mx.eval(state)
            self._pending_custom_passes = 0
            self._pending_custom_io_bytes = 0
            return
        passes = self._pending_custom_passes
        io_bytes = self._pending_custom_io_bytes
        memory_before = metal_memory_snapshot()
        start = _time.perf_counter_ns()
        mx.eval(state)
        elapsed_ms = (_time.perf_counter_ns() - start) / 1e6
        memory_after = metal_memory_snapshot()
        event = {
            "boundary": boundary,
            "optimized_operation_index": optimized_operation_index,
            "reason": reason,
            "adjacent_kernel_family": (
                adjacent_dispatch.get("kernel_family")
                if adjacent_dispatch is not None else None
            ),
            "adjacent_expected_metal_launches": (
                int(adjacent_dispatch["expected_metal_launches"])
                if adjacent_dispatch is not None else None
            ),
            "estimated_passes_evaluated": passes,
            "estimated_input_output_bytes_evaluated": io_bytes,
            "evaluation_ms": elapsed_ms,
            "metal_memory_before": memory_before,
            "metal_memory_after": memory_after,
        }
        record_evaluation_checkpoint(self.last_execution_plan, event)
        self._pending_custom_passes = 0
        self._pending_custom_io_bytes = 0

    def explain(self, operations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Build an execution report without applying any gates."""
        with mx.stream(self._mx_device):
            return self._explain(operations)

    def _explain(self, operations: List[Dict[str, Any]]) -> Dict[str, Any]:
        checkpoint_policy_data = metal_checkpoint_policy(
            self._metal_checkpoint_budget_bytes
        )
        optimized_operations = self._fuse_zz_layers(operations)
        return build_execution_plan(
            n_qubits=self.wires,
            backend=self.backend,
            operations=list(operations),
            optimized_operations=optimized_operations,
            checkpoint_policy_data=checkpoint_policy_data,
            statevector_preflight_data=self.statevector_preflight,
            initial_pending_custom_passes=self._pending_custom_passes,
            initial_pending_custom_io_bytes=self._pending_custom_io_bytes,
        )

    def synchronize(self):
        """Evaluate pending MLX work and mark dispatch evidence complete."""
        with mx.stream(self._mx_device):
            state = getattr(self.sim, 'state', None)
            if state is not None:
                mx.eval(state)
            else:
                tensors = getattr(self.sim, 'tensors', None)
                if tensors:
                    mx.eval(*tensors)
        self._pending_custom_passes = 0
        self._pending_custom_io_bytes = 0
        mark_synchronized(self.last_execution_plan)
        return state

    @staticmethod
    def _dependency_schedule_for_fusion(
        operations: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Expose frontend-independent layers without changing wire order.

        SDK transpilers may emit a valid topological order that starts a
        two-qubit gate as soon as its two wires are ready.  That serialization
        hides otherwise parallel single-qubit layers from the Metal fusion
        pass.  Build the per-wire dependency DAG and choose ready one-qubit
        operations before ready multi-qubit operations.  Operations sharing a
        wire retain their exact input order; only disjoint operations move.
        """
        count = len(operations)
        if count < 2:
            return list(operations)
        indegree = [0] * count
        successors = [set() for _ in range(count)]
        last_on_wire: Dict[int, int] = {}
        for index, operation in enumerate(operations):
            predecessors = {
                last_on_wire[wire]
                for wire in operation.get("wires", [])
                if wire in last_on_wire
            }
            indegree[index] = len(predecessors)
            for predecessor in predecessors:
                successors[predecessor].add(index)
            for wire in operation.get("wires", []):
                last_on_wire[wire] = index

        ready = {index for index, degree in enumerate(indegree) if degree == 0}
        scheduled: List[Dict[str, Any]] = []
        while ready:
            single = [
                index
                for index in ready
                if len(operations[index].get("wires", [])) == 1
            ]
            batch = sorted(single if single else ready)
            for index in batch:
                ready.remove(index)
                scheduled.append(operations[index])
            for index in batch:
                for successor in successors[index]:
                    indegree[successor] -= 1
                    if indegree[successor] == 0:
                        ready.add(successor)
        if len(scheduled) != count:
            raise RuntimeError("operation dependency graph contains a cycle")
        return scheduled

    def _fuse_zz_layers(self, operations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Runtime gate fusion: collapse runs of consecutive ZZPHASE ops that
        share one angle (a Trotter layer) into a single cached diagonal
        multiply. Execution-time optimization of the same input gate sequence,
        analogous to Aer's runtime fusion; disabled on the dense-only ablation
        path and on backends without the fused kernel."""
        if (_os.environ.get("METTLEQ_DENSE_ONLY", "0") == "1"
                or not hasattr(self.sim, "apply_zz_layer")):
            return list(operations)
        try:
            from . import shaders as metal_kernels
            rx_layer_ok = (metal_kernels.metal_enabled(self.wires)
                           and hasattr(self.sim, "state"))
        except Exception:
            rx_layer_ok = False
        # Qiskit commonly collapses rotation pairs to U2/U3 and then emits an
        # ASAP order that hides the layer.  Do not reschedule native structured
        # streams such as QFT: their ordered ladders already have stronger,
        # lower-pass Metal matches.
        if rx_layer_ok and any(
            str(operation.get("name", "")).upper() in {"U2", "U3"}
            for operation in operations
        ):
            operations = self._dependency_schedule_for_fusion(operations)
        fused: List[Dict[str, Any]] = []
        i = 0
        n_ops = len(operations)
        while i < n_ops:
            op = operations[i]
            # Uniform single-qubit layers (opt-in Metal): take the maximal
            # window of consecutive single-qubit ops; ops on distinct wires
            # commute, so if every wire sees the IDENTICAL gate sequence the
            # window equals a stack of all-qubit layers. Handles contiguous
            # layers (TFIM RX), interleaved per-qubit pairs (Grover H,X;
            # QCBM RY,RZ), and leaves non-uniform windows untouched.
            if rx_layer_ok and len(op.get("wires", [])) == 1:
                j = i
                per_wire: Dict[int, list] = {}
                while j < n_ops:
                    nxt = operations[j]
                    w = nxt.get("wires", [])
                    if len(w) != 1 or not (0 <= w[0] < self.wires):
                        break
                    per_wire.setdefault(w[0], []).append(
                        (str(nxt.get("name", "")).upper(),
                         tuple(nxt.get("parameters", []) or [])))
                    j += 1
                k_ops = j - i
                seqs = list(per_wire.values())
                synthesized_u_window = bool(seqs) and all(
                    gate_name in {"U2", "U3"}
                    for sequence in seqs
                    for gate_name, _ in sequence
                )
                uniform = (sorted(per_wire.keys()) == list(range(self.wires))
                           and all(s == seqs[0] for s in seqs))
                if uniform and len(seqs[0]) == 1:
                    gname, gparams = seqs[0][0]
                    if gname == "RX":
                        fused.append({"name": "_RXLAYER", "theta": gparams[0]})
                        i = j
                        continue
                    if gname == "X":
                        # All-qubit X is a pure bit-flip permutation: one
                        # gather pass instead of floor(n/2) U2 passes.
                        rows = [1 << (self.wires - 1 - k)
                                for k in range(self.wires)]
                        fused.append({"name": "_XORLAYER", "rows": rows,
                                      "c": (1 << self.wires) - 1})
                        i = j
                        continue
                    fused.append({"name": "_U2LAYER", "gate": gname,
                                  "params": list(gparams)})
                    i = j
                    continue
                # General window: each wire's op subsequence collapses to ONE
                # 2x2 product, so the whole window is a single per-qubit
                # layer (ceil(n/2) passes). Worth it once the window has more
                # ops than the layer has passes. Also catches algebraic
                # collapses (H*H = I drops out; H,H,X windows become a pure
                # bit-flip gather).
                if (k_ops > self.wires // 2 + 1
                        or (uniform and len(seqs[0]) > 1)
                        or synthesized_u_window):
                    import numpy as _np
                    # Products in complex128 with rtol=0 collapse tests:
                    # true algebraic cancellation (H*H = I) lands within
                    # ~1e-15 while a real RZ(1e-6) stays ~1e-6 away, so
                    # exact semantics survive (codex S3 review: float32 +
                    # default rtol silently dropped small-angle gates).
                    prods = {}
                    ok = True
                    for wq, seq in per_wire.items():
                        m = _np.eye(2, dtype=_np.complex128)
                        for gname, gparams in seq:
                            try:
                                g = self._dense_gate_for(gname, list(gparams))
                            except ValueError:
                                ok = False
                                break
                            m = _np.array(g).reshape(2, 2).astype(
                                _np.complex128) @ m
                        if not ok:
                            break
                        prods[wq] = m
                    if ok:
                        eye2 = _np.eye(2, dtype=_np.complex128)
                        xmat = _np.array([[0, 1], [1, 0]],
                                         dtype=_np.complex128)

                        def _is(m, ref):
                            return _np.allclose(m, ref, rtol=0.0, atol=1e-12)

                        active = {wq for wq, m in prods.items()
                                  if not _is(m, eye2)}
                        if not active:
                            i = j
                            continue  # window is the identity: emit nothing
                        if (active == set(range(self.wires))
                                and all(_is(m, xmat) for m in prods.values())):
                            rows = [1 << (self.wires - 1 - k)
                                    for k in range(self.wires)]
                            fused.append({"name": "_XORLAYER", "rows": rows,
                                          "c": (1 << self.wires) - 1})
                            i = j
                            continue
                        # Passes that will actually launch (identity pairs
                        # are skipped by the host); only fuse when the window
                        # beats them (codex: sparse deep windows can lose).
                        pair_qubits = [(2 * p, 2 * p + 1)
                                       for p in range(self.wires // 2)]
                        est = sum(1 for a, b in pair_qubits
                                  if a in active or b in active)
                        if self.wires % 2 == 1 and (self.wires - 1) in active:
                            est += 1
                        if (k_ops > est
                                or (uniform and len(seqs[0]) > 1)
                                or (synthesized_u_window and k_ops >= est)):
                            mats = _np.tile(
                                _np.eye(2, dtype=_np.complex64).reshape(1, 4),
                                (self.wires, 1))
                            for wq, m in prods.items():
                                mats[wq] = m.astype(_np.complex64).reshape(4)
                            fused.append({"name": "_U2LISTLAYER",
                                          "mats": mx.array(mats),
                                          "active": sorted(active)})
                            i = j
                            continue
            # Controlled single-qubit gates use a sparse target-pair kernel;
            # adjacent gates on four distinct wires share one traversal.
            # Overlapping gates remain ordered as individual groups.
            if rx_layer_ok:
                controlled = {"CH", "CRX", "CRY", "CRZ"}

                def _controlled_entry(candidate):
                    name = str(candidate.get("name", "")).upper()
                    wires = list(candidate.get("wires", []))
                    params = list(candidate.get("parameters", []) or [])
                    if (name not in controlled or len(wires) != 2
                            or wires[0] == wires[1]
                            or not all(0 <= q < self.wires for q in wires)):
                        return None
                    if name == "CH" and not params:
                        target = H()
                    elif name == "CRX" and len(params) == 1:
                        target = RX(params[0])
                    elif name == "CRY" and len(params) == 1:
                        target = RY(params[0])
                    elif name == "CRZ" and len(params) == 1:
                        target = RZ(params[0])
                    else:
                        return None
                    return {"control": wires[0], "target": wires[1],
                            "matrix": mx.reshape(target, (4,))}

                first_controlled = _controlled_entry(op)
                if first_controlled is not None:
                    entries = []
                    j = i
                    while j < n_ops:
                        entry = _controlled_entry(operations[j])
                        if entry is None:
                            break
                        entries.append(entry)
                        j += 1
                    groups = []
                    k = 0
                    while k < len(entries):
                        if (k + 1 < len(entries)
                                and {entries[k]["control"], entries[k]["target"]}.isdisjoint(
                                    {entries[k + 1]["control"], entries[k + 1]["target"]})):
                            groups.append(entries[k:k + 2])
                            k += 2
                        else:
                            groups.append(entries[k:k + 1])
                            k += 1
                    fused.append({"name": "_CULISTLAYER", "groups": groups})
                    i = j
                    continue
            # CNOT/X/SWAP runs (opt-in Metal): the block acts on basis
            # indices as an affine GF(2) map, so the whole run is ONE
            # amplitude permutation executed as a single gather pass.
            if rx_layer_ok:
                xname = str(op.get("name", "")).upper()
                if xname in ("X", "CNOT", "CX", "SWAP"):

                    def _affine_op_ok(o):
                        nn = str(o.get("name", "")).upper()
                        w = o.get("wires", [])
                        if nn == "X":
                            return len(w) == 1 and 0 <= w[0] < self.wires
                        if nn in ("CNOT", "CX", "SWAP"):
                            return (len(w) == 2 and w[0] != w[1]
                                    and all(0 <= q < self.wires for q in w))
                        return False

                    j = i
                    while j < n_ops and _affine_op_ok(operations[j]):
                        j += 1
                    if j - i >= 2:
                        from . import shaders as metal_kernels
                        rows, cflip = metal_kernels.compose_inverse_affine(
                            operations[i:j], self.wires)
                        fused.append({"name": "_XORLAYER", "rows": rows,
                                      "c": cflip})
                        i = j
                        continue
            # QFT stage patterns (opt-in Metal). Forward: H(j) followed by
            # the ascending controlled-phase ladder CPHASE(k, j) with angle
            # +pi/2^(k-j) for k = j+1..m-1; inverse (QPE's inline iQFT): the
            # descending ladder with angle -pi/2^(k-j) followed by H(j).
            # Either fuses to ONE stage pass (butterfly + closed-form
            # ladder). Same gate stream as every other backend receives;
            # only the routing differs.
            if rx_layer_ok:

                def _cphase_of(o):
                    nm = str(o.get("name", "")).upper()
                    if nm not in ("CP", "CPHASE"):
                        return None
                    w = list(o.get("wires", []))
                    pr = list(o.get("parameters", []) or [])
                    if len(w) != 2 or len(pr) != 1:
                        return None
                    return w, pr[0]

                qname = str(op.get("name", "")).upper()
                if (qname == "H" and len(op.get("wires", [])) == 1):
                    jq = op.get("wires", [0])[0]
                    j2 = i + 1
                    kk = jq + 1
                    while j2 < n_ops:
                        cp = _cphase_of(operations[j2])
                        if (cp is not None and cp[0] == [kk, jq]
                                and cp[1] == _math.pi / (1 << (kk - jq))):
                            kk += 1
                            j2 += 1
                        else:
                            break
                    if kk > jq + 1:
                        fused.append({"name": "_QFTSTAGE", "j": jq, "m": kk,
                                      "inverse": False})
                        i = j2
                        continue
                cp0 = _cphase_of(op)
                if cp0 is not None and cp0[1] < 0:
                    (k0, jq), _t = cp0
                    # descending run k0, k0-1, ..., j+1 then H(j)
                    kk = k0
                    j2 = i
                    okrun = True
                    while kk > jq:
                        cp = _cphase_of(operations[j2]) if j2 < n_ops else None
                        if (cp is not None and cp[0] == [kk, jq]
                                and cp[1] == -_math.pi / (1 << (kk - jq))):
                            kk -= 1
                            j2 += 1
                        else:
                            okrun = False
                            break
                    if (okrun and j2 < n_ops
                            and str(operations[j2].get("name", "")).upper() == "H"
                            and operations[j2].get("wires", []) == [jq]):
                        fused.append({"name": "_QFTSTAGE", "j": jq,
                                      "m": k0 + 1, "inverse": True})
                        i = j2 + 1
                        continue
            # Diagonal two-qubit runs (opt-in Metal): CZ / equal-angle CPHASE
            # gates are diagonal and mutually commute; a run over any bond
            # topology (chain, ring, brickwork) collapses to one LUT pass.
            if rx_layer_ok:
                dname = str(op.get("name", "")).upper()
                if dname == "CP":
                    dname = "CPHASE"

                def _diag_wires_ok(o):
                    w = o.get("wires", [])
                    return (len(w) == 2 and w[0] != w[1]
                            and all(0 <= q < self.wires for q in w))

                if dname in ("CZ", "CPHASE") and _diag_wires_ok(op):
                    # Collect the whole diagonal run, CZ and CPHASE mixed,
                    # angles free to differ: all such gates commute. Uniform
                    # angles take the LUT kernel; varying angles (QPE
                    # controlled-power and inline-iQFT ladders) take the
                    # weighted kernel.
                    bonds = []
                    thetas = []
                    j = i
                    while j < n_ops:
                        nxt = operations[j]
                        nname = str(nxt.get("name", "")).upper()
                        if nname == "CP":
                            nname = "CPHASE"
                        if nname == "CZ" and _diag_wires_ok(nxt):
                            thetas.append(_math.pi)
                        elif nname == "CPHASE" and _diag_wires_ok(nxt):
                            thetas.append(list(nxt.get("parameters", [None]))[0])
                        else:
                            break
                        bonds.append(tuple(nxt.get("wires", [])))
                        j += 1
                    if len(bonds) >= 2:
                        if all(t == thetas[0] for t in thetas):
                            fused.append({"name": "_DIAGLAYER",
                                          "theta": thetas[0], "bonds": bonds})
                        else:
                            fused.append({"name": "_DIAGWEIGHTED",
                                          "bonds": bonds, "thetas": thetas})
                        i = j
                        continue
            if str(op.get("name", "")).upper() == "ZZPHASE":
                bonds = []
                thetas = []
                j = i
                while j < n_ops:
                    nxt = operations[j]
                    if str(nxt.get("name", "")).upper() == "ZZPHASE":
                        bonds.append(tuple(nxt.get("wires", [])))
                        thetas.append(list(nxt.get("parameters", [None]))[0])
                        j += 1
                    else:
                        break
                if len(bonds) >= 2:
                    if all(t == thetas[0] for t in thetas):
                        # Uniform layer: cached-phase-vector path (works on
                        # both pure-MLX and Metal modes).
                        fused.append({"name": "_ZZLAYER", "theta": thetas[0],
                                      "bonds": bonds})
                        i = j
                        continue
                    if rx_layer_ok:
                        # Per-bond couplings (long-range Ising): one fused
                        # weighted pass, Metal only.
                        fused.append({"name": "_ZZWEIGHTED", "bonds": bonds,
                                      "thetas": thetas})
                        i = j
                        continue
            # Uniform XX / YY Trotter layers (opt-in Metal): same-family
            # terms all commute; the layer diagonalizes via basis conjugation
            # around the one-pass ZZ kernel. XX and YY must NOT be mixed in
            # one run (cross-family terms on overlapping bonds anticommute).
            if rx_layer_ok:
                pname = str(op.get("name", "")).upper()
                if pname in ("XXPHASE", "YYPHASE"):
                    theta = list(op.get("parameters", [None]))[0]
                    bonds = [tuple(op.get("wires", []))]
                    j = i + 1
                    while j < n_ops:
                        nxt = operations[j]
                        if (str(nxt.get("name", "")).upper() == pname
                                and list(nxt.get("parameters", [None]))[0] == theta):
                            bonds.append(tuple(nxt.get("wires", [])))
                            j += 1
                        else:
                            break
                    if len(bonds) >= 2:
                        fused.append({"name": "_XXLAYER" if pname == "XXPHASE"
                                      else "_YYLAYER",
                                      "theta": theta, "bonds": bonds})
                        i = j
                        continue
            fused.append(op)
            i += 1
        # A decomposed forward QFT arrives as n-1 structured stages followed
        # by the final H. The complete transform can use the existing radix-4
        # kernel and execute two dependent stages per state traversal.
        full_qft_enabled = (
            rx_layer_ok
            and self.wires >= 2
            and _os.environ.get(
                "METTLEQ_FULL_QFT_RADIX4", "1"
            ).strip().lower() not in {"0", "false", "off", "no"}
        )
        if full_qft_enabled:
            collapsed: List[Dict[str, Any]] = []
            index = 0
            while index < len(fused):
                end = index + self.wires
                candidate = fused[index:end]
                stages = candidate[:-1]
                final = candidate[-1] if candidate else None
                if (
                    len(candidate) == self.wires
                    and all(
                        stage.get("name") == "_QFTSTAGE"
                        and stage.get("j") == offset
                        and stage.get("m") == self.wires
                        and not stage.get("inverse", False)
                        for offset, stage in enumerate(stages)
                    )
                    and str((final or {}).get("name", "")).upper() == "H"
                    and list((final or {}).get("wires", []))
                    == [self.wires - 1]
                ):
                    collapsed.append({"name": "_QFTLAYER"})
                    index = end
                    continue
                collapsed.append(fused[index])
                index += 1
            fused = collapsed
        phase_rx_enabled = (
            rx_layer_ok
            and self.wires >= 4
            and _os.environ.get("METTLEQ_SINGLE_QUBIT_RADIX", "16") == "16"
            and _os.environ.get(
                "METTLEQ_CHAINPHASE_RX_FUSION", "1"
            ).strip().lower() not in {"0", "false", "off", "no"}
        )
        if not phase_rx_enabled:
            return fused
        amortized: List[Dict[str, Any]] = []
        index = 0
        while index < len(fused):
            current = fused[index]
            following = fused[index + 1] if index + 1 < len(fused) else None
            if following is not None and following.get("name") == "_RXLAYER":
                if current.get("name") == "_ZZLAYER" and sorted(
                    current.get("bonds", [])
                ) == [(q, q + 1) for q in range(self.wires - 1)]:
                    amortized.append({
                        "name": "_CHAINPHASE_RXLAYER",
                        "phase_kind": "zz",
                        "theta": current["theta"],
                        "n_bonds": self.wires - 1,
                        "edge_mask": (1 << (self.wires - 1)) - 1,
                        "wrap_mask": 0,
                        "rx_theta": following["theta"],
                    })
                    index += 2
                    continue
                if current.get("name") == "_DIAGLAYER":
                    norm = [tuple(sorted(b)) for b in current.get("bonds", [])]
                    if len(norm) == len(set(norm)):
                        edge_mask = 0
                        wrap_mask = 0
                        valid = True
                        for lo, hi in norm:
                            if hi - lo == 1:
                                edge_mask |= 1 << (self.wires - 2 - lo)
                            elif lo == 0 and hi == self.wires - 1:
                                wrap_mask = (1 << (self.wires - 1)) | 1
                            else:
                                valid = False
                                break
                        if valid:
                            amortized.append({
                                "name": "_CHAINPHASE_RXLAYER",
                                "phase_kind": "cphase",
                                "theta": current["theta"],
                                "n_bonds": len(norm),
                                "edge_mask": edge_mask,
                                "wrap_mask": wrap_mask,
                                "rx_theta": following["theta"],
                            })
                            index += 2
                            continue
            amortized.append(current)
            index += 1
        return amortized

    def sample(self, shots: int = None, wires: Optional[List[int]] = None):
        shots = self.shots if shots is None else int(shots)
        with mx.stream(self._mx_device):
            return self.sim.sample(shots, wires)

    def counts(self, shots: int = None, wires: Optional[List[int]] = None):
        shots = self.shots if shots is None else int(shots)
        with mx.stream(self._mx_device):
            return self.sim.sample_counts(shots, wires)

    def _dense_gate_for(self, name: str, params: List[float]) -> mx.array:
        """Dense matrix for a named op (ablation path; mirrors pre-dispatch behavior)."""
        if name in ("SDAG", "S†"):
            name = "SDG"
        elif name in ("TDAG", "T†"):
            name = "TDG"
        elif name == "CX":
            name = "CNOT"
        gate = _CONST_GATES.get(name)
        if gate is not None:
            return gate
        if name == "RX":
            return RX(params[0])
        if name == "RY":
            return RY(params[0])
        if name == "RZ":
            return RZ(params[0])
        if name == "U1":
            return U1(params[0])
        if name == "U2":
            return U2(params[0], params[1])
        if name == "U3":
            return U3(params[0], params[1], params[2])
        if name in ("CP", "CPHASE"):
            return CPHASE(params[0])
        if name == "CRX":
            return CRX(params[0])
        if name == "CRY":
            return CRY(params[0])
        if name == "CRZ":
            return CRZ(params[0])
        if name in ("ZZPHASE", "XXPHASE", "YYPHASE"):
            return _pauli_pair_phase(name, params[0])
        if name in ("CCX",):
            return _CONST_GATES["TOFFOLI"]
        if name in ("CSWAP",):
            return _CONST_GATES["FREDKIN"]
        raise ValueError(f"Unsupported op for dense ablation path: {name}")

    def _apply(
        self,
        name: str,
        wires: List[int],
        params: List[float],
        *,
        mps_lookahead: Optional[List[tuple[int, int]]] = None,
    ):
        # Ablation switch: METTLEQ_DENSE_ONLY=1 routes every gate through the
        # generic dense reshape/transpose/matmul path (pre-dispatch behavior),
        # isolating the structured-kernel speedup from MLX GPU execution.
        if _os.environ.get("METTLEQ_DENSE_ONLY", "0") == "1" and hasattr(self.sim, "apply_dense_gate"):
            gate = self._dense_gate_for(name, params)
            self.sim.apply_dense_gate(gate, wires)
            return
        # Diagonal fast path (SV backend): broadcast phase multiply, no matmul
        diag_ok = hasattr(self.sim, "apply_diagonal")

        if len(wires) == 1:
            q = wires[0]
            if name in ("SDAG", "S†"):
                name = "SDG"
            elif name in ("TDAG", "T†"):
                name = "TDG"
            if diag_ok:
                d = _CONST_DIAGS.get(name)
                if d is None and name == "RZ":
                    d = _rz_diag(params[0])
                elif d is None and name == "U1":
                    d = _phase_diag(params[0])
                if d is not None:
                    self.sim.apply_diagonal(d, [q])
                    return
            gate = _CONST_GATES.get(name)
            if gate is None:
                if name == "RX":
                    gate = RX(params[0])
                elif name == "RY":
                    gate = RY(params[0])
                elif name == "RZ":
                    gate = RZ(params[0])
                elif name == "U1":
                    gate = U1(params[0])
                elif name == "U2":
                    gate = U2(params[0], params[1])
                elif name == "U3":
                    gate = U3(params[0], params[1], params[2])
                else:
                    raise ValueError(f"Unsupported single-qubit op: {name}")
            if hasattr(self.sim, "apply_logical_single"):
                self.sim.apply_logical_single(gate, q)
            else:
                self.sim.apply_single(gate, q)
            return

        if len(wires) == 2:
            c, t = wires
            if name == "CX":
                name = "CNOT"
            if name in ("CP", "CPHASE") and len(params) != 1:
                raise ValueError("CPHASE requires one parameter")
            if diag_ok:
                d = None
                if name == "CZ":
                    d = _CONST_DIAGS["CZ"]
                elif name in ("CP", "CPHASE"):
                    ph = complex(_math.cos(params[0]), _math.sin(params[0]))
                    d = mx.array([1+0j, 1+0j, 1+0j, ph], mx.complex64)
                elif name == "CRZ":
                    rz = _rz_diag(params[0])
                    d = mx.concatenate([mx.array([1+0j, 1+0j], mx.complex64), rz])
                if d is not None:
                    self.sim.apply_diagonal(d, [c, t])
                    return
            if name == "CNOT" and hasattr(self.sim, "apply_logical_cnot"):
                self.sim.apply_logical_cnot(c, t)
                return
            # Structured two-qubit fast paths (SV backend)
            if hasattr(self.sim, "apply_controlled_single"):
                if name == "CNOT":
                    self.sim.apply_controlled_single(_CONST_GATES["X"], c, t)
                    return
                if name == "CH":
                    self.sim.apply_controlled_single(_CONST_GATES["H"], c, t)
                    return
                if name == "CRX":
                    self.sim.apply_controlled_single(RX(params[0]), c, t)
                    return
                if name == "CRY":
                    self.sim.apply_controlled_single(RY(params[0]), c, t)
                    return
            if hasattr(self.sim, "apply_swap_wires"):
                if name == "SWAP":
                    self.sim.apply_swap_wires(c, t)
                    return
                if name == "ISWAP":
                    self.sim.apply_diagonal(mx.array([1+0j, 1j, 1j, 1+0j], mx.complex64), [c, t])
                    self.sim.apply_swap_wires(c, t)
                    return
            if name == "ZZPHASE" and hasattr(self.sim, "apply_zz_phase"):
                if hasattr(self.sim, "apply_logical_zz_phase"):
                    self.sim.apply_logical_zz_phase(
                        params[0], c, t, lookahead=mps_lookahead
                    )
                else:
                    self.sim.apply_zz_phase(params[0], c, t)
                return
            if name == "XXPHASE" and hasattr(self.sim, "apply_xx_phase"):
                self.sim.apply_xx_phase(params[0], c, t)
                return
            if name == "YYPHASE" and hasattr(self.sim, "apply_yy_phase"):
                self.sim.apply_yy_phase(params[0], c, t)
                return
            gate = _CONST_GATES.get(name)
            if gate is None:
                if name in ("CP", "CPHASE"):
                    gate = CPHASE(params[0])
                elif name == "CRX":
                    gate = CRX(params[0])
                elif name == "CRY":
                    gate = CRY(params[0])
                elif name == "CRZ":
                    gate = CRZ(params[0])
                elif name in ("ZZPHASE", "XXPHASE", "YYPHASE"):
                    gate = _pauli_pair_phase(name, params[0])
                else:
                    raise ValueError(f"Unsupported two-qubit op: {name}")
            if hasattr(self.sim, "apply_logical_two"):
                self.sim.apply_logical_two(
                    gate, c, t, lookahead=mps_lookahead
                )
            else:
                self.sim.apply_two(gate, c, t)
            return

        if len(wires) == 3:
            if name in ("CCX", "TOFFOLI"):
                if hasattr(self.sim, "apply_multi_controlled_single"):
                    self.sim.apply_multi_controlled_single(_CONST_GATES["X"], wires[:2], wires[2])
                    return
                self.sim.apply_dense_gate(_CONST_GATES["TOFFOLI"], wires)
                return
            if name in ("CSWAP", "FREDKIN"):
                if hasattr(self.sim, "apply_controlled_swap"):
                    self.sim.apply_controlled_swap(wires[0], wires[1], wires[2])
                    return
                self.sim.apply_dense_gate(_CONST_GATES["FREDKIN"], wires)
                return
            raise ValueError(f"Unsupported three-qubit op: {name}")

        raise ValueError(f"Unsupported operation arity for: {name}")
