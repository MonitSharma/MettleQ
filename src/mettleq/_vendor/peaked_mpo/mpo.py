from contextlib import contextmanager
from functools import lru_cache

from quimb.tensor import MatrixProductOperator, Circuit

from qiskit_quimb import quimb_circuit
from qiskit import QuantumCircuit


# Process-local adapter used by MettleQ to expose quimb's split methods while
# preserving the published pipeline's default when no option is supplied.
_DEFAULT_COMPRESS_OPTS = {}


def _configured_dtype():
    return _DEFAULT_COMPRESS_OPTS.get("dtype")


@contextmanager
def compression_options(**options):
    previous = dict(_DEFAULT_COMPRESS_OPTS)
    _DEFAULT_COMPRESS_OPTS.update(options)
    try:
        yield
    finally:
        _DEFAULT_COMPRESS_OPTS.clear()
        _DEFAULT_COMPRESS_OPTS.update(previous)

# ------------------------------------------------------------------
#  Constructors
# ------------------------------------------------------------------

def mpo_from_circuit(circ: Circuit):
    """Convert a quimb circuit to a simple-chain matrix product operator."""
    for q in range(circ.N):
        circ.u3(0, 0, 0, q)
    tn_uni = circ.get_uni()

    # contract gates per site tag
    for st in list(tn_uni.site_tags):
        tn_uni ^= st

    # make sure bonds are simple 1D chain bonds
    tn_uni.fuse_multibonds_()  

    # cast as MatrixProductOperator
    mpo = tn_uni.view_as_(
        MatrixProductOperator,
        cyclic=False,
        L=circ.N,
    )

    mpo.ensure_bonds_exist()
    dtype = _configured_dtype()
    if dtype is not None:
        mpo = mpo.astype(dtype)
    return mpo


# ------------------------------------------------------------------
#  MPO x MPO composition
# ------------------------------------------------------------------

def apply_mpo(mpo1: MatrixProductOperator, mpo2: MatrixProductOperator,
                side,
                max_bond=None,
                cutoff=0.0,
                contract=True,
                compress=True,
                **compress_opts):
    effective_compress_opts = dict(_DEFAULT_COMPRESS_OPTS)
    effective_compress_opts.update(compress_opts)
    dtype = effective_compress_opts.pop("dtype", None)
    if dtype is not None:
        mpo1 = mpo1.astype(dtype)
        mpo2 = mpo2.astype(dtype)
    if side == "right":
        return mpo1.apply(
            mpo2,
            compress=compress,
            max_bond=max_bond,
            cutoff=cutoff,
            create_bond=True,
            contract=contract,
            **effective_compress_opts,
        )
    elif side == "left":
        return mpo2.apply(
            mpo1,
            compress=compress,
            max_bond=max_bond,
            cutoff=cutoff,
            create_bond=True,
            contract=contract,
            **effective_compress_opts,
        )
    else:
        raise ValueError("side must be 'left' or 'right'.")



# ------------------------------------------------------------------
#  Applying circuits to MPO
# ------------------------------------------------------------------


def apply_circuit(mpo, circ, side, max_bond=None, cutoff=0.0, contract=True, compress=True, **compress_opts):
    return apply_mpo(mpo, mpo_from_circuit(circ), side=side, max_bond=max_bond, cutoff=cutoff, contract=contract, compress=compress, **compress_opts)


@lru_cache(maxsize=4096)
def _cached_swap_mpo(num_qubits, swaps, representation, dtype):
    qc_swaps = QuantumCircuit(num_qubits)
    for q0, q1 in swaps:
        qc_swaps.swap(q0, q1)
    if representation == "cx":
        qc_swaps = qc_swaps.decompose("swap")
    elif representation != "block":
        raise ValueError(f"unsupported SWAP gate representation: {representation}")
    circ_swaps = quimb_circuit(qc_swaps, Circuit, to_backend=None)
    return mpo_from_circuit(circ_swaps)


def _swap_mpo(num_qubits, swaps, to_backend=None, representation="cx"):
    swaps = tuple(tuple(pair) for pair in swaps)
    if to_backend is None:
        return _cached_swap_mpo(
            num_qubits, swaps, representation, _configured_dtype()
        )

    qc_swaps = QuantumCircuit(num_qubits)
    for q0, q1 in swaps:
        qc_swaps.swap(q0, q1)
    if representation == "cx":
        qc_swaps = qc_swaps.decompose("swap")
    elif representation != "block":
        raise ValueError(f"unsupported SWAP gate representation: {representation}")
    circ_swaps = quimb_circuit(qc_swaps, Circuit, to_backend=to_backend)
    return mpo_from_circuit(circ_swaps)


def apply_swaps(
    mpo: MatrixProductOperator,
    swaps_l,
    swaps_r,
    max_bond=None,
    cutoff=0.0,
    to_backend=None,
    inplace=False,
    method="mpo",
    swap_gate_representation="cx",
):
    N = len(mpo.sites)
    if len(swaps_l) == 0 and len(swaps_r) == 0:
        return mpo

    if method != "mpo":
        raise ValueError(f"unsupported SWAP application method: {method}")

    # Quimb's MPO application returns a new network for this path, so copying
    # the input MPO here only duplicates work before every unswap probe.
    mpo_out = mpo

    if len(swaps_l) > 0:
        mpo_out = apply_mpo(
            mpo_out,
            _swap_mpo(
                N,
                swaps_l,
                to_backend=to_backend,
                representation=swap_gate_representation,
            ),
            side="right",
            max_bond=max_bond,
            cutoff=cutoff,
        )
    
    if len(swaps_r) > 0:
        mpo_out = apply_mpo(
            mpo_out,
            _swap_mpo(
                N,
                swaps_r,
                to_backend=to_backend,
                representation=swap_gate_representation,
            ),
            side="left",
            max_bond=max_bond,
            cutoff=cutoff,
        ) 

    return mpo_out
