import cudaq

try:
    @cudaq.kernel
    def test_rz():
        q = cudaq.qvector(1)
        rz(0.5, q[0])

    print("rz(angle, q) works!")
    print(cudaq.draw(test_rz))
except Exception as e:
    print("rz(angle, q) failed:", e)

try:
    @cudaq.kernel
    def test_rzz():
        q = cudaq.qvector(2)
        # RZZ(theta)
        x.ctrl(q[0], q[1])
        rz(0.5, q[1])
        x.ctrl(q[0], q[1])

    print("CNOT-RZ-CNOT works for RZZ!")
    print(cudaq.draw(test_rzz))
except Exception as e:
    print("CNOT-RZ-CNOT failed for RZZ:", e)
