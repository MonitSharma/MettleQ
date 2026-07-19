import cudaq

try:
    @cudaq.kernel
    def test_r1():
        q = cudaq.qvector(2)
        h(q[0])
        r1.ctrl(q[0], q[1], 0.5)

    print("r1.ctrl(control, target, angle) works!")
    print(cudaq.draw(test_r1))
except Exception as e:
    print("r1.ctrl(control, target, angle) failed:", e)

try:
    @cudaq.kernel
    def test_r1_alt():
        q = cudaq.qvector(2)
        h(q[0])
        r1.ctrl(0.5, q[0], q[1])

    print("r1.ctrl(angle, control, target) works!")
    print(cudaq.draw(test_r1_alt))
except Exception as e:
    print("r1.ctrl(angle, control, target) failed:", e)

try:
    @cudaq.kernel
    def test_rzz():
        q = cudaq.qvector(2)
        h(q[0])
        rzz(0.5, q[0], q[1])

    print("rzz(angle, q0, q1) works!")
    print(cudaq.draw(test_rzz))
except Exception as e:
    print("rzz(angle, q0, q1) failed:", e)
