

from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit.visualization import circuit_drawer

def constant_oracle(n: int, output: int) -> QuantumCircuit:
    """
    Oracle for a constant function.

    f(x) = output  for all x  (output is 0 or 1).

    Implementation:
      output=0  ->  do nothing (|q> stays |q>).
      output=1  ->  flip the ancilla qubit unconditionally with an X gate.

    The ancilla is qubit index n (the (n+1)-th qubit).
    """
    qc = QuantumCircuit(n + 1, name=f"Constant-{output} Oracle")

    if output == 1:
        qc.x(n)   # flip ancilla: turns |-> into -|-> (a global phase on marked states)

    return qc


def balanced_oracle(n: int) -> QuantumCircuit:
    """
    Oracle for a balanced function.

    f(x) = parity of x  =  x_0 XOR x_1 XOR ... XOR x_(n-1).

    This is balanced because exactly half of all n-bit strings have even
    parity (f=0) and half have odd parity (f=1).

    Implementation:
      Apply CNOT from each input qubit onto the ancilla.
      The net effect is ancilla <- ancilla XOR (x_0 XOR ... XOR x_(n-1)).
      In the phase-kickback picture this marks the odd-parity strings.
    """
    qc = QuantumCircuit(n + 1, name="Balanced (parity) Oracle")

    for qubit in range(n):
        qc.cx(qubit, n)   # CNOT: control = input qubit, target = ancilla

    return qc


def balanced_oracle_custom(n: int, flip_mask: int) -> QuantumCircuit:
    """
    A more general balanced oracle using a bit-flip mask.

    Before querying, flip the input qubits indicated by flip_mask.
    After querying, flip them back (uncompute).
    This changes *which* half of inputs map to 1, while keeping f balanced.

    flip_mask:  an integer whose binary representation selects which input
                qubits to wrap with X gates.  E.g. flip_mask=0b101 flips
                qubits 0 and 2.
    """
    qc = QuantumCircuit(n + 1, name=f"Balanced Oracle (mask={bin(flip_mask)})")

    # Pre-flip selected input qubits
    for qubit in range(n):
        if flip_mask & (1 << qubit):
            qc.x(qubit)

    # Parity CNOT ladder onto ancilla
    for qubit in range(n):
        qc.cx(qubit, n)

    # Uncompute (restore input qubits — keeps oracle self-inverse)
    for qubit in range(n):
        if flip_mask & (1 << qubit):
            qc.x(qubit)

    return qc


# ── Core Algorithm ────────────────────────────────────────────────────────────

def deutsch_jozsa(oracle: QuantumCircuit, n: int) -> QuantumCircuit:
    """
    Build the full Deutsch-Jozsa circuit for n input qubits + 1 ancilla.

    Circuit structure:
      1. Initialise ancilla to |1> via X, then put it in |-> via H.
         This enables phase kickback from the oracle.
      2. Apply H^{otimes n} to all input qubits -> uniform superposition.
      3. Apply the oracle U_f.
      4. Apply H^{otimes n} to all input qubits again.
      5. Measure the n input qubits.

    Reading the result:
      All-zeros  ->  f is CONSTANT  (all amplitudes interfered constructively
                                     back into |00...0>).
      Any 1      ->  f is BALANCED  (|00...0> amplitude cancelled to 0 by
                                     destructive interference).
    """
    qc = QuantumCircuit(n + 1, n)    # n+1 qubits, n classical bits

    # -- Step 1: Prepare ancilla in |-> = (|0> - |1>) / sqrt(2) --------------
    qc.x(n)        # |0> -> |1>
    qc.h(n)        # |1> -> |->

    qc.barrier()   # visual separator only; no physical effect

    # -- Step 2: Uniform superposition over all n input qubits ----------------
    for qubit in range(n):
        qc.h(qubit)

    qc.barrier()

    # -- Step 3: Oracle -------------------------------------------------------
    qc.compose(oracle, inplace=True)

    qc.barrier()

    # -- Step 4: Interfere (Hadamard again on inputs) -------------------------
    for qubit in range(n):
        qc.h(qubit)

    qc.barrier()

    # -- Step 5: Measure input qubits -----------------------------------------
    qc.measure(range(n), range(n))

    return qc


# ── Runner ────────────────────────────────────────────────────────────────────

def run_and_interpret(label: str, oracle: QuantumCircuit, n: int,
                      shots: int = 1024):
    """
    Run one Deutsch-Jozsa experiment and print a human-readable result.
    """
    qc = deutsch_jozsa(oracle, n)

    simulator = AerSimulator()
    job = simulator.run(qc, shots=shots)
    counts = job.result().get_counts()

    # The algorithm is deterministic: one outcome dominates.
    top_outcome = max(counts, key=counts.get)
    verdict = "CONSTANT" if all(b == "0" for b in top_outcome) else "BALANCED"

    print(f"\n{'='*60}")
    print(f"  Experiment : {label}")
    print(f"  n (input qubits) : {n}")
    print(f"  Measurement counts : {counts}")
    print(f"  Dominant bitstring : |{top_outcome}>")
    print(f"  Verdict    : f is {verdict}")
    print(f"{'='*60}")

    return qc, counts


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    N = 10    # number of input qubits  (change freely: 1, 2, 3, 4, ...)

    print(f"\nDeutsch-Jozsa Algorithm  |  n = {N} input qubits")
    print(f"Classical queries needed (worst case): 2^(n-1)+1 = {2**(N-1)+1}")
    print("Quantum queries needed : 1 (always)")

    # ── Experiment 1: Constant-0 ─────────────────────────────────────────────
    oracle_c0 = constant_oracle(N, output=0)
    qc1, _ = run_and_interpret("Constant oracle  f(x)=0", oracle_c0, N)

    # ── Experiment 2: Constant-1 ─────────────────────────────────────────────
    oracle_c1 = constant_oracle(N, output=1)
    qc2, _ = run_and_interpret("Constant oracle  f(x)=1", oracle_c1, N)

    # ── Experiment 3: Balanced (parity) ──────────────────────────────────────
    oracle_b = balanced_oracle(N)
    qc3, _ = run_and_interpret("Balanced oracle  f(x) = parity(x)", oracle_b, N)

    # ── Experiment 4: Balanced (custom mask) ─────────────────────────────────
    mask = 0b1010   # flip qubits 1 and 3 before and after the parity ladder
    oracle_bm = balanced_oracle_custom(N, flip_mask=mask)
    qc4, _ = run_and_interpret(
        f"Balanced oracle  f(x) = parity(x XOR {bin(mask)})", oracle_bm, N)

    # ── Print circuits ────────────────────────────────────────────────────────
    print("\n\nCircuit for Experiment 3 (Balanced / parity oracle):\n")
    print(qc3.draw(output="text", fold=120))

    print("\nOracle sub-circuit:\n")
    print(oracle_b.draw(output="text"))

    # ── Save circuits to file (optional) ─────────────────────────────────────
    # Uncomment to save an image (requires matplotlib):
    #
    #   qc3.draw(output="mpl", filename="dj_circuit.png", fold=120)
    #   print("\nCircuit saved to dj_circuit.png")