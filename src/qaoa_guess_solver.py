"""Utility class to build and solve a QUBO via QAOA for a battery dispatch skeleton.

This module encapsulates the construction of a quadratic unconstrained binary
optimization (QUBO) matrix tailored to the battery arbitrage problem and uses
Qiskit's optimisation and variational quantum algorithms to obtain a
warm‑start guess for the 24‑hour discharge schedule.

It follows a similar design philosophy to the rest of this repository, where
functionality is grouped into classes with clearly scoped methods.  The class
exposed here enables:

* Building a QUBO matrix from hourly price data and user‑defined penalty
  parameters.
* Assembling the corresponding :class:`~qiskit_optimization.QuadraticProgram`.
* Solving the QUBO using a variational quantum algorithm (QAOA) with a
  classical optimiser to produce a binary discharge vector.
* Converting the resulting binary string into a numeric discharge schedule
  suitable as an initial guess for the classical MILP solver.

As with other modules in this package, heavy dependencies are imported lazily.
If Qiskit is not installed in your environment, an informative error is raised
at runtime.

Example
-------
>>> prices = [50.0] * 24  # dummy price curve
>>> solver = QAOAGuessSolver(gamma=10.0, A=1.0, K=8)
>>> guess = solver.solve(prices)
>>> print(guess['bitstring'], guess['discharge_guess'])
"""
from __future__ import annotations

import numpy as np

try:
    # Qiskit is only imported when needed; if unavailable an informative
    # ImportError is raised in the solve() method.
    import qiskit  # type: ignore
except ImportError:
    qiskit = None


class QAOAGuessSolver:
    """Construct and solve a battery-dispatch QUBO via QAOA.

    Parameters
    ----------
    gamma : float, optional
        Weight of the switching penalty in the QUBO.  Larger values
        discourage frequent on/off transitions.  Default is 0.0.
    A : float, optional
        Weight of the cardinality penalty in the QUBO.  Must be positive.
        Default is 1.0.
    K : int, optional
        Target number of discharge hours in the skeleton (cycle proxy).
        Under a 4 MWh discharge per hour, the MILP cycle budget of 32 MWh
        suggests ``K=8``.  Default is 8.
    reps : int, optional
        Number of QAOA layers (p in the literature).  More repetitions
        typically increase expressivity but also circuit depth.  Default is 1.
    optimizer : object, optional
        A classical optimiser instance from :mod:`qiskit_algorithms.optimizers`.
        If ``None``, a default :class:`~qiskit_algorithms.optimizers.COBYLA`
        with a limited number of iterations is used.
    maxiter : int, optional
        Maximum number of iterations for the default optimiser.
        Ignored if ``optimizer`` is provided explicitly.  Default is 40.

    Notes
    -----
    The solver constructs the QUBO matrix using the formulas described in the
    accompanying tutorial.  It then sets up a
    :class:`~qiskit_optimization.QuadraticProgram` and solves it using a
    :class:`~qiskit_optimization.algorithms.MinimumEigenOptimizer` wrapping
    :class:`~qiskit_algorithms.QAOA`.  The QUBO is minimised; the output
    bitstring is decoded so that ``1`` indicates ``discharge`` and ``0``
    indicates ``no discharge`` for each hour.  A coarse discharge magnitude
    of 4 MWh per hour is used to form the initial guess.
    """

    def __init__(
        self,
        gamma: float = 0.0,
        A: float = 1.0,
        K: int = 8,
        reps: int = 1,
        optimizer: object | None = None,
        maxiter: int = 40,
    ) -> None:
        self.gamma = float(gamma)
        self.A = float(A)
        self.K = int(K)
        self.reps = int(reps)
        self.maxiter = int(maxiter)
        self.optimizer = optimizer  # type: ignore[assignment]

    # ------------------------------------------------------------------
    # QUBO construction
    # ------------------------------------------------------------------
    def build_qubo_matrix(self, prices: list[float] | np.ndarray) -> np.ndarray:
        """Construct the symmetric QUBO matrix for the battery skeleton.

        Parameters
        ----------
        prices : array-like of float
            Hourly price vector of length ``n`` (typically ``n=24``).

        Returns
        -------
        numpy.ndarray
            A symmetric ``n x n`` QUBO matrix Q such that
            ``E(x) = x.T @ Q @ x`` corresponds to
            ``-4 * sum(p_t x_t) + γ ∑_{t=2}^{n}(x_t-x_{t-1})^2 + A (sum x_t - K)^2``.
        """
        p = np.array(prices, dtype=float).flatten()
        n = p.size
        # Precompute degrees for the switching penalty: end points have deg=1, inner nodes deg=2.
        deg = np.ones(n)
        if n > 1:
            deg[1:-1] = 2.0

        Q = np.zeros((n, n))
        # Diagonal terms: linear coefficients
        diag = -4.0 * p + self.gamma * deg + self.A * (1.0 - 2.0 * self.K)
        np.fill_diagonal(Q, diag)

        # Off-diagonal terms: cardinality penalty contributes 2A everywhere
        # and switching penalty contributes -2γ for adjacent elements.
        # Exploit symmetry: iterate over i<j.
        for i in range(n):
            for j in range(i + 1, n):
                val = 2.0 * self.A
                # Adjacent hours get an extra -2γ
                if j == i + 1:
                    val -= 2.0 * self.gamma
                Q[i, j] = val
                Q[j, i] = val
        return Q

    def build_quadratic_program(self, Q: np.ndarray) -> "QuadraticProgram":
        """Create a QuadraticProgram from a QUBO matrix.

        Parameters
        ----------
        Q : numpy.ndarray
            Symmetric matrix representing the QUBO objective.

        Returns
        -------
        qiskit_optimization.QuadraticProgram
            The constructed quadratic program minimising ``x.T @ Q @ x``.

        Raises
        ------
        ImportError
            If qiskit optimisation modules are not installed.
        """
        if qiskit is None:
            raise ImportError(
                "Qiskit is required to build and solve the QUBO. "
                "Please install qiskit, qiskit-optimization and qiskit-algorithms."
            )
        from qiskit_optimization import QuadraticProgram  # type: ignore

        n = Q.shape[0]
        qp = QuadraticProgram()
        for i in range(n):
            qp.binary_var(name=f"x{i}")
        # Collect quadratic coefficients in the format expected by QuadraticProgram.minimize
        quadratic: dict[tuple[str, str], float] = {}
        linear: dict[str, float] = {}
        for i in range(n):
            # diagonal entries act as linear terms when i == j
            linear[f"x{i}"] = Q[i, i]
            for j in range(i + 1, n):
                if Q[i, j] != 0.0:
                    quadratic[(f"x{i}", f"x{j}")] = Q[i, j]
        # The objective constant term (which does not affect the minimiser) is dropped.
        qp.minimize(linear=linear, quadratic=quadratic)
        return qp

    # ------------------------------------------------------------------
    # QAOA solver
    # ------------------------------------------------------------------
    def solve(
        self,
        prices: list[float] | np.ndarray | None = None,
        qubo_matrix: np.ndarray | None = None,
        qubo_matrix_path: str | None = None,
        backend: "qiskit.providers.Backend" | None = None,
        sim_backend: "qiskit.providers.Backend" | None = None,
        shots: int = 100_000,
        init_gamma: float = 0.05,
        init_beta: float = 0.01,
        maxiter: int | None = None,
    ) -> dict[str, object]:
        """Solve the QUBO via a manual QAOA routine using `QAOAAnsatz` and a classical optimiser.

        This method adheres closely to the original tutorial implementation: it
        builds the quadratic program, converts it to an Ising Hamiltonian, constructs
        a :class:`~qiskit.circuit.library.QAOAAnsatz`, evaluates the energy via the
        IBM Runtime Estimator, and minimises the cost with COBYLA.  Finally, it
        samples the optimised circuit with a sampler to obtain a probability
        distribution over bitstrings.

        Parameters
        ----------
        prices : array-like of float, optional
            Hourly price vector.  If provided, ``qubo_matrix`` and ``qubo_matrix_path``
            are ignored and a new QUBO matrix is constructed via
            :meth:`build_qubo_matrix`.
        qubo_matrix : numpy.ndarray, optional
            Pre‑computed QUBO matrix.  Ignored if ``prices`` is not None.
        qubo_matrix_path : str, optional
            Path to a CSV file containing a symmetric QUBO matrix.  Ignored if
            ``prices`` is provided or ``qubo_matrix`` is not None.
        backend : qiskit.providers.Backend, optional
            Backend for running the estimator and sampler.  In the original
            tutorial a fake device (``FakeSherbrooke``) is used.  If not
            provided, the default fake device from :mod:`qiskit_ibm_runtime.fake_provider`
            will be used if available.
        sim_backend : qiskit.providers.Backend, optional
            Backend for circuit transpilation (AerSimulator).  If not provided,
            an :class:`~qiskit_aer.AerSimulator` instance is used.
        shots : int, optional
            Number of shots for sampling the optimised circuit.  Default is
            100,000 to replicate the original behaviour.
        init_gamma : float, optional
            Initial value for the gamma parameter.  Used to build the initial
            parameter vector ``[gamma, beta] * reps``.  Default is 0.05.
        init_beta : float, optional
            Initial value for the beta parameter.  Used similarly to
            ``init_gamma``.  Default is 0.01.
        maxiter : int, optional
            Maximum number of optimisation iterations.  If ``None``, the
            instance's ``maxiter`` attribute is used.

        Returns
        -------
        dict
            A dictionary containing:

            - ``bitstring`` (str): the most probable bitstring sampled from the
              optimised circuit.
            - ``distribution`` (dict): mapping of integer bitstrings to
              probabilities for the measured outcomes.
            - ``discharge_guess`` (list[float]): a length‑n vector with
              ``4 * bit`` for each bit in the solution, representing a coarse
              discharge MWh schedule.
            - ``quadratic_program`` (:class:`qiskit_optimization.QuadraticProgram`):
              the underlying optimisation model.
            - ``objective_values`` (list[float]): the sequence of evaluated
              objective values during optimisation.

        Raises
        ------
        ImportError
            If required Qiskit modules are not available.
        """
        if qiskit is None:
            raise ImportError(
                "Qiskit and its optimisation modules are required to solve the QUBO. "
                "Please install qiskit, qiskit-optimization, qiskit-aer, qiskit-ibm-runtime, "
                "and scipy to run the quantum warm start."
            )

        # Determine the QUBO matrix
        if prices is not None:
            qubo = self.build_qubo_matrix(prices)
        elif qubo_matrix is not None:
            qubo = np.array(qubo_matrix, dtype=float)
        elif qubo_matrix_path is not None:
            # Lazy import pandas to avoid mandatory dependency
            import pandas as pd  # type: ignore
            qubo = pd.read_csv(qubo_matrix_path, index_col=0).to_numpy()
        else:
            raise ValueError(
                "At least one of `prices`, `qubo_matrix` or `qubo_matrix_path` must be provided."
            )

        # Build the quadratic program manually (mirrors the original code)
        from qiskit_optimization import QuadraticProgram  # type: ignore
        from qiskit_optimization.converters import QuadraticProgramToQubo  # type: ignore

        n = qubo.shape[0]
        qp = QuadraticProgram()
        for i in range(n):
            qp.binary_var(name=f"x{i}")

        # Populate the quadratic objective as a dictionary of coefficients
        objective = {}
        for i in range(n):
            for j in range(n):
                if qubo[i, j] != 0:
                    objective[(f"x{i}", f"x{j}")] = float(qubo[i, j])

        qp.minimize(quadratic=objective)

        # Convert to QUBO and then to Ising
        converter = QuadraticProgramToQubo()
        qubo_qp = converter.convert(qp)
        operator, offset = qubo_qp.to_ising()

        # Import QAOAAnsatz and related utilities lazily
        from qiskit.circuit.library import QAOAAnsatz  # type: ignore
        from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager  # type: ignore
        from qiskit_aer import AerSimulator  # type: ignore
        from qiskit_ibm_runtime import EstimatorV2 as Estimator, SamplerV2 as Sampler  # type: ignore
        from qiskit_ibm_runtime import Session  # type: ignore
        from qiskit_ibm_runtime.fake_provider import FakeSherbrooke  # type: ignore
        import scipy.optimize  # type: ignore

        # Determine the optimisation iteration limit
        opt_maxiter = maxiter if maxiter is not None else self.maxiter

        # Set up the backend; default to FakeSherbrooke if available
        if backend is None:
            try:
                backend = FakeSherbrooke()
            except Exception:
                backend = None

        # Set up the simulator backend for transpilation
        if sim_backend is None:
            sim_backend = AerSimulator()

        # Build the QAOA ansatz with the cost operator
        ansatz = QAOAAnsatz(cost_operator=operator, reps=self.reps)
        ansatz.measure_all()

        # Transpile the circuit for the simulator backend
        pm = generate_preset_pass_manager(optimization_level=3, backend=sim_backend)
        candidate_circuit = pm.run(ansatz)

        # Prepare storage for objective values during optimisation
        self.objective_func_vals: list[float] = []

        # Define the cost function used by the classical optimiser
        def cost_fun_estimator(params, ansatz_circ, hamiltonian, estimator_obj):
            # Map the Hamiltonian onto the device using the circuit layout
            isa_hamiltonian = hamiltonian.apply_layout(layout=ansatz_circ.layout)
            # Prepare the publication tuple as required by EstimatorV2
            pub = (ansatz_circ, isa_hamiltonian, params)
            job = estimator_obj.run([pub])
            results = job.result()[0]
            cost = results.data.evs
            # Accumulate the cost value for diagnostics
            self.objective_func_vals.append(cost)
            return cost

        # Initialise the parameter vector [gamma, beta] * reps
        init_params = [init_gamma, init_beta] * ansatz.reps

        # Run the optimisation inside an IBM Runtime session
        # Note: running on real hardware requires credentials; FakeSherbrooke is used as default
        distribution_int: dict[int, float] = {}
        with Session(backend=backend) as session:
            estimator = Estimator(mode=session)
            # Set the default number of shots for the estimator (as in the tutorial)
            estimator.options.default_shots = 1024

            # Perform classical optimisation using COBYLA via scipy.optimize
            result = scipy.optimize.minimize(
                cost_fun_estimator,
                x0=init_params,
                args=(candidate_circuit, operator, estimator),
                method='COBYLA',
                options={'maxiter': opt_maxiter, 'disp': True}
            )

            # Bind the optimal parameters to the transpiled circuit
            optimized_circuit = candidate_circuit.assign_parameters(result.x)

            # Use a sampler to obtain measurement outcomes from the optimised circuit
            sampler = Sampler(mode=backend)
            sampler.options.default_shots = shots
            pub = (optimized_circuit,)
            job = sampler.run([pub])
            counts_int = job.result()[0].data.meas.get_int_counts()
            shots_total = sum(counts_int.values())
            distribution_int = {key: val / shots_total for key, val in counts_int.items()}

        # Determine the most probable bitstring
        if distribution_int:
            max_key = max(distribution_int, key=distribution_int.get)
            bitstring = format(max_key, f'0{n}b')
        else:
            # Fallback to an all‑zero bitstring if distribution is empty
            bitstring = '0' * n

        # Convert bitstring to discharge schedule (4 MWh per '1')
        bits = [int(ch) for ch in bitstring]
        discharge_guess = [4.0 * b for b in bits]

        return {
            'bitstring': bitstring,
            'distribution': distribution_int,
            'discharge_guess': discharge_guess,
            'quadratic_program': qp,
            'objective_values': self.objective_func_vals,
        }

    # ------------------------------------------------------------------
    # Convenience for decoding bitstrings
    # ------------------------------------------------------------------
    @staticmethod
    def decode_bitstring(bitstring: str) -> list[int]:
        """Decode a bitstring into a list of integers (0/1).

        Parameters
        ----------
        bitstring : str
            String representation of binary bits.

        Returns
        -------
        list[int]
            A list of integers corresponding to each bit.
        """
        return [int(ch) for ch in bitstring]

    @staticmethod
    def to_discharge_schedule(bits: list[int], discharge_per_hour: float = 4.0) -> list[float]:
        """Convert a binary vector to a discharge schedule.

        Parameters
        ----------
        bits : list of int
            Sequence of 0/1 decisions.
        discharge_per_hour : float, optional
            Discharge magnitude assigned to a ``1`` in the skeleton.

        Returns
        -------
        list[float]
            Discharge energy per hour in MWh.
        """
        return [discharge_per_hour * b for b in bits]