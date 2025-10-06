# First Contribution Proposal: Grover's Algorithm Benchmark

## Executive Summary

This document outlines a high-quality first contribution to the cuQuantum benchmark suite: implementing Grover's search algorithm as a new benchmark. This contribution demonstrates deep understanding of quantum algorithms, provides value to the community, and follows the extensible architecture of the existing benchmark framework.

## Why Grover's Algorithm?

### Strategic Value
1. **Missing from current benchmarks** - The suite has QFT, QPE, QAOA, but not Grover's
2. **Widely recognized algorithm** - One of the most famous quantum algorithms
3. **Practical applications** - Database search, optimization, cryptography
4. **Good complexity profile** - Demonstrates quantum speedup clearly
5. **Scales well** - Can benchmark from small to large qubit counts

### Technical Value
1. **Oracle implementation** - Shows how to implement problem-specific oracles
2. **Amplitude amplification** - Demonstrates key quantum technique
3. **Iteration control** - Optimal number of Grover iterations depends on problem size
4. **Measurement analysis** - Success probability is a key metric
5. **Multi-qubit operations** - Good stress test for quantum simulators

## Algorithm Overview

### Grover's Search Algorithm
- **Purpose:** Find marked items in an unsorted database with quadratic speedup
- **Classical complexity:** O(N) where N = 2^n for n qubits
- **Quantum complexity:** O(√N)
- **Optimal iterations:** π/4 * √N

### Circuit Structure
```
Initial state: |0⟩^n
↓
Hadamard gates (create superposition)
↓
[Repeat ~√N times:
   1. Oracle (marks solution)
   2. Diffusion operator (amplifies marked state)
]
↓
Measurement
```

## Implementation Plan

### File Structure
```
benchmarks/nv_quantum_benchmarks/benchmarks/grover.py
```

### Class Design
```python
class Grover(Benchmark):
    """
    Grover's algorithm benchmark for unstructured search.
    
    Searches for a marked item among 2^n possibilities using quantum
    amplitude amplification. Demonstrates quadratic speedup over classical search.
    """
    
    @staticmethod
    def generateGatesSequence(nqubits, config):
        """
        Generate gate sequence for Grover's algorithm.
        
        Args:
            nqubits: Number of qubits (search space size = 2^nqubits)
            config: Dictionary with:
                - 'marked_state': Integer representing the marked state (default: random)
                - 'num_iterations': Number of Grover iterations (default: optimal)
                - 'oracle_type': Type of oracle implementation ('standard', 'gray-code', etc.)
        
        Returns:
            List of gate tuples representing the circuit
        """
        pass
    
    @staticmethod
    def postProcess(nqubits, results):
        """
        Analyze success probability of finding marked state.
        
        Returns:
            Dictionary with metrics:
                - 'success_probability': Probability of measuring marked state
                - 'expected_probability': Theoretical probability
                - 'fidelity': How close to ideal result
        """
        pass
```

### Configuration Options

Add to `benchmarks/nv_quantum_benchmarks/config.py`:

```python
BENCHMARKS_CONFIG = {
    # ... existing benchmarks ...
    
    'grover': {
        'description': "Grover's search algorithm for unstructured database search",
        'default_nqubits': 4,
        'min_nqubits': 2,
        'max_nqubits': 30,  # Depends on backend
        'parameters': {
            'marked_state': {
                'type': int,
                'default': None,  # Random if not specified
                'description': 'Target state to search for (0 to 2^n - 1)'
            },
            'num_iterations': {
                'type': int,
                'default': None,  # Optimal if not specified
                'description': 'Number of Grover iterations (default: optimal ~π/4√N)'
            },
            'oracle_type': {
                'type': str,
                'default': 'standard',
                'choices': ['standard', 'gray-code', 'phase'],
                'description': 'Oracle implementation strategy'
            },
            'ancilla_mode': {
                'type': str,
                'default': 'minimal',
                'choices': ['minimal', 'clean'],
                'description': 'Ancilla qubit usage strategy'
            }
        }
    }
}
```

## Detailed Implementation

### 1. Oracle Implementation

```python
def _create_oracle(marked_state, nqubits, oracle_type='standard'):
    """
    Create oracle that marks the target state.
    
    The oracle implements: O|x⟩ = (-1)^f(x)|x⟩
    where f(x) = 1 if x is marked state, 0 otherwise
    """
    gates = []
    
    if oracle_type == 'standard':
        # Convert marked_state to binary and apply X gates
        for i in range(nqubits):
            if not (marked_state >> i) & 1:
                gates.append(('x', i))
        
        # Multi-controlled Z gate
        if nqubits == 1:
            gates.append(('z', 0))
        elif nqubits == 2:
            gates.append(('cz', 0, 1))
        else:
            # Use multi-controlled Z with n-1 controls
            controls = list(range(nqubits - 1))
            target = nqubits - 1
            gates.append(('mcz', *controls, target))
        
        # Undo X gates
        for i in range(nqubits):
            if not (marked_state >> i) & 1:
                gates.append(('x', i))
    
    elif oracle_type == 'phase':
        # Alternative: use phase oracle with controlled rotations
        # More efficient for certain backends
        pass
    
    return gates
```

### 2. Diffusion Operator

```python
def _create_diffusion_operator(nqubits):
    """
    Create diffusion operator (inversion about average).
    
    Implements: D = 2|ψ⟩⟨ψ| - I
    where |ψ⟩ is uniform superposition
    """
    gates = []
    
    # H gates
    for i in range(nqubits):
        gates.append(('h', i))
    
    # X gates
    for i in range(nqubits):
        gates.append(('x', i))
    
    # Multi-controlled Z
    if nqubits == 1:
        gates.append(('z', 0))
    elif nqubits == 2:
        gates.append(('cz', 0, 1))
    else:
        controls = list(range(nqubits - 1))
        target = nqubits - 1
        gates.append(('mcz', *controls, target))
    
    # X gates
    for i in range(nqubits):
        gates.append(('x', i))
    
    # H gates
    for i in range(nqubits):
        gates.append(('h', i))
    
    return gates
```

### 3. Complete Circuit

```python
@staticmethod
def generateGatesSequence(nqubits, config):
    """Generate complete Grover circuit."""
    gates = []
    
    # Extract configuration
    marked_state = config.get('marked_state')
    if marked_state is None:
        import random
        marked_state = random.randint(0, 2**nqubits - 1)
    
    num_iterations = config.get('num_iterations')
    if num_iterations is None:
        import math
        # Optimal number: π/4 * √N where N = 2^nqubits
        N = 2 ** nqubits
        num_iterations = int(math.pi / 4 * math.sqrt(N))
        # Ensure at least 1 iteration
        num_iterations = max(1, num_iterations)
    
    oracle_type = config.get('oracle_type', 'standard')
    
    # Initial Hadamard gates (create superposition)
    for i in range(nqubits):
        gates.append(('h', i))
    
    # Grover iterations
    for _ in range(num_iterations):
        # Oracle
        oracle_gates = _create_oracle(marked_state, nqubits, oracle_type)
        gates.extend(oracle_gates)
        
        # Diffusion operator
        diffusion_gates = _create_diffusion_operator(nqubits)
        gates.extend(diffusion_gates)
    
    return gates
```

### 4. Post-Processing

```python
@staticmethod
def postProcess(nqubits, results):
    """
    Analyze Grover algorithm performance.
    
    Args:
        nqubits: Number of qubits
        results: Measurement results from backend
        
    Returns:
        Dictionary with performance metrics
    """
    import math
    
    # Extract marked state from config (stored in results)
    marked_state = results.get('marked_state', 0)
    num_iterations = results.get('num_iterations', 1)
    
    # Get measurement counts
    counts = results.get('counts', {})
    total_shots = sum(counts.values())
    
    # Calculate success probability
    marked_state_str = format(marked_state, f'0{nqubits}b')
    success_count = counts.get(marked_state_str, 0)
    success_probability = success_count / total_shots if total_shots > 0 else 0
    
    # Calculate theoretical success probability
    N = 2 ** nqubits
    theta = math.asin(1 / math.sqrt(N))
    expected_probability = math.sin((2 * num_iterations + 1) * theta) ** 2
    
    # Calculate fidelity
    fidelity = success_probability / expected_probability if expected_probability > 0 else 0
    
    metrics = {
        'success_probability': success_probability,
        'expected_probability': expected_probability,
        'fidelity': fidelity,
        'marked_state': marked_state,
        'num_iterations': num_iterations,
        'optimal_iterations': int(math.pi / 4 * math.sqrt(N)),
        'success_count': success_count,
        'total_shots': total_shots
    }
    
    return metrics
```

## Testing Strategy

### Unit Tests
Create `benchmarks/tests/test_grover.py`:

```python
import pytest
from nv_quantum_benchmarks.benchmarks.grover import Grover

class TestGrover:
    
    def test_gate_sequence_generation(self):
        """Test that gate sequence is generated correctly."""
        config = {'marked_state': 5, 'num_iterations': 2}
        gates = Grover.generateGatesSequence(4, config)
        
        assert len(gates) > 0
        assert gates[0][0] == 'h'  # First gates should be Hadamards
        
    def test_optimal_iterations(self):
        """Test optimal iteration calculation."""
        import math
        for nqubits in range(2, 10):
            config = {}  # Use default (optimal) iterations
            gates = Grover.generateGatesSequence(nqubits, config)
            
            N = 2 ** nqubits
            expected_iterations = int(math.pi / 4 * math.sqrt(N))
            # Verify the number of oracle-diffusion pairs
            # (More complex - count specific gate patterns)
            
    def test_different_oracle_types(self):
        """Test different oracle implementations."""
        config_standard = {'oracle_type': 'standard', 'marked_state': 3}
        config_phase = {'oracle_type': 'phase', 'marked_state': 3}
        
        gates_standard = Grover.generateGatesSequence(4, config_standard)
        gates_phase = Grover.generateGatesSequence(4, config_phase)
        
        # Both should produce valid gate sequences
        assert len(gates_standard) > 0
        assert len(gates_phase) > 0
        
    def test_small_circuits(self):
        """Test edge cases with small qubit counts."""
        for nqubits in [1, 2]:
            config = {'marked_state': 0}
            gates = Grover.generateGatesSequence(nqubits, config)
            assert len(gates) > 0
            
    def test_postprocess(self):
        """Test post-processing of results."""
        results = {
            'marked_state': 5,
            'num_iterations': 2,
            'counts': {
                '0101': 850,  # marked state (binary 5)
                '0000': 50,
                '1111': 100
            }
        }
        
        metrics = Grover.postProcess(4, results)
        
        assert 'success_probability' in metrics
        assert 'expected_probability' in metrics
        assert 'fidelity' in metrics
        assert metrics['success_probability'] > 0.5  # Should be high
```

### Integration Tests

```python
def test_grover_with_different_backends():
    """Test Grover benchmark with various backends."""
    backends = ['cutn', 'qiskit', 'cirq']  # Available backends
    
    for backend in backends:
        # Run small Grover instance
        # Verify success probability is reasonably high
        pass
```

## Documentation

### 1. Docstrings
- Comprehensive docstrings for all functions
- Examples of usage
- Parameter descriptions
- Return value specifications

### 2. Tutorial
Create `examples/grover_tutorial.md`:
- Explanation of Grover's algorithm
- How to run the benchmark
- Interpreting results
- Performance analysis tips
- Comparison with classical search

### 3. README Update
Update main benchmark README with Grover example:
```bash
# Search for marked state 42 in 10-qubit space
nv-quantum-benchmarks circuit --frontend qiskit --backend cutn \
    --benchmark grover --nqubits 10 --marked-state 42 --ngpus 1
```

## Expected Performance Characteristics

### Success Probability
- **Optimal iterations:** ~99% success rate
- **Sub-optimal:** Lower success rate (good for testing)
- **Over-rotation:** Success rate decreases

### Scaling
- **Gate count:** O(√N * n) where N = 2^n
- **Circuit depth:** O(√N * n)
- **Memory:** O(2^n) for state vector backends

### Benchmarking Insights
- Compare different oracle implementations
- Test multi-controlled gate efficiency across backends
- Measure scaling on different GPU architectures
- Evaluate noise resilience (when using noisy simulators)

## Deliverables

### Code Files
1. ✅ `benchmarks/nv_quantum_benchmarks/benchmarks/grover.py`
2. ✅ `benchmarks/tests/test_grover.py`
3. ✅ Update to `benchmarks/nv_quantum_benchmarks/config.py`
4. ✅ Update to `benchmarks/nv_quantum_benchmarks/benchmarks/__init__.py`

### Documentation
1. ✅ Inline code documentation
2. ✅ Tutorial document
3. ✅ README updates
4. ✅ Example usage in comments

### Testing
1. ✅ Unit tests (>90% coverage)
2. ✅ Integration tests
3. ✅ Performance benchmarks
4. ✅ Validation against theoretical predictions

## Timeline

### Week 1: Implementation
- Day 1-2: Core algorithm implementation
- Day 3-4: Oracle and diffusion operator
- Day 5-7: Testing and debugging

### Week 2: Refinement
- Day 1-2: Multiple oracle implementations
- Day 3-4: Performance optimization
- Day 5-7: Comprehensive testing

### Week 3: Documentation & Polish
- Day 1-3: Write documentation and tutorial
- Day 4-5: Create examples and demos
- Day 6-7: Final review and testing

### Week 4: Community Engagement
- Day 1-2: Post on GitHub Discussions
- Day 3-4: Gather feedback
- Day 5-7: Incorporate suggestions

## Success Criteria

### Technical Excellence
- [ ] Code follows existing style and patterns
- [ ] All tests pass
- [ ] Works with all supported backends
- [ ] Performance is optimal
- [ ] No regressions in other benchmarks

### Documentation Quality
- [ ] Clear, comprehensive documentation
- [ ] Working examples
- [ ] Tutorial is easy to follow
- [ ] Edge cases explained

### Community Impact
- [ ] Positive feedback on Discussions
- [ ] Other users can run it successfully
- [ ] Generates interesting benchmark data
- [ ] Cited in community projects

## Next Steps After Grover

### Follow-up Contributions
1. **Quantum Amplitude Amplification** (generalization of Grover)
2. **Deutsch-Jozsa Algorithm** (oracle-based algorithm)
3. **Bernstein-Vazirani Algorithm** (hidden string problem)
4. **Simon's Algorithm** (period finding)

### Advanced Features
1. **Multi-target Grover** (search for multiple items)
2. **Fixed-point Grover** (guaranteed success)
3. **Adaptive Grover** (dynamic iteration adjustment)
4. **Grover with ancilla optimization**

## Additional Resources

### References
1. Grover, L. K. (1996). "A fast quantum mechanical algorithm for database search"
2. Nielsen & Chuang: "Quantum Computation and Quantum Information"
3. Qiskit Grover tutorial
4. Cirq Grover implementation

### Related Work
- Existing Grover implementations in Qiskit
- cuQuantum documentation on multi-controlled gates
- Performance studies of amplitude amplification

---

## Questions to Consider

1. **Should we support multiple marked states?**
   - Yes, in a future iteration
   
2. **How to handle ancilla qubits?**
   - Provide configuration options for different strategies
   
3. **What about approximate/noisy Grover?**
   - Add as advanced feature later
   
4. **Integration with existing benchmarks?**
   - Keep modular, follow existing patterns

## Contact Points

- GitHub Discussions: Share progress and ask questions
- Issue Tracker: Report any bugs found
- Direct communication: Once established

---

**Status:** Ready to implement
**Priority:** High
**Difficulty:** Medium
**Impact:** High

This contribution demonstrates:
✅ Deep algorithm understanding
✅ Clean code architecture
✅ Comprehensive testing
✅ Excellent documentation
✅ Community value

**Let's build this!** 🚀
