import json
import numpy as np
import os
import pytest

GOLDEN_JSON = os.path.join(os.path.dirname(__file__), '..', 'golden', 'golden_instances.json')

def load_golden():
    with open(GOLDEN_JSON, 'r') as f:
        return json.load(f)

@pytest.mark.parametrize("instance", load_golden(), ids=lambda i: i['id'])
def test_golden_instance_verification(instance):
    J = np.array(instance['J'])
    h = np.array(instance['h'])
    s = np.array(instance['ground_state'])
    E_opt = instance['optimal_energy']
    
    # 1. Assert symmetry
    assert np.allclose(J, J.T), "J must be symmetric"
    
    # 2. Assert zero-diagonal
    assert np.all(np.diag(J) == 0), "J diagonal must be 0"
    
    # 3. Verify energy calculation
    energy = 0.5 * s.dot(J).dot(s) + h.dot(s)
    assert np.isclose(energy, E_opt), f"Energy {energy} does not match CP-SAT E_opt {E_opt}"
    
    # 4. Verify spins are +/- 1
    assert np.all(np.abs(s) == 1), "Spins must be +/- 1"
