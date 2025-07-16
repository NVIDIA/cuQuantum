# Copyright (c) 2021-2025, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause

import multiprocessing

from .benchmarks.hidden_shift import HiddenShift
from .benchmarks.ghz import GHZ
from .benchmarks.qaoa import QAOA
from .benchmarks.qft import QFT
from .benchmarks.iqft import IQFT
from .benchmarks.qpe import QPE
from .benchmarks.quantum_volume import QuantumVolume
from .benchmarks.random import Random
from .benchmarks.simon import Simon


#########################################################################################################
########################################### Benchmarks Config ###########################################
#########################################################################################################

benchmarks = {

    'qft': {
        'benchmark': QFT,
        'config': {
            'measure': True,
        },
    },

    'iqft': {
        'benchmark': IQFT,
        'config': {
            'measure': True,
        },
    },

    'ghz': {
        'benchmark': GHZ,
        'config': {
            'measure': True,
        },
    },

    'simon': {
        'benchmark': Simon,
        'config': {
            'measure': True,
        },
    },

    'hidden_shift': {
        'benchmark': HiddenShift,
        'config': {
            'measure': True,
        },
    },

    'qaoa': {
        'benchmark': QAOA,
        'config': {
            'measure': True,
            'p': 1,
        },
    },

    'qpe': {
        'benchmark': QPE,
        'config': {
            'measure': True,
            'unfold': False,
        },
    },

    'quantum_volume': {
        'benchmark': QuantumVolume,
        'config': {
            'measure': True,
        },
    },

    'random': {
        'benchmark': Random,
        'config': {
            'measure': True,
        },
    },
}

#########################################################################################################
############################################ Backends Config ############################################
#########################################################################################################

backends = {

    'cutn': {
        'config': {
            'nshots': 0,
            'nfused': None,
            'ngpus': 1,
            # TODO: even this may not be a good default
            'ncputhreads': multiprocessing.cpu_count() // 2,
            'precision': 'single',
            'compute_mode': 'amplitude',
            'nhypersamples': 32,
        },
    },

    'aer': {
        'config': {
            'nshots': 1024,
            'nfused': 5,
            'ngpus': 0,
            'ncputhreads': multiprocessing.cpu_count(),
            'precision':'single',
            'compute_mode': 'sampling',
        },
    },

    'aer-cuda': {
        'config': {
            'nshots': 1024,
            'nfused': 5,
            'ngpus': 1,
            'ncputhreads': multiprocessing.cpu_count(),
            'precision':'single',
            'compute_mode': 'sampling',
        },
    },

    'aer-cusv': {
        'config': {
            'nshots': 1024,
            'nfused': 5,
            'ngpus': 1,
            'ncputhreads': multiprocessing.cpu_count(),
            'precision':'single',
            'compute_mode': 'sampling',
        },
    },

    'cusvaer': {
        'config': {
            'nshots': 1024,
            'nfused': 4,
            'ngpus': 1,
            'ncputhreads': multiprocessing.cpu_count(),
            'precision':'single',
            'compute_mode': 'sampling',
        },
    },

    'cirq': {
        'config': {
            'nshots': 1024,
            'nfused': 4,
            'ngpus': 0,
            'ncputhreads': 1,
            'precision':'single',
            'compute_mode': 'sampling',
        },
    },

    'qsim': {
        'config': {
            'nshots': 1024,
            'nfused': 2,
            'ngpus': 0,
            'ncputhreads': multiprocessing.cpu_count(),
            'precision':'single',
            'compute_mode': 'sampling',
        },
    },

    'qsim-cuda': {
        'config': {
            'nshots': 1024,
            'nfused': 2,
            'ngpus': 1,
            'ncputhreads': 1,
            'precision':'single',
            'compute_mode': 'sampling',
        },
    },

    'qsim-cusv': {
        'config': {
            'nshots': 1024,
            'nfused': 2,
            'ngpus': 1,
            'ncputhreads': 1,
            'precision':'single',
            'compute_mode': 'sampling',
        },
    },

    'qsim-mgpu': {
        'config': {
            'nshots': 1024,
            'nfused': 4,
            'ngpus': 1,
            'ncputhreads': 1,
            'precision':'single',
            'compute_mode': 'sampling',
        },
    },

    'pennylane': {
        'config': {
            'nshots': 1024,
            'nfused': None,
            'ngpus': 0,
            'ncputhreads': 1,
            'precision': 'double',
            'compute_mode': 'sampling',
        },
    },

    'pennylane-lightning-gpu': {
        'config': {
            'nshots': 1024,
            'nfused': None,
            'ngpus': 1,
            'ncputhreads': 0,
            'precision': 'single',
            'compute_mode': 'sampling',
        },
    },

    'pennylane-lightning-qubit': {
        'config': {
            'nshots': 1024,
            'nfused': None,
            'ngpus': 0,
            'ncputhreads': 1,
            'precision': 'single',
            'compute_mode': 'sampling',
        },
    },

    'pennylane-lightning-kokkos': {
        'config': {
            'nshots': 1024,
            'nfused': None,
            'ngpus': 1,
            'ncputhreads': 0,
            'precision': 'single',
            'compute_mode': 'sampling',
        },
    },

    'qulacs-gpu': {
        'config': {
            'nshots': 1024,
            'nfused': None,
            'ngpus': 1,
            'ncputhreads': 0,
            'precision': 'double',
            'compute_mode': 'sampling',
        },
    },

    'qulacs-cpu': {
        'config': {
            'nshots': 1024,
            'nfused': None,
            'ngpus': 0,
            'ncputhreads': 1,
            'precision': 'double',
            'compute_mode': 'sampling',
        },
    },

    'cudaq-cusv': {
        'config': {
            'nshots': 1024,
            'nfused': 4,
            'ngpus': 1,
            'ncputhreads': 8,
            'precision': 'double',
            'compute_mode': 'sampling',
        },
    },

    'cudaq-mgpu': {
        'config': {
            'nshots': 1024,
            'nfused': 4,
            'ngpus': 1,
            'ncputhreads': 8,
            'precision': 'double',
            'compute_mode': 'sampling',
        },
    },

    'cudaq-cpu': {
        'config': {
            'nshots': 1024,
            'nfused': None,
            'ngpus': 0,
            'ncputhreads': multiprocessing.cpu_count(),
            'precision': 'double',
            'compute_mode': 'sampling',
        },
    },

}
