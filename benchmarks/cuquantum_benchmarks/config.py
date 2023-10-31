# Copyright (c) 2021-2023, NVIDIA CORPORATION & AFFILIATES
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
        },
    },

    'aer-cuda': {
        'config': {
            'nshots': 1024,
            'nfused': 5,
            'ngpus': 1,
            'ncputhreads': multiprocessing.cpu_count(),
            'precision':'single',
        },
    },

    'aer-cusv': {
        'config': {
            'nshots': 1024,
            'nfused': 5,
            'ngpus': 1,
            'ncputhreads': multiprocessing.cpu_count(),
            'precision':'single',
        },
    },

    'cusvaer': {
        'config': {
            'nshots': 1024,
            'nfused': 4,
            'ngpus': 1,
            'ncputhreads': multiprocessing.cpu_count(),
            'precision':'single',
        },
    },

    'cirq': {
        'config': {
            'nshots': 1024,
            'nfused': 4,
            'ngpus': 0,
            'ncputhreads': 1,
            'precision':'single',
        },
    },

    'qsim': {
        'config': {
            'nshots': 1024,
            'nfused': 2,
            'ngpus': 0,
            'ncputhreads': multiprocessing.cpu_count(),
            'precision':'single',
        },
    },

    'qsim-cuda': {
        'config': {
            'nshots': 1024,
            'nfused': 2,
            'ngpus': 1,
            'ncputhreads': 1,
            'precision':'single',
        },
    },

    'qsim-cusv': {
        'config': {
            'nshots': 1024,
            'nfused': 2,
            'ngpus': 1,
            'ncputhreads': 1,
            'precision':'single',
        },
    },

    'qsim-mgpu': {
        'config': {
            'nshots': 1024,
            'nfused': 4,
            'ngpus': 1,
            'ncputhreads': 1,
            'precision':'single',
        },
    },

    'pennylane': {
        'config': {
            'nshots': 1024,
            'nfused': None,
            'ngpus': 0,
            'ncputhreads': 1,
            'precision': 'single',
        },
    },

    'pennylane-lightning-gpu': {
        'config': {
            'nshots': 1024,
            'nfused': None,
            'ngpus': 1,
            'ncputhreads': 0,
            'precision': 'single',
        },
    },

    'pennylane-lightning-qubit': {
        'config': {
            'nshots': 1024,
            'nfused': None,
            'ngpus': 0,
            'ncputhreads': 1,
            'precision': 'single',
        },
    },

    'pennylane-lightning-kokkos': {
        'config': {
            'nshots': 1024,
            'nfused': None,
            'ngpus': 1,
            'ncputhreads': 0,
            'precision': 'single',
        },
    },

    'qulacs-gpu': {
        'config': {
            'nshots': 1024,
            'nfused': None,
            'ngpus': 1,
            'ncputhreads': 0,
            'precision': 'double',
        },
    },

    'qulacs-cpu': {
        'config': {
            'nshots': 1024,
            'nfused': None,
            'ngpus': 0,
            'ncputhreads': 1,
            'precision': 'double',
        },
    },
}
