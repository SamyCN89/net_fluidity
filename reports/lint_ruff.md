UP009 [*] UTF-8 encoding declaration is unnecessary
 --> allegiance/src/1_preprocessed_data_ts_cog_groups.py:2:1
  |
1 | #!/usr/bin/env python3
2 | # -*- coding: utf-8 -*-
  | ^^^^^^^^^^^^^^^^^^^^^^^
3 | """
4 | Created on Mon Sep 23 13:26:30 2024
  |
help: Remove unnecessary coding comment

I001 [*] Import block is un-sorted or un-formatted
  --> allegiance/src/1_preprocessed_data_ts_cog_groups.py:9:1
   |
 7 |   """
 8 |   # %%
 9 | / from pathlib import Path
10 | | import numpy as np
11 | | import os
12 | | import pandas as pd
13 | | import pickle
14 | |
15 | | from shared_code.fun_loaddata import extract_hash_numbers
16 | | from shared_code.fun_utils import (
17 | |     filename_sort_mat,
18 | |     load_matdata,
19 | |     classify_phenotypes,
20 | |     make_combination_masks,
21 | |     make_masks,
22 | | )
23 | | from shared_code.fun_paths import get_paths
24 | | import matplotlib.pyplot as plt
25 | | import time
   | |___________^
26 |
27 |   # =============================================================================
   |
help: Organize imports

F401 [*] `pathlib.Path` imported but unused
  --> allegiance/src/1_preprocessed_data_ts_cog_groups.py:9:21
   |
 7 | """
 8 | # %%
 9 | from pathlib import Path
   |                     ^^^^
10 | import numpy as np
11 | import os
   |
help: Remove unused import: `pathlib.Path`

F401 [*] `matplotlib.pyplot` imported but unused
  --> allegiance/src/1_preprocessed_data_ts_cog_groups.py:24:29
   |
22 | )
23 | from shared_code.fun_paths import get_paths
24 | import matplotlib.pyplot as plt
   |                             ^^^
25 | import time
   |
help: Remove unused import: `matplotlib.pyplot`

F401 [*] `time` imported but unused
  --> allegiance/src/1_preprocessed_data_ts_cog_groups.py:25:8
   |
23 | from shared_code.fun_paths import get_paths
24 | import matplotlib.pyplot as plt
25 | import time
   |        ^^^^
26 |
27 | # =============================================================================
   |
help: Remove unused import: `time`

UP009 [*] UTF-8 encoding declaration is unnecessary
 --> allegiance/src/2_compute_dfc_local.py:2:1
  |
1 | #!/usr/bin/env python3
2 | # -*- coding: utf-8 -*-
  | ^^^^^^^^^^^^^^^^^^^^^^^
3 | """
4 | Created on Mon Sep 23 13:26:30 2024
  |
help: Remove unnecessary coding comment

I001 [*] Import block is un-sorted or un-formatted
  --> allegiance/src/2_compute_dfc_local.py:10:1
   |
 9 |   # %%
10 | / from calendar import c
11 | | from matplotlib import pyplot as pltcomputation
12 | | import numpy as np
13 | | import time
14 | | from pathlib import Path
15 | |
16 | | # from sphinx import ret
17 | | from shared_code.fun_loaddata import *
18 | | from shared_code.fun_dfcspeed import *
19 | | from shared_code.fun_metaconnectivity import *
20 | |
21 | |
22 | | from shared_code.fun_utils import (
23 | |     set_figure_params,
24 | |     #    get_paths,
25 | |     load_cognitive_data,
26 | |     load_timeseries_data,
27 | |     load_grouping_data,
28 | | )
29 | | from shared_code.fun_paths import get_paths
   | |___________________________________________^
30 |
31 |   # =============================================================================
   |
help: Organize imports

F401 [*] `calendar.c` imported but unused
  --> allegiance/src/2_compute_dfc_local.py:10:22
   |
 9 | # %%
10 | from calendar import c
   |                      ^
11 | from matplotlib import pyplot as pltcomputation
12 | import numpy as np
   |
help: Remove unused import: `calendar.c`

F401 [*] `matplotlib.pyplot` imported but unused
  --> allegiance/src/2_compute_dfc_local.py:11:34
   |
 9 | # %%
10 | from calendar import c
11 | from matplotlib import pyplot as pltcomputation
   |                                  ^^^^^^^^^^^^^^
12 | import numpy as np
13 | import time
   |
help: Remove unused import: `matplotlib.pyplot`

F401 [*] `time` imported but unused
  --> allegiance/src/2_compute_dfc_local.py:13:8
   |
11 | from matplotlib import pyplot as pltcomputation
12 | import numpy as np
13 | import time
   |        ^^^^
14 | from pathlib import Path
   |
help: Remove unused import: `time`

F401 [*] `pathlib.Path` imported but unused
  --> allegiance/src/2_compute_dfc_local.py:14:21
   |
12 | import numpy as np
13 | import time
14 | from pathlib import Path
   |                     ^^^^
15 |
16 | # from sphinx import ret
   |
help: Remove unused import: `pathlib.Path`

F403 `from shared_code.fun_loaddata import *` used; unable to detect undefined names
  --> allegiance/src/2_compute_dfc_local.py:17:1
   |
16 | # from sphinx import ret
17 | from shared_code.fun_loaddata import *
   | ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
18 | from shared_code.fun_dfcspeed import *
19 | from shared_code.fun_metaconnectivity import *
   |

F403 `from shared_code.fun_dfcspeed import *` used; unable to detect undefined names
  --> allegiance/src/2_compute_dfc_local.py:18:1
   |
16 | # from sphinx import ret
17 | from shared_code.fun_loaddata import *
18 | from shared_code.fun_dfcspeed import *
   | ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
19 | from shared_code.fun_metaconnectivity import *
   |

F403 `from shared_code.fun_metaconnectivity import *` used; unable to detect undefined names
  --> allegiance/src/2_compute_dfc_local.py:19:1
   |
17 | from shared_code.fun_loaddata import *
18 | from shared_code.fun_dfcspeed import *
19 | from shared_code.fun_metaconnectivity import *
   | ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
   |

F401 [*] `shared_code.fun_utils.set_figure_params` imported but unused
  --> allegiance/src/2_compute_dfc_local.py:23:5
   |
22 | from shared_code.fun_utils import (
23 |     set_figure_params,
   |     ^^^^^^^^^^^^^^^^^
24 |     #    get_paths,
25 |     load_cognitive_data,
   |
help: Remove unused import

F401 [*] `shared_code.fun_utils.load_cognitive_data` imported but unused
  --> allegiance/src/2_compute_dfc_local.py:25:5
   |
23 |     set_figure_params,
24 |     #    get_paths,
25 |     load_cognitive_data,
   |     ^^^^^^^^^^^^^^^^^^^
26 |     load_timeseries_data,
27 |     load_grouping_data,
   |
help: Remove unused import

F401 [*] `shared_code.fun_utils.load_grouping_data` imported but unused
  --> allegiance/src/2_compute_dfc_local.py:27:5
   |
25 |     load_cognitive_data,
26 |     load_timeseries_data,
27 |     load_grouping_data,
   |     ^^^^^^^^^^^^^^^^^^
28 | )
29 | from shared_code.fun_paths import get_paths
   |
help: Remove unused import

F405 `get_tenet4window_range` may be undefined, or defined from star imports
   --> allegiance/src/2_compute_dfc_local.py:160:1
    |
158 | #         paths[prefix], prefix, time_window_range, lag, n_animals, regions
159 | #     )
160 | get_tenet4window_range(time_window_range, prefix="dfc")
    | ^^^^^^^^^^^^^^^^^^^^^^
161 | # %%
    |

E402 Module level import not at top of file
  --> allegiance/src/allegiance_per_animal.py:73:1
   |
72 | # %%
73 | from matplotlib.colors import ListedColormap
   | ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
74 | from mizani.palettes import brewer_pal
   |

E402 Module level import not at top of file
  --> allegiance/src/allegiance_per_animal.py:74:1
   |
72 | # %%
73 | from matplotlib.colors import ListedColormap
74 | from mizani.palettes import brewer_pal
   | ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
75 |
76 | # Choose a categorical palette: 'Set1', 'Set2', 'Pastel1', etc.
   |

E402 Module level import not at top of file
   --> allegiance/src/allegiance_per_animal.py:130:1
    |
128 | # ----------------- Consensus Clustering -----------------
129 | # Compute the consensus clustering from the temporal aggregation of the contingency matrices
130 | from shared_code.fun_metaconnectivity import build_agreement_matrix_vectorized
    | ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
131 |
132 | temporal_aggregation_mat = (
    |

E402 Module level import not at top of file
   --> allegiance/src/allegiance_per_animal.py:149:1
    |
148 | # %%
149 | import brainconn as bct  # or bctpy equivalent
    | ^^^^^^^^^^^^^^^^^^^^^^^
150 | import time
    |

I001 [*] Import block is un-sorted or un-formatted
   --> allegiance/src/allegiance_per_animal.py:149:1
    |
148 |   # %%
149 | / import brainconn as bct  # or bctpy equivalent
150 | | import time
    | |___________^
151 |
152 |   _runs = 100
    |
help: Organize imports

E402 Module level import not at top of file
   --> allegiance/src/allegiance_per_animal.py:150:1
    |
148 | # %%
149 | import brainconn as bct  # or bctpy equivalent
150 | import time
    | ^^^^^^^^^^^
151 |
152 | _runs = 100
    |

B905 [*] `zip()` without an explicit `strict=` parameter
   --> allegiance/src/allegiance_per_animal.py:229:40
    |
227 | )
228 |
229 | community_agreement_labels, q_values = zip(*results)
    |                                        ^^^^^^^^^^^^^
230 | # for partition, q in results:
231 | #     partitions.append(partition)
    |
help: Add explicit value for parameter `strict=`

B905 [*] `zip()` without an explicit `strict=` parameter
   --> allegiance/src/allegiance_per_animal.py:323:17
    |
321 |     # Create new partition with remapped labels
322 |     aligned = np.zeros_like(partition)
323 |     for i, j in zip(row_ind, col_ind):
    |                 ^^^^^^^^^^^^^^^^^^^^^
324 |         aligned[partition == i] = j
325 |     return aligned
    |
help: Add explicit value for parameter `strict=`

F821 Undefined name `StandardScaler`
   --> allegiance/src/allegiance_per_animal.py:395:10
    |
393 | # TSNE on one animal of contingency matrix (contingency_matrices[0])
394 | # Standardize the data
395 | scaler = StandardScaler()
    |          ^^^^^^^^^^^^^^
396 | dfc_communities_sorted_scaled = scaler.fit_transform(
397 |     dfc_communities_sorted.reshape(-1, dfc_communities_sorted.shape[-1])
    |

F821 Undefined name `TSNE`
   --> allegiance/src/allegiance_per_animal.py:401:8
    |
400 | # Perform t-SNE
401 | tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    |        ^^^^
402 | cont_mat_n_pairs_tsne = tsne.fit_transform(cont_mat_n_pairs)
403 | # %%
    |

E402 Module level import not at top of file
   --> allegiance/src/allegiance_per_animal.py:693:1
    |
691 | # %%
692 | # TSNE on one animal of contingency matrix (contingency_matrices[0])
693 | from sklearn.manifold import TSNE
    | ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
694 | from sklearn.preprocessing import StandardScaler
    |

E402 Module level import not at top of file
   --> allegiance/src/allegiance_per_animal.py:694:1
    |
692 | # TSNE on one animal of contingency matrix (contingency_matrices[0])
693 | from sklearn.manifold import TSNE
694 | from sklearn.preprocessing import StandardScaler
    | ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
695 |
696 | # Standardize the data
    |

I001 [*] Import block is un-sorted or un-formatted
  --> allegiance/src/allegiance_per_animal_v2.py:2:1
   |
 1 |   # %%
 2 | / from joblib import Parallel
 3 | | import numpy as np
 4 | | from tqdm import tqdm
 5 | | import matplotlib.pyplot as plt
 6 | | from shared_code.fun_paths import get_paths
 7 | | from shared_code.fun_metaconnectivity import load_merged_allegiance  # %%
 8 | | import numpy as np
 9 | | from tqdm import tqdm
10 | | import matplotlib.pyplot as plt
11 | | from shared_code.fun_paths import get_paths
12 | | from shared_code.fun_metaconnectivity import load_merged_allegiance
13 | | import pickle
   | |_____________^
14 |
15 |   # Set consistent config to match previous run
   |
help: Organize imports

F811 [*] Redefinition of unused `np` from line 3
  --> allegiance/src/allegiance_per_animal_v2.py:3:17
   |
 1 | # %%
 2 | from joblib import Parallel
 3 | import numpy as np
   |                 -- previous definition of `np` here
 4 | from tqdm import tqdm
 5 | import matplotlib.pyplot as plt
 6 | from shared_code.fun_paths import get_paths
 7 | from shared_code.fun_metaconnectivity import load_merged_allegiance  # %%
 8 | import numpy as np
   |                 ^^ `np` redefined here
 9 | from tqdm import tqdm
10 | import matplotlib.pyplot as plt
   |
help: Remove definition: `np`

F811 [*] Redefinition of unused `tqdm` from line 4
  --> allegiance/src/allegiance_per_animal_v2.py:4:18
   |
 2 | from joblib import Parallel
 3 | import numpy as np
 4 | from tqdm import tqdm
   |                  ---- previous definition of `tqdm` here
 5 | import matplotlib.pyplot as plt
 6 | from shared_code.fun_paths import get_paths
 7 | from shared_code.fun_metaconnectivity import load_merged_allegiance  # %%
 8 | import numpy as np
 9 | from tqdm import tqdm
   |                  ^^^^ `tqdm` redefined here
10 | import matplotlib.pyplot as plt
11 | from shared_code.fun_paths import get_paths
   |
help: Remove definition: `tqdm`

F811 [*] Redefinition of unused `plt` from line 5
  --> allegiance/src/allegiance_per_animal_v2.py:5:29
   |
 3 | import numpy as np
 4 | from tqdm import tqdm
 5 | import matplotlib.pyplot as plt
   |                             --- previous definition of `plt` here
 6 | from shared_code.fun_paths import get_paths
 7 | from shared_code.fun_metaconnectivity import load_merged_allegiance  # %%
 8 | import numpy as np
 9 | from tqdm import tqdm
10 | import matplotlib.pyplot as plt
   |                             ^^^ `plt` redefined here
11 | from shared_code.fun_paths import get_paths
12 | from shared_code.fun_metaconnectivity import load_merged_allegiance
   |
help: Remove definition: `plt`

F811 [*] Redefinition of unused `get_paths` from line 6
  --> allegiance/src/allegiance_per_animal_v2.py:6:35
   |
 4 | from tqdm import tqdm
 5 | import matplotlib.pyplot as plt
 6 | from shared_code.fun_paths import get_paths
   |                                   --------- previous definition of `get_paths` here
 7 | from shared_code.fun_metaconnectivity import load_merged_allegiance  # %%
 8 | import numpy as np
 9 | from tqdm import tqdm
10 | import matplotlib.pyplot as plt
11 | from shared_code.fun_paths import get_paths
   |                                   ^^^^^^^^^ `get_paths` redefined here
12 | from shared_code.fun_metaconnectivity import load_merged_allegiance
13 | import pickle
   |
help: Remove definition: `get_paths`

F811 [*] Redefinition of unused `load_merged_allegiance` from line 7
  --> allegiance/src/allegiance_per_animal_v2.py:7:46
   |
 5 | import matplotlib.pyplot as plt
 6 | from shared_code.fun_paths import get_paths
 7 | from shared_code.fun_metaconnectivity import load_merged_allegiance  # %%
   |                                              ---------------------- previous definition of `load_merged_allegiance` here
 8 | import numpy as np
 9 | from tqdm import tqdm
10 | import matplotlib.pyplot as plt
11 | from shared_code.fun_paths import get_paths
12 | from shared_code.fun_metaconnectivity import load_merged_allegiance
   |                                              ^^^^^^^^^^^^^^^^^^^^^^ `load_merged_allegiance` redefined here
13 | import pickle
   |
help: Remove definition: `load_merged_allegiance`

E402 Module level import not at top of file
   --> allegiance/src/allegiance_per_animal_v2.py:188:1
    |
186 | # %%
187 | # the spearman correlation between time points in contingency_matrices_0_triu (time_points x n_pairs)
188 | from scipy.stats import spearmanr
    | ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
189 |
190 | time_corr_agreement = np.zeros(n_windows - 1)  # Initialize the correlation matrix
    |

E402 Module level import not at top of file
   --> allegiance/src/allegiance_per_animal_v2.py:204:1
    |
202 | plt.plot(time_corr_agreement, "o-", markersize=5, alpha=0.7)
203 | # %%
204 | from scipy.stats import spearmanr, pearsonr
    | ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
205 |
206 | # adjusted rand index
    |

I001 [*] Import block is un-sorted or un-formatted
   --> allegiance/src/allegiance_per_animal_v2.py:204:1
    |
202 | plt.plot(time_corr_agreement, "o-", markersize=5, alpha=0.7)
203 | # %%
204 | from scipy.stats import spearmanr, pearsonr
    | ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
205 |
206 | # adjusted rand index
    |
help: Organize imports

E402 Module level import not at top of file
   --> allegiance/src/allegiance_per_animal_v2.py:305:1
    |
303 | # %%
304 | # Mutual Information between columns of the dfc_communities_sorted matrix
305 | from sklearn.metrics import mutual_info_score
    | ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
306 |
307 | mi_mat = np.zeros((n_windows, n_windows))
    |

E402 Module level import not at top of file
   --> allegiance/src/allegiance_per_animal_v2.py:328:1
    |
326 | # %%
327 |
328 | from matplotlib.colors import ListedColormap
    | ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
329 | from mizani.palettes import brewer_pal
    |

E402 Module level import not at top of file
   --> allegiance/src/allegiance_per_animal_v2.py:329:1
    |
328 | from matplotlib.colors import ListedColormap
329 | from mizani.palettes import brewer_pal
    | ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
330 |
331 | # Choose a categorical palette: 'Set1', 'Set2', 'Pastel1', etc.
    |

E402 Module level import not at top of file
   --> allegiance/src/allegiance_per_animal_v2.py:385:1
    |
383 | # ----------------- Consensus Clustering -----------------
384 | # Compute the consensus clustering from the temporal aggregation of the contingency matrices
385 | from shared_code.fun_metaconnectivity import build_agreement_matrix_vectorized
    | ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
386 |
387 | temporal_aggregation_mat = (
    |

E402 Module level import not at top of file
   --> allegiance/src/allegiance_per_animal_v2.py:404:1
    |
403 | # %%
404 | import brainconn as bct  # or bctpy equivalent
    | ^^^^^^^^^^^^^^^^^^^^^^^
405 | from joblib import Parallel, delayed
406 | import time
    |

I001 [*] Import block is un-sorted or un-formatted
   --> allegiance/src/allegiance_per_animal_v2.py:404:1
    |
403 |   # %%
404 | / import brainconn as bct  # or bctpy equivalent
405 | | from joblib import Parallel, delayed
406 | | import time
407 | | from scipy.stats import pearsonr
    | |________________________________^
408 |
409 |   _runs = 100
    |
help: Organize imports

E402 Module level import not at top of file
   --> allegiance/src/allegiance_per_animal_v2.py:405:1
    |
403 | # %%
404 | import brainconn as bct  # or bctpy equivalent
405 | from joblib import Parallel, delayed
    | ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
406 | import time
407 | from scipy.stats import pearsonr
    |

F811 [*] Redefinition of unused `Parallel` from line 2
   --> allegiance/src/allegiance_per_animal_v2.py:405:20
    |
403 | # %%
404 | import brainconn as bct  # or bctpy equivalent
405 | from joblib import Parallel, delayed
    |                    ^^^^^^^^ `Parallel` redefined here
406 | import time
407 | from scipy.stats import pearsonr
    |
   ::: allegiance/src/allegiance_per_animal_v2.py:2:20
    |
  1 | # %%
  2 | from joblib import Parallel
    |                    -------- previous definition of `Parallel` here
  3 | import numpy as np
  4 | from tqdm import tqdm
    |
help: Remove definition: `Parallel`

E402 Module level import not at top of file
   --> allegiance/src/allegiance_per_animal_v2.py:406:1
    |
404 | import brainconn as bct  # or bctpy equivalent
405 | from joblib import Parallel, delayed
406 | import time
    | ^^^^^^^^^^^
407 | from scipy.stats import pearsonr
    |

E402 Module level import not at top of file
   --> allegiance/src/allegiance_per_animal_v2.py:407:1
    |
405 | from joblib import Parallel, delayed
406 | import time
407 | from scipy.stats import pearsonr
    | ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
408 |
409 | _runs = 100
    |

B905 [*] `zip()` without an explicit `strict=` parameter
   --> allegiance/src/allegiance_per_animal_v2.py:486:40
    |
484 | )
485 |
486 | community_agreement_labels, q_values = zip(*results)
    |                                        ^^^^^^^^^^^^^
487 | # for partition, q in results:
488 | #     partitions.append(partition)
    |
help: Add explicit value for parameter `strict=`

E402 Module level import not at top of file
   --> allegiance/src/allegiance_per_animal_v2.py:505:1
    |
505 | from scipy.optimize import linear_sum_assignment
    | ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    |

E402 Module level import not at top of file
   --> allegiance/src/allegiance_per_animal_v2.py:567:1
    |
565 | # Consensus clustering with temporal aggregation matrix as reference
566 |
567 | from scipy.optimize import linear_sum_assignment
    | ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    |

I001 [*] Import block is un-sorted or un-formatted
   --> allegiance/src/allegiance_per_animal_v2.py:567:1
    |
565 | # Consensus clustering with temporal aggregation matrix as reference
566 |
567 | from scipy.optimize import linear_sum_assignment
    | ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    |
help: Organize imports

F811 [*] Redefinition of unused `linear_sum_assignment` from line 505
   --> allegiance/src/allegiance_per_animal_v2.py:567:28
    |
565 | # Consensus clustering with temporal aggregation matrix as reference
566 |
567 | from scipy.optimize import linear_sum_assignment
    |                            ^^^^^^^^^^^^^^^^^^^^^ `linear_sum_assignment` redefined here
    |
   ::: allegiance/src/allegiance_per_animal_v2.py:505:28
    |
505 | from scipy.optimize import linear_sum_assignment
    |                            --------------------- previous definition of `linear_sum_assignment` here
    |
help: Remove definition: `linear_sum_assignment`

B905 [*] `zip()` without an explicit `strict=` parameter
   --> allegiance/src/allegiance_per_animal_v2.py:585:17
    |
583 |     # Create new partition with remapped labels
584 |     aligned = np.zeros_like(partition)
585 |     for i, j in zip(row_ind, col_ind):
    |                 ^^^^^^^^^^^^^^^^^^^^^
586 |         aligned[partition == i] = j
587 |     return aligned
    |
help: Add explicit value for parameter `strict=`

E402 Module level import not at top of file
   --> allegiance/src/allegiance_per_animal_v2.py:656:1
    |
654 | # %%
655 | # TSNE on one animal of contingency matrix (contingency_matrices[0])
656 | from sklearn.manifold import TSNE
    | ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
657 | from sklearn.preprocessing import StandardScaler
    |

E402 Module level import not at top of file
   --> allegiance/src/allegiance_per_animal_v2.py:657:1
    |
655 | # TSNE on one animal of contingency matrix (contingency_matrices[0])
656 | from sklearn.manifold import TSNE
657 | from sklearn.preprocessing import StandardScaler
    | ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
658 |
659 | # Standardize the data
    |

E402 Module level import not at top of file
   --> allegiance/src/allegiance_per_animal_v2.py:858:1
    |
857 | # "consensus" community structure over the whole period with Louvain method
858 | from shared_code.fun_metaconnectivity import contingency_matrix_fun
    | ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
859 |
860 | # contingency_matrix, gamma_qmod_val, gamma_agreement_mat =contingency_matrix_fun(1000, mc_data=allegiance_avg, gamma_range=10, gmin=…
    |

E402 Module level import not at top of file
   --> allegiance/src/allegiance_per_animal_v2.py:919:1
    |
918 | # %%
919 | import pickle
    | ^^^^^^^^^^^^^
920 |
921 | with open(paths["sorted"] / "grouping_data_oip.pkl", "rb") as f:
    |

E402 Module level import not at top of file
   --> allegiance/src/allegiance_per_animal_v2.py:962:1
    |
960 | # %%
961 | # TSNE on one animal of contingency matrix (contingency_matrices[0])
962 | from sklearn.manifold import TSNE
    | ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
963 | from sklearn.preprocessing import StandardScaler
    |

E402 Module level import not at top of file
   --> allegiance/src/allegiance_per_animal_v2.py:963:1
    |
961 | # TSNE on one animal of contingency matrix (contingency_matrices[0])
962 | from sklearn.manifold import TSNE
963 | from sklearn.preprocessing import StandardScaler
    | ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
964 |
965 | # Standardize the data
    |

E402 Module level import not at top of file
    --> allegiance/src/allegiance_per_animal_v2.py:1211:1
     |
1210 | # "consensus" community structure over the whole period with Louvain method
1211 | from shared_code.fun_metaconnectivity import contingency_matrix_fun
     | ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
1212 |
1213 | # contingency_matrix, gamma_qmod_val, gamma_agreement_mat =contingency_matrix_fun(1000, mc_data=allegiance_avg, gamma_range=10, gmin…
     |

UP009 [*] UTF-8 encoding declaration is unnecessary
 --> allegiance/src/burst_detection_PBM.py:2:1
  |
1 | #!/usr/bin/env python3
2 | # -*- coding: utf-8 -*-
  | ^^^^^^^^^^^^^^^^^^^^^^^
3 | """
4 | Created on Wed Apr  2 02:59:41 2025
  |
help: Remove unnecessary coding comment

I001 [*] Import block is un-sorted or un-formatted
  --> allegiance/src/burst_detection_PBM.py:9:1
   |
 7 |   """
 8 |   # %%
 9 | / from os import path
10 | | import time
11 | | from pathlib import Path
12 | |
13 | | import numpy as np
14 | | import pandas as pd
15 | | import matplotlib.pyplot as plt
16 | | import seaborn as sns
17 | | import brainconn as bct
18 | |
19 | |
20 | | from scipy.stats import zscore, pearsonr
21 | |
22 | | # Compute k-means clustering on the z-scored time series
23 | | from sklearn.cluster import KMeans
24 | | from sklearn.metrics import silhouette_score
25 | |
26 | | from shared_code.fun_utils import set_figure_params
27 | | from shared_code.fun_paths import get_paths
28 | | from shared_code.fun_dfcspeed import ts2fc
   | |__________________________________________^
29 |
30 |   # ========================== Figure parameters ================================
   |
help: Organize imports

F401 [*] `os.path` imported but unused
  --> allegiance/src/burst_detection_PBM.py:9:16
   |
 7 | """
 8 | # %%
 9 | from os import path
   |                ^^^^
10 | import time
11 | from pathlib import Path
   |
help: Remove unused import: `os.path`

F401 [*] `pathlib.Path` imported but unused
  --> allegiance/src/burst_detection_PBM.py:11:21
   |
 9 | from os import path
10 | import time
11 | from pathlib import Path
   |                     ^^^^
12 |
13 | import numpy as np
   |
help: Remove unused import: `pathlib.Path`

F401 [*] `scipy.stats.pearsonr` imported but unused
  --> allegiance/src/burst_detection_PBM.py:20:33
   |
20 | from scipy.stats import zscore, pearsonr
   |                                 ^^^^^^^^
21 |
22 | # Compute k-means clustering on the z-scored time series
   |
help: Remove unused import: `scipy.stats.pearsonr`

B905 [*] `zip()` without an explicit `strict=` parameter
   --> allegiance/src/burst_detection_PBM.py:342:32
    |
340 |             events = [
341 |                 {"onset": int(o), "offset": int(f), "duration": int(d)}
342 |                 for o, f, d in zip(onsets, offsets, durations)
    |                                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
343 |             ]
344 |             animal_events.append(events)
    |
help: Add explicit value for parameter `strict=`

F821 Undefined name `save_figure`
   --> allegiance/src/burst_detection_PBM.py:441:5
    |
439 |     plt.title("Burst Duration Distributions per Link")
440 |     plt.tight_layout()
441 |     save_figure(fig_path / "link_burst_durations.png", save_fig)
    |     ^^^^^^^^^^^
442 |     plt.show()
    |

F821 Undefined name `fc_clusters`
   --> allegiance/src/burst_detection_PBM.py:445:38
    |
445 | analyze_link_dynamics(kmeans_labels, fc_clusters, meta, save_fig, fig_path)
    |                                      ^^^^^^^^^^^
446 |
447 | # aux_plot = []
    |

F821 Undefined name `meta`
   --> allegiance/src/burst_detection_PBM.py:445:51
    |
445 | analyze_link_dynamics(kmeans_labels, fc_clusters, meta, save_fig, fig_path)
    |                                                   ^^^^
446 |
447 | # aux_plot = []
    |

F821 Undefined name `fig_path`
   --> allegiance/src/burst_detection_PBM.py:445:67
    |
445 | analyze_link_dynamics(kmeans_labels, fc_clusters, meta, save_fig, fig_path)
    |                                                                   ^^^^^^^^
446 |
447 | # aux_plot = []
    |

UP009 [*] UTF-8 encoding declaration is unnecessary
 --> allegiance/src/compute_allegiance_local.py:2:1
  |
1 | #!/usr/bin/env python3
2 | # -*- coding: utf-8 -*-
  | ^^^^^^^^^^^^^^^^^^^^^^^
3 | """
4 | Created on Mon Sep 23 13:26:30 2024
  |
help: Remove unnecessary coding comment

I001 [*] Import block is un-sorted or un-formatted
  --> allegiance/src/compute_allegiance_local.py:10:1
   |
 9 |   # %%
10 | / from calendar import c
11 | | from matplotlib import pyplot as plt
12 | | import numpy as np
13 | | import time
14 | |
15 | | # from functions_analysis import *
16 | | from pathlib import Path
17 | | import sys
   | |__________^
18 |
19 |   sys.path.append("../../shared_code")
   |
help: Organize imports

F401 [*] `calendar.c` imported but unused
  --> allegiance/src/compute_allegiance_local.py:10:22
   |
 9 | # %%
10 | from calendar import c
   |                      ^
11 | from matplotlib import pyplot as plt
12 | import numpy as np
   |
help: Remove unused import: `calendar.c`

F403 `from shared_code.fun_loaddata import *` used; unable to detect undefined names
  --> allegiance/src/compute_allegiance_local.py:23:1
   |
21 | # from sphinx import ret
22 |
23 | from shared_code.fun_loaddata import *
   | ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
24 | from shared_code.fun_dfcspeed import *
   |

I001 [*] Import block is un-sorted or un-formatted
  --> allegiance/src/compute_allegiance_local.py:23:1
   |
21 |   # from sphinx import ret
22 |
23 | / from shared_code.fun_loaddata import *
24 | | from shared_code.fun_dfcspeed import *
25 | |
26 | | from shared_code.fun_metaconnectivity import (
27 | |     compute_metaconnectivity,
28 | |     intramodule_indices_mask,
29 | |     get_fc_mc_indices,
30 | |     get_mc_region_identities,
31 | |     fun_allegiance_communities,
32 | |     compute_trimers_identity,
33 | |     build_trimer_mask,
34 | | )
35 | |
36 | | from shared_code.fun_utils import (
37 | |     set_figure_params,
38 | |     #    get_paths,
39 | |     load_cognitive_data,
40 | |     load_timeseries_data,
41 | |     load_grouping_data,
42 | | )
43 | | from shared_code.fun_paths import get_paths
   | |___________________________________________^
44 |
45 |   # =============================================================================
   |
help: Organize imports

F403 `from shared_code.fun_dfcspeed import *` used; unable to detect undefined names
  --> allegiance/src/compute_allegiance_local.py:24:1
   |
23 | from shared_code.fun_loaddata import *
24 | from shared_code.fun_dfcspeed import *
   | ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
25 |
26 | from shared_code.fun_metaconnectivity import (
   |

F401 [*] `shared_code.fun_metaconnectivity.compute_metaconnectivity` imported but unused
  --> allegiance/src/compute_allegiance_local.py:27:5
   |
26 | from shared_code.fun_metaconnectivity import (
27 |     compute_metaconnectivity,
   |     ^^^^^^^^^^^^^^^^^^^^^^^^
28 |     intramodule_indices_mask,
29 |     get_fc_mc_indices,
   |
help: Remove unused import

F401 [*] `shared_code.fun_metaconnectivity.intramodule_indices_mask` imported but unused
  --> allegiance/src/compute_allegiance_local.py:28:5
   |
26 | from shared_code.fun_metaconnectivity import (
27 |     compute_metaconnectivity,
28 |     intramodule_indices_mask,
   |     ^^^^^^^^^^^^^^^^^^^^^^^^
29 |     get_fc_mc_indices,
30 |     get_mc_region_identities,
   |
help: Remove unused import

F401 [*] `shared_code.fun_metaconnectivity.get_fc_mc_indices` imported but unused
  --> allegiance/src/compute_allegiance_local.py:29:5
   |
27 |     compute_metaconnectivity,
28 |     intramodule_indices_mask,
29 |     get_fc_mc_indices,
   |     ^^^^^^^^^^^^^^^^^
30 |     get_mc_region_identities,
31 |     fun_allegiance_communities,
   |
help: Remove unused import

F401 [*] `shared_code.fun_metaconnectivity.get_mc_region_identities` imported but unused
  --> allegiance/src/compute_allegiance_local.py:30:5
   |
28 |     intramodule_indices_mask,
29 |     get_fc_mc_indices,
30 |     get_mc_region_identities,
   |     ^^^^^^^^^^^^^^^^^^^^^^^^
31 |     fun_allegiance_communities,
32 |     compute_trimers_identity,
   |
help: Remove unused import

F401 [*] `shared_code.fun_metaconnectivity.compute_trimers_identity` imported but unused
  --> allegiance/src/compute_allegiance_local.py:32:5
   |
30 |     get_mc_region_identities,
31 |     fun_allegiance_communities,
32 |     compute_trimers_identity,
   |     ^^^^^^^^^^^^^^^^^^^^^^^^
33 |     build_trimer_mask,
34 | )
   |
help: Remove unused import

F401 [*] `shared_code.fun_metaconnectivity.build_trimer_mask` imported but unused
  --> allegiance/src/compute_allegiance_local.py:33:5
   |
31 |     fun_allegiance_communities,
32 |     compute_trimers_identity,
33 |     build_trimer_mask,
   |     ^^^^^^^^^^^^^^^^^
34 | )
   |
help: Remove unused import

E402 Module level import not at top of file
   --> allegiance/src/compute_allegiance_local.py:120:1
    |
119 | # %%Metaconnectivity
120 | from joblib import Parallel, delayed, parallel_backend
    | ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    |

F841 Local variable `mc` is assigned to but never used
   --> allegiance/src/compute_allegiance_local.py:171:5
    |
169 |     n_animals, tr_points, nodes = ts_data.shape
170 |     dfc_stream = None
171 |     mc = None
    |     ^^
172 |
173 |     # File path setup
    |
help: Remove assignment to unused variable `mc`

F405 `ts2dfc_stream` may be undefined, or defined from star imports
   --> allegiance/src/compute_allegiance_local.py:199:25
    |
197 |         with parallel_backend("loky", n_jobs=n_jobs):
198 |             dfc_stream_list = Parallel()(
199 |                 delayed(ts2dfc_stream)(
    |                         ^^^^^^^^^^^^^
200 |                     ts_data[i], window_size, lag, format_data=format_data
201 |                 )
    |

I001 [*] Import block is un-sorted or un-formatted
 --> allegiance/src/merge_allegiance_parallel.py:4:1
  |
3 |   # %%
4 | / import numpy as np
5 | | from pathlib import Path
6 | | from shared_code.fun_paths import get_paths
7 | | from shared_code.fun_metaconnectivity import load_merged_allegiance
8 | | from tqdm import tqdm
  | |_____________________^
  |
help: Organize imports

F401 [*] `pathlib.Path` imported but unused
 --> allegiance/src/merge_allegiance_parallel.py:5:21
  |
3 | # %%
4 | import numpy as np
5 | from pathlib import Path
  |                     ^^^^
6 | from shared_code.fun_paths import get_paths
7 | from shared_code.fun_metaconnectivity import load_merged_allegiance
  |
help: Remove unused import: `pathlib.Path`

I001 [*] Import block is un-sorted or un-formatted
  --> allegiance/src/run_all_allegiance_local.py:1:1
   |
 1 | / import numpy as np
 2 | | from pathlib import Path
 3 | | from joblib import Parallel, delayed
 4 | | from shared_code.fun_utils import load_timeseries_data
 5 | | from shared_code.fun_paths import get_paths
 6 | | from shared_code.fun_metaconnectivity import fun_allegiance_communities
 7 | | import os
 8 | | import logging
 9 | |
10 | | import argparse
   | |_______________^
11 |
12 |   # ===================== CLI ARGUMENTS ==========================
   |
help: Organize imports

I001 [*] Import block is un-sorted or un-formatted
  --> allegiance/src/test_plt.py:2:1
   |
 1 |   # %%
 2 | / import numpy as np
 3 | | import matplotlib.pyplot as plt
 4 | | from pathlib import Path
 5 | | from tqdm import tqdm
 6 | | from sklearn.manifold import TSNE
 7 | | from sklearn.preprocessing import StandardScaler
 8 | | import pickle
 9 | |
10 | | from shared_code.fun_paths import get_paths
11 | | from shared_code.fun_metaconnectivity import (
12 | |     load_merged_allegiance,
13 | |     contingency_matrix_fun,
14 | | )
   | |_^
15 |
16 |   # --- CONFIG ---
   |
help: Organize imports

F401 [*] `pathlib.Path` imported but unused
 --> allegiance/src/test_plt.py:4:21
  |
2 | import numpy as np
3 | import matplotlib.pyplot as plt
4 | from pathlib import Path
  |                     ^^^^
5 | from tqdm import tqdm
6 | from sklearn.manifold import TSNE
  |
help: Remove unused import: `pathlib.Path`

E712 Avoid equality comparisons to `True`; use `grp:` for truth checks
   --> allegiance/src/test_plt.py:146:29
    |
145 | ind_grp_sort = np.concatenate(
146 |     [np.squeeze(np.argwhere(grp == True)) for grp in mask_groups[2]]
    |                             ^^^^^^^^^^^
147 | )
148 | # sort the cont_mat_n_pairs_agg_fcd by ind_grp_sort
    |
help: Replace with `grp`

F821 Undefined name `aux_group`
   --> allegiance/src/test_plt.py:264:20
    |
262 | # for ii, aux_group in enumerate(mask_groups[2]):
263 | # plt.subplot(2, 2, 1+ii)
264 | np.shape(agreement[aux_group])
    |                    ^^^^^^^^^
265 | aux_agreement = agreement[91]
266 | plt.imshow(aux_agreement, aspect="auto", interpolation="none", cmap="viridis")
    |

UP009 [*] UTF-8 encoding declaration is unnecessary
 --> julien_data/1_preprocess_data_ts_cog.py:2:1
  |
1 | #!/usr/bin/env python3
2 | # -*- coding: utf-8 -*-
  | ^^^^^^^^^^^^^^^^^^^^^^^
3 | """
4 | Created on Mon Sep 23 13:26:30 2024
  |
help: Remove unnecessary coding comment

I001 [*] Import block is un-sorted or un-formatted
  --> julien_data/1_preprocess_data_ts_cog.py:9:1
   |
 7 |   """
 8 |   # %%
 9 | / from pathlib import Path
10 | | import numpy as np
11 | | import time
12 | | import pandas as pd
13 | | import pickle
14 | | from scipy.io import loadmat
15 | | from collections import defaultdict
16 | |
17 | | from shared_code.fun_loaddata import (
18 | |     extract_hash_numbers,
19 | |     load_mat_timeseries,
20 | |     extract_mouse_ids,
21 | | )
22 | | from shared_code.fun_utils import (
23 | |     filename_sort_mat,
24 | |     load_matdata,
25 | |     classify_phenotypes,
26 | |     make_combination_masks,
27 | |     make_masks,
28 | | )
29 | | from shared_code.fun_paths import get_paths
30 | | import matplotlib.pyplot as plt
   | |_______________________________^
31 |
32 |   # %%
   |
help: Organize imports

F401 [*] `pathlib.Path` imported but unused
  --> julien_data/1_preprocess_data_ts_cog.py:9:21
   |
 7 | """
 8 | # %%
 9 | from pathlib import Path
   |                     ^^^^
10 | import numpy as np
11 | import time
   |
help: Remove unused import: `pathlib.Path`

F401 [*] `time` imported but unused
  --> julien_data/1_preprocess_data_ts_cog.py:11:8
   |
 9 | from pathlib import Path
10 | import numpy as np
11 | import time
   |        ^^^^
12 | import pandas as pd
13 | import pickle
   |
help: Remove unused import: `time`

F401 [*] `scipy.io.loadmat` imported but unused
  --> julien_data/1_preprocess_data_ts_cog.py:14:22
   |
12 | import pandas as pd
13 | import pickle
14 | from scipy.io import loadmat
   |                      ^^^^^^^
15 | from collections import defaultdict
   |
help: Remove unused import: `scipy.io.loadmat`

F401 [*] `collections.defaultdict` imported but unused
  --> julien_data/1_preprocess_data_ts_cog.py:15:25
   |
13 | import pickle
14 | from scipy.io import loadmat
15 | from collections import defaultdict
   |                         ^^^^^^^^^^^
16 |
17 | from shared_code.fun_loaddata import (
   |
help: Remove unused import: `collections.defaultdict`

F401 [*] `shared_code.fun_loaddata.extract_hash_numbers` imported but unused
  --> julien_data/1_preprocess_data_ts_cog.py:18:5
   |
17 | from shared_code.fun_loaddata import (
18 |     extract_hash_numbers,
   |     ^^^^^^^^^^^^^^^^^^^^
19 |     load_mat_timeseries,
20 |     extract_mouse_ids,
   |
help: Remove unused import: `shared_code.fun_loaddata.extract_hash_numbers`

F401 [*] `shared_code.fun_utils.filename_sort_mat` imported but unused
  --> julien_data/1_preprocess_data_ts_cog.py:23:5
   |
21 | )
22 | from shared_code.fun_utils import (
23 |     filename_sort_mat,
   |     ^^^^^^^^^^^^^^^^^
24 |     load_matdata,
25 |     classify_phenotypes,
   |
help: Remove unused import

F401 [*] `shared_code.fun_utils.load_matdata` imported but unused
  --> julien_data/1_preprocess_data_ts_cog.py:24:5
   |
22 | from shared_code.fun_utils import (
23 |     filename_sort_mat,
24 |     load_matdata,
   |     ^^^^^^^^^^^^
25 |     classify_phenotypes,
26 |     make_combination_masks,
   |
help: Remove unused import

F401 [*] `shared_code.fun_utils.classify_phenotypes` imported but unused
  --> julien_data/1_preprocess_data_ts_cog.py:25:5
   |
23 |     filename_sort_mat,
24 |     load_matdata,
25 |     classify_phenotypes,
   |     ^^^^^^^^^^^^^^^^^^^
26 |     make_combination_masks,
27 |     make_masks,
   |
help: Remove unused import

F401 [*] `shared_code.fun_utils.make_combination_masks` imported but unused
  --> julien_data/1_preprocess_data_ts_cog.py:26:5
   |
24 |     load_matdata,
25 |     classify_phenotypes,
26 |     make_combination_masks,
   |     ^^^^^^^^^^^^^^^^^^^^^^
27 |     make_masks,
28 | )
   |
help: Remove unused import

F401 [*] `shared_code.fun_utils.make_masks` imported but unused
  --> julien_data/1_preprocess_data_ts_cog.py:27:5
   |
25 |     classify_phenotypes,
26 |     make_combination_masks,
27 |     make_masks,
   |     ^^^^^^^^^^
28 | )
29 | from shared_code.fun_paths import get_paths
   |
help: Remove unused import

F401 [*] `matplotlib.pyplot` imported but unused
  --> julien_data/1_preprocess_data_ts_cog.py:30:29
   |
28 | )
29 | from shared_code.fun_paths import get_paths
30 | import matplotlib.pyplot as plt
   |                             ^^^
31 |
32 | # %%
   |
help: Remove unused import: `matplotlib.pyplot`

B905 [*] `zip()` without an explicit `strict=` parameter
  --> julien_data/1_preprocess_data_ts_cog.py:74:45
   |
72 |         # Print only the filename and shape of the smallest time series
73 |         min_shape = min(ts_shapes)
74 |         for idx, (file, shape) in enumerate(zip(loaded_files, ts_shapes)):
   |                                             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
75 |             if shape == min_shape:
76 |                 print(f"{file}: {shape}")
   |
help: Add explicit value for parameter `strict=`

B905 [*] `zip()` without an explicit `strict=` parameter
  --> julien_data/1_preprocess_data_ts_cog.py:86:28
   |
84 |         filtered = [
85 |             (ts, id_)
86 |             for ts, id_ in zip(ts_list, ts_ids)
   |                            ^^^^^^^^^^^^^^^^^^^^
87 |             if ts.shape[0] > min_timepoints
88 |         ]
   |
help: Add explicit value for parameter `strict=`

B905 [*] `zip()` without an explicit `strict=` parameter
   --> julien_data/1_preprocess_data_ts_cog.py:120:38
    |
119 |     # List of time series that match the mouse IDs in the cognitive data, preserving the order
120 |     ts_filtered = [ts for ts, id_ in zip(ts_list, ts_ids) if id_ in matched_ids]
    |                                      ^^^^^^^^^^^^^^^^^^^^
121 |     ts_ids_filtered = [id_ for id_ in ts_ids if id_ in matched_ids]
    |
help: Add explicit value for parameter `strict=`

F841 Local variable `ts_ids_filtered` is assigned to but never used
   --> julien_data/1_preprocess_data_ts_cog.py:121:5
    |
119 |     # List of time series that match the mouse IDs in the cognitive data, preserving the order
120 |     ts_filtered = [ts for ts, id_ in zip(ts_list, ts_ids) if id_ in matched_ids]
121 |     ts_ids_filtered = [id_ for id_ in ts_ids if id_ in matched_ids]
    |     ^^^^^^^^^^^^^^^
122 |
123 |     # Print excluded mice
    |
help: Remove assignment to unused variable `ts_ids_filtered`

UP009 [*] UTF-8 encoding declaration is unnecessary
 --> julien_data/2_compute_dfc_stream.py:2:1
  |
1 | #!/usr/bin/env python3
2 | # -*- coding: utf-8 -*-
  | ^^^^^^^^^^^^^^^^^^^^^^^
3 | """
4 | Created on Mon Oct  2 14:42:38 2023
  |
help: Remove unnecessary coding comment

I001 [*] Import block is un-sorted or un-formatted
  --> julien_data/2_compute_dfc_stream.py:11:1
   |
10 |   # %%
11 | / from os import path
12 | | from pathlib import Path
13 | | import comm
14 | | import numpy as np
15 | | import pandas as pd
16 | | from shared_code.fun_loaddata import *
17 | | from shared_code.fun_dfcspeed import get_tenet4window_range
18 | | from shared_code.fun_paths import get_paths
19 | | from tqdm import tqdm
20 | | import pickle
21 | |
22 | | # %% Define paths
23 | |
24 | | from class_dataanalysis_julien import DFCAnalysis
   | |_________________________________________________^
25 |
26 |   data = DFCAnalysis()
   |
help: Organize imports

F401 [*] `os.path` imported but unused
  --> julien_data/2_compute_dfc_stream.py:11:16
   |
10 | # %%
11 | from os import path
   |                ^^^^
12 | from pathlib import Path
13 | import comm
   |
help: Remove unused import: `os.path`

F401 [*] `comm` imported but unused
  --> julien_data/2_compute_dfc_stream.py:13:8
   |
11 | from os import path
12 | from pathlib import Path
13 | import comm
   |        ^^^^
14 | import numpy as np
15 | import pandas as pd
   |
help: Remove unused import: `comm`

F401 [*] `pandas` imported but unused
  --> julien_data/2_compute_dfc_stream.py:15:18
   |
13 | import comm
14 | import numpy as np
15 | import pandas as pd
   |                  ^^
16 | from shared_code.fun_loaddata import *
17 | from shared_code.fun_dfcspeed import get_tenet4window_range
   |
help: Remove unused import: `pandas`

F403 `from shared_code.fun_loaddata import *` used; unable to detect undefined names
  --> julien_data/2_compute_dfc_stream.py:16:1
   |
14 | import numpy as np
15 | import pandas as pd
16 | from shared_code.fun_loaddata import *
   | ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
17 | from shared_code.fun_dfcspeed import get_tenet4window_range
18 | from shared_code.fun_paths import get_paths
   |

F401 [*] `shared_code.fun_paths.get_paths` imported but unused
  --> julien_data/2_compute_dfc_stream.py:18:35
   |
16 | from shared_code.fun_loaddata import *
17 | from shared_code.fun_dfcspeed import get_tenet4window_range
18 | from shared_code.fun_paths import get_paths
   |                                   ^^^^^^^^^
19 | from tqdm import tqdm
20 | import pickle
   |
help: Remove unused import: `shared_code.fun_paths.get_paths`

F401 [*] `tqdm.tqdm` imported but unused
  --> julien_data/2_compute_dfc_stream.py:19:18
   |
17 | from shared_code.fun_dfcspeed import get_tenet4window_range
18 | from shared_code.fun_paths import get_paths
19 | from tqdm import tqdm
   |                  ^^^^
20 | import pickle
   |
help: Remove unused import: `tqdm.tqdm`

F541 [*] f-string without any placeholders
  --> julien_data/2_compute_dfc_stream.py:78:7
   |
76 |     idx += 1
77 |
78 | print(f"\nData shapes after conversion:")
   |       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
79 | print(f"All animals (padded): {all_animals_3d.shape}")
80 | print(f"500-timepoint animals: {ts_500_3d.shape}")
   |
help: Remove extraneous `f` prefix

UP009 [*] UTF-8 encoding declaration is unnecessary
 --> julien_data/3_dfc_local_speed_v1.py:2:1
  |
1 | #!/usr/bin/env python3
2 | # -*- coding: utf-8 -*-
  | ^^^^^^^^^^^^^^^^^^^^^^^
3 | """
4 | Created on Mon Oct  2 14:42:38 2023
  |
help: Remove unnecessary coding comment

I001 [*] Import block is un-sorted or un-formatted
  --> julien_data/3_dfc_local_speed_v1.py:10:1
   |
 9 |   # %%
10 | / from pathlib import Path
11 | | import re
12 | | import numpy as np
13 | | import pickle
14 | | import logging
15 | | from tqdm import tqdm
16 | | import gc
17 | |
18 | | from joblib import Parallel, delayed
19 | | from typing import Dict, List, Tuple, Optional, Union
20 | |
21 | | from class_dataanalysis_julien import DFCAnalysis
22 | | from shared_code.fun_loaddata import save_pickle
23 | | from shared_code.fun_utils import set_figure_params
   | |___________________________________________________^
24 |
25 |   # logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
   |
help: Organize imports

F401 [*] `re` imported but unused
  --> julien_data/3_dfc_local_speed_v1.py:11:8
   |
 9 | # %%
10 | from pathlib import Path
11 | import re
   |        ^^
12 | import numpy as np
13 | import pickle
   |
help: Remove unused import: `re`

F401 [*] `gc` imported but unused
  --> julien_data/3_dfc_local_speed_v1.py:16:8
   |
14 | import logging
15 | from tqdm import tqdm
16 | import gc
   |        ^^
17 |
18 | from joblib import Parallel, delayed
   |
help: Remove unused import: `gc`

UP035 `typing.Dict` is deprecated, use `dict` instead
  --> julien_data/3_dfc_local_speed_v1.py:19:1
   |
18 | from joblib import Parallel, delayed
19 | from typing import Dict, List, Tuple, Optional, Union
   | ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
20 |
21 | from class_dataanalysis_julien import DFCAnalysis
   |

UP035 `typing.List` is deprecated, use `list` instead
  --> julien_data/3_dfc_local_speed_v1.py:19:1
   |
18 | from joblib import Parallel, delayed
19 | from typing import Dict, List, Tuple, Optional, Union
   | ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
20 |
21 | from class_dataanalysis_julien import DFCAnalysis
   |

UP035 `typing.Tuple` is deprecated, use `tuple` instead
  --> julien_data/3_dfc_local_speed_v1.py:19:1
   |
18 | from joblib import Parallel, delayed
19 | from typing import Dict, List, Tuple, Optional, Union
   | ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
20 |
21 | from class_dataanalysis_julien import DFCAnalysis
   |

F401 [*] `typing.Dict` imported but unused
  --> julien_data/3_dfc_local_speed_v1.py:19:20
   |
18 | from joblib import Parallel, delayed
19 | from typing import Dict, List, Tuple, Optional, Union
   |                    ^^^^
20 |
21 | from class_dataanalysis_julien import DFCAnalysis
   |
help: Remove unused import

F401 [*] `typing.List` imported but unused
  --> julien_data/3_dfc_local_speed_v1.py:19:26
   |
18 | from joblib import Parallel, delayed
19 | from typing import Dict, List, Tuple, Optional, Union
   |                          ^^^^
20 |
21 | from class_dataanalysis_julien import DFCAnalysis
   |
help: Remove unused import

F401 [*] `typing.Tuple` imported but unused
  --> julien_data/3_dfc_local_speed_v1.py:19:32
   |
18 | from joblib import Parallel, delayed
19 | from typing import Dict, List, Tuple, Optional, Union
   |                                ^^^^^
20 |
21 | from class_dataanalysis_julien import DFCAnalysis
   |
help: Remove unused import

F401 [*] `typing.Optional` imported but unused
  --> julien_data/3_dfc_local_speed_v1.py:19:39
   |
18 | from joblib import Parallel, delayed
19 | from typing import Dict, List, Tuple, Optional, Union
   |                                       ^^^^^^^^
20 |
21 | from class_dataanalysis_julien import DFCAnalysis
   |
help: Remove unused import

F401 [*] `typing.Union` imported but unused
  --> julien_data/3_dfc_local_speed_v1.py:19:49
   |
18 | from joblib import Parallel, delayed
19 | from typing import Dict, List, Tuple, Optional, Union
   |                                                 ^^^^^
20 |
21 | from class_dataanalysis_julien import DFCAnalysis
   |
help: Remove unused import

F401 [*] `shared_code.fun_utils.set_figure_params` imported but unused
  --> julien_data/3_dfc_local_speed_v1.py:23:35
   |
21 | from class_dataanalysis_julien import DFCAnalysis
22 | from shared_code.fun_loaddata import save_pickle
23 | from shared_code.fun_utils import set_figure_params
   |                                   ^^^^^^^^^^^^^^^^^
24 |
25 | # logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
   |
help: Remove unused import: `shared_code.fun_utils.set_figure_params`

I001 [*] Import block is un-sorted or un-formatted
  --> julien_data/3_dfc_local_speed_v1.py:77:5
   |
75 |       MethodsX 2020, doi: 10.1016/j.mex.2020.101168
76 |       """
77 | /     from shared_code.fun_optimization import (
78 | |         pearson_speed_vectorized,
79 | |         spearman_speed,
80 | |         cosine_speed_vectorized,
81 | |     )
   | |_____^
82 |
83 |       # Input validation
   |
help: Organize imports

F841 Local variable `n_pairs` is assigned to but never used
   --> julien_data/3_dfc_local_speed_v1.py:143:5
    |
142 |     n_speeds = (len(indices) - 1) * np.size(tau_range)
143 |     n_pairs = fc_stream.shape[0]
    |     ^^^^^^^
144 |
145 |     # Pre-allocate output arrays for efficiency
    |
help: Remove assignment to unused variable `n_pairs`

F841 Local variable `fc2_stream` is assigned to but never used
   --> julien_data/3_dfc_local_speed_v1.py:147:5
    |
145 |     # Pre-allocate output arrays for efficiency
146 |     speeds = np.empty((n_speeds, np.size(tau_range)), dtype=np.float32)
147 |     fc2_stream = None
    |     ^^^^^^^^^^
148 |
149 |     # Extract FC matrices for vectorized computation
    |
help: Remove assignment to unused variable `fc2_stream`

F841 Local variable `min_tau_zero` is assigned to but never used
   --> julien_data/3_dfc_local_speed_v1.py:201:5
    |
200 |     # Parameter extraction & checks (same as before)
201 |     min_tau_zero = kwargs.get("min_tau_zero", True)
    |     ^^^^^^^^^^^^
202 |     method = kwargs.get("method", "pearson")
203 |     return_fc2 = kwargs.get("return_fc2", False)
    |
help: Remove assignment to unused variable `min_tau_zero`

F541 [*] f-string without any placeholders
   --> julien_data/3_dfc_local_speed_v1.py:293:48
    |
291 |     Parallel(n_jobs=processors, verbose=1)(
292 |         delayed(process_window)(ws, nodes)
293 |         for ws in tqdm(time_window_range, desc=f"Processing windows for ...")
    |                                                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
294 |     )
    |
help: Remove extraneous `f` prefix

F841 Local variable `min_tau_zero` is assigned to but never used
   --> julien_data/3_dfc_local_speed_v1.py:318:5
    |
317 |     # Parameter extraction & checks (same as before)
318 |     min_tau_zero = kwargs.get("min_tau_zero", True)
    |     ^^^^^^^^^^^^
319 |     method = kwargs.get("method", "pearson")
320 |     return_fc2 = kwargs.get("return_fc2", False)
    |
help: Remove assignment to unused variable `min_tau_zero`

F541 [*] f-string without any placeholders
   --> julien_data/3_dfc_local_speed_v1.py:412:48
    |
410 |     Parallel(n_jobs=processors, verbose=1)(
411 |         delayed(process_window)(ws, nodes)
412 |         for ws in tqdm(time_window_range, desc=f"Processing windows for ...")
    |                                                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
413 |     )
    |
help: Remove extraneous `f` prefix

B007 Loop control variable `idx` not used within loop body
   --> julien_data/3_dfc_local_speed_v1.py:537:9
    |
535 | for ind_reg in range(data.regions):
536 |     save_speed = []
537 |     for idx, window_size in enumerate(time_window_range):
    |         ^^^
538 |         window_file = (
539 |             save_path
    |
help: Rename unused `idx` to `_idx`

E402 Module level import not at top of file
   --> julien_data/3_dfc_local_speed_v1.py:592:1
    |
591 | # %%
592 | import matplotlib.pyplot as plt
    | ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
593 |
594 | plt.hist(merged, bins=150, alpha=0.7, histtype="step")
    |

E402 Module level import not at top of file
   --> julien_data/3_dfc_local_speed_v1.py:609:1
    |
608 | # %%
609 | import matplotlib.pyplot as plt
    | ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
610 | import seaborn as sns
    |

E402 Module level import not at top of file
   --> julien_data/3_dfc_local_speed_v1.py:610:1
    |
608 | # %%
609 | import matplotlib.pyplot as plt
610 | import seaborn as sns
    | ^^^^^^^^^^^^^^^^^^^^^
611 |
612 | sns.set_theme(style="white")
    |

B007 Loop control variable `i` not used within loop body
   --> julien_data/3_dfc_local_speed_v1.py:650:5
    |
648 |     communities = pickle.load(f)
649 |
650 | for i, c in enumerate(np.unique(communities)):
    |     ^
651 |     regions_mod1 = np.sum(communities == c)
    |
help: Rename unused `i` to `_i`

B007 Loop control variable `idx` not used within loop body
   --> julien_data/3_dfc_local_speed_v1.py:673:9
    |
671 |     save_speed = []
672 |     # Load the speed results for each window size
673 |     for idx, window_size in enumerate(time_window_range):
    |         ^^^
674 |         window_file = (
675 |             save_path
    |
help: Rename unused `idx` to `_idx`

UP009 [*] UTF-8 encoding declaration is unnecessary
 --> julien_data/3_dfc_speed_test_v6.py:2:1
  |
1 | #!/usr/bin/env python3
2 | # -*- coding: utf-8 -*-
  | ^^^^^^^^^^^^^^^^^^^^^^^
3 | """
4 | Created on Mon Oct  2 14:42:38 2023
  |
help: Remove unnecessary coding comment

I001 [*] Import block is un-sorted or un-formatted
  --> julien_data/3_dfc_speed_test_v6.py:10:1
   |
 9 |   # %%
10 | / from pathlib import Path
11 | | import numpy as np
12 | | import pickle
13 | | import logging
14 | | from tqdm import tqdm
15 | | import gc
16 | |
17 | | from joblib import Parallel, delayed
18 | | from typing import Dict, List, Tuple, Optional, Union
19 | |
20 | | from class_dataanalysis_julien import DFCAnalysis
21 | | from shared_code.fun_loaddata import save_pickle
22 | | from shared_code.fun_utils import set_figure_params
   | |___________________________________________________^
23 |
24 |   logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
   |
help: Organize imports

F401 [*] `gc` imported but unused
  --> julien_data/3_dfc_speed_test_v6.py:15:8
   |
13 | import logging
14 | from tqdm import tqdm
15 | import gc
   |        ^^
16 |
17 | from joblib import Parallel, delayed
   |
help: Remove unused import: `gc`

UP035 `typing.Dict` is deprecated, use `dict` instead
  --> julien_data/3_dfc_speed_test_v6.py:18:1
   |
17 | from joblib import Parallel, delayed
18 | from typing import Dict, List, Tuple, Optional, Union
   | ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
19 |
20 | from class_dataanalysis_julien import DFCAnalysis
   |

UP035 `typing.List` is deprecated, use `list` instead
  --> julien_data/3_dfc_speed_test_v6.py:18:1
   |
17 | from joblib import Parallel, delayed
18 | from typing import Dict, List, Tuple, Optional, Union
   | ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
19 |
20 | from class_dataanalysis_julien import DFCAnalysis
   |

UP035 `typing.Tuple` is deprecated, use `tuple` instead
  --> julien_data/3_dfc_speed_test_v6.py:18:1
   |
17 | from joblib import Parallel, delayed
18 | from typing import Dict, List, Tuple, Optional, Union
   | ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
19 |
20 | from class_dataanalysis_julien import DFCAnalysis
   |

F401 [*] `typing.Dict` imported but unused
  --> julien_data/3_dfc_speed_test_v6.py:18:20
   |
17 | from joblib import Parallel, delayed
18 | from typing import Dict, List, Tuple, Optional, Union
   |                    ^^^^
19 |
20 | from class_dataanalysis_julien import DFCAnalysis
   |
help: Remove unused import

F401 [*] `typing.List` imported but unused
  --> julien_data/3_dfc_speed_test_v6.py:18:26
   |
17 | from joblib import Parallel, delayed
18 | from typing import Dict, List, Tuple, Optional, Union
   |                          ^^^^
19 |
20 | from class_dataanalysis_julien import DFCAnalysis
   |
help: Remove unused import

F401 [*] `typing.Tuple` imported but unused
  --> julien_data/3_dfc_speed_test_v6.py:18:32
   |
17 | from joblib import Parallel, delayed
18 | from typing import Dict, List, Tuple, Optional, Union
   |                                ^^^^^
19 |
20 | from class_dataanalysis_julien import DFCAnalysis
   |
help: Remove unused import

F401 [*] `typing.Optional` imported but unused
  --> julien_data/3_dfc_speed_test_v6.py:18:39
   |
17 | from joblib import Parallel, delayed
18 | from typing import Dict, List, Tuple, Optional, Union
   |                                       ^^^^^^^^
19 |
20 | from class_dataanalysis_julien import DFCAnalysis
   |
help: Remove unused import

F401 [*] `typing.Union` imported but unused
  --> julien_data/3_dfc_speed_test_v6.py:18:49
   |
17 | from joblib import Parallel, delayed
18 | from typing import Dict, List, Tuple, Optional, Union
   |                                                 ^^^^^
19 |
20 | from class_dataanalysis_julien import DFCAnalysis
   |
help: Remove unused import

F401 [*] `shared_code.fun_utils.set_figure_params` imported but unused
  --> julien_data/3_dfc_speed_test_v6.py:22:35
   |
20 | from class_dataanalysis_julien import DFCAnalysis
21 | from shared_code.fun_loaddata import save_pickle
22 | from shared_code.fun_utils import set_figure_params
   |                                   ^^^^^^^^^^^^^^^^^
23 |
24 | logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
   |
help: Remove unused import: `shared_code.fun_utils.set_figure_params`

I001 [*] Import block is un-sorted or un-formatted
  --> julien_data/3_dfc_speed_test_v6.py:76:5
   |
74 |       MethodsX 2020, doi: 10.1016/j.mex.2020.101168
75 |       """
76 | /     from shared_code.fun_optimization import (
77 | |         pearson_speed_vectorized,
78 | |         spearman_speed,
79 | |         cosine_speed_vectorized,
80 | |     )
   | |_____^
81 |
82 |       # Input validation
   |
help: Organize imports

F841 Local variable `n_pairs` is assigned to but never used
   --> julien_data/3_dfc_speed_test_v6.py:142:5
    |
141 |     n_speeds = (len(indices) - 1) * np.size(tau_range)
142 |     n_pairs = fc_stream.shape[0]
    |     ^^^^^^^
143 |
144 |     # Pre-allocate output arrays for efficiency
    |
help: Remove assignment to unused variable `n_pairs`

F841 Local variable `fc2_stream` is assigned to but never used
   --> julien_data/3_dfc_speed_test_v6.py:146:5
    |
144 |     # Pre-allocate output arrays for efficiency
145 |     speeds = np.empty((n_speeds, np.size(tau_range)), dtype=np.float32)
146 |     fc2_stream = None
    |     ^^^^^^^^^^
147 |
148 |     # Extract FC matrices for vectorized computation
    |
help: Remove assignment to unused variable `fc2_stream`

F841 Local variable `min_tau_zero` is assigned to but never used
   --> julien_data/3_dfc_speed_test_v6.py:200:5
    |
199 |     # Parameter extraction & checks (same as before)
200 |     min_tau_zero = kwargs.get("min_tau_zero", True)
    |     ^^^^^^^^^^^^
201 |     method = kwargs.get("method", "pearson")
202 |     return_fc2 = kwargs.get("return_fc2", False)
    |
help: Remove assignment to unused variable `min_tau_zero`

F541 [*] f-string without any placeholders
   --> julien_data/3_dfc_speed_test_v6.py:287:48
    |
285 |     Parallel(n_jobs=processors, verbose=1)(
286 |         delayed(process_window)(ws, nodes)
287 |         for ws in tqdm(time_window_range, desc=f"Processing windows for ...")
    |                                                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
288 |     )
    |
help: Remove extraneous `f` prefix

B007 Loop control variable `idx` not used within loop body
   --> julien_data/3_dfc_speed_test_v6.py:369:5
    |
367 | save_speed = []
368 | # Load the speed results for each window size
369 | for idx, window_size in enumerate(time_window_range):
    |     ^^^
370 |     window_file = (
371 |         save_path
    |
help: Rename unused `idx` to `_idx`

B007 Loop control variable `i` not used within loop body
   --> julien_data/3_dfc_speed_test_v6.py:423:5
    |
421 |     communities = pickle.load(f)
422 |
423 | for i, c in enumerate(np.unique(communities)):
    |     ^
424 |     regions_mod1 = np.sum(communities == c)
    |
help: Rename unused `i` to `_i`

B007 Loop control variable `idx` not used within loop body
   --> julien_data/3_dfc_speed_test_v6.py:446:9
    |
444 |     save_speed = []
445 |     # Load the speed results for each window size
446 |     for idx, window_size in enumerate(time_window_range):
    |         ^^^
447 |         window_file = (
448 |             save_path
    |
help: Rename unused `idx` to `_idx`

I001 [*] Import block is un-sorted or un-formatted
  --> julien_data/Plot_speed_figures.py:2:1
   |
 1 |   # %%
 2 | / import pickle
 3 | | import numpy as np
 4 | | import pandas as pd
 5 | | import matplotlib.pyplot as plt
 6 | | import seaborn as sns
 7 | | from pathlib import Path
 8 | | from scipy.stats import mannwhitneyu, kruskal
 9 | |
10 | | # Load analysis class and preprocessed data
11 | | from class_dataanalysis_julien import DFCAnalysis
   | |_________________________________________________^
12 |
13 |   data = DFCAnalysis()
   |
help: Organize imports

E402 Module level import not at top of file
   --> julien_data/Plot_speed_figures.py:150:1
    |
149 | # %%
150 | import numpy as np
    | ^^^^^^^^^^^^^^^^^^
151 | import matplotlib.pyplot as plt
152 | import seaborn as sns
    |

I001 [*] Import block is un-sorted or un-formatted
   --> julien_data/Plot_speed_figures.py:150:1
    |
149 |   # %%
150 | / import numpy as np
151 | | import matplotlib.pyplot as plt
152 | | import seaborn as sns
153 | |
154 | | import numpy as np
155 | | import matplotlib.pyplot as plt
156 | | import seaborn as sns
    | |_____________________^
    |
help: Organize imports

E402 Module level import not at top of file
   --> julien_data/Plot_speed_figures.py:151:1
    |
149 | # %%
150 | import numpy as np
151 | import matplotlib.pyplot as plt
    | ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
152 | import seaborn as sns
    |

E402 Module level import not at top of file
   --> julien_data/Plot_speed_figures.py:152:1
    |
150 | import numpy as np
151 | import matplotlib.pyplot as plt
152 | import seaborn as sns
    | ^^^^^^^^^^^^^^^^^^^^^
153 |
154 | import numpy as np
    |

E402 Module level import not at top of file
   --> julien_data/Plot_speed_figures.py:154:1
    |
152 | import seaborn as sns
153 |
154 | import numpy as np
    | ^^^^^^^^^^^^^^^^^^
155 | import matplotlib.pyplot as plt
156 | import seaborn as sns
    |

E402 Module level import not at top of file
   --> julien_data/Plot_speed_figures.py:155:1
    |
154 | import numpy as np
155 | import matplotlib.pyplot as plt
    | ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
156 | import seaborn as sns
    |

E402 Module level import not at top of file
   --> julien_data/Plot_speed_figures.py:156:1
    |
154 | import numpy as np
155 | import matplotlib.pyplot as plt
156 | import seaborn as sns
    | ^^^^^^^^^^^^^^^^^^^^^
    |

B007 Loop control variable `group` not used within loop body
   --> julien_data/Plot_speed_figures.py:175:19
    |
173 |     for comm in range(n_communities):
174 |         plt.figure(figsize=(9, 5))
175 |         for idx, (group, animal_idxs) in enumerate(groups.items()):
    |                   ^^^^^
176 |             pooled = []
177 |             for animal_idx in animal_idxs:
    |
help: Rename unused `group` to `_group`

I001 [*] Import block is un-sorted or un-formatted
  --> julien_data/class_dataanalysis_julien.py:7:1
   |
 5 |   and processes the data to compute DFC streams and speeds.
 6 |   """
 7 | / from importlib import metadata
 8 | | from math import e
 9 | | from pathlib import Path
10 | | import numpy as np
11 | | import pandas as pd
12 | | import scipy as sp
13 | | import pickle
14 | |
15 | | from shared_code.fun_paths import get_paths
16 | | from shared_code.fun_loaddata import (
17 | |     load_mat_timeseries,
18 | |     extract_mouse_ids,
19 | |     load_npz_dict,
20 | |     make_file_path,
21 | |     load_pickle,
22 | |     load_fc2_npz,
23 | | )
24 | | from shared_code.shared_code.fun_loaddata import load_pickle
   | |____________________________________________________________^
   |
help: Organize imports

F401 [*] `importlib.metadata` imported but unused
 --> julien_data/class_dataanalysis_julien.py:7:23
  |
5 | and processes the data to compute DFC streams and speeds.
6 | """
7 | from importlib import metadata
  |                       ^^^^^^^^
8 | from math import e
9 | from pathlib import Path
  |
help: Remove unused import: `importlib.metadata`

F401 [*] `math.e` imported but unused
  --> julien_data/class_dataanalysis_julien.py:8:18
   |
 6 | """
 7 | from importlib import metadata
 8 | from math import e
   |                  ^
 9 | from pathlib import Path
10 | import numpy as np
   |
help: Remove unused import: `math.e`

F401 [*] `scipy` imported but unused
  --> julien_data/class_dataanalysis_julien.py:12:17
   |
10 | import numpy as np
11 | import pandas as pd
12 | import scipy as sp
   |                 ^^
13 | import pickle
   |
help: Remove unused import: `scipy`

F811 Redefinition of unused `load_pickle` from line 21
  --> julien_data/class_dataanalysis_julien.py:21:5
   |
19 |     load_npz_dict,
20 |     make_file_path,
21 |     load_pickle,
   |     ----------- previous definition of `load_pickle` here
22 |     load_fc2_npz,
23 | )
24 | from shared_code.shared_code.fun_loaddata import load_pickle
   |                                                  ^^^^^^^^^^^ `load_pickle` redefined here
   |
help: Remove definition: `load_pickle`

F841 Local variable `results` is assigned to but never used
   --> julien_data/class_dataanalysis_julien.py:172:13
    |
170 |                 self.regions,
171 |             )
172 |             results = load_npz_dict(file_path)
    |             ^^^^^^^
173 |             self.dfc_streams[window_size] = self.load_dfc_1_window(lag, window_size)
    |
help: Remove assignment to unused variable `results`

B008 Do not perform function call `np.arange` in argument defaults; instead, perform the call within the function, or read the default from a module-level singleton variable
   --> julien_data/class_dataanalysis_julien.py:177:26
    |
175 |     # 3.4 Load speed analysis
176 |     def get_speed_analysis(
177 |         self, tau_arange=np.arange(4), time_window_range=np.arange(5, 50 + 1, 1)
    |                          ^^^^^^^^^^^^
178 |     ):
179 |         prefix = "speed"
    |

B008 Do not perform function call `np.arange` in argument defaults; instead, perform the call within the function, or read the default from a module-level singleton variable
   --> julien_data/class_dataanalysis_julien.py:177:58
    |
175 |     # 3.4 Load speed analysis
176 |     def get_speed_analysis(
177 |         self, tau_arange=np.arange(4), time_window_range=np.arange(5, 50 + 1, 1)
    |                                                          ^^^^^^^^^^^^^^^^^^^^^^^
178 |     ):
179 |         prefix = "speed"
    |

B008 Do not perform function call `np.arange` in argument defaults; instead, perform the call within the function, or read the default from a module-level singleton variable
   --> julien_data/class_dataanalysis_julien.py:189:26
    |
187 |     # 3.5 Load speed fc analysis
188 |     def get_speed_fc_analysis(
189 |         self, tau_arange=np.arange(4), time_window_range=np.arange(5, 50 + 1, 1)
    |                          ^^^^^^^^^^^^
190 |     ):
191 |         prefix = "speed_fc"
    |

B008 Do not perform function call `np.arange` in argument defaults; instead, perform the call within the function, or read the default from a module-level singleton variable
   --> julien_data/class_dataanalysis_julien.py:189:58
    |
187 |     # 3.5 Load speed fc analysis
188 |     def get_speed_fc_analysis(
189 |         self, tau_arange=np.arange(4), time_window_range=np.arange(5, 50 + 1, 1)
    |                                                          ^^^^^^^^^^^^^^^^^^^^^^^
190 |     ):
191 |         prefix = "speed_fc"
    |

UP009 [*] UTF-8 encoding declaration is unnecessary
 --> julien_data/dfc_windows_pooling.py:2:1
  |
1 | #!/usr/bin/env python3
2 | # -*- coding: utf-8 -*-
  | ^^^^^^^^^^^^^^^^^^^^^^^
3 | """
4 | Created on Mon Oct  2 14:42:38 2023
  |
help: Remove unnecessary coding comment

I001 [*] Import block is un-sorted or un-formatted
  --> julien_data/dfc_windows_pooling.py:9:1
   |
 7 |   """
 8 |   # %% Import libraries
 9 | / from cProfile import label
10 | | from pathlib import Path
11 | | from tkinter import W
12 | | from tqdm import tqdm
13 | | import numpy as np
14 | | import matplotlib.pyplot as plt
15 | | import brainconn as bct
16 | | import os
17 | | import time
18 | | import pandas as pd
19 | |
20 | | # import sys
21 | | # sys.path.append('../shared_code')
22 | | # from functions_analysis import *
23 | | from scipy.io import loadmat, savemat
24 | | from scipy.special import erfc
25 | | from scipy.stats import pearsonr, spearmanr
26 | |
27 | | from shared_code.fun_loaddata import *
28 | | from shared_code.fun_dfcspeed import pool_vel_windows, get_population_wpooling
29 | |
30 | | # from fun_utils import set_figure_params
31 | | from shared_code.fun_bootstrap import handler_bootstrap_permutation
32 | | from shared_code.fun_utils import set_figure_params
33 | | from shared_code.fun_paths import get_paths
34 | |
35 | | from joblib import Parallel, delayed
   | |____________________________________^
   |
help: Organize imports

F401 [*] `cProfile.label` imported but unused
  --> julien_data/dfc_windows_pooling.py:9:22
   |
 7 | """
 8 | # %% Import libraries
 9 | from cProfile import label
   |                      ^^^^^
10 | from pathlib import Path
11 | from tkinter import W
   |
help: Remove unused import: `cProfile.label`

F401 [*] `tkinter.W` imported but unused
  --> julien_data/dfc_windows_pooling.py:11:21
   |
 9 | from cProfile import label
10 | from pathlib import Path
11 | from tkinter import W
   |                     ^
12 | from tqdm import tqdm
13 | import numpy as np
   |
help: Remove unused import: `tkinter.W`

F401 [*] `tqdm.tqdm` imported but unused
  --> julien_data/dfc_windows_pooling.py:12:18
   |
10 | from pathlib import Path
11 | from tkinter import W
12 | from tqdm import tqdm
   |                  ^^^^
13 | import numpy as np
14 | import matplotlib.pyplot as plt
   |
help: Remove unused import: `tqdm.tqdm`

F401 [*] `brainconn` imported but unused
  --> julien_data/dfc_windows_pooling.py:15:21
   |
13 | import numpy as np
14 | import matplotlib.pyplot as plt
15 | import brainconn as bct
   |                     ^^^
16 | import os
17 | import time
   |
help: Remove unused import: `brainconn`

F401 [*] `os` imported but unused
  --> julien_data/dfc_windows_pooling.py:16:8
   |
14 | import matplotlib.pyplot as plt
15 | import brainconn as bct
16 | import os
   |        ^^
17 | import time
18 | import pandas as pd
   |
help: Remove unused import: `os`

F401 [*] `time` imported but unused
  --> julien_data/dfc_windows_pooling.py:17:8
   |
15 | import brainconn as bct
16 | import os
17 | import time
   |        ^^^^
18 | import pandas as pd
   |
help: Remove unused import: `time`

F401 [*] `scipy.io.loadmat` imported but unused
  --> julien_data/dfc_windows_pooling.py:23:22
   |
21 | # sys.path.append('../shared_code')
22 | # from functions_analysis import *
23 | from scipy.io import loadmat, savemat
   |                      ^^^^^^^
24 | from scipy.special import erfc
25 | from scipy.stats import pearsonr, spearmanr
   |
help: Remove unused import

F401 [*] `scipy.io.savemat` imported but unused
  --> julien_data/dfc_windows_pooling.py:23:31
   |
21 | # sys.path.append('../shared_code')
22 | # from functions_analysis import *
23 | from scipy.io import loadmat, savemat
   |                               ^^^^^^^
24 | from scipy.special import erfc
25 | from scipy.stats import pearsonr, spearmanr
   |
help: Remove unused import

F401 [*] `scipy.special.erfc` imported but unused
  --> julien_data/dfc_windows_pooling.py:24:27
   |
22 | # from functions_analysis import *
23 | from scipy.io import loadmat, savemat
24 | from scipy.special import erfc
   |                           ^^^^
25 | from scipy.stats import pearsonr, spearmanr
   |
help: Remove unused import: `scipy.special.erfc`

F401 [*] `scipy.stats.pearsonr` imported but unused
  --> julien_data/dfc_windows_pooling.py:25:25
   |
23 | from scipy.io import loadmat, savemat
24 | from scipy.special import erfc
25 | from scipy.stats import pearsonr, spearmanr
   |                         ^^^^^^^^
26 |
27 | from shared_code.fun_loaddata import *
   |
help: Remove unused import

F401 [*] `scipy.stats.spearmanr` imported but unused
  --> julien_data/dfc_windows_pooling.py:25:35
   |
23 | from scipy.io import loadmat, savemat
24 | from scipy.special import erfc
25 | from scipy.stats import pearsonr, spearmanr
   |                                   ^^^^^^^^^
26 |
27 | from shared_code.fun_loaddata import *
   |
help: Remove unused import

F403 `from shared_code.fun_loaddata import *` used; unable to detect undefined names
  --> julien_data/dfc_windows_pooling.py:27:1
   |
25 | from scipy.stats import pearsonr, spearmanr
26 |
27 | from shared_code.fun_loaddata import *
   | ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
28 | from shared_code.fun_dfcspeed import pool_vel_windows, get_population_wpooling
   |

F401 [*] `joblib.Parallel` imported but unused
  --> julien_data/dfc_windows_pooling.py:35:20
   |
33 | from shared_code.fun_paths import get_paths
34 |
35 | from joblib import Parallel, delayed
   |                    ^^^^^^^^
   |
help: Remove unused import

F401 [*] `joblib.delayed` imported but unused
  --> julien_data/dfc_windows_pooling.py:35:30
   |
33 | from shared_code.fun_paths import get_paths
34 |
35 | from joblib import Parallel, delayed
   |                              ^^^^^^^
   |
help: Remove unused import

UP031 Use format specifiers instead of percent format
   --> julien_data/dfc_windows_pooling.py:152:5
    |
151 | vel_label = (
152 |     "%s-%ss (short)" % (aux_timewr[0], aux_timewr[limits[0]]),
    |     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
153 |     "%s-%ss (mid)" % (aux_timewr[limits[0]], aux_timewr[limits[1]]),
154 |     "%s-%ss (long)" % (aux_timewr[limits[1]], aux_timewr[-1]),
    |
help: Replace with format specifiers

UP031 Use format specifiers instead of percent format
   --> julien_data/dfc_windows_pooling.py:153:5
    |
151 | vel_label = (
152 |     "%s-%ss (short)" % (aux_timewr[0], aux_timewr[limits[0]]),
153 |     "%s-%ss (mid)" % (aux_timewr[limits[0]], aux_timewr[limits[1]]),
    |     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
154 |     "%s-%ss (long)" % (aux_timewr[limits[1]], aux_timewr[-1]),
155 | )
    |
help: Replace with format specifiers

UP031 Use format specifiers instead of percent format
   --> julien_data/dfc_windows_pooling.py:154:5
    |
152 |     "%s-%ss (short)" % (aux_timewr[0], aux_timewr[limits[0]]),
153 |     "%s-%ss (mid)" % (aux_timewr[limits[0]], aux_timewr[limits[1]]),
154 |     "%s-%ss (long)" % (aux_timewr[limits[1]], aux_timewr[-1]),
    |     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
155 | )
156 | # n_animals = int(np.sum(male_index))
    |
help: Replace with format specifiers

F402 Import `label` from line 9 shadowed by loop variable
   --> julien_data/dfc_windows_pooling.py:294:21
    |
292 |         # Linear scale plot
293 |         plt.subplot(3, 2, 2 * i + 1)
294 |         for counts, label in zip(norm_counts_list, labels):
    |                     ^^^^^
295 |             plt.plot(bin_centers, counts, label=label, alpha=0.9)
296 |         plt.title(f"{vel_label[i]} {name_data}" if i == 0 else vel_label[i])
    |

B905 [*] `zip()` without an explicit `strict=` parameter
   --> julien_data/dfc_windows_pooling.py:294:30
    |
292 |         # Linear scale plot
293 |         plt.subplot(3, 2, 2 * i + 1)
294 |         for counts, label in zip(norm_counts_list, labels):
    |                              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
295 |             plt.plot(bin_centers, counts, label=label, alpha=0.9)
296 |         plt.title(f"{vel_label[i]} {name_data}" if i == 0 else vel_label[i])
    |
help: Add explicit value for parameter `strict=`

B905 [*] `zip()` without an explicit `strict=` parameter
   --> julien_data/dfc_windows_pooling.py:303:30
    |
301 |         # Log scale plot
302 |         plt.subplot(3, 2, 2 * i + 2)
303 |         for counts, label in zip(norm_counts_list, labels):
    |                              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
304 |             plt.plot(bin_centers, counts, label=label, alpha=0.9)
305 |         plt.title(f"{vel_label[i]} {name_data}" if i == 0 else vel_label[i])
    |
help: Add explicit value for parameter `strict=`

F402 Import `label` from line 9 shadowed by loop variable
   --> julien_data/dfc_windows_pooling.py:397:18
    |
395 |         plt.subplot(3, 1, i + 1)
396 |         plt.title(f"Cumulative Distribution Function ({vel_label[i]})")
397 |         for cdf, label in zip(wp_vars, labels):
    |                  ^^^^^
398 |             plt.plot(cdf[i][0], cdf[i][1], label=label)
399 |         # plt.yscale('log')
    |

B905 [*] `zip()` without an explicit `strict=` parameter
   --> julien_data/dfc_windows_pooling.py:397:27
    |
395 |         plt.subplot(3, 1, i + 1)
396 |         plt.title(f"Cumulative Distribution Function ({vel_label[i]})")
397 |         for cdf, label in zip(wp_vars, labels):
    |                           ^^^^^^^^^^^^^^^^^^^^
398 |             plt.plot(cdf[i][0], cdf[i][1], label=label)
399 |         # plt.yscale('log')
    |
help: Add explicit value for parameter `strict=`

F841 Local variable `num_group` is assigned to but never used
   --> julien_data/dfc_windows_pooling.py:503:5
    |
501 |         qq_data, (1, 0, 2)
502 |     )  # Shape: (n_groups, n_wpools, n_quantiles)
503 |     num_group = qq_data.shape[0]  # Number of groups
    |     ^^^^^^^^^
504 |     qq_diff = []  # Initialize list to store slopes
505 |     for qq_aux in qq_data:
    |
help: Remove assignment to unused variable `num_group`

F841 Local variable `num_group` is assigned to but never used
   --> julien_data/dfc_windows_pooling.py:517:5
    |
515 |         qq_data, (1, 0, 2)
516 |     )  # Shape: (n_groups, n_wpools, n_quantiles)
517 |     num_group = qq_data.shape[0]  # Number of groups
    |     ^^^^^^^^^
518 |     qq_slope = []  # Initialize list to store slopes
519 |     for qq_aux in qq_data:
    |
help: Remove assignment to unused variable `num_group`

UP031 Use format specifiers instead of percent format
   --> julien_data/dfc_windows_pooling.py:618:15
    |
616 |     print(vv)
617 |     plt.subplot(1, 3, vv + 1)
618 |     plt.title("Q-Q plot %s" % label_vel[vv])
    |               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
619 |     plt.scatter(
620 |         qq_genotype[0, vv],
    |
help: Replace with format specifiers

UP031 Use format specifiers instead of percent format
   --> julien_data/dfc_windows_pooling.py:622:15
    |
620 |         qq_genotype[0, vv],
621 |         qq_genotype[1, vv],
622 |         label="%s" % label_vel[vv],
    |               ^^^^^^^^^^^^^^^^^^^^
623 |         facecolors="none",
624 |         edgecolors="C%s" % vv,
    |
help: Replace with format specifiers

UP031 Use format specifiers instead of percent format
   --> julien_data/dfc_windows_pooling.py:624:20
    |
622 |         label="%s" % label_vel[vv],
623 |         facecolors="none",
624 |         edgecolors="C%s" % vv,
    |                    ^^^^^^^^^^
625 |         s=40,
626 |     )
    |
help: Replace with format specifiers

F402 Import `label` from line 9 shadowed by loop variable
   --> julien_data/dfc_windows_pooling.py:668:16
    |
666 |     for i in range(n_windows):
667 |         plt.subplot(1, n_windows, i + 1)
668 |         for g, label in enumerate(group_labels):
    |                ^^^^^
669 |             # Plot median quantile line
670 |             plt.plot(q_range, qq_data[g, :, i], label=f"{label} median", color=f"C{g}")
    |

I001 [*] Import block is un-sorted or un-formatted
  --> julien_data/laod_las_speed.py:2:1
   |
 1 |   # %%
 2 | / import pickle
 3 | | from pathlib import Path
 4 | | from networkx import density
 5 | | import numpy as np
 6 | | from class_dataanalysis_julien import DFCAnalysis
 7 | | import matplotlib.pyplot as plt
 8 | | import seaborn as sns
   | |_____________________^
 9 |
10 |   data = DFCAnalysis()
   |
help: Organize imports

F401 [*] `networkx.density` imported but unused
 --> julien_data/laod_las_speed.py:4:22
  |
2 | import pickle
3 | from pathlib import Path
4 | from networkx import density
  |                      ^^^^^^^
5 | import numpy as np
6 | from class_dataanalysis_julien import DFCAnalysis
  |
help: Remove unused import: `networkx.density`

B007 Loop control variable `i` not used within loop body
  --> julien_data/laod_las_speed.py:60:5
   |
58 | save_path = data.paths["speed"]
59 | # time_window_range
60 | for i, c in enumerate(np.unique(communities)):
   |     ^
61 |     regions_mod1 = np.sum(communities == c)
62 |     print(regions_mod1)
   |
help: Rename unused `i` to `_i`

E402 Module level import not at top of file
  --> julien_data/laod_las_speed.py:74:1
   |
72 | # %%
73 |
74 | import numpy as np
   | ^^^^^^^^^^^^^^^^^^
75 |
76 | n_communities = 3  # Update if more
   |

E402 Module level import not at top of file
   --> julien_data/laod_las_speed.py:117:1
    |
115 | # %%
116 |
117 | import seaborn as sns
    | ^^^^^^^^^^^^^^^^^^^^^
118 | import matplotlib.pyplot as plt
119 | import pandas as pd
    |

I001 [*] Import block is un-sorted or un-formatted
   --> julien_data/laod_las_speed.py:117:1
    |
115 |   # %%
116 |
117 | / import seaborn as sns
118 | | import matplotlib.pyplot as plt
119 | | import pandas as pd
    | |___________________^
120 |
121 |   community = 2  # or 1, 2, ...
    |
help: Organize imports

F811 [*] Redefinition of unused `sns` from line 8
   --> julien_data/laod_las_speed.py:117:19
    |
115 | # %%
116 |
117 | import seaborn as sns
    |                   ^^^ `sns` redefined here
118 | import matplotlib.pyplot as plt
119 | import pandas as pd
    |
   ::: julien_data/laod_las_speed.py:8:19
    |
  6 | from class_dataanalysis_julien import DFCAnalysis
  7 | import matplotlib.pyplot as plt
  8 | import seaborn as sns
    |                   --- previous definition of `sns` here
  9 |
 10 | data = DFCAnalysis()
    |
help: Remove definition: `sns`

E402 Module level import not at top of file
   --> julien_data/laod_las_speed.py:118:1
    |
117 | import seaborn as sns
118 | import matplotlib.pyplot as plt
    | ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
119 | import pandas as pd
    |

F811 [*] Redefinition of unused `plt` from line 7
   --> julien_data/laod_las_speed.py:118:29
    |
117 | import seaborn as sns
118 | import matplotlib.pyplot as plt
    |                             ^^^ `plt` redefined here
119 | import pandas as pd
    |
   ::: julien_data/laod_las_speed.py:7:29
    |
  5 | import numpy as np
  6 | from class_dataanalysis_julien import DFCAnalysis
  7 | import matplotlib.pyplot as plt
    |                             --- previous definition of `plt` here
  8 | import seaborn as sns
    |
help: Remove definition: `plt`

E402 Module level import not at top of file
   --> julien_data/laod_las_speed.py:119:1
    |
117 | import seaborn as sns
118 | import matplotlib.pyplot as plt
119 | import pandas as pd
    | ^^^^^^^^^^^^^^^^^^^
120 |
121 | community = 2  # or 1, 2, ...
    |

E402 Module level import not at top of file
   --> julien_data/laod_las_speed.py:142:1
    |
140 | # %%
141 |
142 | from scipy.stats import mannwhitneyu
    | ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
143 |
144 | arr_lctb = community_speeds[(("Dp1Yey", "LCTB92"), 0)]
    |

E402 Module level import not at top of file
   --> julien_data/laod_las_speed.py:188:1
    |
187 | # %%
188 | import pandas as pd
    | ^^^^^^^^^^^^^^^^^^^
189 |
190 | df_plot = pd.concat([df_all, df_short, df_long], ignore_index=True)
    |

F821 Undefined name `df_all`
   --> julien_data/laod_las_speed.py:190:22
    |
188 | import pandas as pd
189 |
190 | df_plot = pd.concat([df_all, df_short, df_long], ignore_index=True)
    |                      ^^^^^^
191 | df_plot = df_plot.loc[:, ~df_plot.columns.duplicated()]
    |

F821 Undefined name `df_short`
   --> julien_data/laod_las_speed.py:190:30
    |
188 | import pandas as pd
189 |
190 | df_plot = pd.concat([df_all, df_short, df_long], ignore_index=True)
    |                              ^^^^^^^^
191 | df_plot = df_plot.loc[:, ~df_plot.columns.duplicated()]
    |

F821 Undefined name `df_long`
   --> julien_data/laod_las_speed.py:190:40
    |
188 | import pandas as pd
189 |
190 | df_plot = pd.concat([df_all, df_short, df_long], ignore_index=True)
    |                                        ^^^^^^^
191 | df_plot = df_plot.loc[:, ~df_plot.columns.duplicated()]
    |

B007 Loop control variable `i` not used within loop body
   --> julien_data/laod_las_speed.py:200:17
    |
198 |             pooled = pooled_community_speeds(community, animal_idxs, window_idx_list)
199 |             group_label = f"{group[0]}/{group[1]}"
200 |             for i, arr in enumerate(pooled):
    |                 ^
201 |                 if arr.size > 0:
202 |                     for val in arr:
    |
help: Rename unused `i` to `_i`

E402 Module level import not at top of file
   --> julien_data/laod_las_speed.py:214:1
    |
214 | import seaborn as sns
    | ^^^^^^^^^^^^^^^^^^^^^
215 | import matplotlib.pyplot as plt
    |

I001 [*] Import block is un-sorted or un-formatted
   --> julien_data/laod_las_speed.py:214:1
    |
214 | / import seaborn as sns
215 | | import matplotlib.pyplot as plt
    | |_______________________________^
216 |
217 |   # Prepare data
    |
help: Organize imports

E402 Module level import not at top of file
   --> julien_data/laod_las_speed.py:215:1
    |
214 | import seaborn as sns
215 | import matplotlib.pyplot as plt
    | ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
216 |
217 | # Prepare data
    |

B905 [*] `zip()` without an explicit `strict=` parameter
   --> julien_data/laod_las_speed.py:326:44
    |
324 | plt.figure(figsize=(10, 6))
325 |
326 | for (group_name, animal_indices), color in zip(data.groups.items(), palette):
    |                                            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
327 |     pooled = []
328 |     for win_list in all_speed:  # Each window size
    |
help: Add explicit value for parameter `strict=`

E402 Module level import not at top of file
   --> julien_data/laod_las_speed.py:367:1
    |
366 | # %%
367 | import numpy as np
    | ^^^^^^^^^^^^^^^^^^
368 | import matplotlib.pyplot as plt
369 | import seaborn as sns
    |

I001 [*] Import block is un-sorted or un-formatted
   --> julien_data/laod_las_speed.py:367:1
    |
366 |   # %%
367 | / import numpy as np
368 | | import matplotlib.pyplot as plt
369 | | import seaborn as sns
    | |_____________________^
370 |
371 |   plt.figure(figsize=(10, 6))
    |
help: Organize imports

E402 Module level import not at top of file
   --> julien_data/laod_las_speed.py:368:1
    |
366 | # %%
367 | import numpy as np
368 | import matplotlib.pyplot as plt
    | ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
369 | import seaborn as sns
    |

E402 Module level import not at top of file
   --> julien_data/laod_las_speed.py:369:1
    |
367 | import numpy as np
368 | import matplotlib.pyplot as plt
369 | import seaborn as sns
    | ^^^^^^^^^^^^^^^^^^^^^
370 |
371 | plt.figure(figsize=(10, 6))
    |

E402 Module level import not at top of file
   --> julien_data/laod_las_speed.py:433:1
    |
431 | # %%
432 |
433 | import numpy as np
    | ^^^^^^^^^^^^^^^^^^
434 | import matplotlib.pyplot as plt
435 | import seaborn as sns
    |

I001 [*] Import block is un-sorted or un-formatted
   --> julien_data/laod_las_speed.py:433:1
    |
431 |   # %%
432 |
433 | / import numpy as np
434 | | import matplotlib.pyplot as plt
435 | | import seaborn as sns
    | |_____________________^
436 |
437 |   window_sizes = time_window_range  # Your array/list of window sizes
    |
help: Organize imports

E402 Module level import not at top of file
   --> julien_data/laod_las_speed.py:434:1
    |
433 | import numpy as np
434 | import matplotlib.pyplot as plt
    | ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
435 | import seaborn as sns
    |

E402 Module level import not at top of file
   --> julien_data/laod_las_speed.py:435:1
    |
433 | import numpy as np
434 | import matplotlib.pyplot as plt
435 | import seaborn as sns
    | ^^^^^^^^^^^^^^^^^^^^^
436 |
437 | window_sizes = time_window_range  # Your array/list of window sizes
    |

E402 Module level import not at top of file
   --> julien_data/laod_las_speed.py:494:1
    |
493 | # %%
494 | import numpy as np
    | ^^^^^^^^^^^^^^^^^^
495 | import matplotlib.pyplot as plt
    |

I001 [*] Import block is un-sorted or un-formatted
   --> julien_data/laod_las_speed.py:494:1
    |
493 |   # %%
494 | / import numpy as np
495 | | import matplotlib.pyplot as plt
    | |_______________________________^
496 |
497 |   quantile_levels = np.linspace(0, 1, 20)
    |
help: Organize imports

E402 Module level import not at top of file
   --> julien_data/laod_las_speed.py:495:1
    |
493 | # %%
494 | import numpy as np
495 | import matplotlib.pyplot as plt
    | ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
496 |
497 | quantile_levels = np.linspace(0, 1, 20)
    |

E402 Module level import not at top of file
   --> julien_data/laod_las_speed.py:584:1
    |
582 | # %%
583 |
584 | import numpy as np
    | ^^^^^^^^^^^^^^^^^^
585 | import matplotlib.pyplot as plt
    |

I001 [*] Import block is un-sorted or un-formatted
   --> julien_data/laod_las_speed.py:584:1
    |
582 |   # %%
583 |
584 | / import numpy as np
585 | | import matplotlib.pyplot as plt
    | |_______________________________^
586 |
587 |   quantile_levels = np.linspace(0, 1, 100)
    |
help: Organize imports

E402 Module level import not at top of file
   --> julien_data/laod_las_speed.py:585:1
    |
584 | import numpy as np
585 | import matplotlib.pyplot as plt
    | ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
586 |
587 | quantile_levels = np.linspace(0, 1, 100)
    |

E402 Module level import not at top of file
   --> julien_data/laod_las_speed.py:674:1
    |
673 | # %%
674 | import numpy as np
    | ^^^^^^^^^^^^^^^^^^
675 | import matplotlib.pyplot as plt
    |

I001 [*] Import block is un-sorted or un-formatted
   --> julien_data/laod_las_speed.py:674:1
    |
673 |   # %%
674 | / import numpy as np
675 | | import matplotlib.pyplot as plt
    | |_______________________________^
    |
help: Organize imports

E402 Module level import not at top of file
   --> julien_data/laod_las_speed.py:675:1
    |
673 | # %%
674 | import numpy as np
675 | import matplotlib.pyplot as plt
    | ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    |

E402 Module level import not at top of file
   --> julien_data/laod_las_speed.py:836:1
    |
835 | # %%
836 | import numpy as np
    | ^^^^^^^^^^^^^^^^^^
837 | import matplotlib.pyplot as plt
838 | import itertools
    |

I001 [*] Import block is un-sorted or un-formatted
   --> julien_data/laod_las_speed.py:836:1
    |
835 |   # %%
836 | / import numpy as np
837 | | import matplotlib.pyplot as plt
838 | | import itertools
839 | | import math
    | |___________^
840 |
841 |   # Assume speed_matrices, group_names, window_sizes, quantile_levels are defined
    |
help: Organize imports

E402 Module level import not at top of file
   --> julien_data/laod_las_speed.py:837:1
    |
835 | # %%
836 | import numpy as np
837 | import matplotlib.pyplot as plt
    | ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
838 | import itertools
839 | import math
    |

E402 Module level import not at top of file
   --> julien_data/laod_las_speed.py:838:1
    |
836 | import numpy as np
837 | import matplotlib.pyplot as plt
838 | import itertools
    | ^^^^^^^^^^^^^^^^
839 | import math
    |

E402 Module level import not at top of file
   --> julien_data/laod_las_speed.py:839:1
    |
837 | import matplotlib.pyplot as plt
838 | import itertools
839 | import math
    | ^^^^^^^^^^^
840 |
841 | # Assume speed_matrices, group_names, window_sizes, quantile_levels are defined
    |

E402 Module level import not at top of file
   --> julien_data/laod_las_speed.py:925:1
    |
923 | # %%
924 |
925 | import pandas as pd
    | ^^^^^^^^^^^^^^^^^^^
926 | from scipy.stats import spearmanr
    |

E402 Module level import not at top of file
   --> julien_data/laod_las_speed.py:926:1
    |
925 | import pandas as pd
926 | from scipy.stats import spearmanr
    | ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
927 |
928 | # ------------------------ NOR scores vs dFC speed ------------------------
    |

E402 Module level import not at top of file
   --> julien_data/laod_las_speed.py:978:1
    |
976 | # %%
977 |
978 | import numpy as np
    | ^^^^^^^^^^^^^^^^^^
979 | import matplotlib.pyplot as plt
980 | import pandas as pd
    |

I001 [*] Import block is un-sorted or un-formatted
   --> julien_data/laod_las_speed.py:978:1
    |
976 |   # %%
977 |
978 | / import numpy as np
979 | | import matplotlib.pyplot as plt
980 | | import pandas as pd
981 | | from scipy.stats import theilslopes, spearmanr
982 | | import seaborn as sns
    | |_____________________^
983 |
984 |   cog_data_filtered = (
    |
help: Organize imports

E402 Module level import not at top of file
   --> julien_data/laod_las_speed.py:979:1
    |
978 | import numpy as np
979 | import matplotlib.pyplot as plt
    | ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
980 | import pandas as pd
981 | from scipy.stats import theilslopes, spearmanr
    |

E402 Module level import not at top of file
   --> julien_data/laod_las_speed.py:980:1
    |
978 | import numpy as np
979 | import matplotlib.pyplot as plt
980 | import pandas as pd
    | ^^^^^^^^^^^^^^^^^^^
981 | from scipy.stats import theilslopes, spearmanr
982 | import seaborn as sns
    |

E402 Module level import not at top of file
   --> julien_data/laod_las_speed.py:981:1
    |
979 | import matplotlib.pyplot as plt
980 | import pandas as pd
981 | from scipy.stats import theilslopes, spearmanr
    | ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
982 | import seaborn as sns
    |

E402 Module level import not at top of file
   --> julien_data/laod_las_speed.py:982:1
    |
980 | import pandas as pd
981 | from scipy.stats import theilslopes, spearmanr
982 | import seaborn as sns
    | ^^^^^^^^^^^^^^^^^^^^^
983 |
984 | cog_data_filtered = (
    |

B905 [*] `zip()` without an explicit `strict=` parameter
    --> julien_data/laod_las_speed.py:1009:21
     |
1007 | cog_df = cog_data_filtered.reset_index(drop=True)
1008 | cog_scores = cog_df["index_NOR"].values
1009 | group_labels = list(zip(cog_df["genotype"], cog_df["treatment"]))
     |                     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
1010 |
1011 | # Assign color/marker per group
     |
help: Add explicit value for parameter `strict=`

E402 Module level import not at top of file
    --> julien_data/laod_las_speed.py:1067:1
     |
1065 | # %%
1066 | # %%
1067 | import numpy as np
     | ^^^^^^^^^^^^^^^^^^
1068 | import matplotlib.pyplot as plt
1069 | from scipy.stats import spearmanr
     |

I001 [*] Import block is un-sorted or un-formatted
    --> julien_data/laod_las_speed.py:1067:1
     |
1065 |   # %%
1066 |   # %%
1067 | / import numpy as np
1068 | | import matplotlib.pyplot as plt
1069 | | from scipy.stats import spearmanr
1070 | | import seaborn as sns
     | |_____________________^
1071 |
1072 |   window_sizes = time_window_range
     |
help: Organize imports

E402 Module level import not at top of file
    --> julien_data/laod_las_speed.py:1068:1
     |
1066 | # %%
1067 | import numpy as np
1068 | import matplotlib.pyplot as plt
     | ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
1069 | from scipy.stats import spearmanr
1070 | import seaborn as sns
     |

E402 Module level import not at top of file
    --> julien_data/laod_las_speed.py:1069:1
     |
1067 | import numpy as np
1068 | import matplotlib.pyplot as plt
1069 | from scipy.stats import spearmanr
     | ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
1070 | import seaborn as sns
     |

E402 Module level import not at top of file
    --> julien_data/laod_las_speed.py:1070:1
     |
1068 | import matplotlib.pyplot as plt
1069 | from scipy.stats import spearmanr
1070 | import seaborn as sns
     | ^^^^^^^^^^^^^^^^^^^^^
1071 |
1072 | window_sizes = time_window_range
     |

E402 Module level import not at top of file
    --> julien_data/laod_las_speed.py:1146:1
     |
1144 | # ---------------------------- Two timescales --------------------------------
1145 |
1146 | import numpy as np
     | ^^^^^^^^^^^^^^^^^^
1147 | import pandas as pd
     |

E402 Module level import not at top of file
    --> julien_data/laod_las_speed.py:1147:1
     |
1146 | import numpy as np
1147 | import pandas as pd
     | ^^^^^^^^^^^^^^^^^^^
1148 |
1149 | # Split window indices into two pools (first half, second half)
     |

B905 [*] `zip()` without an explicit `strict=` parameter
    --> julien_data/laod_las_speed.py:1198:19
     |
1196 | all_speeds_long = []
1197 |
1198 | for idxs, pool in zip([short_idx, long_idx], ["short", "long"]):
     |                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
1199 |     pool_speeds = []
1200 |     for win_idx in idxs:
     |
help: Add explicit value for parameter `strict=`

E402 Module level import not at top of file
    --> julien_data/laod_las_speed.py:1214:1
     |
1212 |         all_speeds_long = flat
1213 |
1214 | import matplotlib.pyplot as plt
     | ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
1215 | import seaborn as sns
     |

E402 Module level import not at top of file
    --> julien_data/laod_las_speed.py:1215:1
     |
1214 | import matplotlib.pyplot as plt
1215 | import seaborn as sns
     | ^^^^^^^^^^^^^^^^^^^^^
1216 |
1217 | plt.figure(figsize=(8, 5))
     |

E402 Module level import not at top of file
    --> julien_data/laod_las_speed.py:1252:1
     |
1250 | # %%
1251 | # %%
1252 | import matplotlib.pyplot as plt
     | ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
1253 | import seaborn as sns
1254 | import numpy as np
     |

I001 [*] Import block is un-sorted or un-formatted
    --> julien_data/laod_las_speed.py:1252:1
     |
1250 |   # %%
1251 |   # %%
1252 | / import matplotlib.pyplot as plt
1253 | | import seaborn as sns
1254 | | import numpy as np
     | |__________________^
1255 |
1256 |   # Assume: groups = df_summary.groupby(['genotype', 'treatment']).groups
     |
help: Organize imports

E402 Module level import not at top of file
    --> julien_data/laod_las_speed.py:1253:1
     |
1251 | # %%
1252 | import matplotlib.pyplot as plt
1253 | import seaborn as sns
     | ^^^^^^^^^^^^^^^^^^^^^
1254 | import numpy as np
     |

E402 Module level import not at top of file
    --> julien_data/laod_las_speed.py:1254:1
     |
1252 | import matplotlib.pyplot as plt
1253 | import seaborn as sns
1254 | import numpy as np
     | ^^^^^^^^^^^^^^^^^^
1255 |
1256 | # Assume: groups = df_summary.groupby(['genotype', 'treatment']).groups
     |

E402 Module level import not at top of file
    --> julien_data/laod_las_speed.py:1319:1
     |
1317 | # %%
1318 |
1319 | import numpy as np
     | ^^^^^^^^^^^^^^^^^^
1320 | import matplotlib.pyplot as plt
1321 | from itertools import combinations
     |

I001 [*] Import block is un-sorted or un-formatted
    --> julien_data/laod_las_speed.py:1319:1
     |
1317 |   # %%
1318 |
1319 | / import numpy as np
1320 | | import matplotlib.pyplot as plt
1321 | | from itertools import combinations
1322 | | import string
     | |_____________^
1323 |
1324 |   # Suppose you have: all_speed, window_sizes, groups from previous code
     |
help: Organize imports

E402 Module level import not at top of file
    --> julien_data/laod_las_speed.py:1320:1
     |
1319 | import numpy as np
1320 | import matplotlib.pyplot as plt
     | ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
1321 | from itertools import combinations
1322 | import string
     |

E402 Module level import not at top of file
    --> julien_data/laod_las_speed.py:1321:1
     |
1319 | import numpy as np
1320 | import matplotlib.pyplot as plt
1321 | from itertools import combinations
     | ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
1322 | import string
     |

E402 Module level import not at top of file
    --> julien_data/laod_las_speed.py:1322:1
     |
1320 | import matplotlib.pyplot as plt
1321 | from itertools import combinations
1322 | import string
     | ^^^^^^^^^^^^^
1323 |
1324 | # Suppose you have: all_speed, window_sizes, groups from previous code
     |

B905 [*] `zip()` without an explicit `strict=` parameter
    --> julien_data/laod_las_speed.py:1370:5
     |
1369 | for panel_idx, (ax, (g1, g2)) in enumerate(
1370 |     zip(axes.flat, combinations(groups_list, 2))
     |     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
1371 | ):
1372 |     arr1 = group_speeds_dict[g1]
     |
help: Add explicit value for parameter `strict=`

E402 Module level import not at top of file
    --> julien_data/laod_las_speed.py:1493:1
     |
1492 | # Now re-use the Q–Q grid code (with supertitle tweak)
1493 | import numpy as np
     | ^^^^^^^^^^^^^^^^^^
1494 | import matplotlib.pyplot as plt
1495 | from itertools import combinations
     |

I001 [*] Import block is un-sorted or un-formatted
    --> julien_data/laod_las_speed.py:1493:1
     |
1492 |   # Now re-use the Q–Q grid code (with supertitle tweak)
1493 | / import numpy as np
1494 | | import matplotlib.pyplot as plt
1495 | | from itertools import combinations
1496 | | import string
     | |_____________^
     |
help: Organize imports

E402 Module level import not at top of file
    --> julien_data/laod_las_speed.py:1494:1
     |
1492 | # Now re-use the Q–Q grid code (with supertitle tweak)
1493 | import numpy as np
1494 | import matplotlib.pyplot as plt
     | ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
1495 | from itertools import combinations
1496 | import string
     |

E402 Module level import not at top of file
    --> julien_data/laod_las_speed.py:1495:1
     |
1493 | import numpy as np
1494 | import matplotlib.pyplot as plt
1495 | from itertools import combinations
     | ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
1496 | import string
     |

E402 Module level import not at top of file
    --> julien_data/laod_las_speed.py:1496:1
     |
1494 | import matplotlib.pyplot as plt
1495 | from itertools import combinations
1496 | import string
     | ^^^^^^^^^^^^^
     |

B905 [*] `zip()` without an explicit `strict=` parameter
    --> julien_data/laod_las_speed.py:1522:5
     |
1521 | for panel_idx, (ax, (g1, g2)) in enumerate(
1522 |     zip(axes.flat, combinations(groups_list, 2))
     |     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
1523 | ):
1524 |     arr1 = group_speeds_dict_long[g1]
     |
help: Add explicit value for parameter `strict=`

E402 Module level import not at top of file
    --> julien_data/laod_las_speed.py:1619:1
     |
1617 | # %%
1618 | # %%
1619 | import matplotlib.pyplot as plt
     | ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
1620 | import seaborn as sns
1621 | import numpy as np
     |

I001 [*] Import block is un-sorted or un-formatted
    --> julien_data/laod_las_speed.py:1619:1
     |
1617 |   # %%
1618 |   # %%
1619 | / import matplotlib.pyplot as plt
1620 | | import seaborn as sns
1621 | | import numpy as np
     | |__________________^
1622 |
1623 |   # Assume: groups = df_summary.groupby(['genotype', 'treatment']).groups
     |
help: Organize imports

E402 Module level import not at top of file
    --> julien_data/laod_las_speed.py:1620:1
     |
1618 | # %%
1619 | import matplotlib.pyplot as plt
1620 | import seaborn as sns
     | ^^^^^^^^^^^^^^^^^^^^^
1621 | import numpy as np
     |

E402 Module level import not at top of file
    --> julien_data/laod_las_speed.py:1621:1
     |
1619 | import matplotlib.pyplot as plt
1620 | import seaborn as sns
1621 | import numpy as np
     | ^^^^^^^^^^^^^^^^^^
1622 |
1623 | # Assume: groups = df_summary.groupby(['genotype', 'treatment']).groups
     |

E402 Module level import not at top of file
    --> julien_data/laod_las_speed.py:1686:1
     |
1684 | # ----------- Kruskal-Wallis test for long window speeds -----------
1685 | # %%
1686 | from scipy.stats import kruskal
     | ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
1687 |
1688 | # Prepare data for test (lists of arrays)
     |

E402 Module level import not at top of file
    --> julien_data/laod_las_speed.py:1696:1
     |
1695 | # %%
1696 | from scipy.stats import mannwhitneyu
     | ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
1697 | from itertools import combinations
     |

I001 [*] Import block is un-sorted or un-formatted
    --> julien_data/laod_las_speed.py:1696:1
     |
1695 |   # %%
1696 | / from scipy.stats import mannwhitneyu
1697 | | from itertools import combinations
     | |__________________________________^
1698 |
1699 |   # Bonferroni correction for multiple comparisons
     |
help: Organize imports

E402 Module level import not at top of file
    --> julien_data/laod_las_speed.py:1697:1
     |
1695 | # %%
1696 | from scipy.stats import mannwhitneyu
1697 | from itertools import combinations
     | ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
1698 |
1699 | # Bonferroni correction for multiple comparisons
     |

E402 Module level import not at top of file
    --> julien_data/laod_las_speed.py:1710:1
     |
1709 | # %%
1710 | import pandas as pd
     | ^^^^^^^^^^^^^^^^^^^
1711 | import statsmodels.api as sm
     |

E402 Module level import not at top of file
    --> julien_data/laod_las_speed.py:1711:1
     |
1709 | # %%
1710 | import pandas as pd
1711 | import statsmodels.api as sm
     | ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
1712 |
1713 | groups = df_summary.groupby(["genotype", "treatment"])
     |

I001 [*] Import block is un-sorted or un-formatted
  --> julien_data/local_speed_plot.py:2:1
   |
 1 |   # %%
 2 | / import pickle
 3 | | from pathlib import Path
 4 | |
 5 | | # from matplotlib import scale
 6 | | # from networkx import density
 7 | | import numpy as np
 8 | | from class_dataanalysis_julien import DFCAnalysis
 9 | | import matplotlib.pyplot as plt
10 | | import seaborn as sns
11 | | import pandas as pd
12 | | from scipy.stats import theilslopes, spearmanr, kruskal
13 | | from itertools import combinations
14 | | import string
15 | | from scipy.stats import mannwhitneyu
16 | | import statsmodels.api as sm
17 | |
18 | | from dataclasses import dataclass
   | |_________________________________^
19 |
20 |   data = DFCAnalysis()
   |
help: Organize imports

F401 [*] `pathlib.Path` imported but unused
 --> julien_data/local_speed_plot.py:3:21
  |
1 | # %%
2 | import pickle
3 | from pathlib import Path
  |                     ^^^^
4 |
5 | # from matplotlib import scale
  |
help: Remove unused import: `pathlib.Path`

B007 Loop control variable `color` not used within loop body
   --> julien_data/local_speed_plot.py:245:35
    |
245 | for (group_name, animal_indices), color in zip(data.groups.items(), palette):
    |                                   ^^^^^
246 |     print(f"Processing group {group_name} with n animals {len(animal_indices)}")
247 |     pooled = []
    |
help: Rename unused `color` to `_color`

B905 [*] `zip()` without an explicit `strict=` parameter
   --> julien_data/local_speed_plot.py:245:44
    |
245 | for (group_name, animal_indices), color in zip(data.groups.items(), palette):
    |                                            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
246 |     print(f"Processing group {group_name} with n animals {len(animal_indices)}")
247 |     pooled = []
    |
help: Add explicit value for parameter `strict=`

B905 [*] `zip()` without an explicit `strict=` parameter
   --> julien_data/local_speed_plot.py:355:48
    |
353 |     """Flatten speed array for a given animal index across all taus."""
354 |
355 |     for (group_name, animal_indices), color in zip(data.groups.items(), palette):
    |                                                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
356 |         pooled = []
357 |         for win_list in all_speed:  # Each window size
    |
help: Add explicit value for parameter `strict=`

B905 [*] `zip()` without an explicit `strict=` parameter
   --> julien_data/local_speed_plot.py:579:25
    |
577 |     cog_df = cog_data_filtered.reset_index(drop=True)
578 |     cog_scores = cog_df["index_NOR"].values
579 |     group_labels = list(zip(cog_df["genotype"], cog_df["treatment"]))
    |                         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
580 |
581 |     # Assign color/marker per group
    |
help: Add explicit value for parameter `strict=`

B905 [*] `zip()` without an explicit `strict=` parameter
   --> julien_data/local_speed_plot.py:846:9
    |
845 |     for panel_idx, (ax, (g1, g2)) in enumerate(
846 |         zip(axes.flat, combinations(groups_list, 2))
    |         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
847 |     ):
848 |         arr1 = group_speeds_dict[g1]
    |
help: Add explicit value for parameter `strict=`

E712 Avoid equality comparisons to `True`; use `save_fig:` for truth checks
   --> julien_data/local_speed_plot.py:946:8
    |
944 |         left=0.09, right=0.96, bottom=0.13, top=0.94, wspace=0.25, hspace=0.28
945 |     )
946 |     if save_fig == True:
    |        ^^^^^^^^^^^^^^^^
947 |         plt.savefig(
948 |             data.paths["f_speed"]
    |
help: Replace with `save_fig`

B905 [*] `zip()` without an explicit `strict=` parameter
    --> julien_data/local_speed_plot.py:1006:9
     |
1005 |     for panel_idx, (ax, (g1, g2)) in enumerate(
1006 |         zip(axes.flat, combinations(groups_list, 2))
     |         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
1007 |     ):
1008 |         arr1 = group_speeds_dict_long[g1]
     |
help: Add explicit value for parameter `strict=`

F821 Undefined name `group_speeds_dict`
    --> julien_data/local_speed_plot.py:1681:33
     |
1680 | # Prepare data for test (lists of arrays)
1681 | data_for_test = [arr for arr in group_speeds_dict.values()]
     |                                 ^^^^^^^^^^^^^^^^^
1682 | stat, pval = kruskal(*data_for_test)
     |

F821 Undefined name `group_speeds_dict`
    --> julien_data/local_speed_plot.py:1691:15
     |
1690 | # Bonferroni correction for multiple comparisons
1691 | n_comps = len(group_speeds_dict) * (len(group_speeds_dict) - 1) // 2
     |               ^^^^^^^^^^^^^^^^^
1692 | for g1, g2 in combinations(group_speeds_dict.keys(), 2):
1693 |     u, p = mannwhitneyu(
     |

F821 Undefined name `group_speeds_dict`
    --> julien_data/local_speed_plot.py:1691:41
     |
1690 | # Bonferroni correction for multiple comparisons
1691 | n_comps = len(group_speeds_dict) * (len(group_speeds_dict) - 1) // 2
     |                                         ^^^^^^^^^^^^^^^^^
1692 | for g1, g2 in combinations(group_speeds_dict.keys(), 2):
1693 |     u, p = mannwhitneyu(
     |

F821 Undefined name `group_speeds_dict`
    --> julien_data/local_speed_plot.py:1692:28
     |
1690 | # Bonferroni correction for multiple comparisons
1691 | n_comps = len(group_speeds_dict) * (len(group_speeds_dict) - 1) // 2
1692 | for g1, g2 in combinations(group_speeds_dict.keys(), 2):
     |                            ^^^^^^^^^^^^^^^^^
1693 |     u, p = mannwhitneyu(
1694 |         group_speeds_dict[g1], group_speeds_dict[g2], alternative="two-sided"
     |

F821 Undefined name `group_speeds_dict`
    --> julien_data/local_speed_plot.py:1694:9
     |
1692 | for g1, g2 in combinations(group_speeds_dict.keys(), 2):
1693 |     u, p = mannwhitneyu(
1694 |         group_speeds_dict[g1], group_speeds_dict[g2], alternative="two-sided"
     |         ^^^^^^^^^^^^^^^^^
1695 |     )
1696 |     print(
     |

F821 Undefined name `group_speeds_dict`
    --> julien_data/local_speed_plot.py:1694:32
     |
1692 | for g1, g2 in combinations(group_speeds_dict.keys(), 2):
1693 |     u, p = mannwhitneyu(
1694 |         group_speeds_dict[g1], group_speeds_dict[g2], alternative="two-sided"
     |                                ^^^^^^^^^^^^^^^^^
1695 |     )
1696 |     print(
     |

F821 Undefined name `df_summary`
    --> julien_data/local_speed_plot.py:1702:10
     |
1700 | # %%
1701 |
1702 | groups = df_summary.groupby(["genotype", "treatment"])
     |          ^^^^^^^^^^
1703 |
1704 | results = []
     |

I001 [*] Import block is un-sorted or un-formatted
  --> julien_data/local_speed_plot_v2.py:2:1
   |
 1 |   # %%
 2 | / import pickle
 3 | | import numpy as np
 4 | | import pandas as pd
 5 | | import matplotlib.pyplot as plt
 6 | | import seaborn as sns
 7 | | from dataclasses import dataclass
 8 | | from pathlib import Path
 9 | | from typing import Optional, List, Dict, Tuple, Union
10 | |
11 | | from scipy.stats import theilslopes, spearmanr, kruskal, mannwhitneyu
12 | | from scipy import stats
13 | | import statsmodels.api as sm
14 | |
15 | | # from matplotlib import scale
16 | | # from networkx import density
17 | | from class_dataanalysis_julien import DFCAnalysis
18 | | from itertools import combinations
19 | | import string
   | |_____________^
   |
help: Organize imports

F401 [*] `pathlib.Path` imported but unused
 --> julien_data/local_speed_plot_v2.py:8:21
  |
6 | import seaborn as sns
7 | from dataclasses import dataclass
8 | from pathlib import Path
  |                     ^^^^
9 | from typing import Optional, List, Dict, Tuple, Union
  |
help: Remove unused import: `pathlib.Path`

UP035 `typing.List` is deprecated, use `list` instead
  --> julien_data/local_speed_plot_v2.py:9:1
   |
 7 | from dataclasses import dataclass
 8 | from pathlib import Path
 9 | from typing import Optional, List, Dict, Tuple, Union
   | ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
10 |
11 | from scipy.stats import theilslopes, spearmanr, kruskal, mannwhitneyu
   |

UP035 `typing.Dict` is deprecated, use `dict` instead
  --> julien_data/local_speed_plot_v2.py:9:1
   |
 7 | from dataclasses import dataclass
 8 | from pathlib import Path
 9 | from typing import Optional, List, Dict, Tuple, Union
   | ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
10 |
11 | from scipy.stats import theilslopes, spearmanr, kruskal, mannwhitneyu
   |

UP035 `typing.Tuple` is deprecated, use `tuple` instead
  --> julien_data/local_speed_plot_v2.py:9:1
   |
 7 | from dataclasses import dataclass
 8 | from pathlib import Path
 9 | from typing import Optional, List, Dict, Tuple, Union
   | ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
10 |
11 | from scipy.stats import theilslopes, spearmanr, kruskal, mannwhitneyu
   |

F401 [*] `typing.Dict` imported but unused
  --> julien_data/local_speed_plot_v2.py:9:36
   |
 7 | from dataclasses import dataclass
 8 | from pathlib import Path
 9 | from typing import Optional, List, Dict, Tuple, Union
   |                                    ^^^^
10 |
11 | from scipy.stats import theilslopes, spearmanr, kruskal, mannwhitneyu
   |
help: Remove unused import

F401 [*] `typing.Tuple` imported but unused
  --> julien_data/local_speed_plot_v2.py:9:42
   |
 7 | from dataclasses import dataclass
 8 | from pathlib import Path
 9 | from typing import Optional, List, Dict, Tuple, Union
   |                                          ^^^^^
10 |
11 | from scipy.stats import theilslopes, spearmanr, kruskal, mannwhitneyu
   |
help: Remove unused import

UP006 [*] Use `list` instead of `List` for type annotation
  --> julien_data/local_speed_plot_v2.py:85:16
   |
84 | def pool_speeds(
85 |     all_speed: List[np.ndarray],
   |                ^^^^
86 |     animals: List[int],
87 |     windows: Optional[Union[List[int], np.ndarray]] = None,
   |
help: Replace with `list`

UP006 [*] Use `list` instead of `List` for type annotation
  --> julien_data/local_speed_plot_v2.py:86:14
   |
84 | def pool_speeds(
85 |     all_speed: List[np.ndarray],
86 |     animals: List[int],
   |              ^^^^
87 |     windows: Optional[Union[List[int], np.ndarray]] = None,
88 |     taus: Optional[Union[List[int], np.ndarray]] = None,
   |
help: Replace with `list`

UP045 [*] Use `X | None` for type annotations
  --> julien_data/local_speed_plot_v2.py:87:14
   |
85 |     all_speed: List[np.ndarray],
86 |     animals: List[int],
87 |     windows: Optional[Union[List[int], np.ndarray]] = None,
   |              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
88 |     taus: Optional[Union[List[int], np.ndarray]] = None,
89 |     weighting: str = "sample",  # "sample" (current behavior) or "animal"
   |
help: Convert to `X | None`

UP007 [*] Use `X | Y` for type annotations
  --> julien_data/local_speed_plot_v2.py:87:23
   |
85 |     all_speed: List[np.ndarray],
86 |     animals: List[int],
87 |     windows: Optional[Union[List[int], np.ndarray]] = None,
   |                       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
88 |     taus: Optional[Union[List[int], np.ndarray]] = None,
89 |     weighting: str = "sample",  # "sample" (current behavior) or "animal"
   |
help: Convert to `X | Y`

UP006 [*] Use `list` instead of `List` for type annotation
  --> julien_data/local_speed_plot_v2.py:87:29
   |
85 |     all_speed: List[np.ndarray],
86 |     animals: List[int],
87 |     windows: Optional[Union[List[int], np.ndarray]] = None,
   |                             ^^^^
88 |     taus: Optional[Union[List[int], np.ndarray]] = None,
89 |     weighting: str = "sample",  # "sample" (current behavior) or "animal"
   |
help: Replace with `list`

UP045 [*] Use `X | None` for type annotations
  --> julien_data/local_speed_plot_v2.py:88:11
   |
86 |     animals: List[int],
87 |     windows: Optional[Union[List[int], np.ndarray]] = None,
88 |     taus: Optional[Union[List[int], np.ndarray]] = None,
   |           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
89 |     weighting: str = "sample",  # "sample" (current behavior) or "animal"
90 | ) -> Union[np.ndarray, List[np.ndarray]]:
   |
help: Convert to `X | None`

UP007 [*] Use `X | Y` for type annotations
  --> julien_data/local_speed_plot_v2.py:88:20
   |
86 |     animals: List[int],
87 |     windows: Optional[Union[List[int], np.ndarray]] = None,
88 |     taus: Optional[Union[List[int], np.ndarray]] = None,
   |                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
89 |     weighting: str = "sample",  # "sample" (current behavior) or "animal"
90 | ) -> Union[np.ndarray, List[np.ndarray]]:
   |
help: Convert to `X | Y`

UP006 [*] Use `list` instead of `List` for type annotation
  --> julien_data/local_speed_plot_v2.py:88:26
   |
86 |     animals: List[int],
87 |     windows: Optional[Union[List[int], np.ndarray]] = None,
88 |     taus: Optional[Union[List[int], np.ndarray]] = None,
   |                          ^^^^
89 |     weighting: str = "sample",  # "sample" (current behavior) or "animal"
90 | ) -> Union[np.ndarray, List[np.ndarray]]:
   |
help: Replace with `list`

UP007 [*] Use `X | Y` for type annotations
  --> julien_data/local_speed_plot_v2.py:90:6
   |
88 |     taus: Optional[Union[List[int], np.ndarray]] = None,
89 |     weighting: str = "sample",  # "sample" (current behavior) or "animal"
90 | ) -> Union[np.ndarray, List[np.ndarray]]:
   |      ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
91 |     """
92 |     Gather speed values for given animals, window indices, and tau indices.
   |
help: Convert to `X | Y`

UP006 [*] Use `list` instead of `List` for type annotation
  --> julien_data/local_speed_plot_v2.py:90:24
   |
88 |     taus: Optional[Union[List[int], np.ndarray]] = None,
89 |     weighting: str = "sample",  # "sample" (current behavior) or "animal"
90 | ) -> Union[np.ndarray, List[np.ndarray]]:
   |                        ^^^^
91 |     """
92 |     Gather speed values for given animals, window indices, and tau indices.
   |
help: Replace with `list`

E741 Ambiguous variable name: `l`
   --> julien_data/local_speed_plot_v2.py:188:29
    |
186 |                 pooled_a = np.concatenate(pooled_a) if pooled_a else np.array([])
187 |             lengths.append(len(pooled_a))
188 |         min_len = min(l for l in lengths if l > 0)
    |                             ^
189 |
190 |     for a in range(n_animals):
    |

B905 [*] `zip()` without an explicit `strict=` parameter
   --> julien_data/local_speed_plot_v2.py:437:46
    |
435 |     fig, ax = plt.subplots(figsize=(7, 5))
436 |
437 |     for (grp_name, animal_indices), color in zip(group_items, palette):
    |                                              ^^^^^^^^^^^^^^^^^^^^^^^^^
438 |         idx = np.array(animal_indices)
439 |         mask = ~np.isnan(per_animal_vals[idx]) & ~np.isnan(cog_scores[idx])
    |
help: Add explicit value for parameter `strict=`

B905 [*] `zip()` without an explicit `strict=` parameter
   --> julien_data/local_speed_plot_v2.py:577:27
    |
575 |         order = ["-".join(g) for g in ctx.groups.keys()]
576 |         palette = sns.color_palette("tab10", n_colors=len(order))
577 |         for color, lab in zip(palette, order):
    |                           ^^^^^^^^^^^^^^^^^^^
578 |             sub = df[df["group"] == lab].sort_values("window_size")
579 |             if sub.empty:
    |
help: Add explicit value for parameter `strict=`

F811 Redefinition of unused `plot_group_distributions` from line 216
   --> julien_data/local_speed_plot_v2.py:637:5
    |
637 | def plot_group_distributions(
    |     ^^^^^^^^^^^^^^^^^^^^^^^^ `plot_group_distributions` redefined here
638 |     ctx: SpeedContext,
639 |     windows=None,
    |
   ::: julien_data/local_speed_plot_v2.py:216:5
    |
216 | def plot_group_distributions(
    |     ------------------------ previous definition of `plot_group_distributions` here
217 |     ctx: SpeedContext,
218 |     windows=None,
    |
help: Remove definition: `plot_group_distributions`

E402 Module level import not at top of file
   --> julien_data/local_speed_plot_v2.py:946:1
    |
946 | from scipy import stats
    | ^^^^^^^^^^^^^^^^^^^^^^^
    |

F811 [*] Redefinition of unused `stats` from line 12
   --> julien_data/local_speed_plot_v2.py:946:19
    |
946 | from scipy import stats
    |                   ^^^^^ `stats` redefined here
    |
   ::: julien_data/local_speed_plot_v2.py:12:19
    |
 11 | from scipy.stats import theilslopes, spearmanr, kruskal, mannwhitneyu
 12 | from scipy import stats
    |                   ----- previous definition of `stats` here
 13 | import statsmodels.api as sm
    |
help: Remove definition: `stats`

B905 [*] `zip()` without an explicit `strict=` parameter
   --> julien_data/local_speed_plot_v2.py:957:36
    |
955 |     arrays = [np.asarray(group_arrays[g], float) for g in labels]
956 |     arrays = [a[~np.isnan(a)] for a in arrays]
957 |     nonempty = [(g, a) for g, a in zip(labels, arrays) if a.size > 0]
    |                                    ^^^^^^^^^^^^^^^^^^^
958 |
959 |     if len(nonempty) < 2:
    |
help: Add explicit value for parameter `strict=`

B905 [*] `zip()` without an explicit `strict=` parameter
   --> julien_data/local_speed_plot_v2.py:962:18
    |
960 |         return {"ok": False, "reason": "Need at least two groups with data."}
961 |
962 |     labs, arrs = zip(*nonempty)
    |                  ^^^^^^^^^^^^^^
963 |     H, p = stats.kruskal(*arrs)
964 |     Ns = {g: int(a.size) for g, a in nonempty}
    |
help: Add explicit value for parameter `strict=`

E402 Module level import not at top of file
   --> julien_data/local_speed_plot_v2.py:976:1
    |
976 | import numpy as np
    | ^^^^^^^^^^^^^^^^^^
977 | import pandas as pd
978 | from itertools import combinations
    |

I001 [*] Import block is un-sorted or un-formatted
   --> julien_data/local_speed_plot_v2.py:976:1
    |
976 | / import numpy as np
977 | | import pandas as pd
978 | | from itertools import combinations
    | |__________________________________^
    |
help: Organize imports

E402 Module level import not at top of file
   --> julien_data/local_speed_plot_v2.py:977:1
    |
976 | import numpy as np
977 | import pandas as pd
    | ^^^^^^^^^^^^^^^^^^^
978 | from itertools import combinations
    |

E402 Module level import not at top of file
   --> julien_data/local_speed_plot_v2.py:978:1
    |
976 | import numpy as np
977 | import pandas as pd
978 | from itertools import combinations
    | ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    |

F811 [*] Redefinition of unused `combinations` from line 18
   --> julien_data/local_speed_plot_v2.py:978:23
    |
976 | import numpy as np
977 | import pandas as pd
978 | from itertools import combinations
    |                       ^^^^^^^^^^^^ `combinations` redefined here
    |
   ::: julien_data/local_speed_plot_v2.py:18:23
    |
 16 | # from networkx import density
 17 | from class_dataanalysis_julien import DFCAnalysis
 18 | from itertools import combinations
    |                       ------------ previous definition of `combinations` here
 19 | import string
    |
help: Remove definition: `combinations`

E402 Module level import not at top of file
    --> julien_data/local_speed_plot_v2.py:1095:1
     |
1095 | from itertools import combinations
     | ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
1096 | import string
     |

F811 [*] Redefinition of unused `combinations` from line 978
    --> julien_data/local_speed_plot_v2.py:1095:23
     |
1095 | from itertools import combinations
     |                       ^^^^^^^^^^^^ `combinations` redefined here
1096 | import string
     |
    ::: julien_data/local_speed_plot_v2.py:978:23
     |
 976 | import numpy as np
 977 | import pandas as pd
 978 | from itertools import combinations
     |                       ------------ previous definition of `combinations` here
     |
help: Remove definition: `combinations`

E402 Module level import not at top of file
    --> julien_data/local_speed_plot_v2.py:1096:1
     |
1095 | from itertools import combinations
1096 | import string
     | ^^^^^^^^^^^^^
     |

F811 [*] Redefinition of unused `string` from line 19
    --> julien_data/local_speed_plot_v2.py:1096:8
     |
1095 | from itertools import combinations
1096 | import string
     |        ^^^^^^ `string` redefined here
     |
    ::: julien_data/local_speed_plot_v2.py:19:8
     |
  17 | from class_dataanalysis_julien import DFCAnalysis
  18 | from itertools import combinations
  19 | import string
     |        ------ previous definition of `string` here
     |
help: Remove definition: `string`

B905 [*] `zip()` without an explicit `strict=` parameter
    --> julien_data/local_speed_plot_v2.py:1148:42
     |
1147 |     handles = None
1148 |     for idx, ((g1, g2), ax) in enumerate(zip(pairs, axes.flat)):
     |                                          ^^^^^^^^^^^^^^^^^^^^^
1149 |         a1, a2 = gvals[g1], gvals[g2]
1150 |         if a1.size == 0 or a2.size == 0:
     |
help: Add explicit value for parameter `strict=`

E402 Module level import not at top of file
    --> julien_data/local_speed_plot_v2.py:1334:1
     |
1333 | # %%
1334 | import numpy as np
     | ^^^^^^^^^^^^^^^^^^
1335 | import pandas as pd
1336 | from itertools import combinations
     |

I001 [*] Import block is un-sorted or un-formatted
    --> julien_data/local_speed_plot_v2.py:1334:1
     |
1333 |   # %%
1334 | / import numpy as np
1335 | | import pandas as pd
1336 | | from itertools import combinations
     | |__________________________________^
     |
help: Organize imports

E402 Module level import not at top of file
    --> julien_data/local_speed_plot_v2.py:1335:1
     |
1333 | # %%
1334 | import numpy as np
1335 | import pandas as pd
     | ^^^^^^^^^^^^^^^^^^^
1336 | from itertools import combinations
     |

E402 Module level import not at top of file
    --> julien_data/local_speed_plot_v2.py:1336:1
     |
1334 | import numpy as np
1335 | import pandas as pd
1336 | from itertools import combinations
     | ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
     |

F811 [*] Redefinition of unused `combinations` from line 1095
    --> julien_data/local_speed_plot_v2.py:1336:23
     |
1334 | import numpy as np
1335 | import pandas as pd
1336 | from itertools import combinations
     |                       ^^^^^^^^^^^^ `combinations` redefined here
     |
    ::: julien_data/local_speed_plot_v2.py:1095:23
     |
1095 | from itertools import combinations
     |                       ------------ previous definition of `combinations` here
1096 | import string
     |
help: Remove definition: `combinations`

F811 Redefinition of unused `pairwise_mwu_speed_groups` from line 1004
    --> julien_data/local_speed_plot_v2.py:1362:5
     |
1362 | def pairwise_mwu_speed_groups(
     |     ^^^^^^^^^^^^^^^^^^^^^^^^^ `pairwise_mwu_speed_groups` redefined here
1363 |     group_arrays: dict, correction="bonferroni"
1364 | ) -> pd.DataFrame:
     |
    ::: julien_data/local_speed_plot_v2.py:1004:5
     |
1004 | def pairwise_mwu_speed_groups(
     |     ------------------------- previous definition of `pairwise_mwu_speed_groups` here
1005 |     group_arrays: dict, correction="bonferroni"
1006 | ) -> pd.DataFrame:
     |
help: Remove definition: `pairwise_mwu_speed_groups`

B007 Loop control variable `color` not used within loop body
    --> julien_data/local_speed_plot_v2.py:1532:35
     |
1532 | for (group_name, animal_indices), color in zip(data.groups.items(), palette):
     |                                   ^^^^^
1533 |     print(f"Processing group {group_name} with n animals {len(animal_indices)}")
1534 |     pooled = []
     |
help: Rename unused `color` to `_color`

B905 [*] `zip()` without an explicit `strict=` parameter
    --> julien_data/local_speed_plot_v2.py:1532:44
     |
1532 | for (group_name, animal_indices), color in zip(data.groups.items(), palette):
     |                                            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
1533 |     print(f"Processing group {group_name} with n animals {len(animal_indices)}")
1534 |     pooled = []
     |
help: Add explicit value for parameter `strict=`

B905 [*] `zip()` without an explicit `strict=` parameter
    --> julien_data/local_speed_plot_v2.py:1642:48
     |
1640 |     """Flatten speed array for a given animal index across all taus."""
1641 |
1642 |     for (group_name, animal_indices), color in zip(data.groups.items(), palette):
     |                                                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
1643 |         pooled = []
1644 |         for win_list in all_speed:  # Each window size
     |
help: Add explicit value for parameter `strict=`

B905 [*] `zip()` without an explicit `strict=` parameter
    --> julien_data/local_speed_plot_v2.py:1866:25
     |
1864 |     cog_df = cog_data_filtered.reset_index(drop=True)
1865 |     cog_scores = cog_df["index_NOR"].values
1866 |     group_labels = list(zip(cog_df["genotype"], cog_df["treatment"]))
     |                         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
1867 |
1868 |     # Assign color/marker per group
     |
help: Add explicit value for parameter `strict=`

B905 [*] `zip()` without an explicit `strict=` parameter
    --> julien_data/local_speed_plot_v2.py:2133:9
     |
2132 |     for panel_idx, (ax, (g1, g2)) in enumerate(
2133 |         zip(axes.flat, combinations(groups_list, 2))
     |         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
2134 |     ):
2135 |         arr1 = group_speeds_dict[g1]
     |
help: Add explicit value for parameter `strict=`

E712 Avoid equality comparisons to `True`; use `save_fig:` for truth checks
    --> julien_data/local_speed_plot_v2.py:2233:8
     |
2231 |         left=0.09, right=0.96, bottom=0.13, top=0.94, wspace=0.25, hspace=0.28
2232 |     )
2233 |     if save_fig == True:
     |        ^^^^^^^^^^^^^^^^
2234 |         plt.savefig(
2235 |             data.paths["f_speed"]
     |
help: Replace with `save_fig`

B905 [*] `zip()` without an explicit `strict=` parameter
    --> julien_data/local_speed_plot_v2.py:2293:9
     |
2292 |     for panel_idx, (ax, (g1, g2)) in enumerate(
2293 |         zip(axes.flat, combinations(groups_list, 2))
     |         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
2294 |     ):
2295 |         arr1 = group_speeds_dict_long[g1]
     |
help: Add explicit value for parameter `strict=`

F821 Undefined name `group_speeds_dict`
    --> julien_data/local_speed_plot_v2.py:2968:33
     |
2967 | # Prepare data for test (lists of arrays)
2968 | data_for_test = [arr for arr in group_speeds_dict.values()]
     |                                 ^^^^^^^^^^^^^^^^^
2969 | stat, pval = kruskal(*data_for_test)
     |

F821 Undefined name `group_speeds_dict`
    --> julien_data/local_speed_plot_v2.py:2978:15
     |
2977 | # Bonferroni correction for multiple comparisons
2978 | n_comps = len(group_speeds_dict) * (len(group_speeds_dict) - 1) // 2
     |               ^^^^^^^^^^^^^^^^^
2979 | for g1, g2 in combinations(group_speeds_dict.keys(), 2):
2980 |     u, p = mannwhitneyu(
     |

F821 Undefined name `group_speeds_dict`
    --> julien_data/local_speed_plot_v2.py:2978:41
     |
2977 | # Bonferroni correction for multiple comparisons
2978 | n_comps = len(group_speeds_dict) * (len(group_speeds_dict) - 1) // 2
     |                                         ^^^^^^^^^^^^^^^^^
2979 | for g1, g2 in combinations(group_speeds_dict.keys(), 2):
2980 |     u, p = mannwhitneyu(
     |

F821 Undefined name `group_speeds_dict`
    --> julien_data/local_speed_plot_v2.py:2979:28
     |
2977 | # Bonferroni correction for multiple comparisons
2978 | n_comps = len(group_speeds_dict) * (len(group_speeds_dict) - 1) // 2
2979 | for g1, g2 in combinations(group_speeds_dict.keys(), 2):
     |                            ^^^^^^^^^^^^^^^^^
2980 |     u, p = mannwhitneyu(
2981 |         group_speeds_dict[g1], group_speeds_dict[g2], alternative="two-sided"
     |

F821 Undefined name `group_speeds_dict`
    --> julien_data/local_speed_plot_v2.py:2981:9
     |
2979 | for g1, g2 in combinations(group_speeds_dict.keys(), 2):
2980 |     u, p = mannwhitneyu(
2981 |         group_speeds_dict[g1], group_speeds_dict[g2], alternative="two-sided"
     |         ^^^^^^^^^^^^^^^^^
2982 |     )
2983 |     print(
     |

F821 Undefined name `group_speeds_dict`
    --> julien_data/local_speed_plot_v2.py:2981:32
     |
2979 | for g1, g2 in combinations(group_speeds_dict.keys(), 2):
2980 |     u, p = mannwhitneyu(
2981 |         group_speeds_dict[g1], group_speeds_dict[g2], alternative="two-sided"
     |                                ^^^^^^^^^^^^^^^^^
2982 |     )
2983 |     print(
     |

F821 Undefined name `df_summary`
    --> julien_data/local_speed_plot_v2.py:2989:10
     |
2987 | # %%
2988 |
2989 | groups = df_summary.groupby(["genotype", "treatment"])
     |          ^^^^^^^^^^
2990 |
2991 | results = []
     |

I001 [*] Import block is un-sorted or un-formatted
  --> julien_data/modularity.py:3:1
   |
 1 |   # %%
 2 |   #
 3 | / from pathlib import Path
 4 | | import numpy as np
 5 | | import pickle
 6 | | import logging
 7 | | from tqdm import tqdm
 8 | | import gc
 9 | |
10 | | from joblib import Parallel, delayed
11 | | from typing import Dict, List, Tuple, Optional, Union
12 | |
13 | | from webcolors import name_to_rgb_percent
14 | |
15 | | from class_dataanalysis_julien import DFCAnalysis
16 | | from shared_code.fun_loaddata import save_pickle
17 | | from shared_code.fun_utils import set_figure_params
18 | | from shared_code.fun_dfcspeed import ts2fc, ts2dfc_stream
19 | | from shared_code.fun_metaconnectivity import contingency_matrix_fun
   | |___________________________________________________________________^
20 |
21 |   processors = -1  # Use all available processors
   |
help: Organize imports

F401 [*] `logging` imported but unused
 --> julien_data/modularity.py:6:8
  |
4 | import numpy as np
5 | import pickle
6 | import logging
  |        ^^^^^^^
7 | from tqdm import tqdm
8 | import gc
  |
help: Remove unused import: `logging`

F401 [*] `gc` imported but unused
  --> julien_data/modularity.py:8:8
   |
 6 | import logging
 7 | from tqdm import tqdm
 8 | import gc
   |        ^^
 9 |
10 | from joblib import Parallel, delayed
   |
help: Remove unused import: `gc`

UP035 `typing.Dict` is deprecated, use `dict` instead
  --> julien_data/modularity.py:11:1
   |
10 | from joblib import Parallel, delayed
11 | from typing import Dict, List, Tuple, Optional, Union
   | ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
12 |
13 | from webcolors import name_to_rgb_percent
   |

UP035 `typing.List` is deprecated, use `list` instead
  --> julien_data/modularity.py:11:1
   |
10 | from joblib import Parallel, delayed
11 | from typing import Dict, List, Tuple, Optional, Union
   | ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
12 |
13 | from webcolors import name_to_rgb_percent
   |

UP035 `typing.Tuple` is deprecated, use `tuple` instead
  --> julien_data/modularity.py:11:1
   |
10 | from joblib import Parallel, delayed
11 | from typing import Dict, List, Tuple, Optional, Union
   | ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
12 |
13 | from webcolors import name_to_rgb_percent
   |

F401 [*] `typing.Dict` imported but unused
  --> julien_data/modularity.py:11:20
   |
10 | from joblib import Parallel, delayed
11 | from typing import Dict, List, Tuple, Optional, Union
   |                    ^^^^
12 |
13 | from webcolors import name_to_rgb_percent
   |
help: Remove unused import

F401 [*] `typing.List` imported but unused
  --> julien_data/modularity.py:11:26
   |
10 | from joblib import Parallel, delayed
11 | from typing import Dict, List, Tuple, Optional, Union
   |                          ^^^^
12 |
13 | from webcolors import name_to_rgb_percent
   |
help: Remove unused import

F401 [*] `typing.Tuple` imported but unused
  --> julien_data/modularity.py:11:32
   |
10 | from joblib import Parallel, delayed
11 | from typing import Dict, List, Tuple, Optional, Union
   |                                ^^^^^
12 |
13 | from webcolors import name_to_rgb_percent
   |
help: Remove unused import

F401 [*] `typing.Optional` imported but unused
  --> julien_data/modularity.py:11:39
   |
10 | from joblib import Parallel, delayed
11 | from typing import Dict, List, Tuple, Optional, Union
   |                                       ^^^^^^^^
12 |
13 | from webcolors import name_to_rgb_percent
   |
help: Remove unused import

F401 [*] `typing.Union` imported but unused
  --> julien_data/modularity.py:11:49
   |
10 | from joblib import Parallel, delayed
11 | from typing import Dict, List, Tuple, Optional, Union
   |                                                 ^^^^^
12 |
13 | from webcolors import name_to_rgb_percent
   |
help: Remove unused import

F401 [*] `webcolors.name_to_rgb_percent` imported but unused
  --> julien_data/modularity.py:13:23
   |
11 | from typing import Dict, List, Tuple, Optional, Union
12 |
13 | from webcolors import name_to_rgb_percent
   |                       ^^^^^^^^^^^^^^^^^^^
14 |
15 | from class_dataanalysis_julien import DFCAnalysis
   |
help: Remove unused import: `webcolors.name_to_rgb_percent`

F401 [*] `shared_code.fun_utils.set_figure_params` imported but unused
  --> julien_data/modularity.py:17:35
   |
15 | from class_dataanalysis_julien import DFCAnalysis
16 | from shared_code.fun_loaddata import save_pickle
17 | from shared_code.fun_utils import set_figure_params
   |                                   ^^^^^^^^^^^^^^^^^
18 | from shared_code.fun_dfcspeed import ts2fc, ts2dfc_stream
19 | from shared_code.fun_metaconnectivity import contingency_matrix_fun
   |
help: Remove unused import: `shared_code.fun_utils.set_figure_params`

E402 Module level import not at top of file
  --> julien_data/modularity.py:36:1
   |
34 | fc = np.array([ts2fc(ts[xx]) for xx in tqdm(range(len(ts)))])
35 |
36 | import matplotlib.pyplot as plt
   | ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
37 |
38 | plt.figure(figsize=(10, 8))
   |

E402 Module level import not at top of file
  --> julien_data/modularity.py:76:1
   |
74 | gamma = np.linspace(gmin, gmax, gamma_n)
75 | # %%
76 | import numpy as np
   | ^^^^^^^^^^^^^^^^^^
77 | from sklearn.metrics import normalized_mutual_info_score
78 | from itertools import combinations
   |

I001 [*] Import block is un-sorted or un-formatted
  --> julien_data/modularity.py:76:1
   |
74 |   gamma = np.linspace(gmin, gmax, gamma_n)
75 |   # %%
76 | / import numpy as np
77 | | from sklearn.metrics import normalized_mutual_info_score
78 | | from itertools import combinations
79 | | from joblib import Parallel, delayed
80 | | from tqdm import tqdm
   | |_____________________^
   |
help: Organize imports

E402 Module level import not at top of file
  --> julien_data/modularity.py:77:1
   |
75 | # %%
76 | import numpy as np
77 | from sklearn.metrics import normalized_mutual_info_score
   | ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
78 | from itertools import combinations
79 | from joblib import Parallel, delayed
   |

E402 Module level import not at top of file
  --> julien_data/modularity.py:78:1
   |
76 | import numpy as np
77 | from sklearn.metrics import normalized_mutual_info_score
78 | from itertools import combinations
   | ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
79 | from joblib import Parallel, delayed
80 | from tqdm import tqdm
   |

E402 Module level import not at top of file
  --> julien_data/modularity.py:79:1
   |
77 | from sklearn.metrics import normalized_mutual_info_score
78 | from itertools import combinations
79 | from joblib import Parallel, delayed
   | ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
80 | from tqdm import tqdm
   |

F811 [*] Redefinition of unused `Parallel` from line 10
  --> julien_data/modularity.py:79:20
   |
77 | from sklearn.metrics import normalized_mutual_info_score
78 | from itertools import combinations
79 | from joblib import Parallel, delayed
   |                    ^^^^^^^^ `Parallel` redefined here
80 | from tqdm import tqdm
   |
  ::: julien_data/modularity.py:10:20
   |
 8 | import gc
 9 |
10 | from joblib import Parallel, delayed
   |                    -------- previous definition of `Parallel` here
11 | from typing import Dict, List, Tuple, Optional, Union
   |
help: Remove definition: `Parallel`

F811 [*] Redefinition of unused `delayed` from line 10
  --> julien_data/modularity.py:79:30
   |
77 | from sklearn.metrics import normalized_mutual_info_score
78 | from itertools import combinations
79 | from joblib import Parallel, delayed
   |                              ^^^^^^^ `delayed` redefined here
80 | from tqdm import tqdm
   |
  ::: julien_data/modularity.py:10:30
   |
 8 | import gc
 9 |
10 | from joblib import Parallel, delayed
   |                              ------- previous definition of `delayed` here
11 | from typing import Dict, List, Tuple, Optional, Union
   |
help: Remove definition: `delayed`

E402 Module level import not at top of file
  --> julien_data/modularity.py:80:1
   |
78 | from itertools import combinations
79 | from joblib import Parallel, delayed
80 | from tqdm import tqdm
   | ^^^^^^^^^^^^^^^^^^^^^
   |

F821 Undefined name `consensus_pipeline`
   --> julien_data/modularity.py:130:51
    |
128 | gammas = np.linspace(0.5, 2.0, gamma_n)
129 |
130 | best_gamma_idx, consensus_partition, allegiance = consensus_pipeline(
    |                                                   ^^^^^^^^^^^^^^^^^^
131 |     communities_mat, gammas
132 | )
    |

F821 Undefined name `gamma_qmod_val2`
   --> julien_data/modularity.py:150:25
    |
148 | )
149 |
150 | plt.plot(gamma, np.mean(gamma_qmod_val2, axis=1), label="gamma_qmod_val2")
    |                         ^^^^^^^^^^^^^^^
151 | plt.fill_between(
152 |     gamma,
    |

F821 Undefined name `gamma_qmod_val2`
   --> julien_data/modularity.py:153:13
    |
151 | plt.fill_between(
152 |     gamma,
153 |     np.mean(gamma_qmod_val2, axis=1) - np.std(gamma_qmod_val2, axis=1),
    |             ^^^^^^^^^^^^^^^
154 |     np.mean(gamma_qmod_val2, axis=1) + np.std(gamma_qmod_val2, axis=1),
155 |     alpha=0.2,
    |

F821 Undefined name `gamma_qmod_val2`
   --> julien_data/modularity.py:153:47
    |
151 | plt.fill_between(
152 |     gamma,
153 |     np.mean(gamma_qmod_val2, axis=1) - np.std(gamma_qmod_val2, axis=1),
    |                                               ^^^^^^^^^^^^^^^
154 |     np.mean(gamma_qmod_val2, axis=1) + np.std(gamma_qmod_val2, axis=1),
155 |     alpha=0.2,
    |

F821 Undefined name `gamma_qmod_val2`
   --> julien_data/modularity.py:154:13
    |
152 |     gamma,
153 |     np.mean(gamma_qmod_val2, axis=1) - np.std(gamma_qmod_val2, axis=1),
154 |     np.mean(gamma_qmod_val2, axis=1) + np.std(gamma_qmod_val2, axis=1),
    |             ^^^^^^^^^^^^^^^
155 |     alpha=0.2,
156 | )
    |

F821 Undefined name `gamma_qmod_val2`
   --> julien_data/modularity.py:154:47
    |
152 |     gamma,
153 |     np.mean(gamma_qmod_val2, axis=1) - np.std(gamma_qmod_val2, axis=1),
154 |     np.mean(gamma_qmod_val2, axis=1) + np.std(gamma_qmod_val2, axis=1),
    |                                               ^^^^^^^^^^^^^^^
155 |     alpha=0.2,
156 | )
    |

F821 Undefined name `gamma_qmod_val3`
   --> julien_data/modularity.py:157:25
    |
155 |     alpha=0.2,
156 | )
157 | plt.plot(gamma, np.mean(gamma_qmod_val3, axis=1), label="gamma_qmod_val3")
    |                         ^^^^^^^^^^^^^^^
158 | plt.fill_between(
159 |     gamma,
    |

F821 Undefined name `gamma_qmod_val3`
   --> julien_data/modularity.py:160:13
    |
158 | plt.fill_between(
159 |     gamma,
160 |     np.mean(gamma_qmod_val3, axis=1) - np.std(gamma_qmod_val3, axis=1),
    |             ^^^^^^^^^^^^^^^
161 |     np.mean(gamma_qmod_val3, axis=1) + np.std(gamma_qmod_val3, axis=1),
162 |     alpha=0.2,
    |

F821 Undefined name `gamma_qmod_val3`
   --> julien_data/modularity.py:160:47
    |
158 | plt.fill_between(
159 |     gamma,
160 |     np.mean(gamma_qmod_val3, axis=1) - np.std(gamma_qmod_val3, axis=1),
    |                                               ^^^^^^^^^^^^^^^
161 |     np.mean(gamma_qmod_val3, axis=1) + np.std(gamma_qmod_val3, axis=1),
162 |     alpha=0.2,
    |

F821 Undefined name `gamma_qmod_val3`
   --> julien_data/modularity.py:161:13
    |
159 |     gamma,
160 |     np.mean(gamma_qmod_val3, axis=1) - np.std(gamma_qmod_val3, axis=1),
161 |     np.mean(gamma_qmod_val3, axis=1) + np.std(gamma_qmod_val3, axis=1),
    |             ^^^^^^^^^^^^^^^
162 |     alpha=0.2,
163 | )
    |

F821 Undefined name `gamma_qmod_val3`
   --> julien_data/modularity.py:161:47
    |
159 |     gamma,
160 |     np.mean(gamma_qmod_val3, axis=1) - np.std(gamma_qmod_val3, axis=1),
161 |     np.mean(gamma_qmod_val3, axis=1) + np.std(gamma_qmod_val3, axis=1),
    |                                               ^^^^^^^^^^^^^^^
162 |     alpha=0.2,
163 | )
    |

E402 Module level import not at top of file
   --> julien_data/modularity.py:172:1
    |
170 | # %%
171 | # Spearman Correlation between allegiance matrices
172 | from scipy.stats import pearsonr, spearmanr
    | ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    |

F401 [*] `scipy.stats.pearsonr` imported but unused
   --> julien_data/modularity.py:172:25
    |
170 | # %%
171 | # Spearman Correlation between allegiance matrices
172 | from scipy.stats import pearsonr, spearmanr
    |                         ^^^^^^^^
    |
help: Remove unused import: `scipy.stats.pearsonr`

B018 Found useless expression. Either assign it to a variable or remove it.
   --> julien_data/modularity.py:201:1
    |
199 | agreement_matrices = np.sum(allegiance_mat, axis=0) / len(fc_wt_veh)
200 |
201 | data.region_labels_preprocessed
    | ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
202 | # plot the agreement matrix
203 | plt.figure(figsize=(10, 8))
    |

E402 Module level import not at top of file
   --> julien_data/modularity.py:217:1
    |
217 | import brainconn as bct
    | ^^^^^^^^^^^^^^^^^^^^^^^
    |

I001 [*] Import block is un-sorted or un-formatted
   --> julien_data/modularity.py:217:1
    |
217 | import brainconn as bct
    | ^^^^^^^^^^^^^^^^^^^^^^^
    |
help: Organize imports

F821 Undefined name `matrix2vec`
   --> julien_data/modularity.py:248:25
    |
247 |     if dfc_stream.ndim == 3:
248 |         dfc_stream_2D = matrix2vec(dfc_stream)
    |                         ^^^^^^^^^^
249 |     else:
250 |         dfc_stream_2D = dfc_stream
    |

B018 Found useless expression. Either assign it to a variable or remove it.
   --> julien_data/modularity.py:276:1
    |
274 | plt.yticks([])
275 | plt.colorbar(label=r"CC($FC(t_i)$, $FC(t_j)$)")
276 | data.cog_data_filtered
    | ^^^^^^^^^^^^^^^^^^^^^^
    |

I001 [*] Import block is un-sorted or un-formatted
  --> julien_data/plot_cog_data.py:2:1
   |
 1 |   # %%
 2 | / from os import path
 3 | | from pathlib import Path
 4 | | from tkinter import font
 5 | | from matplotlib.pylab import f
 6 | | import numpy as np
 7 | | import matplotlib.pyplot as plt
 8 | | import seaborn as sns
 9 | | import pandas as pd
10 | |
11 | | from statannotations.Annotator import Annotator
12 | | import seaborn as sns
13 | | from scipy.stats import kruskal
14 | |
15 | | from shared_code.fun_utils import set_figure_params
16 | | from shared_code.fun_paths import get_paths
17 | |
18 | |
19 | | from class_dataanalysis_julien import DFCAnalysis
   | |_________________________________________________^
20 |
21 |   data = DFCAnalysis()
   |
help: Organize imports

F401 [*] `os.path` imported but unused
 --> julien_data/plot_cog_data.py:2:16
  |
1 | # %%
2 | from os import path
  |                ^^^^
3 | from pathlib import Path
4 | from tkinter import font
  |
help: Remove unused import: `os.path`

F401 [*] `pathlib.Path` imported but unused
 --> julien_data/plot_cog_data.py:3:21
  |
1 | # %%
2 | from os import path
3 | from pathlib import Path
  |                     ^^^^
4 | from tkinter import font
5 | from matplotlib.pylab import f
  |
help: Remove unused import: `pathlib.Path`

F401 [*] `tkinter.font` imported but unused
 --> julien_data/plot_cog_data.py:4:21
  |
2 | from os import path
3 | from pathlib import Path
4 | from tkinter import font
  |                     ^^^^
5 | from matplotlib.pylab import f
6 | import numpy as np
  |
help: Remove unused import: `tkinter.font`

F401 [*] `matplotlib.pylab.f` imported but unused
 --> julien_data/plot_cog_data.py:5:30
  |
3 | from pathlib import Path
4 | from tkinter import font
5 | from matplotlib.pylab import f
  |                              ^
6 | import numpy as np
7 | import matplotlib.pyplot as plt
  |
help: Remove unused import: `matplotlib.pylab.f`

F401 [*] `pandas` imported but unused
  --> julien_data/plot_cog_data.py:9:18
   |
 7 | import matplotlib.pyplot as plt
 8 | import seaborn as sns
 9 | import pandas as pd
   |                  ^^
10 |
11 | from statannotations.Annotator import Annotator
   |
help: Remove unused import: `pandas`

F811 [*] Redefinition of unused `sns` from line 8
  --> julien_data/plot_cog_data.py:12:19
   |
11 | from statannotations.Annotator import Annotator
12 | import seaborn as sns
   |                   ^^^ `sns` redefined here
13 | from scipy.stats import kruskal
   |
  ::: julien_data/plot_cog_data.py:8:19
   |
 6 | import numpy as np
 7 | import matplotlib.pyplot as plt
 8 | import seaborn as sns
   |                   --- previous definition of `sns` here
 9 | import pandas as pd
   |
help: Remove definition: `sns`

F401 [*] `shared_code.fun_paths.get_paths` imported but unused
  --> julien_data/plot_cog_data.py:16:35
   |
15 | from shared_code.fun_utils import set_figure_params
16 | from shared_code.fun_paths import get_paths
   |                                   ^^^^^^^^^
   |
help: Remove unused import: `shared_code.fun_paths.get_paths`

E402 Module level import not at top of file
   --> julien_data/plot_cog_data.py:171:1
    |
169 | print(f"Statistic: {stat:.4f}, p-value: {p:.4g}")
170 |
171 | from scipy.stats import mannwhitneyu
    | ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
172 | from statsmodels.stats.multitest import multipletests
173 | import itertools
    |

I001 [*] Import block is un-sorted or un-formatted
   --> julien_data/plot_cog_data.py:171:1
    |
169 |   print(f"Statistic: {stat:.4f}, p-value: {p:.4g}")
170 |
171 | / from scipy.stats import mannwhitneyu
172 | | from statsmodels.stats.multitest import multipletests
173 | | import itertools
    | |________________^
174 |
175 |   pairs = list(itertools.combinations(groups, 2))
    |
help: Organize imports

E402 Module level import not at top of file
   --> julien_data/plot_cog_data.py:172:1
    |
171 | from scipy.stats import mannwhitneyu
172 | from statsmodels.stats.multitest import multipletests
    | ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
173 | import itertools
    |

E402 Module level import not at top of file
   --> julien_data/plot_cog_data.py:173:1
    |
171 | from scipy.stats import mannwhitneyu
172 | from statsmodels.stats.multitest import multipletests
173 | import itertools
    | ^^^^^^^^^^^^^^^^
174 |
175 | pairs = list(itertools.combinations(groups, 2))
    |

B905 [*] `zip()` without an explicit `strict=` parameter
   --> julien_data/plot_cog_data.py:187:30
    |
185 | reject, pvals_corrected, _, _ = multipletests(pvals, method="fdr_bh")
186 | print("\nFDR-corrected p-values:")
187 | for (g1, g2), p_corr, sig in zip(pairs, pvals_corrected, reject):
    |                              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
188 |     print(f"{g1} vs {g2}: corrected p={p_corr:.4g}, {'significant' if sig else 'ns'}")
    |
help: Add explicit value for parameter `strict=`

B007 Loop control variable `sig` not used within loop body
   --> julien_data/plot_cog_data.py:230:5
    |
228 | # Prepare annotation labels for the plot
229 | star_labels = []
230 | for sig, p_corr in zip(reject, pvals_corrected):
    |     ^^^
231 |     if p_corr < 0.001:
232 |         star = "***"
    |
help: Rename unused `sig` to `_sig`

B905 [*] `zip()` without an explicit `strict=` parameter
   --> julien_data/plot_cog_data.py:230:20
    |
228 | # Prepare annotation labels for the plot
229 | star_labels = []
230 | for sig, p_corr in zip(reject, pvals_corrected):
    |                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
231 |     if p_corr < 0.001:
232 |         star = "***"
    |
help: Add explicit value for parameter `strict=`

UP032 [*] Use f-string instead of `format` call
   --> julien_data/plot_cog_data.py:251:11
    |
249 | )
250 | annotator.set_pvalues_and_annotate(pvals_corrected)
251 | plt.title("NOR values by Genotype and Treatment\n(Kruskal-Wallis: p={:.3g})".format(p))
    |           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
252 | plt.tight_layout()
253 | (
    |
help: Convert to f-string

F821 Undefined name `speeds_all`
   --> julien_data/plot_cog_data.py:291:25
    |
289 | tau_count = 3  # adjust as needed
290 | animal_count = len(filtered_df)
291 | time_window_count = len(speeds_all)  # now using len instead of shape
    |                         ^^^^^^^^^^
292 |
293 | plt.figure(figsize=(8, 5))
    |

F821 Undefined name `speeds_all`
   --> julien_data/plot_cog_data.py:302:23
    |
300 |                 # speeds_all is a list: speeds_all[time_win][animal_idx * tau_count + tau, :]
301 |                 print(animal_idx, animal_idx * tau_count + tau)
302 |                 arr = speeds_all[time_win][animal_idx * tau_count + tau]
    |                       ^^^^^^^^^^
303 |                 arr = np.asarray(arr, dtype=float)
304 |                 arr = arr[~np.isnan(arr)]
    |

F821 Undefined name `speeds_all_T`
   --> julien_data/plot_cog_data.py:340:26
    |
339 | all_speeds = np.concatenate(
340 |     [speed for animal in speeds_all_T for speed in animal]
    |                          ^^^^^^^^^^^^
341 | ).astype(np.float32)
342 | all_speeds = all_speeds[~np.isnan(all_speeds)]
    |

F821 Undefined name `speeds_all_T`
   --> julien_data/plot_cog_data.py:347:22
    |
345 | # Pool all tau for each animal (removing NaNs)
346 | pooled_speeds_per_animal = []
347 | for animal_speeds in speeds_all_T:
    |                      ^^^^^^^^^^^^
348 |     # animal_speeds is a list of arrays, one per tau
349 |     animal_pool = np.concatenate(
    |

E402 Module level import not at top of file
   --> julien_data/plot_cog_data.py:375:1
    |
375 | import numpy as np
    | ^^^^^^^^^^^^^^^^^^
376 | import matplotlib.pyplot as plt
    |

I001 [*] Import block is un-sorted or un-formatted
   --> julien_data/plot_cog_data.py:375:1
    |
375 | / import numpy as np
376 | | import matplotlib.pyplot as plt
    | |_______________________________^
377 |
378 |   filtered_df = cog_data_filtered[cog_data_filtered["n_timepoints"] >= 500].reset_index(
    |
help: Organize imports

E402 Module level import not at top of file
   --> julien_data/plot_cog_data.py:376:1
    |
375 | import numpy as np
376 | import matplotlib.pyplot as plt
    | ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
377 |
378 | filtered_df = cog_data_filtered[cog_data_filtered["n_timepoints"] >= 500].reset_index(
    |

F821 Undefined name `speeds_all_T`
   --> julien_data/plot_cog_data.py:391:30
    |
389 |         animal_tau_speeds = [
390 |             speed_arr.astype(float)[~np.isnan(speed_arr)]
391 |             for speed_arr in speeds_all_T[animal_idx]
    |                              ^^^^^^^^^^^^
392 |             if speed_arr is not None
393 |         ]
    |

F821 Undefined name `speeds_all_T`
   --> julien_data/plot_cog_data.py:426:26
    |
424 |             speed.astype(float)
425 |             for animal_idx in animal_idxs
426 |             for speed in speeds_all_T[animal_idx]
    |                          ^^^^^^^^^^^^
427 |         ]
428 |     )
    |

E402 Module level import not at top of file
   --> julien_data/plot_cog_data.py:449:1
    |
448 | # %%
449 | import matplotlib.pyplot as plt
    | ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
450 | import seaborn as sns
    |

E402 Module level import not at top of file
   --> julien_data/plot_cog_data.py:450:1
    |
448 | # %%
449 | import matplotlib.pyplot as plt
450 | import seaborn as sns
    | ^^^^^^^^^^^^^^^^^^^^^
451 |
452 | filtered_df = cog_data_filtered[cog_data_filtered["n_timepoints"] >= 500]
    |

F821 Undefined name `speeds_all_T`
   --> julien_data/plot_cog_data.py:463:26
    |
461 |             speed.astype(float)
462 |             for animal_idx in animal_idxs
463 |             for speed in speeds_all_T[animal_idx]
    |                          ^^^^^^^^^^^^
464 |         ]
465 |     )
    |

E402 Module level import not at top of file
   --> julien_data/plot_cog_data.py:501:1
    |
500 | # %%
501 | import matplotlib.pyplot as plt
    | ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
502 | import seaborn as sns
503 | import numpy as np
    |

I001 [*] Import block is un-sorted or un-formatted
   --> julien_data/plot_cog_data.py:501:1
    |
500 |   # %%
501 | / import matplotlib.pyplot as plt
502 | | import seaborn as sns
503 | | import numpy as np
    | |__________________^
504 |
505 |   filtered_df = cog_data_filtered[cog_data_filtered["n_timepoints"] >= 500]
    |
help: Organize imports

E402 Module level import not at top of file
   --> julien_data/plot_cog_data.py:502:1
    |
500 | # %%
501 | import matplotlib.pyplot as plt
502 | import seaborn as sns
    | ^^^^^^^^^^^^^^^^^^^^^
503 | import numpy as np
    |

E402 Module level import not at top of file
   --> julien_data/plot_cog_data.py:503:1
    |
501 | import matplotlib.pyplot as plt
502 | import seaborn as sns
503 | import numpy as np
    | ^^^^^^^^^^^^^^^^^^
504 |
505 | filtered_df = cog_data_filtered[cog_data_filtered["n_timepoints"] >= 500]
    |

F821 Undefined name `speeds_all_T`
   --> julien_data/plot_cog_data.py:516:26
    |
514 |             speed.astype(float)
515 |             for animal_idx in animal_idxs
516 |             for speed in speeds_all_T[animal_idx]
    |                          ^^^^^^^^^^^^
517 |         ]
518 |     )
    |

E402 Module level import not at top of file
   --> julien_data/plot_cog_data.py:552:1
    |
551 | # %%
552 | import matplotlib.pyplot as plt
    | ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
553 | import seaborn as sns
554 | import numpy as np
    |

I001 [*] Import block is un-sorted or un-formatted
   --> julien_data/plot_cog_data.py:552:1
    |
551 |   # %%
552 | / import matplotlib.pyplot as plt
553 | | import seaborn as sns
554 | | import numpy as np
    | |__________________^
555 |
556 |   filtered_df = cog_data_filtered[cog_data_filtered["n_timepoints"] >= 500]
    |
help: Organize imports

E402 Module level import not at top of file
   --> julien_data/plot_cog_data.py:553:1
    |
551 | # %%
552 | import matplotlib.pyplot as plt
553 | import seaborn as sns
    | ^^^^^^^^^^^^^^^^^^^^^
554 | import numpy as np
    |

E402 Module level import not at top of file
   --> julien_data/plot_cog_data.py:554:1
    |
552 | import matplotlib.pyplot as plt
553 | import seaborn as sns
554 | import numpy as np
    | ^^^^^^^^^^^^^^^^^^
555 |
556 | filtered_df = cog_data_filtered[cog_data_filtered["n_timepoints"] >= 500]
    |

F821 Undefined name `speeds_all_T`
   --> julien_data/plot_cog_data.py:567:26
    |
565 |             speed.astype(float)
566 |             for animal_idx in animal_idxs
567 |             for speed in speeds_all_T[animal_idx]
    |                          ^^^^^^^^^^^^
568 |         ]
569 |     )
    |

E402 Module level import not at top of file
   --> julien_data/plot_cog_data.py:602:1
    |
600 | # window size/ median speed
601 |
602 | import numpy as np
    | ^^^^^^^^^^^^^^^^^^
603 | import matplotlib.pyplot as plt
604 | import seaborn as sns
    |

I001 [*] Import block is un-sorted or un-formatted
   --> julien_data/plot_cog_data.py:602:1
    |
600 |   # window size/ median speed
601 |
602 | / import numpy as np
603 | | import matplotlib.pyplot as plt
604 | | import seaborn as sns
    | |_____________________^
605 |
606 |   filtered_df = cog_data_filtered[cog_data_filtered["n_timepoints"] >= 500]
    |
help: Organize imports

E402 Module level import not at top of file
   --> julien_data/plot_cog_data.py:603:1
    |
602 | import numpy as np
603 | import matplotlib.pyplot as plt
    | ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
604 | import seaborn as sns
    |

E402 Module level import not at top of file
   --> julien_data/plot_cog_data.py:604:1
    |
602 | import numpy as np
603 | import matplotlib.pyplot as plt
604 | import seaborn as sns
    | ^^^^^^^^^^^^^^^^^^^^^
605 |
606 | filtered_df = cog_data_filtered[cog_data_filtered["n_timepoints"] >= 500]
    |

F821 Undefined name `time_window_range`
   --> julien_data/plot_cog_data.py:609:16
    |
607 | groups = filtered_df.groupby(["genotype", "treatment"]).groups
608 |
609 | window_sizes = time_window_range  # fill with your actual window sizes
    |                ^^^^^^^^^^^^^^^^^
610 | n_windows = len(speeds_all)  # Or len(window_sizes)
    |

F821 Undefined name `speeds_all`
   --> julien_data/plot_cog_data.py:610:17
    |
609 | window_sizes = time_window_range  # fill with your actual window sizes
610 | n_windows = len(speeds_all)  # Or len(window_sizes)
    |                 ^^^^^^^^^^
    |

F821 Undefined name `speeds_all`
   --> julien_data/plot_cog_data.py:622:13
    |
620 |         # Pool all animals' speeds for this window and group
621 |         speeds_this_window = [
622 |             speeds_all[win_idx][animal_idx].astype(float)
    |             ^^^^^^^^^^
623 |             for animal_idx in animal_idxs
624 |             if len(speeds_all[win_idx][animal_idx]) > 0  # skip empty arrays
    |

F821 Undefined name `speeds_all`
   --> julien_data/plot_cog_data.py:624:20
    |
622 |             speeds_all[win_idx][animal_idx].astype(float)
623 |             for animal_idx in animal_idxs
624 |             if len(speeds_all[win_idx][animal_idx]) > 0  # skip empty arrays
    |                    ^^^^^^^^^^
625 |         ]
626 |         if speeds_this_window:
    |

E402 Module level import not at top of file
   --> julien_data/plot_cog_data.py:650:1
    |
648 | # %%
649 |
650 | import matplotlib.pyplot as plt
    | ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
651 | import seaborn as sns
    |

E402 Module level import not at top of file
   --> julien_data/plot_cog_data.py:651:1
    |
650 | import matplotlib.pyplot as plt
651 | import seaborn as sns
    | ^^^^^^^^^^^^^^^^^^^^^
652 |
653 | filtered_df = cog_data_filtered[cog_data_filtered["n_timepoints"] >= 500]
    |

F821 Undefined name `speeds_all`
   --> julien_data/plot_cog_data.py:656:17
    |
654 | groups = filtered_df.groupby(["genotype", "treatment"]).groups
655 |
656 | n_windows = len(speeds_all)
    |                 ^^^^^^^^^^
657 | palette = sns.color_palette("tab10", n_colors=len(groups))
    |

F821 Undefined name `speeds_all`
   --> julien_data/plot_cog_data.py:668:13
    |
666 |         # Pool all animals' speeds for this window and group
667 |         speeds_this_window = [
668 |             speeds_all[win_idx][animal_idx].astype(float)
    |             ^^^^^^^^^^
669 |             for animal_idx in animal_idxs
670 |             if len(speeds_all[win_idx][animal_idx]) > 0  # skip empty
    |

F821 Undefined name `speeds_all`
   --> julien_data/plot_cog_data.py:670:20
    |
668 |             speeds_all[win_idx][animal_idx].astype(float)
669 |             for animal_idx in animal_idxs
670 |             if len(speeds_all[win_idx][animal_idx]) > 0  # skip empty
    |                    ^^^^^^^^^^
671 |         ]
672 |         if speeds_this_window:
    |

F821 Undefined name `time_window_range`
   --> julien_data/plot_cog_data.py:703:16
    |
702 | quantile_levels = np.linspace(0.05, 0.95, 19)  # e.g. 0.05, 0.1, ..., 0.95
703 | window_sizes = time_window_range  # fill with your actual window sizes
    |                ^^^^^^^^^^^^^^^^^
704 | n_windows = len(window_sizes)
705 | n_q = len(quantile_levels)
    |

F821 Undefined name `speeds_all`
   --> julien_data/plot_cog_data.py:715:9
    |
713 |     # Pool all speeds for this window and group
714 |     speeds_this_window = [
715 |         speeds_all[win_idx][animal_idx].astype(float)
    |         ^^^^^^^^^^
716 |         for animal_idx in animal_idxs
717 |         if len(speeds_all[win_idx][animal_idx]) > 0
    |

F821 Undefined name `speeds_all`
   --> julien_data/plot_cog_data.py:717:16
    |
715 |         speeds_all[win_idx][animal_idx].astype(float)
716 |         for animal_idx in animal_idxs
717 |         if len(speeds_all[win_idx][animal_idx]) > 0
    |                ^^^^^^^^^^
718 |     ]
719 |     if speeds_this_window:
    |

E402 Module level import not at top of file
   --> julien_data/plot_cog_data.py:728:1
    |
726 |         speed_matrix[:, win_idx] = np.nan
727 |
728 | import matplotlib.pyplot as plt
    | ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
729 |
730 | plt.figure(figsize=(10, 5))
    |

UP032 [*] Use f-string instead of `format` call
   --> julien_data/plot_cog_data.py:744:11
    |
742 | plt.xlabel("Window Size")
743 | plt.ylabel("Quantile")
744 | plt.title("Speed quantile matrix\n(Group: {})".format(list(groups.keys())[0]))
    |           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
745 | plt.tight_layout()
746 | plt.show()
    |
help: Convert to f-string

E402 Module level import not at top of file
   --> julien_data/plot_cog_data.py:750:1
    |
748 | # %%
749 |
750 | import matplotlib.colors as mcolors
    | ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
751 |
752 | speed_matrix_clipped = np.clip(speed_matrix, 1e-10, None)
    |

UP032 [*] Use f-string instead of `format` call
   --> julien_data/plot_cog_data.py:769:11
    |
767 | plt.xlabel("Window Size")
768 | plt.ylabel("Quantile")
769 | plt.title("Speed quantile matrix\n(Group: {})".format(list(groups.keys())[0]))
    |           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
770 | plt.tight_layout()
771 | plt.show()
    |
help: Convert to f-string

E402 Module level import not at top of file
   --> julien_data/plot_cog_data.py:776:1
    |
774 | # %%
775 |
776 | import numpy as np
    | ^^^^^^^^^^^^^^^^^^
777 | import matplotlib.pyplot as plt
778 | import seaborn as sns
    |

I001 [*] Import block is un-sorted or un-formatted
   --> julien_data/plot_cog_data.py:776:1
    |
774 |   # %%
775 |
776 | / import numpy as np
777 | | import matplotlib.pyplot as plt
778 | | import seaborn as sns
779 | | import matplotlib.colors as mcolors
    | |___________________________________^
780 |
781 |   filtered_df = cog_data_filtered[cog_data_filtered["n_timepoints"] >= 500]
    |
help: Organize imports

E402 Module level import not at top of file
   --> julien_data/plot_cog_data.py:777:1
    |
776 | import numpy as np
777 | import matplotlib.pyplot as plt
    | ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
778 | import seaborn as sns
779 | import matplotlib.colors as mcolors
    |

E402 Module level import not at top of file
   --> julien_data/plot_cog_data.py:778:1
    |
776 | import numpy as np
777 | import matplotlib.pyplot as plt
778 | import seaborn as sns
    | ^^^^^^^^^^^^^^^^^^^^^
779 | import matplotlib.colors as mcolors
    |

E402 Module level import not at top of file
   --> julien_data/plot_cog_data.py:779:1
    |
777 | import matplotlib.pyplot as plt
778 | import seaborn as sns
779 | import matplotlib.colors as mcolors
    | ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
780 |
781 | filtered_df = cog_data_filtered[cog_data_filtered["n_timepoints"] >= 500]
    |

F821 Undefined name `time_window_range`
   --> julien_data/plot_cog_data.py:785:16
    |
784 | quantile_levels = np.linspace(0.05, 0.95, 19)  # or your preferred quantiles
785 | window_sizes = time_window_range  # fill with your actual window sizes
    |                ^^^^^^^^^^^^^^^^^
786 | n_windows = len(window_sizes)
787 | n_q = len(quantile_levels)
    |

F821 Undefined name `speeds_all`
   --> julien_data/plot_cog_data.py:793:13
    |
791 |     for win_idx in range(n_windows):
792 |         speeds_this_window = [
793 |             speeds_all[win_idx][animal_idx].astype(float)
    |             ^^^^^^^^^^
794 |             for animal_idx in animal_idxs
795 |             if len(speeds_all[win_idx][animal_idx]) > 0
    |

F821 Undefined name `speeds_all`
   --> julien_data/plot_cog_data.py:795:20
    |
793 |             speeds_all[win_idx][animal_idx].astype(float)
794 |             for animal_idx in animal_idxs
795 |             if len(speeds_all[win_idx][animal_idx]) > 0
    |                    ^^^^^^^^^^
796 |         ]
797 |         if speeds_this_window:
    |

E402 Module level import not at top of file
   --> julien_data/plot_cog_data.py:839:1
    |
837 | # %%
838 |
839 | import numpy as np
    | ^^^^^^^^^^^^^^^^^^
840 | import matplotlib.pyplot as plt
    |

I001 [*] Import block is un-sorted or un-formatted
   --> julien_data/plot_cog_data.py:839:1
    |
837 |   # %%
838 |
839 | / import numpy as np
840 | | import matplotlib.pyplot as plt
    | |_______________________________^
841 |
842 |   filtered_df = cog_data_filtered[cog_data_filtered["n_timepoints"] >= 500]
    |
help: Organize imports

E402 Module level import not at top of file
   --> julien_data/plot_cog_data.py:840:1
    |
839 | import numpy as np
840 | import matplotlib.pyplot as plt
    | ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
841 |
842 | filtered_df = cog_data_filtered[cog_data_filtered["n_timepoints"] >= 500]
    |

F821 Undefined name `time_window_range`
   --> julien_data/plot_cog_data.py:846:16
    |
845 | quantile_levels = np.linspace(0, 1, 100)  # e.g. 19 quantiles
846 | window_sizes = time_window_range  # fill with your actual window sizes
    |                ^^^^^^^^^^^^^^^^^
847 | n_windows = len(window_sizes)
848 | n_q = len(quantile_levels)
    |

F821 Undefined name `speeds_all`
   --> julien_data/plot_cog_data.py:859:13
    |
857 |     for win_idx in range(n_windows):
858 |         speeds_this_window = [
859 |             speeds_all[win_idx][animal_idx].astype(float)
    |             ^^^^^^^^^^
860 |             for animal_idx in animal_idxs
861 |             if len(speeds_all[win_idx][animal_idx]) > 0
    |

F821 Undefined name `speeds_all`
   --> julien_data/plot_cog_data.py:861:20
    |
859 |             speeds_all[win_idx][animal_idx].astype(float)
860 |             for animal_idx in animal_idxs
861 |             if len(speeds_all[win_idx][animal_idx]) > 0
    |                    ^^^^^^^^^^
862 |         ]
863 |         if speeds_this_window:
    |

E402 Module level import not at top of file
   --> julien_data/plot_cog_data.py:909:1
    |
908 | # %%
909 | import numpy as np
    | ^^^^^^^^^^^^^^^^^^
910 | import matplotlib.pyplot as plt
    |

I001 [*] Import block is un-sorted or un-formatted
   --> julien_data/plot_cog_data.py:909:1
    |
908 |   # %%
909 | / import numpy as np
910 | | import matplotlib.pyplot as plt
    | |_______________________________^
911 |
912 |   filtered_df = cog_data_filtered[cog_data_filtered["n_timepoints"] >= 500]
    |
help: Organize imports

E402 Module level import not at top of file
   --> julien_data/plot_cog_data.py:910:1
    |
908 | # %%
909 | import numpy as np
910 | import matplotlib.pyplot as plt
    | ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
911 |
912 | filtered_df = cog_data_filtered[cog_data_filtered["n_timepoints"] >= 500]
    |

F821 Undefined name `speeds_all`
   --> julien_data/plot_cog_data.py:936:13
    |
934 |     for win_idx in range(n_windows):
935 |         speeds_this_window = [
936 |             speeds_all[win_idx][animal_idx].astype(float)
    |             ^^^^^^^^^^
937 |             for animal_idx in animal_idxs
938 |             if len(speeds_all[win_idx][animal_idx]) > 0
    |

F821 Undefined name `speeds_all`
   --> julien_data/plot_cog_data.py:938:20
    |
936 |             speeds_all[win_idx][animal_idx].astype(float)
937 |             for animal_idx in animal_idxs
938 |             if len(speeds_all[win_idx][animal_idx]) > 0
    |                    ^^^^^^^^^^
939 |         ]
940 |         if speeds_this_window:
    |

E402 Module level import not at top of file
   --> julien_data/plot_cog_data.py:985:1
    |
983 | # %%
984 |
985 | import numpy as np
    | ^^^^^^^^^^^^^^^^^^
986 | import matplotlib.pyplot as plt
    |

I001 [*] Import block is un-sorted or un-formatted
   --> julien_data/plot_cog_data.py:985:1
    |
983 |   # %%
984 |
985 | / import numpy as np
986 | | import matplotlib.pyplot as plt
    | |_______________________________^
987 |
988 |   filtered_df = cog_data_filtered[cog_data_filtered["n_timepoints"] >= 500]
    |
help: Organize imports

E402 Module level import not at top of file
   --> julien_data/plot_cog_data.py:986:1
    |
985 | import numpy as np
986 | import matplotlib.pyplot as plt
    | ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
987 |
988 | filtered_df = cog_data_filtered[cog_data_filtered["n_timepoints"] >= 500]
    |

F821 Undefined name `speeds_all`
    --> julien_data/plot_cog_data.py:1012:13
     |
1010 |     for win_idx in range(n_windows):
1011 |         speeds_this_window = [
1012 |             speeds_all[win_idx][animal_idx].astype(float)
     |             ^^^^^^^^^^
1013 |             for animal_idx in animal_idxs
1014 |             if len(speeds_all[win_idx][animal_idx]) > 0
     |

F821 Undefined name `speeds_all`
    --> julien_data/plot_cog_data.py:1014:20
     |
1012 |             speeds_all[win_idx][animal_idx].astype(float)
1013 |             for animal_idx in animal_idxs
1014 |             if len(speeds_all[win_idx][animal_idx]) > 0
     |                    ^^^^^^^^^^
1015 |         ]
1016 |         if speeds_this_window:
     |

E402 Module level import not at top of file
    --> julien_data/plot_cog_data.py:1063:1
     |
1062 | # %%
1063 | import numpy as np
     | ^^^^^^^^^^^^^^^^^^
1064 | import matplotlib.pyplot as plt
     |

I001 [*] Import block is un-sorted or un-formatted
    --> julien_data/plot_cog_data.py:1063:1
     |
1062 |   # %%
1063 | / import numpy as np
1064 | | import matplotlib.pyplot as plt
     | |_______________________________^
1065 |
1066 |   diff_AB = speed_matrices[0] - speed_matrices[1]  # A - B
     |
help: Organize imports

E402 Module level import not at top of file
    --> julien_data/plot_cog_data.py:1064:1
     |
1062 | # %%
1063 | import numpy as np
1064 | import matplotlib.pyplot as plt
     | ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
1065 |
1066 | diff_AB = speed_matrices[0] - speed_matrices[1]  # A - B
     |

UP009 [*] UTF-8 encoding declaration is unnecessary
 --> julien_data/plots.py:2:1
  |
1 | #!/usr/bin/env python3
2 | # -*- coding: utf-8 -*-
  | ^^^^^^^^^^^^^^^^^^^^^^^
3 | """
4 | Created on Mon Oct  2 14:42:38 2023
  |
help: Remove unnecessary coding comment

I001 [*] Import block is un-sorted or un-formatted
  --> julien_data/plots.py:10:1
   |
 9 |   # %%
10 | / from pathlib import Path
11 | | import numpy as np
12 | | import matplotlib.pyplot as plt
13 | | import brainconn as bct
14 | | import os
15 | | import time
16 | |
17 | | from joblib import Parallel, delayed, parallel_backend
18 | | import pandas as pd
19 | |
20 | | # from functions_analysis import *
21 | | from scipy.io import loadmat, savemat
22 | | from scipy.special import erfc
23 | | from scipy.stats import pearsonr, spearmanr
24 | |
25 | | from shared_code.fun_loaddata import *  # Import only needed functions
26 | | from shared_code.fun_dfcspeed import parallel_dfc_speed_oversampled_series
27 | | from shared_code.fun_utils import set_figure_params
28 | | from shared_code.fun_paths import get_paths
29 | | from tqdm import tqdm
30 | |
31 | | from shared_code.shared_code.fun_dfcspeed import get_tenet4window_range
   | |_______________________________________________________________________^
   |
help: Organize imports

F401 [*] `brainconn` imported but unused
  --> julien_data/plots.py:13:21
   |
11 | import numpy as np
12 | import matplotlib.pyplot as plt
13 | import brainconn as bct
   |                     ^^^
14 | import os
15 | import time
   |
help: Remove unused import: `brainconn`

F401 [*] `os` imported but unused
  --> julien_data/plots.py:14:8
   |
12 | import matplotlib.pyplot as plt
13 | import brainconn as bct
14 | import os
   |        ^^
15 | import time
   |
help: Remove unused import: `os`

F401 [*] `time` imported but unused
  --> julien_data/plots.py:15:8
   |
13 | import brainconn as bct
14 | import os
15 | import time
   |        ^^^^
16 |
17 | from joblib import Parallel, delayed, parallel_backend
   |
help: Remove unused import: `time`

F401 [*] `joblib.Parallel` imported but unused
  --> julien_data/plots.py:17:20
   |
15 | import time
16 |
17 | from joblib import Parallel, delayed, parallel_backend
   |                    ^^^^^^^^
18 | import pandas as pd
   |
help: Remove unused import

F401 [*] `joblib.delayed` imported but unused
  --> julien_data/plots.py:17:30
   |
15 | import time
16 |
17 | from joblib import Parallel, delayed, parallel_backend
   |                              ^^^^^^^
18 | import pandas as pd
   |
help: Remove unused import

F401 [*] `joblib.parallel_backend` imported but unused
  --> julien_data/plots.py:17:39
   |
15 | import time
16 |
17 | from joblib import Parallel, delayed, parallel_backend
   |                                       ^^^^^^^^^^^^^^^^
18 | import pandas as pd
   |
help: Remove unused import

F401 [*] `scipy.io.loadmat` imported but unused
  --> julien_data/plots.py:21:22
   |
20 | # from functions_analysis import *
21 | from scipy.io import loadmat, savemat
   |                      ^^^^^^^
22 | from scipy.special import erfc
23 | from scipy.stats import pearsonr, spearmanr
   |
help: Remove unused import

F401 [*] `scipy.io.savemat` imported but unused
  --> julien_data/plots.py:21:31
   |
20 | # from functions_analysis import *
21 | from scipy.io import loadmat, savemat
   |                               ^^^^^^^
22 | from scipy.special import erfc
23 | from scipy.stats import pearsonr, spearmanr
   |
help: Remove unused import

F401 [*] `scipy.special.erfc` imported but unused
  --> julien_data/plots.py:22:27
   |
20 | # from functions_analysis import *
21 | from scipy.io import loadmat, savemat
22 | from scipy.special import erfc
   |                           ^^^^
23 | from scipy.stats import pearsonr, spearmanr
   |
help: Remove unused import: `scipy.special.erfc`

F401 [*] `scipy.stats.pearsonr` imported but unused
  --> julien_data/plots.py:23:25
   |
21 | from scipy.io import loadmat, savemat
22 | from scipy.special import erfc
23 | from scipy.stats import pearsonr, spearmanr
   |                         ^^^^^^^^
24 |
25 | from shared_code.fun_loaddata import *  # Import only needed functions
   |
help: Remove unused import

F401 [*] `scipy.stats.spearmanr` imported but unused
  --> julien_data/plots.py:23:35
   |
21 | from scipy.io import loadmat, savemat
22 | from scipy.special import erfc
23 | from scipy.stats import pearsonr, spearmanr
   |                                   ^^^^^^^^^
24 |
25 | from shared_code.fun_loaddata import *  # Import only needed functions
   |
help: Remove unused import

F403 `from shared_code.fun_loaddata import *` used; unable to detect undefined names
  --> julien_data/plots.py:25:1
   |
23 | from scipy.stats import pearsonr, spearmanr
24 |
25 | from shared_code.fun_loaddata import *  # Import only needed functions
   | ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
26 | from shared_code.fun_dfcspeed import parallel_dfc_speed_oversampled_series
27 | from shared_code.fun_utils import set_figure_params
   |

F401 [*] `shared_code.fun_dfcspeed.parallel_dfc_speed_oversampled_series` imported but unused
  --> julien_data/plots.py:26:38
   |
25 | from shared_code.fun_loaddata import *  # Import only needed functions
26 | from shared_code.fun_dfcspeed import parallel_dfc_speed_oversampled_series
   |                                      ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
27 | from shared_code.fun_utils import set_figure_params
28 | from shared_code.fun_paths import get_paths
   |
help: Remove unused import: `shared_code.fun_dfcspeed.parallel_dfc_speed_oversampled_series`

F401 [*] `shared_code.fun_utils.set_figure_params` imported but unused
  --> julien_data/plots.py:27:35
   |
25 | from shared_code.fun_loaddata import *  # Import only needed functions
26 | from shared_code.fun_dfcspeed import parallel_dfc_speed_oversampled_series
27 | from shared_code.fun_utils import set_figure_params
   |                                   ^^^^^^^^^^^^^^^^^
28 | from shared_code.fun_paths import get_paths
29 | from tqdm import tqdm
   |
help: Remove unused import: `shared_code.fun_utils.set_figure_params`

F401 [*] `tqdm.tqdm` imported but unused
  --> julien_data/plots.py:29:18
   |
27 | from shared_code.fun_utils import set_figure_params
28 | from shared_code.fun_paths import get_paths
29 | from tqdm import tqdm
   |                  ^^^^
30 |
31 | from shared_code.shared_code.fun_dfcspeed import get_tenet4window_range
   |
help: Remove unused import: `tqdm.tqdm`

F401 [*] `shared_code.shared_code.fun_dfcspeed.get_tenet4window_range` imported but unused
  --> julien_data/plots.py:31:50
   |
29 | from tqdm import tqdm
30 |
31 | from shared_code.shared_code.fun_dfcspeed import get_tenet4window_range
   |                                                  ^^^^^^^^^^^^^^^^^^^^^^
   |
help: Remove unused import: `shared_code.shared_code.fun_dfcspeed.get_tenet4window_range`

F405 `load_npz_dict` may be undefined, or defined from star imports
  --> julien_data/plots.py:50:15
   |
48 | # ------------------------ Load Data ------------------------
49 |
50 | data_ts_pre = load_npz_dict(paths["preprocessed"] / Path("ts_filtered_unstacked.npz"))
   |               ^^^^^^^^^^^^^
51 | ts = data_ts_pre["ts"]
52 | n_animals = data_ts_pre["n_animals"]
   |

E402 Module level import not at top of file
  --> julien_data/plots.py:82:1
   |
81 | # %%
82 | from shared_code.fun_loaddata import make_file_path, load_from_cache
   | ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
83 |
84 | # Plot FCD of the first animal
   |

I001 [*] Import block is un-sorted or un-formatted
  --> julien_data/plots.py:82:1
   |
81 | # %%
82 | from shared_code.fun_loaddata import make_file_path, load_from_cache
   | ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
83 |
84 | # Plot FCD of the first animal
   |
help: Organize imports

E402 Module level import not at top of file
  --> julien_data/plots.py:96:1
   |
95 | # %%
96 | from shared_code.fun_utils import dfc_stream2fcd
   | ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
97 |
98 | fcd = [dfc_stream2fcd(dfc_stream[n]) for n in range(n_animals)]
   |

E712 Avoid equality comparisons to `True`; use `group:` for truth checks
   --> julien_data/plots.py:137:18
    |
135 | for i, group in enumerate(group_gen_treat):
136 |     plt.subplot(2, 2, i + 1)
137 |     print(np.sum(group == True), "animals in group", labels[i])
    |                  ^^^^^^^^^^^^^
138 |     # Select the FCDs for the current group
139 |     fcd_group = [fcd[idx] for idx, val in enumerate(group.values) if val]
    |
help: Replace with `group`

I001 [*] Import block is un-sorted or un-formatted
  --> julien_data/plts_speed.py:2:1
   |
 1 |   # %%
 2 | / from pathlib import Path
 3 | | import numpy as np
 4 | | import pandas as pd
 5 | | import time
 6 | |
 7 | | from shared_code.fun_paths import get_paths
 8 | | from shared_code.fun_loaddata import (
 9 | |     load_mat_timeseries,
10 | |     extract_mouse_ids,
11 | |     load_npz_dict,
12 | |     make_file_path,
13 | | )
   | |_^
14 |
15 |   # %%
   |
help: Organize imports

F401 [*] `time` imported but unused
 --> julien_data/plts_speed.py:5:8
  |
3 | import numpy as np
4 | import pandas as pd
5 | import time
  |        ^^^^
6 |
7 | from shared_code.fun_paths import get_paths
  |
help: Remove unused import: `time`

I001 [*] Import block is un-sorted or un-formatted
 --> julien_data/simple_speed_analysis.py:6:1
  |
4 |   """
5 |
6 | / import numpy as np
7 | | import matplotlib.pyplot as plt
8 | | from pathlib import Path
  | |________________________^
  |
help: Organize imports

F541 [*] f-string without any placeholders
  --> julien_data/simple_speed_analysis.py:27:11
   |
26 |     # Simple statistics on speed medians
27 |     print(f"\n=== SPEED MEDIAN ANALYSIS ===")
   |           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
28 |     print(
29 |         f"Overall median speed range: [{np.nanmin(speed_medians):.6f}, {np.nanmax(speed_medians):.6f}]"
   |
help: Remove extraneous `f` prefix

F541 [*] f-string without any placeholders
  --> julien_data/simple_speed_analysis.py:34:11
   |
33 |     # Try to analyze individual animal speeds
34 |     print(f"\n=== INDIVIDUAL ANIMAL ANALYSIS ===")
   |           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
35 |     valid_animals = 0
36 |     total_measurements = 0
   |
help: Remove extraneous `f` prefix

F541 [*] f-string without any placeholders
  --> julien_data/simple_speed_analysis.py:65:11
   |
64 |     # Create simple visualization
65 |     print(f"\n=== CREATING VISUALIZATION ===")
   |           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
66 |
67 |     fig, axes = plt.subplots(2, 2, figsize=(12, 10))
   |
help: Remove extraneous `f` prefix

I001 [*] Import block is un-sorted or un-formatted
 --> julien_data/test_func_speed.py:3:1
  |
1 |   # %%
2 |   # import pickle
3 | / from pathlib import Path
4 | | from networkx import density
5 | | import numpy as np
6 | | from class_dataanalysis_julien import DFCAnalysis
7 | | import pickle
  | |_____________^
8 |
9 |   data = DFCAnalysis()
  |
help: Organize imports

F401 [*] `pathlib.Path` imported but unused
 --> julien_data/test_func_speed.py:3:21
  |
1 | # %%
2 | # import pickle
3 | from pathlib import Path
  |                     ^^^^
4 | from networkx import density
5 | import numpy as np
  |
help: Remove unused import: `pathlib.Path`

F401 [*] `networkx.density` imported but unused
 --> julien_data/test_func_speed.py:4:22
  |
2 | # import pickle
3 | from pathlib import Path
4 | from networkx import density
  |                      ^^^^^^^
5 | import numpy as np
6 | from class_dataanalysis_julien import DFCAnalysis
  |
help: Remove unused import: `networkx.density`

I001 [*] Import block is un-sorted or un-formatted
   --> julien_data/test_func_speed.py:129:5
    |
127 |       MethodsX 2020, doi: 10.1016/j.mex.2020.101168
128 |       """
129 | /     from shared_code.fun_dfcspeed import (
130 | |         pearson_speed_vectorized,
131 | |         spearman_speed,
132 | |         cosine_speed_vectorized,
133 | |     )
    | |_____^
134 |
135 |       # Input validation
    |
help: Organize imports

E402 Module level import not at top of file
   --> julien_data/test_func_speed.py:229:1
    |
228 | # hist of all_speed_5_0 abd speed
229 | import matplotlib.pyplot as plt
    | ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
230 |
231 | plt.hist(all_speed_5_0, bins=10, alpha=0.5, label="all_speed_5_0")
    |

E402 Module level import not at top of file
   --> julien_data/test_func_speed.py:240:1
    |
239 | # %%
240 | import logging
    | ^^^^^^^^^^^^^^
241 | from joblib import Parallel, delayed
242 | from tqdm import tqdm
    |

I001 [*] Import block is un-sorted or un-formatted
   --> julien_data/test_func_speed.py:240:1
    |
239 |   # %%
240 | / import logging
241 | | from joblib import Parallel, delayed
242 | | from tqdm import tqdm
    | |_____________________^
243 |
244 |   logging.basicConfig(level=logging.WARNING, format="%(levelname)s:%(name)s:%(message)s")
    |
help: Organize imports

E402 Module level import not at top of file
   --> julien_data/test_func_speed.py:241:1
    |
239 | # %%
240 | import logging
241 | from joblib import Parallel, delayed
    | ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
242 | from tqdm import tqdm
    |

F401 [*] `joblib.Parallel` imported but unused
   --> julien_data/test_func_speed.py:241:20
    |
239 | # %%
240 | import logging
241 | from joblib import Parallel, delayed
    |                    ^^^^^^^^
242 | from tqdm import tqdm
    |
help: Remove unused import

F401 [*] `joblib.delayed` imported but unused
   --> julien_data/test_func_speed.py:241:30
    |
239 | # %%
240 | import logging
241 | from joblib import Parallel, delayed
    |                              ^^^^^^^
242 | from tqdm import tqdm
    |
help: Remove unused import

E402 Module level import not at top of file
   --> julien_data/test_func_speed.py:242:1
    |
240 | import logging
241 | from joblib import Parallel, delayed
242 | from tqdm import tqdm
    | ^^^^^^^^^^^^^^^^^^^^^
243 |
244 | logging.basicConfig(level=logging.WARNING, format="%(levelname)s:%(name)s:%(message)s")
    |

F401 [*] `tqdm.tqdm` imported but unused
   --> julien_data/test_func_speed.py:242:18
    |
240 | import logging
241 | from joblib import Parallel, delayed
242 | from tqdm import tqdm
    |                  ^^^^
243 |
244 | logging.basicConfig(level=logging.WARNING, format="%(levelname)s:%(name)s:%(message)s")
    |
help: Remove unused import: `tqdm.tqdm`

I001 [*] Import block is un-sorted or un-formatted
   --> julien_data/test_func_speed.py:294:5
    |
292 |       MethodsX 2020, doi: 10.1016/j.mex.2020.101168
293 |       """
294 | /     from shared_code.fun_optimization import (
295 | |         pearson_speed_vectorized,
296 | |         spearman_speed,
297 | |         cosine_speed_vectorized,
298 | |     )
    | |_____^
299 |
300 |       # Input validation
    |
help: Organize imports

F841 Local variable `n_pairs` is assigned to but never used
   --> julien_data/test_func_speed.py:360:5
    |
359 |     n_speeds = (len(indices) - 1) * np.size(tau_range)
360 |     n_pairs = fc_stream.shape[0]
    |     ^^^^^^^
361 |
362 |     # Pre-allocate output arrays for efficiency
    |
help: Remove assignment to unused variable `n_pairs`

F841 Local variable `fc2_stream` is assigned to but never used
   --> julien_data/test_func_speed.py:364:5
    |
362 |     # Pre-allocate output arrays for efficiency
363 |     speeds = np.empty((n_speeds, np.size(tau_range)), dtype=np.float32)
364 |     fc2_stream = None
    |     ^^^^^^^^^^
365 |
366 |     # Extract FC matrices for vectorized computation
    |
help: Remove assignment to unused variable `fc2_stream`

UP009 [*] UTF-8 encoding declaration is unnecessary
 --> metaconnectivity/cognitive_data_ts_sorted.py:2:1
  |
1 | #!/usr/bin/env python3
2 | # -*- coding: utf-8 -*-
  | ^^^^^^^^^^^^^^^^^^^^^^^
3 | """
4 | Created on Mon Sep 23 13:26:30 2024
  |
help: Remove unnecessary coding comment

I001 [*] Import block is un-sorted or un-formatted
  --> metaconnectivity/cognitive_data_ts_sorted.py:9:1
   |
 7 |   """
 8 |   # %%
 9 | / from pathlib import Path
10 | | import os
11 | | import numpy as np
12 | | import matplotlib.pyplot as plt
13 | | import pandas as pd
14 | | import pickle
15 | |
16 | | from shared_code.fun_loaddata import extract_hash_numbers
17 | | from shared_code.fun_utils import (
18 | |     filename_sort_mat,
19 | |     load_matdata,
20 | |     classify_phenotypes,
21 | |     make_combination_masks,
22 | |     make_masks,
23 | | )
24 | | from shared_code.fun_paths import get_paths
25 | | import time
   | |___________^
26 |
27 |   # =============================================================================
   |
help: Organize imports

F401 [*] `pathlib.Path` imported but unused
  --> metaconnectivity/cognitive_data_ts_sorted.py:9:21
   |
 7 | """
 8 | # %%
 9 | from pathlib import Path
   |                     ^^^^
10 | import os
11 | import numpy as np
   |
help: Remove unused import: `pathlib.Path`

F401 [*] `matplotlib.pyplot` imported but unused
  --> metaconnectivity/cognitive_data_ts_sorted.py:12:29
   |
10 | import os
11 | import numpy as np
12 | import matplotlib.pyplot as plt
   |                             ^^^
13 | import pandas as pd
14 | import pickle
   |
help: Remove unused import: `matplotlib.pyplot`

F401 [*] `time` imported but unused
  --> metaconnectivity/cognitive_data_ts_sorted.py:25:8
   |
23 | )
24 | from shared_code.fun_paths import get_paths
25 | import time
   |        ^^^^
26 |
27 | # =============================================================================
   |
help: Remove unused import: `time`

UP009 [*] UTF-8 encoding declaration is unnecessary
 --> metaconnectivity/compute_fluidity.py:2:1
  |
1 | #!/usr/bin/env python3
2 | # -*- coding: utf-8 -*-
  | ^^^^^^^^^^^^^^^^^^^^^^^
3 | """
4 | Created on Mon Sep 23 13:26:30 2024
  |
help: Remove unnecessary coding comment

I001 [*] Import block is un-sorted or un-formatted
  --> metaconnectivity/compute_fluidity.py:11:1
   |
 9 |   # %%
10 |   # from click import group
11 | / import numpy as np
12 | | import time
13 | |
14 | | # from functions_analysis import *
15 | | from pathlib import Path
16 | |
17 | | import scipy
18 | |
19 | | from shared_code.fun_loaddata import *
20 | | from shared_code.fun_dfcspeed import *
21 | |
22 | | from shared_code.fun_metaconnectivity import (
23 | |     compute_metaconnectivity,
24 | |     intramodule_indices_mask,
25 | |     get_fc_mc_indices,
26 | |     get_mc_region_identities,
27 | |     fun_allegiance_communities,
28 | |     compute_trimers_identity,
29 | |     build_trimer_mask,
30 | | )
31 | |
32 | | from shared_code.fun_utils import (
33 | |     set_figure_params,
34 | |     #    get_paths,
35 | |     load_cognitive_data,
36 | |     load_timeseries_data,
37 | |     load_grouping_data,
38 | | )
39 | | from shared_code.fun_paths import get_paths
   | |___________________________________________^
40 |
41 |   # =============================================================================
   |
help: Organize imports

F401 [*] `time` imported but unused
  --> metaconnectivity/compute_fluidity.py:12:8
   |
10 | # from click import group
11 | import numpy as np
12 | import time
   |        ^^^^
13 |
14 | # from functions_analysis import *
   |
help: Remove unused import: `time`

F401 [*] `pathlib.Path` imported but unused
  --> metaconnectivity/compute_fluidity.py:15:21
   |
14 | # from functions_analysis import *
15 | from pathlib import Path
   |                     ^^^^
16 |
17 | import scipy
   |
help: Remove unused import: `pathlib.Path`

F401 [*] `scipy` imported but unused
  --> metaconnectivity/compute_fluidity.py:17:8
   |
15 | from pathlib import Path
16 |
17 | import scipy
   |        ^^^^^
18 |
19 | from shared_code.fun_loaddata import *
   |
help: Remove unused import: `scipy`

F403 `from shared_code.fun_loaddata import *` used; unable to detect undefined names
  --> metaconnectivity/compute_fluidity.py:19:1
   |
17 | import scipy
18 |
19 | from shared_code.fun_loaddata import *
   | ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
20 | from shared_code.fun_dfcspeed import *
   |

F403 `from shared_code.fun_dfcspeed import *` used; unable to detect undefined names
  --> metaconnectivity/compute_fluidity.py:20:1
   |
19 | from shared_code.fun_loaddata import *
20 | from shared_code.fun_dfcspeed import *
   | ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
21 |
22 | from shared_code.fun_metaconnectivity import (
   |

F401 [*] `shared_code.fun_metaconnectivity.compute_metaconnectivity` imported but unused
  --> metaconnectivity/compute_fluidity.py:23:5
   |
22 | from shared_code.fun_metaconnectivity import (
23 |     compute_metaconnectivity,
   |     ^^^^^^^^^^^^^^^^^^^^^^^^
24 |     intramodule_indices_mask,
25 |     get_fc_mc_indices,
   |
help: Remove unused import

F401 [*] `shared_code.fun_metaconnectivity.intramodule_indices_mask` imported but unused
  --> metaconnectivity/compute_fluidity.py:24:5
   |
22 | from shared_code.fun_metaconnectivity import (
23 |     compute_metaconnectivity,
24 |     intramodule_indices_mask,
   |     ^^^^^^^^^^^^^^^^^^^^^^^^
25 |     get_fc_mc_indices,
26 |     get_mc_region_identities,
   |
help: Remove unused import

F401 [*] `shared_code.fun_metaconnectivity.get_fc_mc_indices` imported but unused
  --> metaconnectivity/compute_fluidity.py:25:5
   |
23 |     compute_metaconnectivity,
24 |     intramodule_indices_mask,
25 |     get_fc_mc_indices,
   |     ^^^^^^^^^^^^^^^^^
26 |     get_mc_region_identities,
27 |     fun_allegiance_communities,
   |
help: Remove unused import

F401 [*] `shared_code.fun_metaconnectivity.get_mc_region_identities` imported but unused
  --> metaconnectivity/compute_fluidity.py:26:5
   |
24 |     intramodule_indices_mask,
25 |     get_fc_mc_indices,
26 |     get_mc_region_identities,
   |     ^^^^^^^^^^^^^^^^^^^^^^^^
27 |     fun_allegiance_communities,
28 |     compute_trimers_identity,
   |
help: Remove unused import

F401 [*] `shared_code.fun_metaconnectivity.fun_allegiance_communities` imported but unused
  --> metaconnectivity/compute_fluidity.py:27:5
   |
25 |     get_fc_mc_indices,
26 |     get_mc_region_identities,
27 |     fun_allegiance_communities,
   |     ^^^^^^^^^^^^^^^^^^^^^^^^^^
28 |     compute_trimers_identity,
29 |     build_trimer_mask,
   |
help: Remove unused import

F401 [*] `shared_code.fun_metaconnectivity.compute_trimers_identity` imported but unused
  --> metaconnectivity/compute_fluidity.py:28:5
   |
26 |     get_mc_region_identities,
27 |     fun_allegiance_communities,
28 |     compute_trimers_identity,
   |     ^^^^^^^^^^^^^^^^^^^^^^^^
29 |     build_trimer_mask,
30 | )
   |
help: Remove unused import

F401 [*] `shared_code.fun_metaconnectivity.build_trimer_mask` imported but unused
  --> metaconnectivity/compute_fluidity.py:29:5
   |
27 |     fun_allegiance_communities,
28 |     compute_trimers_identity,
29 |     build_trimer_mask,
   |     ^^^^^^^^^^^^^^^^^
30 | )
   |
help: Remove unused import

E402 Module level import not at top of file
  --> metaconnectivity/compute_fluidity.py:98:1
   |
98 | from scipy.spatial.distance import cdist
   | ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
99 | from scipy.stats import genpareto
   |

E402 Module level import not at top of file
  --> metaconnectivity/compute_fluidity.py:99:1
   |
98 | from scipy.spatial.distance import cdist
99 | from scipy.stats import genpareto
   | ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
   |

F405 `Parallel` may be undefined, or defined from star imports
   --> metaconnectivity/compute_fluidity.py:143:11
    |
141 | dimension = np.zeros((n_animals, len(ts[0])))
142 |
143 | results = Parallel(n_jobs=-1)(
    |           ^^^^^^^^
144 |     delayed(MA_EEG_Man_Dim_Flui)(ts[xx]) for xx in tqdm(range(n_animals))
145 | )
    |

F405 `delayed` may be undefined, or defined from star imports
   --> metaconnectivity/compute_fluidity.py:144:5
    |
143 | results = Parallel(n_jobs=-1)(
144 |     delayed(MA_EEG_Man_Dim_Flui)(ts[xx]) for xx in tqdm(range(n_animals))
    |     ^^^^^^^
145 | )
146 | for xx, (f, d) in enumerate(results):
    |

F405 `tqdm` may be undefined, or defined from star imports
   --> metaconnectivity/compute_fluidity.py:144:52
    |
143 | results = Parallel(n_jobs=-1)(
144 |     delayed(MA_EEG_Man_Dim_Flui)(ts[xx]) for xx in tqdm(range(n_animals))
    |                                                    ^^^^
145 | )
146 | for xx, (f, d) in enumerate(results):
    |

E402 Module level import not at top of file
   --> metaconnectivity/compute_fluidity.py:151:1
    |
150 | # %%
151 | import matplotlib.pyplot as plt
    | ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    |

I001 [*] Import block is un-sorted or un-formatted
   --> metaconnectivity/compute_fluidity.py:151:1
    |
150 | # %%
151 | import matplotlib.pyplot as plt
    | ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    |
help: Organize imports

B905 [*] `zip()` without an explicit `strict=` parameter
   --> metaconnectivity/compute_fluidity.py:197:20
    |
195 | qq_fluidity_group = {}
196 |
197 | for label, mask in zip(label_variables[0], mask_groups[0]):
    |                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
198 |     dimension_group[label] = dimension[mask]
199 |     fluidity_group[label] = fluidity[mask]
    |
help: Add explicit value for parameter `strict=`

UP009 [*] UTF-8 encoding declaration is unnecessary
 --> metaconnectivity/compute_metaconnectivity_modularity.py:2:1
  |
1 | #!/usr/bin/env python3
2 | # -*- coding: utf-8 -*-
  | ^^^^^^^^^^^^^^^^^^^^^^^
3 | """
4 | Created on Mon Sep 23 13:26:30 2024
  |
help: Remove unnecessary coding comment

I001 [*] Import block is un-sorted or un-formatted
  --> metaconnectivity/compute_metaconnectivity_modularity.py:10:1
   |
 9 |   # %%
10 | / import numpy as np
11 | | import time
12 | |
13 | | # from functions_analysis import *
14 | | from pathlib import Path
15 | |
16 | | from shared_code.fun_loaddata import *
17 | | from shared_code.fun_dfcspeed import *
18 | |
19 | | from shared_code.fun_metaconnectivity import *
20 | |
21 | | from shared_code.fun_utils import (
22 | |     set_figure_params,
23 | |     load_cognitive_data,
24 | |     load_timeseries_data,
25 | |     load_grouping_data,
26 | | )
27 | | from shared_code.fun_paths import get_paths
   | |___________________________________________^
28 |
29 |   # ===============================================================================
   |
help: Organize imports

F401 [*] `pathlib.Path` imported but unused
  --> metaconnectivity/compute_metaconnectivity_modularity.py:14:21
   |
13 | # from functions_analysis import *
14 | from pathlib import Path
   |                     ^^^^
15 |
16 | from shared_code.fun_loaddata import *
   |
help: Remove unused import: `pathlib.Path`

F403 `from shared_code.fun_loaddata import *` used; unable to detect undefined names
  --> metaconnectivity/compute_metaconnectivity_modularity.py:16:1
   |
14 | from pathlib import Path
15 |
16 | from shared_code.fun_loaddata import *
   | ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
17 | from shared_code.fun_dfcspeed import *
   |

F403 `from shared_code.fun_dfcspeed import *` used; unable to detect undefined names
  --> metaconnectivity/compute_metaconnectivity_modularity.py:17:1
   |
16 | from shared_code.fun_loaddata import *
17 | from shared_code.fun_dfcspeed import *
   | ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
18 |
19 | from shared_code.fun_metaconnectivity import *
   |

F403 `from shared_code.fun_metaconnectivity import *` used; unable to detect undefined names
  --> metaconnectivity/compute_metaconnectivity_modularity.py:19:1
   |
17 | from shared_code.fun_dfcspeed import *
18 |
19 | from shared_code.fun_metaconnectivity import *
   | ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
20 |
21 | from shared_code.fun_utils import (
   |

F405 `compute_metaconnectivity` may be undefined, or defined from star imports
  --> metaconnectivity/compute_metaconnectivity_modularity.py:84:6
   |
82 | # %%compute metaconnectivity
83 | start = time.time()
84 | mc = compute_metaconnectivity(
   |      ^^^^^^^^^^^^^^^^^^^^^^^^
85 |     ts,
86 |     window_size=window_size,
   |

F405 `fun_allegiance_communities` may be undefined, or defined from star imports
   --> metaconnectivity/compute_metaconnectivity_modularity.py:110:5
    |
108 | # %% Compute allegiance
109 | mc_ref_allegiance_communities, sort_allegiance, contingency_matrix = (
110 |     fun_allegiance_communities(
    |     ^^^^^^^^^^^^^^^^^^^^^^^^^^
111 |         mc_ref,
112 |         n_runs=n_runs_allegiance,
    |

F405 `intramodule_indices_mask` may be undefined, or defined from star imports
   --> metaconnectivity/compute_metaconnectivity_modularity.py:131:58
    |
129 | # ========================Modules==========================================
130 |
131 | intramodules_idx, intramodule_indices, mc_modules_mask = intramodule_indices_mask(
    |                                                          ^^^^^^^^^^^^^^^^^^^^^^^^
132 |     mc_ref_allegiance_communities
133 | )
    |

F405 `get_fc_mc_indices` may be undefined, or defined from star imports
   --> metaconnectivity/compute_metaconnectivity_modularity.py:137:18
    |
136 | # Build basic indices
137 | fc_idx, mc_idx = get_fc_mc_indices(regions, allegiance_sort=sort_allegiance)
    |                  ^^^^^^^^^^^^^^^^^
138 |
139 | # Get the indices of the regions in the functional connectivity matrix
    |

F405 `get_mc_region_identities` may be undefined, or defined from star imports
   --> metaconnectivity/compute_metaconnectivity_modularity.py:140:26
    |
139 | # Get the indices of the regions in the functional connectivity matrix
140 | mc_reg_idx, fc_reg_idx = get_mc_region_identities(fc_idx, mc_idx)  # , sort_allegiance)
    |                          ^^^^^^^^^^^^^^^^^^^^^^^^
141 |
142 | # Get the indices of the regions in the metaconnectivity matrix
    |

UP009 [*] UTF-8 encoding declaration is unnecessary
 --> metaconnectivity/compute_trimers.py:2:1
  |
1 | #!/usr/bin/env python3
2 | # -*- coding: utf-8 -*-
  | ^^^^^^^^^^^^^^^^^^^^^^^
3 | """
4 | Created on Mon Sep 23 13:26:30 2024
  |
help: Remove unnecessary coding comment

I001 [*] Import block is un-sorted or un-formatted
  --> metaconnectivity/compute_trimers.py:10:1
   |
 9 |   # %%
10 | / import numpy as np
11 | | import time
12 | |
13 | | # from functions_analysis import *
14 | | from pathlib import Path
15 | |
16 | | from fun_loaddata import *
17 | | from fun_dfcspeed import *
18 | |
19 | | from fun_metaconnectivity import (
20 | |     compute_mc_nplets_mask_and_index,
21 | |     compute_metaconnectivity,
22 | |     intramodule_indices_mask,
23 | |     get_fc_mc_indices,
24 | |     get_mc_region_identities,
25 | |     fun_allegiance_communities,
26 | |     compute_trimers_identity,
27 | |     build_trimer_mask,
28 | |     trimers_leaves_fc,
29 | |     trimers_root_fc,
30 | |     compute_mc_nplets_mask_and_index,
31 | | )
32 | |
33 | | from fun_utils import (
34 | |     set_figure_params,
35 | |     get_paths,
36 | |     load_cognitive_data,
37 | |     load_timeseries_data,
38 | |     load_grouping_data,
39 | | )
   | |_^
40 |
41 |   # =============================================================================
   |
help: Organize imports

F401 [*] `time` imported but unused
  --> metaconnectivity/compute_trimers.py:11:8
   |
 9 | # %%
10 | import numpy as np
11 | import time
   |        ^^^^
12 |
13 | # from functions_analysis import *
   |
help: Remove unused import: `time`

F403 `from fun_loaddata import *` used; unable to detect undefined names
  --> metaconnectivity/compute_trimers.py:16:1
   |
14 | from pathlib import Path
15 |
16 | from fun_loaddata import *
   | ^^^^^^^^^^^^^^^^^^^^^^^^^^
17 | from fun_dfcspeed import *
   |

F403 `from fun_dfcspeed import *` used; unable to detect undefined names
  --> metaconnectivity/compute_trimers.py:17:1
   |
16 | from fun_loaddata import *
17 | from fun_dfcspeed import *
   | ^^^^^^^^^^^^^^^^^^^^^^^^^^
18 |
19 | from fun_metaconnectivity import (
   |

F401 [*] `fun_metaconnectivity.compute_metaconnectivity` imported but unused
  --> metaconnectivity/compute_trimers.py:21:5
   |
19 | from fun_metaconnectivity import (
20 |     compute_mc_nplets_mask_and_index,
21 |     compute_metaconnectivity,
   |     ^^^^^^^^^^^^^^^^^^^^^^^^
22 |     intramodule_indices_mask,
23 |     get_fc_mc_indices,
   |
help: Remove unused import

F401 [*] `fun_metaconnectivity.intramodule_indices_mask` imported but unused
  --> metaconnectivity/compute_trimers.py:22:5
   |
20 |     compute_mc_nplets_mask_and_index,
21 |     compute_metaconnectivity,
22 |     intramodule_indices_mask,
   |     ^^^^^^^^^^^^^^^^^^^^^^^^
23 |     get_fc_mc_indices,
24 |     get_mc_region_identities,
   |
help: Remove unused import

F401 [*] `fun_metaconnectivity.get_fc_mc_indices` imported but unused
  --> metaconnectivity/compute_trimers.py:23:5
   |
21 |     compute_metaconnectivity,
22 |     intramodule_indices_mask,
23 |     get_fc_mc_indices,
   |     ^^^^^^^^^^^^^^^^^
24 |     get_mc_region_identities,
25 |     fun_allegiance_communities,
   |
help: Remove unused import

F401 [*] `fun_metaconnectivity.get_mc_region_identities` imported but unused
  --> metaconnectivity/compute_trimers.py:24:5
   |
22 |     intramodule_indices_mask,
23 |     get_fc_mc_indices,
24 |     get_mc_region_identities,
   |     ^^^^^^^^^^^^^^^^^^^^^^^^
25 |     fun_allegiance_communities,
26 |     compute_trimers_identity,
   |
help: Remove unused import

F401 [*] `fun_metaconnectivity.fun_allegiance_communities` imported but unused
  --> metaconnectivity/compute_trimers.py:25:5
   |
23 |     get_fc_mc_indices,
24 |     get_mc_region_identities,
25 |     fun_allegiance_communities,
   |     ^^^^^^^^^^^^^^^^^^^^^^^^^^
26 |     compute_trimers_identity,
27 |     build_trimer_mask,
   |
help: Remove unused import

F401 [*] `fun_metaconnectivity.compute_trimers_identity` imported but unused
  --> metaconnectivity/compute_trimers.py:26:5
   |
24 |     get_mc_region_identities,
25 |     fun_allegiance_communities,
26 |     compute_trimers_identity,
   |     ^^^^^^^^^^^^^^^^^^^^^^^^
27 |     build_trimer_mask,
28 |     trimers_leaves_fc,
   |
help: Remove unused import

F401 [*] `fun_metaconnectivity.build_trimer_mask` imported but unused
  --> metaconnectivity/compute_trimers.py:27:5
   |
25 |     fun_allegiance_communities,
26 |     compute_trimers_identity,
27 |     build_trimer_mask,
   |     ^^^^^^^^^^^^^^^^^
28 |     trimers_leaves_fc,
29 |     trimers_root_fc,
   |
help: Remove unused import

F811 [*] Redefinition of unused `compute_mc_nplets_mask_and_index` from line 20
  --> metaconnectivity/compute_trimers.py:30:5
   |
28 |     trimers_leaves_fc,
29 |     trimers_root_fc,
30 |     compute_mc_nplets_mask_and_index,
   |     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ `compute_mc_nplets_mask_and_index` redefined here
31 | )
   |
  ::: metaconnectivity/compute_trimers.py:20:5
   |
19 | from fun_metaconnectivity import (
20 |     compute_mc_nplets_mask_and_index,
   |     -------------------------------- previous definition of `compute_mc_nplets_mask_and_index` here
21 |     compute_metaconnectivity,
22 |     intramodule_indices_mask,
   |
help: Remove definition: `compute_mc_nplets_mask_and_index`

E712 Avoid equality comparisons to `True`; use `external_disk:` for truth checks
  --> metaconnectivity/compute_trimers.py:51:4
   |
49 | timeseries_folder = "Timecourses_updated_03052024"
50 | external_disk = True
51 | if external_disk == True:
   |    ^^^^^^^^^^^^^^^^^^^^^
52 |     root = Path("/media/samy/Elements1/Proyectos/LauraHarsan/script_mc/")
53 | else:
   |
help: Replace with `external_disk`

F405 `ts2fc` may be undefined, or defined from star imports
   --> metaconnectivity/compute_trimers.py:120:9
    |
118 | fc = np.array(
119 |     [
120 |         ts2fc(ts[animal], format_data="2D", method="pearson")
    |         ^^^^^
121 |         for animal in range(n_animals)
122 |     ]
    |

F405 `ts2dfc_stream` may be undefined, or defined from star imports
   --> metaconnectivity/compute_trimers.py:130:9
    |
128 | dfc_stream = np.array(
129 |     [
130 |         ts2dfc_stream(
    |         ^^^^^^^^^^^^^
131 |             ts[animal], window_size, lag=lag, format_data="3D", method="pearson"
132 |         )
    |

E402 Module level import not at top of file
   --> metaconnectivity/compute_trimers.py:247:1
    |
246 | # %%
247 | from matplotlib import pyplot as plt
    | ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
248 |
249 | label_fc_root_fc_leaves = r"$min(FC_{i,r}, FC_{j,r}) > FC_{i,j}$"
    |

F841 Local variable `mc` is assigned to but never used
  --> metaconnectivity/deprecated_fun.py:53:5
   |
51 |     n_animals, tr_points, nodes = ts_data.shape
52 |     dfc_stream = None
53 |     mc = None
   |     ^^
54 |     dfc_stream_loaded = False  # <- initialize this early
   |
help: Remove assignment to unused variable `mc`

F821 Undefined name `Path`
  --> metaconnectivity/deprecated_fun.py:57:17
   |
56 |     # File path setup
57 |     save_path = Path(save_path) if save_path else None
   |                 ^^^^
58 |     file_path = (
59 |         save_path
   |

F821 Undefined name `np`
  --> metaconnectivity/deprecated_fun.py:73:20
   |
71 |         try:
72 |             print(f"Loading dFC stream from: {file_path}")
73 |             data = np.load(file_path, allow_pickle=True)
   |                    ^^
74 |             dfc_stream = data["dfc_stream"]
75 |             dfc_stream_loaded = True
   |

F821 Undefined name `parallel_backend`
  --> metaconnectivity/deprecated_fun.py:85:14
   |
84 |         # Parallel DFC stream computation per animal
85 |         with parallel_backend("loky", n_jobs=n_jobs):
   |              ^^^^^^^^^^^^^^^^
86 |             dfc_stream_list = Parallel()(
87 |                 delayed(ts2dfc_stream)(
   |

F821 Undefined name `Parallel`
  --> metaconnectivity/deprecated_fun.py:86:31
   |
84 |         # Parallel DFC stream computation per animal
85 |         with parallel_backend("loky", n_jobs=n_jobs):
86 |             dfc_stream_list = Parallel()(
   |                               ^^^^^^^^
87 |                 delayed(ts2dfc_stream)(
88 |                     ts_data[i], window_size, lag, format_data=format_data
   |

F821 Undefined name `delayed`
  --> metaconnectivity/deprecated_fun.py:87:17
   |
85 |         with parallel_backend("loky", n_jobs=n_jobs):
86 |             dfc_stream_list = Parallel()(
87 |                 delayed(ts2dfc_stream)(
   |                 ^^^^^^^
88 |                     ts_data[i], window_size, lag, format_data=format_data
89 |                 )
   |

F821 Undefined name `ts2dfc_stream`
  --> metaconnectivity/deprecated_fun.py:87:25
   |
85 |         with parallel_backend("loky", n_jobs=n_jobs):
86 |             dfc_stream_list = Parallel()(
87 |                 delayed(ts2dfc_stream)(
   |                         ^^^^^^^^^^^^^
88 |                     ts_data[i], window_size, lag, format_data=format_data
89 |                 )
   |

F821 Undefined name `np`
  --> metaconnectivity/deprecated_fun.py:93:22
   |
91 |                 for i in range(n_animals)
92 |             )
93 |         dfc_stream = np.stack(dfc_stream_list)
   |                      ^^
94 |
95 |     # Save results if path is provided
   |

F821 Undefined name `np`
  --> metaconnectivity/deprecated_fun.py:98:9
   |
96 |     if file_path:
97 |         print(f"Saving dFC stream to: {file_path}")
98 |         np.savez_compressed(file_path, dfc_stream=dfc_stream)
   |         ^^
99 |     return dfc_stream
   |

F821 Undefined name `copy`
   --> metaconnectivity/deprecated_fun.py:135:12
    |
133 |     n_trials = data.shape[0]
134 |
135 |     data = copy.deepcopy(data)
    |            ^^^^
136 |     mc_viscocity_mask = data < 0
137 |     mc_viscocity_val = np.array(
    |

F821 Undefined name `np`
   --> metaconnectivity/deprecated_fun.py:137:24
    |
135 |     data = copy.deepcopy(data)
136 |     mc_viscocity_mask = data < 0
137 |     mc_viscocity_val = np.array(
    |                        ^^
138 |         [data[i, mc_viscocity_mask[i]] for i in range(n_trials)], dtype="object"
139 |     )
    |

F821 Undefined name `np`
   --> metaconnectivity/deprecated_fun.py:158:17
    |
157 |     n_nodes = mc_data.shape[0]
158 |     gamma_mod = np.linspace(gmin, gmax, gamma_range)
    |                 ^^
159 |
160 |     if cache_path:
    |

F821 Undefined name `Path`
   --> metaconnectivity/deprecated_fun.py:161:21
    |
160 |     if cache_path:
161 |         cache_dir = Path(cache_path)
    |                     ^^^^
162 |         cache_dir.mkdir(parents=True, exist_ok=True)
163 |         full_cache_path = (
    |

F821 Undefined name `pickle`
   --> metaconnectivity/deprecated_fun.py:170:24
    |
168 |             with full_cache_path.open("rb") as f:
169 |                 print(f"[cache] Loading contingency matrix from {full_cache_path}")
170 |                 return pickle.load(f)
    |                        ^^^^^^
171 |     else:
172 |         full_cache_path = None
    |

F821 Undefined name `np`
   --> metaconnectivity/deprecated_fun.py:174:26
    |
172 |         full_cache_path = None
173 |
174 |     contingency_matrix = np.zeros((n_nodes, n_nodes), dtype=np.float64)
    |                          ^^
175 |     gamma_qmod_val = np.zeros((gamma_range, n_runs), dtype=np.float64)
176 |     gamma_agreement_mat = np.zeros((gamma_range, n_nodes, n_nodes), dtype=np.float64)
    |

F821 Undefined name `np`
   --> metaconnectivity/deprecated_fun.py:174:61
    |
172 |         full_cache_path = None
173 |
174 |     contingency_matrix = np.zeros((n_nodes, n_nodes), dtype=np.float64)
    |                                                             ^^
175 |     gamma_qmod_val = np.zeros((gamma_range, n_runs), dtype=np.float64)
176 |     gamma_agreement_mat = np.zeros((gamma_range, n_nodes, n_nodes), dtype=np.float64)
    |

F821 Undefined name `np`
   --> metaconnectivity/deprecated_fun.py:175:22
    |
174 |     contingency_matrix = np.zeros((n_nodes, n_nodes), dtype=np.float64)
175 |     gamma_qmod_val = np.zeros((gamma_range, n_runs), dtype=np.float64)
    |                      ^^
176 |     gamma_agreement_mat = np.zeros((gamma_range, n_nodes, n_nodes), dtype=np.float64)
    |

F821 Undefined name `np`
   --> metaconnectivity/deprecated_fun.py:175:60
    |
174 |     contingency_matrix = np.zeros((n_nodes, n_nodes), dtype=np.float64)
175 |     gamma_qmod_val = np.zeros((gamma_range, n_runs), dtype=np.float64)
    |                                                            ^^
176 |     gamma_agreement_mat = np.zeros((gamma_range, n_nodes, n_nodes), dtype=np.float64)
    |

F821 Undefined name `np`
   --> metaconnectivity/deprecated_fun.py:176:27
    |
174 |     contingency_matrix = np.zeros((n_nodes, n_nodes), dtype=np.float64)
175 |     gamma_qmod_val = np.zeros((gamma_range, n_runs), dtype=np.float64)
176 |     gamma_agreement_mat = np.zeros((gamma_range, n_nodes, n_nodes), dtype=np.float64)
    |                           ^^
177 |
178 |     for idx, gamma in enumerate(tqdm(gamma_mod, desc="Gamma values")):
    |

F821 Undefined name `np`
   --> metaconnectivity/deprecated_fun.py:176:75
    |
174 |     contingency_matrix = np.zeros((n_nodes, n_nodes), dtype=np.float64)
175 |     gamma_qmod_val = np.zeros((gamma_range, n_runs), dtype=np.float64)
176 |     gamma_agreement_mat = np.zeros((gamma_range, n_nodes, n_nodes), dtype=np.float64)
    |                                                                           ^^
177 |
178 |     for idx, gamma in enumerate(tqdm(gamma_mod, desc="Gamma values")):
    |

F821 Undefined name `tqdm`
   --> metaconnectivity/deprecated_fun.py:178:33
    |
176 |     gamma_agreement_mat = np.zeros((gamma_range, n_nodes, n_nodes), dtype=np.float64)
177 |
178 |     for idx, gamma in enumerate(tqdm(gamma_mod, desc="Gamma values")):
    |                                 ^^^^
179 |         # Louvain with per-run progress bar
180 |         results = list(
    |

F821 Undefined name `tqdm`
   --> metaconnectivity/deprecated_fun.py:181:13
    |
179 |         # Louvain with per-run progress bar
180 |         results = list(
181 |             tqdm(
    |             ^^^^
182 |                 Parallel(n_jobs=n_jobs)(
183 |                     delayed(_run_louvain)(mc_data, gamma) for _ in range(n_runs)
    |

F821 Undefined name `Parallel`
   --> metaconnectivity/deprecated_fun.py:182:17
    |
180 |         results = list(
181 |             tqdm(
182 |                 Parallel(n_jobs=n_jobs)(
    |                 ^^^^^^^^
183 |                     delayed(_run_louvain)(mc_data, gamma) for _ in range(n_runs)
184 |                 ),
    |

F821 Undefined name `delayed`
   --> metaconnectivity/deprecated_fun.py:183:21
    |
181 |             tqdm(
182 |                 Parallel(n_jobs=n_jobs)(
183 |                     delayed(_run_louvain)(mc_data, gamma) for _ in range(n_runs)
    |                     ^^^^^^^
184 |                 ),
185 |                 total=n_runs,
    |

F821 Undefined name `_run_louvain`
   --> metaconnectivity/deprecated_fun.py:183:29
    |
181 |             tqdm(
182 |                 Parallel(n_jobs=n_jobs)(
183 |                     delayed(_run_louvain)(mc_data, gamma) for _ in range(n_runs)
    |                             ^^^^^^^^^^^^
184 |                 ),
185 |                 total=n_runs,
    |

B905 [*] `zip()` without an explicit `strict=` parameter
   --> metaconnectivity/deprecated_fun.py:190:37
    |
188 |         )
189 |
190 |         communities, modularities = zip(*results)
    |                                     ^^^^^^^^^^^^^
191 |         communities = np.array([np.array(c) for c in communities])
192 |         gamma_qmod_val[idx] = modularities
    |
help: Add explicit value for parameter `strict=`

F821 Undefined name `np`
   --> metaconnectivity/deprecated_fun.py:191:23
    |
190 |         communities, modularities = zip(*results)
191 |         communities = np.array([np.array(c) for c in communities])
    |                       ^^
192 |         gamma_qmod_val[idx] = modularities
    |

F821 Undefined name `np`
   --> metaconnectivity/deprecated_fun.py:191:33
    |
190 |         communities, modularities = zip(*results)
191 |         communities = np.array([np.array(c) for c in communities])
    |                                 ^^
192 |         gamma_qmod_val[idx] = modularities
    |

F821 Undefined name `build_agreement_matrix_vectorized`
   --> metaconnectivity/deprecated_fun.py:195:21
    |
194 |         # Efficient agreement accumulation
195 |         agreement = build_agreement_matrix_vectorized(communities)
    |                     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
196 |         gamma_agreement_mat[idx] = agreement
    |

F821 Undefined name `pickle`
   --> metaconnectivity/deprecated_fun.py:205:13
    |
203 |     if full_cache_path is not None:
204 |         with full_cache_path.open("wb") as f:
205 |             pickle.dump((contingency_matrix, gamma_qmod_val, gamma_agreement_mat), f)
    |             ^^^^^^
206 |             print(f"[cache] Saved to {full_cache_path}")
    |

F821 Undefined name `allegiance_matrix_analysis`
   --> metaconnectivity/deprecated_fun.py:243:49
    |
241 |     ):  # , n_runs = 10, gamma_pt = 10, ref_name='', save_path=None, n_jobs=-1): # gamma number of points in the defined range
242 |         # allegiance index, argsort, Q value
243 |         communities, sort_idx, _, contingency = allegiance_matrix_analysis(
    |                                                 ^^^^^^^^^^^^^^^^^^^^^^^^^^
244 |             mc_matrix,
245 |             n_runs=n_runs,
    |

F821 Undefined name `np`
   --> metaconnectivity/deprecated_fun.py:259:27
    |
257 |             allegiance, _, _ = process_single(mc_data[i])
258 |             allegiances.append(allegiance)
259 |         mean_allegiance = np.mean(allegiances, axis=0)
    |                           ^^
260 |         communities, sort_idx, contingency = process_single(mean_allegiance)
261 |     elif mc_data.ndim == 2:
    |

F821 Undefined name `np`
   --> metaconnectivity/deprecated_fun.py:269:9
    |
268 |     if save_path and ref_name:
269 |         np.savez_compressed(
    |         ^^
270 |             Path(save_path) / f"allegiance_{ref_name}.npz",
271 |             communities=communities,
    |

F821 Undefined name `Path`
   --> metaconnectivity/deprecated_fun.py:270:13
    |
268 |     if save_path and ref_name:
269 |         np.savez_compressed(
270 |             Path(save_path) / f"allegiance_{ref_name}.npz",
    |             ^^^^
271 |             communities=communities,
272 |             sort_idx=sort_idx,
    |

UP009 [*] UTF-8 encoding declaration is unnecessary
 --> metaconnectivity/fun_dfcspeed.py:2:1
  |
1 | #!/usr/bin/env python3
2 | # -*- coding: utf-8 -*-
  | ^^^^^^^^^^^^^^^^^^^^^^^
3 | """
4 | Created on Fri Mar  8 15:45:43 2024
  |
help: Remove unnecessary coding comment

I001 [*] Import block is un-sorted or un-formatted
  --> metaconnectivity/fun_dfcspeed.py:22:1
   |
20 |   # =============================================================================
21 |
22 | / import numpy as np
23 | | import brainconn as bct
24 | | from tqdm import tqdm
25 | | import numexpr as ne
26 | | from joblib import Parallel, delayed
27 | | from fun_optimization import fast_corrcoef, fast_corrcoef_numba
   | |_______________________________________________________________^
28 |
29 |   # =============================================================================
   |
help: Organize imports

F401 [*] `fun_optimization.fast_corrcoef_numba` imported but unused
  --> metaconnectivity/fun_dfcspeed.py:27:45
   |
25 | import numexpr as ne
26 | from joblib import Parallel, delayed
27 | from fun_optimization import fast_corrcoef, fast_corrcoef_numba
   |                                             ^^^^^^^^^^^^^^^^^^^
28 |
29 | # =============================================================================
   |
help: Remove unused import: `fun_optimization.fast_corrcoef_numba`

F841 Local variable `num_channels` is assigned to but never used
  --> metaconnectivity/fun_dfcspeed.py:47:5
   |
45 |         A 2D array of shape (channels, channels) representing PLV between each pair of channels.
46 |     """
47 |     num_channels = data.shape[0]
   |     ^^^^^^^^^^^^
48 |
49 |     # Compute the phase for each channel
   |
help: Remove assignment to unused variable `num_channels`

F841 Local variable `phase_diff` is assigned to but never used
  --> metaconnectivity/fun_dfcspeed.py:54:5
   |
52 |     # Compute pairwise phase differences for all channels at once using broadcasting
53 |     # The result is an array of shape (channels, channels, timepoints)
54 |     phase_diff = phase_data[:, np.newaxis, :] - phase_data[np.newaxis, :, :]
   |     ^^^^^^^^^^
55 |
56 |     # Compute the complex exponential of the phase differences for all pairs
   |
help: Remove assignment to unused variable `phase_diff`

E712 Avoid equality comparisons to `True`; use `min_tau_zero:` for truth checks
   --> metaconnectivity/fun_dfcspeed.py:301:8
    |
299 |     """
300 |
301 |     if min_tau_zero == True:
    |        ^^^^^^^^^^^^^^^^^^^^
302 |         min_tau = 0
303 |     else:
    |
help: Replace with `min_tau_zero`

E712 Avoid equality comparisons to `True`; use `get_speed_dist:` for truth checks
   --> metaconnectivity/fun_dfcspeed.py:330:12
    |
328 |         speed_windows_tau[idx_tt] = np.median(speed_oversampl, axis=1)
329 |
330 |         if get_speed_dist == True:  # speed_dist = np.mean(speed_oversampl,axis=1)
    |            ^^^^^^^^^^^^^^^^^^^^^^
331 |             speed_dist.append(speed_oversampl.flatten())
    |
help: Replace with `get_speed_dist`

E712 Avoid equality comparisons to `True`; use `get_speed_dist:` for truth checks
   --> metaconnectivity/fun_dfcspeed.py:333:8
    |
331 |             speed_dist.append(speed_oversampl.flatten())
332 |
333 |     if get_speed_dist == True:  # speed_dist = np.mean(speed_oversampl,axis=1)
    |        ^^^^^^^^^^^^^^^^^^^^^^
334 |         return speed_windows_tau, speed_dist
335 |     else:
    |
help: Replace with `get_speed_dist`

E712 Avoid equality comparisons to `True`; use `min_tau_zero:` for truth checks
   --> metaconnectivity/fun_dfcspeed.py:386:8
    |
384 |     Samy Castro 2024
385 |     """
386 |     if min_tau_zero == True:
    |        ^^^^^^^^^^^^^^^^^^^^
387 |         min_tau = 0
388 |     else:
    |
help: Replace with `min_tau_zero`

B905 [*] `zip()` without an explicit `strict=` parameter
   --> metaconnectivity/fun_dfcspeed.py:412:9
    |
410 |     )
411 |     speed_windows_tau, speed_dist = (
412 |         zip(*results) if get_speed_dist else (zip(*results), None)
    |         ^^^^^^^^^^^^^
413 |     )
    |
help: Add explicit value for parameter `strict=`

B905 [*] `zip()` without an explicit `strict=` parameter
   --> metaconnectivity/fun_dfcspeed.py:412:47
    |
410 |     )
411 |     speed_windows_tau, speed_dist = (
412 |         zip(*results) if get_speed_dist else (zip(*results), None)
    |                                               ^^^^^^^^^^^^^
413 |     )
    |
help: Add explicit value for parameter `strict=`

E712 Avoid equality comparisons to `True`; use `filter_listed:` for truth checks
   --> metaconnectivity/fun_dfcspeed.py:428:28
    |
426 |     long_vel_list = []
427 |
428 |     filter_list = np.where(filter_listed == True)[0]
    |                            ^^^^^^^^^^^^^^^^^^^^^
429 |
430 |     # for tt in range(29):
    |
help: Replace with `filter_listed`

UP009 [*] UTF-8 encoding declaration is unnecessary
 --> metaconnectivity/fun_loaddata.py:2:1
  |
1 | #!/usr/bin/env python3
2 | # -*- coding: utf-8 -*-
  | ^^^^^^^^^^^^^^^^^^^^^^^
3 | """
4 | Created on Fri Mar  8 15:56:50 2024
  |
help: Remove unnecessary coding comment

UP009 [*] UTF-8 encoding declaration is unnecessary
 --> metaconnectivity/fun_metaconnectivity.py:2:1
  |
1 | #!/usr/bin/env python3
2 | # -*- coding: utf-8 -*-
  | ^^^^^^^^^^^^^^^^^^^^^^^
3 | """
4 | Created on Wed Mar 26 00:16:53 2025
  |
help: Remove unnecessary coding comment

I001 [*] Import block is un-sorted or un-formatted
  --> metaconnectivity/fun_metaconnectivity.py:9:1
   |
 7 |   """
 8 |
 9 | / import joblib
10 | | import numpy as np
11 | | import matplotlib.pyplot as plt
12 | | import brainconn as bct
13 | | import os
14 | | import pandas as pd
15 | | from pathlib import Path
16 | | import copy
17 | | import pickle
18 | | from tqdm import tqdm
19 | |
20 | | from itertools import combinations_with_replacement
21 | | from joblib import Parallel, delayed, parallel_backend
22 | |
23 | | from fun_dfcspeed import ts2dfc_stream
24 | | from fun_loaddata import *
25 | | from fun_optimization import (
26 | |     fast_corrcoef,
27 | | )  # , fast_corrcoef_numba, fast_corrcoef_numba_parallel
   | |_^
28 |
29 |   # import time
   |
help: Organize imports

F401 [*] `matplotlib.pyplot` imported but unused
  --> metaconnectivity/fun_metaconnectivity.py:11:29
   |
 9 | import joblib
10 | import numpy as np
11 | import matplotlib.pyplot as plt
   |                             ^^^
12 | import brainconn as bct
13 | import os
   |
help: Remove unused import: `matplotlib.pyplot`

F401 [*] `os` imported but unused
  --> metaconnectivity/fun_metaconnectivity.py:13:8
   |
11 | import matplotlib.pyplot as plt
12 | import brainconn as bct
13 | import os
   |        ^^
14 | import pandas as pd
15 | from pathlib import Path
   |
help: Remove unused import: `os`

F401 [*] `pandas` imported but unused
  --> metaconnectivity/fun_metaconnectivity.py:14:18
   |
12 | import brainconn as bct
13 | import os
14 | import pandas as pd
   |                  ^^
15 | from pathlib import Path
16 | import copy
   |
help: Remove unused import: `pandas`

F401 [*] `copy` imported but unused
  --> metaconnectivity/fun_metaconnectivity.py:16:8
   |
14 | import pandas as pd
15 | from pathlib import Path
16 | import copy
   |        ^^^^
17 | import pickle
18 | from tqdm import tqdm
   |
help: Remove unused import: `copy`

F403 `from fun_loaddata import *` used; unable to detect undefined names
  --> metaconnectivity/fun_metaconnectivity.py:24:1
   |
23 | from fun_dfcspeed import ts2dfc_stream
24 | from fun_loaddata import *
   | ^^^^^^^^^^^^^^^^^^^^^^^^^^
25 | from fun_optimization import (
26 |     fast_corrcoef,
   |

B905 [*] `zip()` without an explicit `strict=` parameter
   --> metaconnectivity/fun_metaconnectivity.py:258:31
    |
256 |     # Reshape into [gamma_index][runs]
257 |     results_by_gamma = [[] for _ in range(gamma_range)]
258 |     for (gamma, _), result in zip(job_list, all_results):
    |                               ^^^^^^^^^^^^^^^^^^^^^^^^^^
259 |         gamma_idx = np.argmin(np.abs(gamma_mod - gamma))  # match gamma to index
260 |         results_by_gamma[gamma_idx].append(result)
    |
help: Add explicit value for parameter `strict=`

B007 Loop control variable `gamma` not used within loop body
   --> metaconnectivity/fun_metaconnectivity.py:268:14
    |
267 |     # Process per gamma
268 |     for idx, gamma in enumerate(tqdm(gamma_mod, desc="Processing gammas")):
    |              ^^^^^
269 |         results = results_by_gamma[idx]
270 |         communities, modularities = zip(*results)
    |
help: Rename unused `gamma` to `_gamma`

B905 [*] `zip()` without an explicit `strict=` parameter
   --> metaconnectivity/fun_metaconnectivity.py:270:37
    |
268 |     for idx, gamma in enumerate(tqdm(gamma_mod, desc="Processing gammas")):
269 |         results = results_by_gamma[idx]
270 |         communities, modularities = zip(*results)
    |                                     ^^^^^^^^^^^^^
271 |         communities = np.array(communities, dtype=np.int32)
272 |         gamma_qmod_val[idx] = modularities
    |
help: Add explicit value for parameter `strict=`

B905 [*] `zip()` without an explicit `strict=` parameter
   --> metaconnectivity/fun_metaconnectivity.py:449:61
    |
447 |       if mc_data.ndim == 3:
448 |           # Process multiple MC matrices
449 |           communities_list, sort_idx_list, contingency_list = zip(
    |  _____________________________________________________________^
450 | |             *(process_single(mc_data[i]) for i in range(mc_data.shape[0]))
451 | |         )
    | |_________^
452 |           communities = np.mean(communities_list, axis=0)
453 |           sort_idx = np.argsort(communities)
    |
help: Add explicit value for parameter `strict=`

F841 Local variable `repeated` is assigned to but never used
   --> metaconnectivity/fun_metaconnectivity.py:835:5
    |
833 |     unique, counts = np.unique(flat, return_counts=True)
834 |     non_repeated = unique[counts == 1]
835 |     repeated = unique[counts == 2]
    |     ^^^^^^^^
836 |     return non_repeated
    |
help: Remove assignment to unused variable `repeated`

UP009 [*] UTF-8 encoding declaration is unnecessary
 --> metaconnectivity/fun_optimization.py:2:1
  |
1 | #!/usr/bin/env python3
2 | # -*- coding: utf-8 -*-
  | ^^^^^^^^^^^^^^^^^^^^^^^
3 | """
4 | Created on Thu Apr  3 12:47:31 2025
  |
help: Remove unnecessary coding comment

I001 [*] Import block is un-sorted or un-formatted
  --> metaconnectivity/fun_optimization.py:9:1
   |
 7 |   """
 8 |
 9 | / import numpy as np
10 | | from scipy.stats import zscore
11 | | from numba import njit, prange
   | |______________________________^
12 |
13 |   # =============================================================================
   |
help: Organize imports

F401 [*] `scipy.stats.zscore` imported but unused
  --> metaconnectivity/fun_optimization.py:10:25
   |
 9 | import numpy as np
10 | from scipy.stats import zscore
   |                         ^^^^^^
11 | from numba import njit, prange
   |
help: Remove unused import: `scipy.stats.zscore`

UP009 [*] UTF-8 encoding declaration is unnecessary
 --> metaconnectivity/fun_utils.py:2:1
  |
1 | #!/usr/bin/env python3
2 | # -*- coding: utf-8 -*-
  | ^^^^^^^^^^^^^^^^^^^^^^^
3 | """
4 | Created on Sat Apr  5 00:18:49 2025
  |
help: Remove unnecessary coding comment

I001 [*] Import block is un-sorted or un-formatted
  --> metaconnectivity/fun_utils.py:9:1
   |
 7 |   """
 8 |
 9 | / import numpy as np
10 | | import os
11 | | from pathlib import Path
12 | | from scipy.io import loadmat
13 | | import pandas as pd
14 | | import matplotlib.pyplot as plt
15 | | import pickle
16 | | import logging
17 | | from typing import Any, Union
   | |_____________________________^
18 |
19 |   # =============================================================================
   |
help: Organize imports

F401 [*] `logging` imported but unused
  --> metaconnectivity/fun_utils.py:16:8
   |
14 | import matplotlib.pyplot as plt
15 | import pickle
16 | import logging
   |        ^^^^^^^
17 | from typing import Any, Union
   |
help: Remove unused import: `logging`

F401 [*] `typing.Any` imported but unused
  --> metaconnectivity/fun_utils.py:17:20
   |
15 | import pickle
16 | import logging
17 | from typing import Any, Union
   |                    ^^^
18 |
19 | # =============================================================================
   |
help: Remove unused import

F401 [*] `typing.Union` imported but unused
  --> metaconnectivity/fun_utils.py:17:25
   |
15 | import pickle
16 | import logging
17 | from typing import Any, Union
   |                         ^^^^^
18 |
19 | # =============================================================================
   |
help: Remove unused import

E712 Avoid equality comparisons to `True`; use `savefig:` for truth checks
  --> metaconnectivity/fun_utils.py:33:8
   |
31 |         }
32 |     )
33 |     if savefig == True:
   |        ^^^^^^^^^^^^^^^
34 |         return savefig
   |
help: Replace with `savefig`

B007 Loop control variable `idx` not used within loop body
   --> metaconnectivity/fun_utils.py:163:9
    |
161 |     hash_dir = os.path.join(folder_data, specific_folder)
162 |
163 |     for idx, file_name in enumerate(files_name):
    |         ^^^
164 |         file_path = os.path.join(hash_dir, file_name)
    |
help: Rename unused `idx` to `_idx`

B905 [*] `zip()` without an explicit `strict=` parameter
   --> metaconnectivity/fun_utils.py:208:28
    |
206 |     labels = []
207 |
208 |     for g_mask, g_label in zip(group_masks, group_labels):
    |                            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
209 |         for is_2m, age_label in zip([True, False], age_labels):
210 |             cond_mask = np.logical_and(g_mask, age_mask == is_2m)
    |
help: Add explicit value for parameter `strict=`

B905 [*] `zip()` without an explicit `strict=` parameter
   --> metaconnectivity/fun_utils.py:209:33
    |
208 |     for g_mask, g_label in zip(group_masks, group_labels):
209 |         for is_2m, age_label in zip([True, False], age_labels):
    |                                 ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
210 |             cond_mask = np.logical_and(g_mask, age_mask == is_2m)
211 |             masks.append(cond_mask)
    |
help: Add explicit value for parameter `strict=`

F541 [*] f-string without any placeholders
   --> metaconnectivity/fun_utils.py:239:10
    |
237 |     labels = np.select(
238 |         [good, learners, impaired, bad],
239 |         [f"good", f"learners", f"impaired", f"bad"],
    |          ^^^^^^^
240 |         default=f"undefined",
241 |     )
    |
help: Remove extraneous `f` prefix

F541 [*] f-string without any placeholders
   --> metaconnectivity/fun_utils.py:239:19
    |
237 |     labels = np.select(
238 |         [good, learners, impaired, bad],
239 |         [f"good", f"learners", f"impaired", f"bad"],
    |                   ^^^^^^^^^^^
240 |         default=f"undefined",
241 |     )
    |
help: Remove extraneous `f` prefix

F541 [*] f-string without any placeholders
   --> metaconnectivity/fun_utils.py:239:32
    |
237 |     labels = np.select(
238 |         [good, learners, impaired, bad],
239 |         [f"good", f"learners", f"impaired", f"bad"],
    |                                ^^^^^^^^^^^
240 |         default=f"undefined",
241 |     )
    |
help: Remove extraneous `f` prefix

F541 [*] f-string without any placeholders
   --> metaconnectivity/fun_utils.py:239:45
    |
237 |     labels = np.select(
238 |         [good, learners, impaired, bad],
239 |         [f"good", f"learners", f"impaired", f"bad"],
    |                                             ^^^^^^
240 |         default=f"undefined",
241 |     )
    |
help: Remove extraneous `f` prefix

F541 [*] f-string without any placeholders
   --> metaconnectivity/fun_utils.py:240:17
    |
238 |         [good, learners, impaired, bad],
239 |         [f"good", f"learners", f"impaired", f"bad"],
240 |         default=f"undefined",
    |                 ^^^^^^^^^^^^
241 |     )
    |
help: Remove extraneous `f` prefix

F541 [*] f-string without any placeholders
   --> metaconnectivity/fun_utils.py:246:29
    |
244 |     df = df.copy()
245 |     df[phenotype_column] = pd.Categorical(
246 |         labels, categories=[f"good", f"learners", f"impaired", f"bad"], ordered=False
    |                             ^^^^^^^
247 |     )
    |
help: Remove extraneous `f` prefix

F541 [*] f-string without any placeholders
   --> metaconnectivity/fun_utils.py:246:38
    |
244 |     df = df.copy()
245 |     df[phenotype_column] = pd.Categorical(
246 |         labels, categories=[f"good", f"learners", f"impaired", f"bad"], ordered=False
    |                                      ^^^^^^^^^^^
247 |     )
    |
help: Remove extraneous `f` prefix

F541 [*] f-string without any placeholders
   --> metaconnectivity/fun_utils.py:246:51
    |
244 |     df = df.copy()
245 |     df[phenotype_column] = pd.Categorical(
246 |         labels, categories=[f"good", f"learners", f"impaired", f"bad"], ordered=False
    |                                                   ^^^^^^^^^^^
247 |     )
    |
help: Remove extraneous `f` prefix

F541 [*] f-string without any placeholders
   --> metaconnectivity/fun_utils.py:246:64
    |
244 |     df = df.copy()
245 |     df[phenotype_column] = pd.Categorical(
246 |         labels, categories=[f"good", f"learners", f"impaired", f"bad"], ordered=False
    |                                                                ^^^^^^
247 |     )
    |
help: Remove extraneous `f` prefix

UP009 [*] UTF-8 encoding declaration is unnecessary
 --> metaconnectivity/master_mc.py:2:1
  |
1 | #!/usr/bin/env python3
2 | # -*- coding: utf-8 -*-
  | ^^^^^^^^^^^^^^^^^^^^^^^
3 | """
4 | Created on Tue Apr  8 23:13:49 2025
  |
help: Remove unnecessary coding comment

I001 [*] Import block is un-sorted or un-formatted
  --> metaconnectivity/master_mc.py:9:1
   |
 7 |   """
 8 |   # %%
 9 | / import numpy as np
10 | | import os
11 | | import pandas as pd
12 | | import pickle
13 | | from pathlib import Path
14 | | from shared_code.fun_utils import (
15 | |     filename_sort_mat,
16 | |     split_groups_by_age,
17 | |     load_matdata,
18 | |     classify_phenotypes,
19 | |     set_figure_params,
20 | |     #    load_cogdata_sorted,
21 | |     get_paths,
22 | | )
23 | | import matplotlib.pyplot as plt
24 | | import time
   | |___________^
25 |
26 |   # %% Figure parameters
   |
help: Organize imports

F401 [*] `os` imported but unused
  --> metaconnectivity/master_mc.py:10:8
   |
 8 | # %%
 9 | import numpy as np
10 | import os
   |        ^^
11 | import pandas as pd
12 | import pickle
   |
help: Remove unused import: `os`

F401 [*] `shared_code.fun_utils.filename_sort_mat` imported but unused
  --> metaconnectivity/master_mc.py:15:5
   |
13 | from pathlib import Path
14 | from shared_code.fun_utils import (
15 |     filename_sort_mat,
   |     ^^^^^^^^^^^^^^^^^
16 |     split_groups_by_age,
17 |     load_matdata,
   |
help: Remove unused import

F401 [*] `shared_code.fun_utils.split_groups_by_age` imported but unused
  --> metaconnectivity/master_mc.py:16:5
   |
14 | from shared_code.fun_utils import (
15 |     filename_sort_mat,
16 |     split_groups_by_age,
   |     ^^^^^^^^^^^^^^^^^^^
17 |     load_matdata,
18 |     classify_phenotypes,
   |
help: Remove unused import

F401 [*] `shared_code.fun_utils.load_matdata` imported but unused
  --> metaconnectivity/master_mc.py:17:5
   |
15 |     filename_sort_mat,
16 |     split_groups_by_age,
17 |     load_matdata,
   |     ^^^^^^^^^^^^
18 |     classify_phenotypes,
19 |     set_figure_params,
   |
help: Remove unused import

F401 [*] `shared_code.fun_utils.classify_phenotypes` imported but unused
  --> metaconnectivity/master_mc.py:18:5
   |
16 |     split_groups_by_age,
17 |     load_matdata,
18 |     classify_phenotypes,
   |     ^^^^^^^^^^^^^^^^^^^
19 |     set_figure_params,
20 |     #    load_cogdata_sorted,
   |
help: Remove unused import

F401 [*] `matplotlib.pyplot` imported but unused
  --> metaconnectivity/master_mc.py:23:29
   |
21 |     get_paths,
22 | )
23 | import matplotlib.pyplot as plt
   |                             ^^^
24 | import time
   |
help: Remove unused import: `matplotlib.pyplot`

F401 [*] `time` imported but unused
  --> metaconnectivity/master_mc.py:24:8
   |
22 | )
23 | import matplotlib.pyplot as plt
24 | import time
   |        ^^^^
25 |
26 | # %% Figure parameters
   |
help: Remove unused import: `time`

F403 `from .fun_bootstrap import *` used; unable to detect undefined names
  --> shared_code/shared_code/__init__.py:8:1
   |
 7 | # Optional: expose key functions
 8 | from .fun_bootstrap import *
   | ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
 9 | from .fun_dfcspeed import *
10 | from .fun_loaddata import *
   |

I001 [*] Import block is un-sorted or un-formatted
  --> shared_code/shared_code/__init__.py:8:1
   |
 7 |   # Optional: expose key functions
 8 | / from .fun_bootstrap import *
 9 | | from .fun_dfcspeed import *
10 | | from .fun_loaddata import *
11 | | from .fun_metaconnectivity import *
12 | | from .fun_network import *
13 | | from .fun_optimization import *
14 | | from .fun_utils import *
15 | | from .fun_paths import *
   | |________________________^
   |
help: Organize imports

F403 `from .fun_dfcspeed import *` used; unable to detect undefined names
  --> shared_code/shared_code/__init__.py:9:1
   |
 7 | # Optional: expose key functions
 8 | from .fun_bootstrap import *
 9 | from .fun_dfcspeed import *
   | ^^^^^^^^^^^^^^^^^^^^^^^^^^^
10 | from .fun_loaddata import *
11 | from .fun_metaconnectivity import *
   |

F403 `from .fun_loaddata import *` used; unable to detect undefined names
  --> shared_code/shared_code/__init__.py:10:1
   |
 8 | from .fun_bootstrap import *
 9 | from .fun_dfcspeed import *
10 | from .fun_loaddata import *
   | ^^^^^^^^^^^^^^^^^^^^^^^^^^^
11 | from .fun_metaconnectivity import *
12 | from .fun_network import *
   |

F403 `from .fun_metaconnectivity import *` used; unable to detect undefined names
  --> shared_code/shared_code/__init__.py:11:1
   |
 9 | from .fun_dfcspeed import *
10 | from .fun_loaddata import *
11 | from .fun_metaconnectivity import *
   | ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
12 | from .fun_network import *
13 | from .fun_optimization import *
   |

F403 `from .fun_network import *` used; unable to detect undefined names
  --> shared_code/shared_code/__init__.py:12:1
   |
10 | from .fun_loaddata import *
11 | from .fun_metaconnectivity import *
12 | from .fun_network import *
   | ^^^^^^^^^^^^^^^^^^^^^^^^^^
13 | from .fun_optimization import *
14 | from .fun_utils import *
   |

F403 `from .fun_optimization import *` used; unable to detect undefined names
  --> shared_code/shared_code/__init__.py:13:1
   |
11 | from .fun_metaconnectivity import *
12 | from .fun_network import *
13 | from .fun_optimization import *
   | ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
14 | from .fun_utils import *
15 | from .fun_paths import *
   |

F403 `from .fun_utils import *` used; unable to detect undefined names
  --> shared_code/shared_code/__init__.py:14:1
   |
12 | from .fun_network import *
13 | from .fun_optimization import *
14 | from .fun_utils import *
   | ^^^^^^^^^^^^^^^^^^^^^^^^
15 | from .fun_paths import *
   |

F403 `from .fun_paths import *` used; unable to detect undefined names
  --> shared_code/shared_code/__init__.py:15:1
   |
13 | from .fun_optimization import *
14 | from .fun_utils import *
15 | from .fun_paths import *
   | ^^^^^^^^^^^^^^^^^^^^^^^^
   |

I001 [*] Import block is un-sorted or un-formatted
 --> shared_code/shared_code/fun_bootstrap.py:1:1
  |
1 | / import numpy as np
2 | | import time
3 | | from joblib import Parallel, delayed
4 | | from tqdm import tqdm
  | |_____________________^
  |
help: Organize imports

F841 Local variable `n_type` is assigned to but never used
  --> shared_code/shared_code/fun_bootstrap.py:42:5
   |
40 |     wp_type, q_range, replicas=10, n_jobs=-1, bootstrap_fn=bootstrap_permutation_joblib
41 | ):
42 |     n_type = np.array(wp_type).shape[0]
   |     ^^^^^^
43 |     aux_qq_data = []
44 |     for wp_ in tqdm(wp_type):
   |
help: Remove assignment to unused variable `n_type`

UP009 [*] UTF-8 encoding declaration is unnecessary
 --> shared_code/shared_code/fun_dfcspeed.py:2:1
  |
1 | #!/usr/bin/env python3
2 | # -*- coding: utf-8 -*-
  | ^^^^^^^^^^^^^^^^^^^^^^^
3 | """
4 | Created on Fri Mar  8 15:45:43 2024
  |
help: Remove unnecessary coding comment

I001 [*] Import block is un-sorted or un-formatted
  --> shared_code/shared_code/fun_dfcspeed.py:22:1
   |
20 |   # =============================================================================
21 |
22 | / from pathlib import Path
23 | | from sys import prefix
24 | | import numpy as np
25 | | import brainconn as bct
26 | | import time
27 | |
28 | | from tqdm import tqdm
29 | | import numexpr as ne
30 | | from joblib import Parallel, delayed, parallel_backend, cpu_count
31 | | from collections import Counter
32 | | import logging
33 | | from scipy.stats import rankdata
34 | |
35 | | from .fun_optimization import (
36 | |     fast_corrcoef,
37 | |     fast_corrcoef_numba,
38 | |     fast_corrcoef_numba_parallel,
39 | |     pearson_speed_vectorized,
40 | |     cosine_speed_vectorized,
41 | |     spearman_speed,
42 | | )
43 | | from .fun_loaddata import *
   | |___________________________^
44 |
45 |   logger = logging.getLogger(__name__)
   |
help: Organize imports

F401 [*] `pathlib.Path` imported but unused
  --> shared_code/shared_code/fun_dfcspeed.py:22:21
   |
20 | # =============================================================================
21 |
22 | from pathlib import Path
   |                     ^^^^
23 | from sys import prefix
24 | import numpy as np
   |
help: Remove unused import: `pathlib.Path`

F401 [*] `sys.prefix` imported but unused
  --> shared_code/shared_code/fun_dfcspeed.py:23:17
   |
22 | from pathlib import Path
23 | from sys import prefix
   |                 ^^^^^^
24 | import numpy as np
25 | import brainconn as bct
   |
help: Remove unused import: `sys.prefix`

F401 [*] `brainconn` imported but unused
  --> shared_code/shared_code/fun_dfcspeed.py:25:21
   |
23 | from sys import prefix
24 | import numpy as np
25 | import brainconn as bct
   |                     ^^^
26 | import time
   |
help: Remove unused import: `brainconn`

F401 [*] `joblib.parallel_backend` imported but unused
  --> shared_code/shared_code/fun_dfcspeed.py:30:39
   |
28 | from tqdm import tqdm
29 | import numexpr as ne
30 | from joblib import Parallel, delayed, parallel_backend, cpu_count
   |                                       ^^^^^^^^^^^^^^^^
31 | from collections import Counter
32 | import logging
   |
help: Remove unused import: `joblib.parallel_backend`

F401 [*] `scipy.stats.rankdata` imported but unused
  --> shared_code/shared_code/fun_dfcspeed.py:33:25
   |
31 | from collections import Counter
32 | import logging
33 | from scipy.stats import rankdata
   |                         ^^^^^^^^
34 |
35 | from .fun_optimization import (
   |
help: Remove unused import: `scipy.stats.rankdata`

F401 [*] `.fun_optimization.fast_corrcoef_numba` imported but unused
  --> shared_code/shared_code/fun_dfcspeed.py:37:5
   |
35 | from .fun_optimization import (
36 |     fast_corrcoef,
37 |     fast_corrcoef_numba,
   |     ^^^^^^^^^^^^^^^^^^^
38 |     fast_corrcoef_numba_parallel,
39 |     pearson_speed_vectorized,
   |
help: Remove unused import

F401 [*] `.fun_optimization.fast_corrcoef_numba_parallel` imported but unused
  --> shared_code/shared_code/fun_dfcspeed.py:38:5
   |
36 |     fast_corrcoef,
37 |     fast_corrcoef_numba,
38 |     fast_corrcoef_numba_parallel,
   |     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
39 |     pearson_speed_vectorized,
40 |     cosine_speed_vectorized,
   |
help: Remove unused import

F403 `from .fun_loaddata import *` used; unable to detect undefined names
  --> shared_code/shared_code/fun_dfcspeed.py:43:1
   |
41 |     spearman_speed,
42 | )
43 | from .fun_loaddata import *
   | ^^^^^^^^^^^^^^^^^^^^^^^^^^^
44 |
45 | logger = logging.getLogger(__name__)
   |

F841 Local variable `num_channels` is assigned to but never used
   --> shared_code/shared_code/fun_dfcspeed.py:102:5
    |
100 |         A 2D array of shape (channels, channels) representing PLV between each pair of channels.
101 |     """
102 |     num_channels = data.shape[0]
    |     ^^^^^^^^^^^^
103 |
104 |     # Compute the phase for each channel
    |
help: Remove assignment to unused variable `num_channels`

F841 Local variable `phase_diff` is assigned to but never used
   --> shared_code/shared_code/fun_dfcspeed.py:109:5
    |
107 |     # Compute pairwise phase differences for all channels at once using broadcasting
108 |     # The result is an array of shape (channels, channels, timepoints)
109 |     phase_diff = phase_data[:, np.newaxis, :] - phase_data[np.newaxis, :, :]
    |     ^^^^^^^^^^
110 |
111 |     # Compute the complex exponential of the phase differences for all pairs
    |
help: Remove assignment to unused variable `phase_diff`

F841 Local variable `min_tau_zero` is assigned to but never used
   --> shared_code/shared_code/fun_dfcspeed.py:200:5
    |
198 |     # Extract DFC speed specific parameters
199 |     tau = kwargs.get("tau", 3)
200 |     min_tau_zero = kwargs.get("min_tau_zero", False)
    |     ^^^^^^^^^^^^
201 |     method = kwargs.get("method", "pearson")
    |
help: Remove assignment to unused variable `min_tau_zero`

F811 Redefinition of unused `prefix` from line 23
   --> shared_code/shared_code/fun_dfcspeed.py:204:5
    |
203 |     # Define prefix for DFC speed (following test version convention)
204 |     prefix = "speed"
    |     ^^^^^^ `prefix` redefined here
205 |
206 |     # Create custom file path for DFC speed (uses different naming convention like test version)
    |
   ::: shared_code/shared_code/fun_dfcspeed.py:23:17
    |
 22 | from pathlib import Path
 23 | from sys import prefix
    |                 ------ previous definition of `prefix` here
 24 | import numpy as np
 25 | import brainconn as bct
    |
help: Remove definition: `prefix`

F405 `make_file_path` may be undefined, or defined from star imports
   --> shared_code/shared_code/fun_dfcspeed.py:208:21
    |
206 |     # Create custom file path for DFC speed (uses different naming convention like test version)
207 |     if save_path:
208 |         file_path = make_file_path(
    |                     ^^^^^^^^^^^^^^
209 |             save_path / prefix, prefix, window_size, lag, n_animals, nodes
210 |         )
    |

F405 `make_file_path` may be undefined, or defined from star imports
   --> shared_code/shared_code/fun_dfcspeed.py:232:21
    |
231 |     # First, load DFC streams (following test version approach)
232 |     dfc_file_path = make_file_path(
    |                     ^^^^^^^^^^^^^^
233 |         save_path / "dfc", "dfc", window_size, lag, n_animals, nodes
234 |     )
    |

F405 `load_from_cache` may be undefined, or defined from star imports
   --> shared_code/shared_code/fun_dfcspeed.py:240:26
    |
238 |         try:
239 |             logger.info(f"Loading DFC stream from cache: {dfc_file_path}")
240 |             dfc_stream = load_from_cache(
    |                          ^^^^^^^^^^^^^^^
241 |                 dfc_file_path, key="dfc", logger=logger, label="dfc"
242 |             )
    |

B905 [*] `zip()` without an explicit `strict=` parameter
   --> shared_code/shared_code/fun_dfcspeed.py:263:47
    |
261 |     ]
262 |
263 |     median_speeds, speed_arrays, fc2_arrays = zip(*results)
    |                                               ^^^^^^^^^^^^^
264 |     median_speeds = np.array(median_speeds)  # This works, because all are scalar
    |
help: Add explicit value for parameter `strict=`

F405 `save2disk` may be undefined, or defined from star imports
   --> shared_code/shared_code/fun_dfcspeed.py:269:13
    |
267 |     if file_path:
268 |         try:
269 |             save2disk(file_path, prefix, **{"speed": speed_arrays, "fc": fc2_arrays})
    |             ^^^^^^^^^
270 |             logger.info(f"Saved results to {file_path} using key {prefix}")
271 |         except Exception as e:
    |

F811 Redefinition of unused `prefix` from line 23
   --> shared_code/shared_code/fun_dfcspeed.py:279:5
    |
277 | def handler_get_tenet(
278 |     ts_data,
279 |     prefix,
    |     ^^^^^^ `prefix` redefined here
280 |     window_size,
281 |     lag,
    |
   ::: shared_code/shared_code/fun_dfcspeed.py:23:17
    |
 22 | from pathlib import Path
 23 | from sys import prefix
    |                 ------ previous definition of `prefix` here
 24 | import numpy as np
 25 | import brainconn as bct
    |
help: Remove definition: `prefix`

F405 `make_file_path` may be undefined, or defined from star imports
   --> shared_code/shared_code/fun_dfcspeed.py:317:17
    |
315 |     # Handle DFC and meta-connectivity analysis
316 |     # Define the full save path based on parameters and save_path folder
317 |     file_path = make_file_path(save_path, prefix, window_size, lag, n_animals, nodes)
    |                 ^^^^^^^^^^^^^^
318 |     logger.info(f"file path: {file_path}")
    |

E712 Avoid equality comparisons to `True`; use `load_cache:` for truth checks
   --> shared_code/shared_code/fun_dfcspeed.py:326:12
    |
324 |     # label = "dfc-stream" if prefix == "dfc" else "meta-connectivity"
325 |     if file_path is not None and file_path.exists():
326 |         if load_cache == True:
    |            ^^^^^^^^^^^^^^^^^^
327 |             logger.info(f"Loading from cache: {file_path} and key: {key}")
328 |             try:
    |
help: Replace with `load_cache`

F405 `load_from_cache` may be undefined, or defined from star imports
   --> shared_code/shared_code/fun_dfcspeed.py:329:24
    |
327 |             logger.info(f"Loading from cache: {file_path} and key: {key}")
328 |             try:
329 |                 return load_from_cache(file_path, key=key, label=label)
    |                        ^^^^^^^^^^^^^^^
330 |             except Exception as e:
331 |                 logger.error(f"Failed to load {label} (reason: {e}). Recomputing...")
    |

F405 `save2disk` may be undefined, or defined from star imports
   --> shared_code/shared_code/fun_dfcspeed.py:345:9
    |
343 |     # Save results
344 |     try:
345 |         save2disk(file_path, prefix, **{key: results})
    |         ^^^^^^^^^
346 |         logger.info(f"Saved results to {file_path} using key as {key}")
347 |     except Exception as e:
    |

F811 Redefinition of unused `prefix` from line 23
   --> shared_code/shared_code/fun_dfcspeed.py:352:28
    |
352 | def compute4window(ws, ts, prefix, lag, save_path, load_cache, **kwargs):
    |                            ^^^^^^ `prefix` redefined here
353 |     """
354 |     Compute the analysis for a single window size.
    |
   ::: shared_code/shared_code/fun_dfcspeed.py:23:17
    |
 22 | from pathlib import Path
 23 | from sys import prefix
    |                 ------ previous definition of `prefix` here
 24 | import numpy as np
 25 | import brainconn as bct
    |
help: Remove definition: `prefix`

F811 Redefinition of unused `prefix` from line 23
   --> shared_code/shared_code/fun_dfcspeed.py:382:5
    |
380 |     ts: np.ndarray,
381 |     time_window_range: list,
382 |     prefix: str,
    |     ^^^^^^ `prefix` redefined here
383 |     paths: dict,
384 |     lag: int,
    |
   ::: shared_code/shared_code/fun_dfcspeed.py:23:17
    |
 22 | from pathlib import Path
 23 | from sys import prefix
    |                 ------ previous definition of `prefix` here
 24 | import numpy as np
 25 | import brainconn as bct
    |
help: Remove definition: `prefix`

F541 [*] f-string without any placeholders
   --> shared_code/shared_code/fun_dfcspeed.py:440:52
    |
438 |                 ws, ts, prefix, lag, save_path, load_cache, **kwargs
439 |             )
440 |             for ws in tqdm(time_window_range, desc=f"Window sizes")
    |                                                    ^^^^^^^^^^^^^^^
441 |         )
442 |         logging.info(f"{prefix} computation time {time.time()-start:.2f} seconds")
    |
help: Remove extraneous `f` prefix

E712 Avoid equality comparisons to `True`; use `min_tau_zero:` for truth checks
   --> shared_code/shared_code/fun_dfcspeed.py:676:8
    |
674 |     """
675 |
676 |     if min_tau_zero == True:
    |        ^^^^^^^^^^^^^^^^^^^^
677 |         min_tau = 0
678 |     else:
    |
help: Replace with `min_tau_zero`

E712 Avoid equality comparisons to `True`; use `get_speed_dist:` for truth checks
   --> shared_code/shared_code/fun_dfcspeed.py:707:12
    |
705 |         speed_windows_tau[idx_tt] = np.median(speed_oversampl, axis=1)
706 |
707 |         if get_speed_dist == True:  # speed_dist = np.mean(speed_oversampl,axis=1)
    |            ^^^^^^^^^^^^^^^^^^^^^^
708 |             speed_dist.append(speed_oversampl.flatten())
    |
help: Replace with `get_speed_dist`

E712 Avoid equality comparisons to `True`; use `get_speed_dist:` for truth checks
   --> shared_code/shared_code/fun_dfcspeed.py:710:8
    |
708 |             speed_dist.append(speed_oversampl.flatten())
709 |
710 |     if get_speed_dist == True:  # speed_dist = np.mean(speed_oversampl,axis=1)
    |        ^^^^^^^^^^^^^^^^^^^^^^
711 |         return speed_windows_tau, speed_dist
712 |     else:
    |
help: Replace with `get_speed_dist`

F811 Redefinition of unused `prefix` from line 23
   --> shared_code/shared_code/fun_dfcspeed.py:726:5
    |
724 |     n_jobs=-1,
725 |     path=None,
726 |     prefix="dfc",
    |     ^^^^^^ `prefix` redefined here
727 | ):
728 |     """
    |
   ::: shared_code/shared_code/fun_dfcspeed.py:23:17
    |
 22 | from pathlib import Path
 23 | from sys import prefix
    |                 ------ previous definition of `prefix` here
 24 | import numpy as np
 25 | import brainconn as bct
    |
help: Remove definition: `prefix`

F405 `make_file_path` may be undefined, or defined from star imports
   --> shared_code/shared_code/fun_dfcspeed.py:756:21
    |
754 |     def load_from_cache(ws, prefix):
755 |         n_animals, regions = ts.shape[0], ts.shape[1]
756 |         file_path = make_file_path(path, prefix, ws, lag, n_animals, regions)
    |                     ^^^^^^^^^^^^^^
757 |
758 |         # try loading from cache
    |

B905 [*] `zip()` without an explicit `strict=` parameter
   --> shared_code/shared_code/fun_dfcspeed.py:787:38
    |
786 |     if get_speed_dist:
787 |         speed_medians, speed_dists = zip(*results)
    |                                      ^^^^^^^^^^^^^
788 |         # Flatten the speed_dist list of lists to a single list
789 |         speed_dists = [
    |
help: Add explicit value for parameter `strict=`

F811 Redefinition of unused `prefix` from line 23
   --> shared_code/shared_code/fun_dfcspeed.py:881:12
    |
879 | # %%
880 | def check_and_rerun_missing_files(
881 |     paths, prefix, time_window_range, lag, n_animals, roi, processors=1
    |            ^^^^^^ `prefix` redefined here
882 | ):
883 |     """
    |
   ::: shared_code/shared_code/fun_dfcspeed.py:23:17
    |
 22 | from pathlib import Path
 23 | from sys import prefix
    |                 ------ previous definition of `prefix` here
 24 | import numpy as np
 25 | import brainconn as bct
    |
help: Remove definition: `prefix`

F405 `get_missing_files` may be undefined, or defined from star imports
   --> shared_code/shared_code/fun_dfcspeed.py:896:21
    |
894 |     """
895 |     # from shared_code.fun_dfcspeeZ>d import compute4window
896 |     missing_files = get_missing_files(
    |                     ^^^^^^^^^^^^^^^^^
897 |         paths, prefix, time_window_range, lag, n_animals, roi
898 |     )
    |

UP009 [*] UTF-8 encoding declaration is unnecessary
 --> shared_code/shared_code/fun_loaddata.py:2:1
  |
1 | #!/usr/bin/env python3
2 | # -*- coding: utf-8 -*-
  | ^^^^^^^^^^^^^^^^^^^^^^^
3 | """
4 | Created on Fri Mar  8 15:56:50 2024
  |
help: Remove unnecessary coding comment

I001 [*] Import block is un-sorted or un-formatted
  --> shared_code/shared_code/fun_loaddata.py:14:1
   |
12 |   # =============================================================================
13 |
14 | / from pathlib import Path
15 | | import numpy as np
16 | | import os
17 | | from scipy.io import loadmat
18 | | from joblib import Parallel, delayed, parallel_backend
19 | | import re
20 | | import logging
21 | | from typing import Any, Union
22 | | import pickle
   | |_____________^
23 |
24 |   # from .fun_dfcspeed import compute4window_new
   |
help: Organize imports

F401 [*] `joblib.Parallel` imported but unused
  --> shared_code/shared_code/fun_loaddata.py:18:20
   |
16 | import os
17 | from scipy.io import loadmat
18 | from joblib import Parallel, delayed, parallel_backend
   |                    ^^^^^^^^
19 | import re
20 | import logging
   |
help: Remove unused import

F401 [*] `joblib.delayed` imported but unused
  --> shared_code/shared_code/fun_loaddata.py:18:30
   |
16 | import os
17 | from scipy.io import loadmat
18 | from joblib import Parallel, delayed, parallel_backend
   |                              ^^^^^^^
19 | import re
20 | import logging
   |
help: Remove unused import

F401 [*] `joblib.parallel_backend` imported but unused
  --> shared_code/shared_code/fun_loaddata.py:18:39
   |
16 | import os
17 | from scipy.io import loadmat
18 | from joblib import Parallel, delayed, parallel_backend
   |                                       ^^^^^^^^^^^^^^^^
19 | import re
20 | import logging
   |
help: Remove unused import

UP007 [*] Use `X | Y` for type annotations
  --> shared_code/shared_code/fun_loaddata.py:91:33
   |
89 | # Save data functions
90 | # =============================================================================
91 | def save_pickle(obj: Any, path: Union[str, Path]) -> None:
   |                                 ^^^^^^^^^^^^^^^^
92 |     """Save a Python object to a file using pickle."""
93 |     path = Path(path)  # always use Path object
   |
help: Convert to `X | Y`

E402 Module level import not at top of file
   --> shared_code/shared_code/fun_loaddata.py:106:1
    |
106 | import numpy as np
    | ^^^^^^^^^^^^^^^^^^
    |

F811 [*] Redefinition of unused `np` from line 15
   --> shared_code/shared_code/fun_loaddata.py:106:17
    |
106 | import numpy as np
    |                 ^^ `np` redefined here
    |
   ::: shared_code/shared_code/fun_loaddata.py:15:17
    |
 14 | from pathlib import Path
 15 | import numpy as np
    |                 -- previous definition of `np` here
 16 | import os
 17 | from scipy.io import loadmat
    |
help: Remove definition: `np`

UP007 [*] Use `X | Y` for type annotations
   --> shared_code/shared_code/fun_loaddata.py:109:24
    |
109 | def load_fc2_npz(path: Union[str, Path]) -> Any:
    |                        ^^^^^^^^^^^^^^^^
110 |     """Load fc2 results from a .npz file."""
111 |     path = Path(path)
    |
help: Convert to `X | Y`

B007 Loop control variable `idx` not used within loop body
   --> shared_code/shared_code/fun_loaddata.py:302:9
    |
300 |     hash_dir = Path(folder_data) / specific_folder
301 |
302 |     for idx, file_name in enumerate(files_name):
    |         ^^^
303 |         file_path = hash_dir / file_name
    |
help: Rename unused `idx` to `_idx`

UP009 [*] UTF-8 encoding declaration is unnecessary
 --> shared_code/shared_code/fun_metaconnectivity.py:2:1
  |
1 | #!/usr/bin/env python3
2 | # -*- coding: utf-8 -*-
  | ^^^^^^^^^^^^^^^^^^^^^^^
3 | """
4 | Created on Wed Mar 26 00:16:53 2025
  |
help: Remove unnecessary coding comment

I001 [*] Import block is un-sorted or un-formatted
  --> shared_code/shared_code/fun_metaconnectivity.py:9:1
   |
 7 |   """
 8 |
 9 | / import joblib
10 | | import numpy as np
11 | | import matplotlib.pyplot as plt
12 | | import brainconn as bct
13 | | import os
14 | | import pandas as pd
15 | | from pathlib import Path
16 | | import copy
17 | | import pickle
18 | | from tqdm import tqdm
19 | |
20 | | from itertools import combinations_with_replacement
21 | | from joblib import Parallel, delayed, parallel_backend
22 | |
23 | | from .fun_dfcspeed import ts2dfc_stream
24 | | from .fun_loaddata import *
25 | | from .fun_optimization import (
26 | |     fast_corrcoef,
27 | | )  # , fast_corrcoef_numba, fast_corrcoef_numba_parallel
28 | |
29 | | import logging
   | |______________^
30 |
31 |   # import time
   |
help: Organize imports

F401 [*] `matplotlib.pyplot` imported but unused
  --> shared_code/shared_code/fun_metaconnectivity.py:11:29
   |
 9 | import joblib
10 | import numpy as np
11 | import matplotlib.pyplot as plt
   |                             ^^^
12 | import brainconn as bct
13 | import os
   |
help: Remove unused import: `matplotlib.pyplot`

F401 [*] `os` imported but unused
  --> shared_code/shared_code/fun_metaconnectivity.py:13:8
   |
11 | import matplotlib.pyplot as plt
12 | import brainconn as bct
13 | import os
   |        ^^
14 | import pandas as pd
15 | from pathlib import Path
   |
help: Remove unused import: `os`

F401 [*] `pandas` imported but unused
  --> shared_code/shared_code/fun_metaconnectivity.py:14:18
   |
12 | import brainconn as bct
13 | import os
14 | import pandas as pd
   |                  ^^
15 | from pathlib import Path
16 | import copy
   |
help: Remove unused import: `pandas`

F401 [*] `copy` imported but unused
  --> shared_code/shared_code/fun_metaconnectivity.py:16:8
   |
14 | import pandas as pd
15 | from pathlib import Path
16 | import copy
   |        ^^^^
17 | import pickle
18 | from tqdm import tqdm
   |
help: Remove unused import: `copy`

F403 `from .fun_loaddata import *` used; unable to detect undefined names
  --> shared_code/shared_code/fun_metaconnectivity.py:24:1
   |
23 | from .fun_dfcspeed import ts2dfc_stream
24 | from .fun_loaddata import *
   | ^^^^^^^^^^^^^^^^^^^^^^^^^^^
25 | from .fun_optimization import (
26 |     fast_corrcoef,
   |

F405 `make_file_path` may be undefined, or defined from star imports
  --> shared_code/shared_code/fun_metaconnectivity.py:65:17
   |
63 |     """
64 |     n_animals, _, nodes = ts_data.shape
65 |     file_path = make_file_path(save_path, "mc", window_size, lag, n_animals, nodes)
   |                 ^^^^^^^^^^^^^^
66 |     # Load from cache if available
67 |     if file_path is not None and file_path.exists():
   |

F405 `load_from_cache` may be undefined, or defined from star imports
  --> shared_code/shared_code/fun_metaconnectivity.py:68:16
   |
66 |     # Load from cache if available
67 |     if file_path is not None and file_path.exists():
68 |         return load_from_cache(file_path, key="mc", label="meta-connectivity")
   |                ^^^^^^^^^^^^^^^
69 |
70 |     # Compute meta-connectivity in parallel
   |

F405 `save2disk` may be undefined, or defined from star imports
  --> shared_code/shared_code/fun_metaconnectivity.py:79:5
   |
77 |     mc = np.stack(results)
78 |     # Save results if a save path is provided
79 |     save2disk(file_path, prefix="mc", mc=mc)
   |     ^^^^^^^^^
80 |     if save_path:
81 |         logger.info(f"Saving meta-connectivity to: {file_path}")
   |

B905 [*] `zip()` without an explicit `strict=` parameter
   --> shared_code/shared_code/fun_metaconnectivity.py:308:31
    |
306 |     # Reshape into [gamma_index][runs]
307 |     results_by_gamma = [[] for _ in range(gamma_range)]
308 |     for (gamma, _), result in zip(job_list, all_results):
    |                               ^^^^^^^^^^^^^^^^^^^^^^^^^^
309 |         gamma_idx = np.argmin(np.abs(gamma_mod - gamma))  # match gamma to index
310 |         results_by_gamma[gamma_idx].append(result)
    |
help: Add explicit value for parameter `strict=`

B007 Loop control variable `gamma` not used within loop body
   --> shared_code/shared_code/fun_metaconnectivity.py:319:14
    |
318 |     # Process per gamma
319 |     for idx, gamma in enumerate(tqdm(gamma_mod, desc="Processing gammas")):
    |              ^^^^^
320 |         results = results_by_gamma[idx]
321 |         communities, modularities = zip(*results)
    |
help: Rename unused `gamma` to `_gamma`

B905 [*] `zip()` without an explicit `strict=` parameter
   --> shared_code/shared_code/fun_metaconnectivity.py:321:37
    |
319 |     for idx, gamma in enumerate(tqdm(gamma_mod, desc="Processing gammas")):
320 |         results = results_by_gamma[idx]
321 |         communities, modularities = zip(*results)
    |                                     ^^^^^^^^^^^^^
322 |         communities = np.array(communities, dtype=np.int32)
323 |         communities_mat[idx] = communities
    |
help: Add explicit value for parameter `strict=`

B905 [*] `zip()` without an explicit `strict=` parameter
   --> shared_code/shared_code/fun_metaconnectivity.py:503:61
    |
501 |       if mc_data.ndim == 3:
502 |           # Process multiple MC matrices
503 |           communities_list, sort_idx_list, contingency_list = zip(
    |  _____________________________________________________________^
504 | |             *(process_single(mc_data[i]) for i in range(mc_data.shape[0]))
505 | |         )
    | |_________^
506 |           communities = np.mean(communities_list, axis=0)
507 |           sort_idx = np.argsort(communities)
    |
help: Add explicit value for parameter `strict=`

F841 Local variable `repeated` is assigned to but never used
   --> shared_code/shared_code/fun_metaconnectivity.py:900:5
    |
898 |     unique, counts = np.unique(flat, return_counts=True)
899 |     non_repeated = unique[counts == 1]
900 |     repeated = unique[counts == 2]
    |     ^^^^^^^^
901 |     return non_repeated
    |
help: Remove assignment to unused variable `repeated`

F821 Undefined name `bct`
  --> shared_code/shared_code/fun_network.py:24:24
   |
22 |     """
23 |     # modules, louvain = bct.modularity.modularity_louvain_dir(fc)
24 |     modules, louvain = bct.modularity.modularity_louvain_und_sign(fc, gamma=1.1)
   |                        ^^^
25 |
26 |     # Sort FC according to module labels
   |

F821 Undefined name `np`
  --> shared_code/shared_code/fun_network.py:27:13
   |
26 |     # Sort FC according to module labels
27 |     order = np.argsort(modules)
   |             ^^
28 |     fc_sorted = fc[:, order][order, :]
29 |     return fc_sorted
   |

UP009 [*] UTF-8 encoding declaration is unnecessary
 --> shared_code/shared_code/fun_optimization.py:2:1
  |
1 | #!/usr/bin/env python3
2 | # -*- coding: utf-8 -*-
  | ^^^^^^^^^^^^^^^^^^^^^^^
3 | """
4 | Created on Thu Apr  3 12:47:31 2025
  |
help: Remove unnecessary coding comment

I001 [*] Import block is un-sorted or un-formatted
  --> shared_code/shared_code/fun_optimization.py:9:1
   |
 7 |   """
 8 |
 9 | / import numpy as np
10 | | from scipy.stats import zscore, rankdata
11 | | from numba import njit, prange
   | |______________________________^
12 |
13 |   # =============================================================================
   |
help: Organize imports

F401 [*] `scipy.stats.zscore` imported but unused
  --> shared_code/shared_code/fun_optimization.py:10:25
   |
 9 | import numpy as np
10 | from scipy.stats import zscore, rankdata
   |                         ^^^^^^
11 | from numba import njit, prange
   |
help: Remove unused import: `scipy.stats.zscore`

I001 [*] Import block is un-sorted or un-formatted
  --> shared_code/shared_code/fun_paths.py:13:1
   |
11 |   """
12 |
13 | / from pathlib import Path
14 | | import os
15 | | from dotenv import load_dotenv
16 | | from typing import Dict, Optional
   | |_________________________________^
17 |
18 |   # Load environment variables from ../../.env if present
   |
help: Organize imports

UP035 `typing.Dict` is deprecated, use `dict` instead
  --> shared_code/shared_code/fun_paths.py:16:1
   |
14 | import os
15 | from dotenv import load_dotenv
16 | from typing import Dict, Optional
   | ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
17 |
18 | # Load environment variables from ../../.env if present
   |

UP024 [*] Replace aliased errors with `OSError`
  --> shared_code/shared_code/fun_paths.py:41:15
   |
39 |     root = os.getenv(f"PROJECT_ROOT_{env}")
40 |     if not root:
41 |         raise EnvironmentError(f"Environment variable PROJECT_ROOT_{env} is not set.")
   |               ^^^^^^^^^^^^^^^^
42 |     return Path(root)
   |
help: Replace `EnvironmentError` with builtin `OSError`

UP006 [*] Use `dict` instead of `Dict` for type annotation
  --> shared_code/shared_code/fun_paths.py:52:6
   |
50 |     cognitive_data_file: str,
51 |     anat_labels_file: str,
52 | ) -> Dict[str, Path]:
   |      ^^^^
53 |     """
54 |     Build canonical dataset/results/figures subpaths under a given root.
   |
help: Replace with `dict`

UP006 [*] Use `dict` instead of `Dict` for type annotation
  --> shared_code/shared_code/fun_paths.py:90:31
   |
89 | # =============================================================================
90 | def create_directories(paths: Dict[str, Path]) -> None:
   |                               ^^^^
91 |     """
92 |     Create all directories in the mapping that are not files (no suffix).
   |
help: Replace with `dict`

UP006 [*] Use `dict` instead of `Dict` for type annotation
   --> shared_code/shared_code/fun_paths.py:100:36
    |
 99 | # =============================================================================
100 | def check_write_permissions(paths: Dict[str, Path]) -> None:
    |                                    ^^^^
101 |     """
102 |     Check basic write permissions for directories in `paths`.
    |
help: Replace with `dict`

UP045 [*] Use `X | None` for type annotations
   --> shared_code/shared_code/fun_paths.py:123:19
    |
121 | # =============================================================================
122 | def get_paths(
123 |     dataset_name: Optional[str] = None,
    |                   ^^^^^^^^^^^^^
124 |     timecourse_folder: str = "Timecourses_updated_03052024",
125 |     cognitive_data_file: str = "ROIs.xlsx",
    |
help: Convert to `X | None`

UP006 [*] Use `dict` instead of `Dict` for type annotation
   --> shared_code/shared_code/fun_paths.py:130:6
    |
128 |     check_write: bool = False,
129 |     env: str = "LOCAL",
130 | ) -> Dict[str, Path]:
    |      ^^^^
131 |     """
132 |     Generate a dictionary of canonical paths for data, results, and figures.
    |
help: Replace with `dict`

UP009 [*] UTF-8 encoding declaration is unnecessary
 --> shared_code/shared_code/fun_utils.py:2:1
  |
1 | #!/usr/bin/env python3
2 | # -*- coding: utf-8 -*-
  | ^^^^^^^^^^^^^^^^^^^^^^^
3 | """
4 | Created on Sat Apr  5 00:18:49 2025
  |
help: Remove unnecessary coding comment

I001 [*] Import block is un-sorted or un-formatted
  --> shared_code/shared_code/fun_utils.py:9:1
   |
 7 |   """
 8 |   # %%
 9 | / from pathlib import Path
10 | | import numpy as np
11 | | import os
12 | | from scipy.io import loadmat
13 | | import pandas as pd
14 | | import pickle
15 | | from dotenv import load_dotenv
16 | | import matplotlib.pyplot as plt
   | |_______________________________^
17 |
18 |   # # Load environment variables from ../../.env if present
   |
help: Organize imports

F401 [*] `os` imported but unused
  --> shared_code/shared_code/fun_utils.py:11:8
   |
 9 | from pathlib import Path
10 | import numpy as np
11 | import os
   |        ^^
12 | from scipy.io import loadmat
13 | import pandas as pd
   |
help: Remove unused import: `os`

F401 [*] `dotenv.load_dotenv` imported but unused
  --> shared_code/shared_code/fun_utils.py:15:20
   |
13 | import pandas as pd
14 | import pickle
15 | from dotenv import load_dotenv
   |                    ^^^^^^^^^^^
16 | import matplotlib.pyplot as plt
   |
help: Remove unused import: `dotenv.load_dotenv`

F541 [*] f-string without any placeholders
   --> shared_code/shared_code/fun_utils.py:361:10
    |
359 |     labels = np.select(
360 |         [good, learners, impaired, bad],
361 |         [f"good", f"learners", f"impaired", f"bad"],
    |          ^^^^^^^
362 |         default=f"undefined",
363 |     )
    |
help: Remove extraneous `f` prefix

F541 [*] f-string without any placeholders
   --> shared_code/shared_code/fun_utils.py:361:19
    |
359 |     labels = np.select(
360 |         [good, learners, impaired, bad],
361 |         [f"good", f"learners", f"impaired", f"bad"],
    |                   ^^^^^^^^^^^
362 |         default=f"undefined",
363 |     )
    |
help: Remove extraneous `f` prefix

F541 [*] f-string without any placeholders
   --> shared_code/shared_code/fun_utils.py:361:32
    |
359 |     labels = np.select(
360 |         [good, learners, impaired, bad],
361 |         [f"good", f"learners", f"impaired", f"bad"],
    |                                ^^^^^^^^^^^
362 |         default=f"undefined",
363 |     )
    |
help: Remove extraneous `f` prefix

F541 [*] f-string without any placeholders
   --> shared_code/shared_code/fun_utils.py:361:45
    |
359 |     labels = np.select(
360 |         [good, learners, impaired, bad],
361 |         [f"good", f"learners", f"impaired", f"bad"],
    |                                             ^^^^^^
362 |         default=f"undefined",
363 |     )
    |
help: Remove extraneous `f` prefix

F541 [*] f-string without any placeholders
   --> shared_code/shared_code/fun_utils.py:362:17
    |
360 |         [good, learners, impaired, bad],
361 |         [f"good", f"learners", f"impaired", f"bad"],
362 |         default=f"undefined",
    |                 ^^^^^^^^^^^^
363 |     )
    |
help: Remove extraneous `f` prefix

F541 [*] f-string without any placeholders
   --> shared_code/shared_code/fun_utils.py:373:29
    |
371 |     # Store results in a new column
372 |     df_out[phenotype_column] = pd.Categorical(
373 |         labels, categories=[f"good", f"learners", f"impaired", f"bad"], ordered=False
    |                             ^^^^^^^
374 |     )
    |
help: Remove extraneous `f` prefix

F541 [*] f-string without any placeholders
   --> shared_code/shared_code/fun_utils.py:373:38
    |
371 |     # Store results in a new column
372 |     df_out[phenotype_column] = pd.Categorical(
373 |         labels, categories=[f"good", f"learners", f"impaired", f"bad"], ordered=False
    |                                      ^^^^^^^^^^^
374 |     )
    |
help: Remove extraneous `f` prefix

F541 [*] f-string without any placeholders
   --> shared_code/shared_code/fun_utils.py:373:51
    |
371 |     # Store results in a new column
372 |     df_out[phenotype_column] = pd.Categorical(
373 |         labels, categories=[f"good", f"learners", f"impaired", f"bad"], ordered=False
    |                                                   ^^^^^^^^^^^
374 |     )
    |
help: Remove extraneous `f` prefix

F541 [*] f-string without any placeholders
   --> shared_code/shared_code/fun_utils.py:373:64
    |
371 |     # Store results in a new column
372 |     df_out[phenotype_column] = pd.Categorical(
373 |         labels, categories=[f"good", f"learners", f"impaired", f"bad"], ordered=False
    |                                                                ^^^^^^
374 |     )
    |
help: Remove extraneous `f` prefix

B905 [*] `zip()` without an explicit `strict=` parameter
   --> shared_code/shared_code/fun_utils.py:411:28
    |
409 |     labels = []
410 |
411 |     for g_mask, g_label in zip(group_masks, group_labels):
    |                            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
412 |         for is_2m, age_label in zip([True, False], age_labels):
413 |             cond_mask = np.logical_and(g_mask, age_mask == is_2m)
    |
help: Add explicit value for parameter `strict=`

B905 [*] `zip()` without an explicit `strict=` parameter
   --> shared_code/shared_code/fun_utils.py:412:33
    |
411 |     for g_mask, g_label in zip(group_masks, group_labels):
412 |         for is_2m, age_label in zip([True, False], age_labels):
    |                                 ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
413 |             cond_mask = np.logical_and(g_mask, age_mask == is_2m)
414 |             masks.append(cond_mask)
    |
help: Add explicit value for parameter `strict=`

E712 Avoid equality comparisons to `True`; use `savefig:` for truth checks
   --> shared_code/shared_code/fun_utils.py:511:8
    |
509 |         }
510 |     )
511 |     if savefig == True:
    |        ^^^^^^^^^^^^^^^
512 |         return savefig
    |
help: Replace with `savefig`

Found 786 errors.
[*] 389 fixable with the `--fix` option (56 hidden fixes can be enabled with the `--unsafe-fixes` option).
