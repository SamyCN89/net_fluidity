# Inspecting Preprocessing Outputs

```python
from pathlib import Path
import numpy as np
import pandas as pd
import pickle

project_root = Path("/path/to/net_fluidity")
dataset = "julien_caillette"  # or "ines_abdallah"
pre_dir = project_root / "results" / dataset / "preprocessed"

# Load canonical time-series bundle
bundle = np.load(pre_dir / f"ts_and_meta_{dataset}.npz")
ts = bundle["ts"]            # time-series array (animals × TR × regions)
mouse_ids = bundle["mouse_ids"]
anat_labels = bundle["anat_labels"]

# Review metadata manifest
meta_path = next(pre_dir.glob("metadata_animals_*_regions_*_tr_*.pkl"))
with meta_path.open("rb") as handle:
    metadata = pickle.load(handle)

# Peek at aligned cognition table
cog_csv = next(pre_dir.glob("cog_data_filtered_*.csv"), None)
if cog_csv:
    cog_df = pd.read_csv(cog_csv)
    print(cog_df.head())
```

- Ines runs also emit `grouping_data_*.pkl` files containing mask dictionaries; open them with `pickle.load` to list available groupings.
- Julien runs produce `ts_filtered_*.npz` alongside the canonical bundle when time-series lengths differ; load these with `np.load` as shown above.
