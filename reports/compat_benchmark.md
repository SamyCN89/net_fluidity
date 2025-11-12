# Compatibility and Performance Benchmark

This compares selected functions between `shared_code` and `metaconnectivity` on small synthetic inputs.

| Category | Name | Shape Equal | Allclose | Max | Mean | Shared ms | Meta ms | Speedup (meta/shared) | Note |
|---|---|:--:|:--:|---:|---:|---:|---:|---:|---|
| optimization | fast_corrcoef | 1 | 1 | 0.0 | 0.0 | 0.05525439992197789 | 0.029196599643910304 | 0.5284031621940956 |  |
| dfc | ts2dfc_stream(2D) | 1 | 1 | 0.0 | 0.0 | 0.5377663997933269 | 0.4860248001932632 | 0.9037842460593506 |  |
| dfc | dfc_speed(vstep=1) | 0 | 0 | 0.009547799825668335 | 0.0005304333236483703 | 0.0785441996413283 | 0.7216450001578778 | 9.187756746561377 | length_mismatch:(17, 18) |
| dfc | dfc_speed(3D, vstep=1) | 0 | 0 | 0.2332408474489852 | 0.12334156469658542 | 0.09687479978310876 | 0.5406240001320839 | 5.580646373901956 | length_mismatch:(17, 18) |
| metaconnectivity | compute_metaconnectivity | 1 | 1 | 0.0 | 0.0 | 0.7484950001526158 | 0.7187609990069177 | 0.9602749502139153 |  |

## Notes on dFC Speed Differences
- metaconnectivity.dfc_speed(2D/3D) computes speeds for each t vs t+vstep, yielding n_frames - vstep values.
- shared_code.fun_dfcspeed.dfc_speed uses an internal index stride that currently produces n_frames - vstep - 1 values for vstep=1 on 2D input; reports align after trimming to the shorter length.
- For 3D input, shared_code extracts lower-triangular FC values (excluding diagonal), while some legacy implementations may include diagonals or duplicates when reshaping.
