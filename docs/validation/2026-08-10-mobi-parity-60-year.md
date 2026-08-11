# MOBI parity 60-year validation

## Provenance

- Feature commit: `5c327d894e810c7373c7f7f9a00d76a69e15476f`
- Plugin content SHA-256: `efb9072c88cde93b694df4e4f40d3a40dd2336e6344016d968f8e013aa9f1a65`
- Veros commit: `bcbead95d56341f3ee9f30311c05a546674095e7` (v0.2.3)
- Runtime: Python 3.11.15; NumPy 1.23.5; SciPy 1.10.1; h5netcdf 1.1.0; h5py 3.8.0; NumPy backend; SciPy streamfunction solver.
- Production command: `.venv/bin/python scripts/run_segmented_60_year.py new /home/nutrik/Documents/bugfix/validation/mobi-parity-60y`
- Independent validator commands: `.venv/bin/python scripts/validate_60_year_run.py "$MERGED_SNAPSHOT"` and `.venv/bin/python scripts/validate_60_year_run.py "$MERGED_SNAPSHOT" --markdown`, where `MERGED_SNAPSHOT` was derived from `progress.json`.
- Manifest-derived merged snapshot: `/home/nutrik/Documents/bugfix/validation/mobi-parity-60y/merged/mobi-parity.snapshot-attempt-001.nc`
- Merged snapshot SHA-256: `0add41c5b474dcb86f220cbe84cb5401b48293fe8c50d4028514c7b67c4493b6`
- Manifest status: `complete`; committed segments: 60; merged `Time` records: 60.
- The intentional stress-derived wind reconstruction was retained unchanged.

Snapshot: `/home/nutrik/Documents/bugfix/validation/mobi-parity-60y/merged/mobi-parity.snapshot-attempt-001.nc`

| Check | Value |
| --- | ---: |
| Snapshots | 60 |
| DIC all-years min | 1631.8099093963583 |
| DIC all-years max | 2751.232594131835 |
| DIC year-1 max | 2500.785730389979 |
| DIC year-60 max | 2738.782591720624 |
| Alkalinity all-years min | 1927.7914958420358 |
| Alkalinity all-years max | 2704.2159264928873 |
| Alkalinity year-1 max | 2426.273699446465 |
| Alkalinity year-60 max | 2704.2159264928873 |
| pCO2 all-years min | 87.39077480989094 |
| pCO2 all-years max | 1648.0278177199625 |
| hSWS all-years min | 2.5624248638369075e-09 |
| hSWS all-years max | 3.494794496218218e-08 |
| cflux all-years min | -0.0009143260247590597 |
| cflux all-years max | 0.00025084694176365566 |

Acceptance result: **PASS**
