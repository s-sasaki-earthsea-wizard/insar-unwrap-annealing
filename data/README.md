# data/

Local input data. Everything in this directory except this file is ignored by
Git. Never copy real products into the repository history and never delete
existing data from here as part of automated work.

## Layout (proposed)

```
data/
  raw/        downloaded products as delivered (e.g. NISAR L2 GUNW .h5)
  derived/    crops or intermediate arrays produced by this project
  synthetic/  generated test fields (small, reproducible from code + seed)
```

## Provenance

`insar-unwrap fetch-sample NAME` downloads approved datasets
(`configs/datasets.yaml`) into `raw/NAME/`, verifies size and checksums and
writes the sidecar `<file>.provenance.yaml` automatically (also for a file that
was already present and verifies). Per product the record should cover:

* data source and granule identifier
* product version / processing version
* download date and checksum (sha256)
* frequency and polarization actually used
* HDF5 dataset paths used for wrapped phase, coherence and mask, and their
  shapes, after confirming that they share one grid

Do not substitute the product's unwrapped phase, rewrapped, for the observed
wrapped interferogram. Do not resample wrapped phase to reach a target size;
crop on the native grid instead.
