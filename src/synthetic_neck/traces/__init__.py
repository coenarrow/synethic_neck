"""Ground-truth traces, one module per trace, all on the shared grid in `grid` and sharing one respiratory waveform
and one set of R-wave times: `respiratory`, `ecg`, `abp`, `cvp` and `ppg`. Each module owns the config dataclass for
its section of priors/base.yaml; `priors` holds the distribution types. See trace_generation.md."""
