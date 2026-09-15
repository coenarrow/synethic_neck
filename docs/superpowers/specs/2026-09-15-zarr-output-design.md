# `generate --zarr` — design

Date: 2026-09-15
Status: approved in discussion, awaiting written review

## 1. Purpose

`synthetic-neck generate --zarr` writes the dataset as zarr stores that
satisfy remote-physiology's cache contract (`docs/cache-contract.md` in that
repository), so a synthetic dataset loads through the same reader as Neckflix
and PURE with no conversion step. The acceptance mechanism is that
repository's validator, `tools/validate_cache.py`: every store it is pointed
at must PASS.

Without `--zarr` nothing changes: the per-sample folder layout (section 7 of
the generator design) is written exactly as today.

## 2. Decisions

| Question | Decision |
| --- | --- |
| Zarr instead of, or as well as, the folder output? | **Instead of.** No `trace.csv`, MKVs, `vessel_ids.npy` or `metadata.json`; ffmpeg is not needed. |
| `participant` root attr | **The bare sample index as a string** (`"1"`, `"2"`, ...). Mixed-dataset runs disambiguate with remote-physiology's `--test-participant-dataset`. |
| Depth representation | **float32 millimetres.** Same unit as Neckflix's uint16 mm, without rounding away the 0.3–0.5 mm pulse lift. |
| `vessel_ids` | **A root-level array** in the store. The validator and the reader walk root *groups* only, so it is invisible to both. |
| Implementation shape | **One render loop, pluggable sink** (section 4). Rejected: a duplicated zarr-only generate path (the SNR retry logic would exist twice), and convert-after-generate (keeps ffmpeg, passes depth through 0.02 mm uint16). |
| Testing | **No new tests.** Verification is the validator over a generated cache (section 7). |

## 3. Store layout

```text
{out}/
├── dataset.json                  as today, plus "format": "zarr" | "folder"
└── {i}.zarr                      one per sample, i = sample index
    ├── attrs
    │   participant    "{i}"
    │   recording      "{i}"
    │   posture        "supine" | "recumbent" | "sitting"
    │   monk_tone      int 1-10, or null
    │   preset         str
    │   seed           int, the sample seed (base seed + i)
    │   synthetic_neck {...}      the dict metadata.json holds today, verbatim
    ├── vessel_ids                (H, W) uint8
    │                             attrs {labels: {"0": "background", "1": "artery", "2": "vein"}}
    └── 1/                        attrs {fps: float(video.fps)}
        ├── rgb/
        │   ├── video/data         (3, T, H, W) uint8
        │   ├── timestamps_us/data (T,) int64
        │   ├── abp/data           (T,) float64, attrs {units: "mmHg"}
        │   └── cvp/data           (T,) float64, attrs {units: "mmHg"}
        ├── ir/                    only when the draw has IR + depth
        │   └── video/data (1, T, H, W) uint8, plus the same three siblings as rgb
        └── depth/                 only when the draw has IR + depth
            └── video/data (1, T, H, W) float32 mm, plus the same three siblings as rgb
```

Pinned details:

- **Perspective.** Always exactly one, named `"1"`.
- **Timestamps.** `np.round(np.arange(T) * 1e6 / fps).astype(np.int64)`,
  identical in every modality, so the contract's first-frame alignment check
  holds by construction. `fps >= 20` is already enforced, so they are strictly
  increasing.
- **Traces.** The in-memory 1 kHz trace (`generate_trace`'s `(N, 3)` array)
  is linearly interpolated at the frame times:
  `np.interp(np.arange(T) / fps, trace[:, 0], trace[:, k])`. The same ABP and
  CVP arrays are written under every modality (the contract's identical-trace-
  set clause). No `rr`, `ecg` or `ppg`: the generator produces none.
- **Posture.** `params.trace.posture_deg` maps `0 -> supine`, `45 ->
  recumbent`, `90 -> sitting` (Neckflix's vocabulary, so one `FILTERS` entry
  serves both datasets). Any other angle raises `ConfigError` before rendering
  when `--zarr` is set; today only a `--set trace.posture_deg=...` override can
  produce one.
- **Monk tone.** `params.appearance.monk_tone`, which is `null` unless the
  config draws one. Deliberately not Neckflix's `skin_tone` dict: a synthetic
  sample has no self/recorder/clinician ratings to fill it with.
- **Every other drawn value** stays filterable through the dotted-path
  `FILTERS` syntax, e.g. `synthetic_neck.params.trace.heart_rate_bpm`.
- **Not written.** No `gr` modality (it is the luma of `rgb`), no
  `complete`/`resized_to`/`tool_version` bookkeeping (section 5 makes every
  visible store complete), no `--resize` (`video.frame_size` already sets it).
- **Storage settings** mirror remote-physiology's cacher `writer.py`, so every
  cache reads alike: video arrays use blosc-zstd level 9 with bitshuffle and
  chunks `(C, min(32, T), H, W)`. The integer video arrays (`rgb`, `ir`) also
  get a `Delta` filter. Float32 `depth` does not: on floats, Delta round-trips
  exactly or not depending on the values. A zarr 3.3 probe showed depth-like
  values near 700 mm surviving exactly while values in `[0, 1)` did not, and
  synthetic depth mixes near and far pixels.
  Timestamps, traces and `vessel_ids` use zarr's defaults.
- **Participant type.** A non-string `participant` is refused at write time,
  as the cachers do.

## 4. Generation flow

**CLI.** `generate` gains `--zarr` (`store_true`). Every other flag is
unchanged. `generate_dataset` and `generate_sample` keep their current
signatures and gain a keyword-only `zarr: bool = False`.

**Sink interface.** `_render` no longer knows about files. It drives a sink:

```python
trace = sink.begin(trace, streams, n_frames, frame_size, fps)  # once per attempt; wipes any previous attempt
sink.frame(rgb, ir, depth_mm)       # once per frame; ir and depth_mm are None without IR + depth
sink.commit(vessel_ids, meta)       # once, after the SNR check passes
sink.discard()                      # when the sample fails
```

`begin` receives the freshly generated 1 kHz trace and returns the trace to
render from. `FolderSink` writes `trace.csv` and returns the CSV read back, so
its pixels match the ground truth as written. `ZarrSink` returns the trace
unchanged. The sink keeps the trace from `begin`, which is why `commit` does
not take it.

`rgb` is `(H, W, 3)` uint8, `ir` is `(H, W)` uint8, `depth_mm` is `(H, W)`
float in mm, straight from `Renderer`. `generate.py` keeps the per-sample
seeding, the SNR measurement and retry loop, metadata building and
`dataset.json`, and picks the sink.

## 5. Components

| Module | Holds |
| --- | --- |
| `folder_store.py` (new) | `FolderSink`: today's `_render` file handling, **moved, not rewritten**. It writes `trace.csv` and renders from the CSV read back, as now, and handles the temporary MKVs and mux, `gray_video.mkv`, `vessel_ids.npy` and `metadata.json`. It is the only caller of `require_ffmpeg()`. |
| `zarr_store.py` (new) | `ZarrSink` and the store-writing helpers for section 3. |
| `generate.py` | Orchestration, as above. |
| `cli.py` | The `--zarr` flag. |

`ZarrSink` behaviour:

- **`begin`** opens `{out}/{i}.zarr.partial` with `mode="w"`, which drops
  any earlier attempt. The `.partial` suffix keeps it out of every `*.zarr`
  glob. It creates the perspective and modality groups, the full-shape video
  arrays and the timestamps.
- **`frame`** appends each modality's frame to a buffer. When a buffer holds
  one chunk's worth of frames (`min(32, T)`), it is stacked, moved to
  `(C, k, H, W)` and assigned as one slice, so each chunk is compressed once.
  Peak memory is one chunk per modality, about 28 MB at 300×300.
- **`commit`** flushes any remaining partial chunk. It then writes the traces,
  `vessel_ids` and the root attrs, removes an existing `{i}.zarr` if there is
  one, and renames `.partial` to `{i}.zarr`. A store the validator or reader
  can see is therefore always complete.
- **`discard`** removes `.partial`.
- **Trace source.** The zarr path renders from the in-memory trace and stores
  exactly that trace. Folder mode renders from the 3-decimal CSV, so one seed
  gives very slightly different pixels in the two modes.
- **Axis moves.** numpy `stack`/`transpose`, following this repository's
  numpy-only idiom.

## 6. Error handling

Unchanged in behaviour.

- A sample that raises, or never reaches green SNR 10 in 20 draws, is
  discarded. In zarr mode that means `discard()` removes `.partial`; in folder
  mode the sample directory is removed, as now.
- The sample is listed under `failed` in `dataset.json`, and the CLI exits 1.
- Config errors, including an unmappable posture, exit 2 before anything
  renders.

## 7. Verification

No new tests. The existing suite is neither run nor updated by this change.
The public signatures in section 4 stay compatible, so it should not need to
be.

1. **Build a small mixed cache**, run from the remote-physiology root with
   the output in a scratch directory:

   ```bash
   uv run --project tools/synthetic_datasets/synthetic_neck synthetic-neck generate --zarr \
       --preset lesson --n 4 --set video.frame_size=64 \
       --set streams.depth_ir_probability=0.5 --out <scratch>/synthetic_zarr
   ```

   Keep the default 10 s duration. The SNR guarantee measures cardiac power
   with an FFT, and a much shorter clip risks failing every draw. 64 px at
   10 s is what the existing generator test runs. Check that both shapes are
   present: at least one rgb-only store and at least one with IR + depth. If a
   seed gives only one shape, change `--seed`.
2. **Validate**:
   `uv run python tools/validate_cache.py <scratch>/synthetic_zarr` must print
   `4/4 stores pass`.
3. **Confirm folder mode still works.** Run the same command without `--zarr`
   and check that each sample directory still holds exactly `trace.csv`,
   `gray_video.mkv`, `rgbid_video.mkv`, `vessel_ids.npy` and `metadata.json`.
   This is a run of the tool, not a test, and it is the only check on the
   move into `FolderSink`.

## 8. Changes outside this repository

These are in remote-physiology, which carries this repository as a submodule
at `tools/synthetic_datasets/synthetic_neck`:

- **`configs/datasets/synthetic_neck.yaml`**: `CACHED_PATH` and
  `FILTERS: {}`, with a header comment listing the filterable root attrs in the
  style of `pure.yaml`. The file name makes `synthetic_neck` the
  `--test-participant-dataset` label.
- **`dataset/data_loader/SYNTHETIC_NECK.md`**: the cache spec the contract
  asks every new dataset for. It covers how generator output maps onto the
  layout, the attr vocabularies, and the liberties taken: synthesised
  timestamps, interpolated traces, float32 depth, and the root `vessel_ids`
  array.
- **The submodule pointer**, bumped to the commit that lands this change.

The validator, the reader and `docs/cache-contract.md` are not changed.

## 9. Also in this change

- `uv add "zarr>=3.3,<4"`, a core dependency with the same bound as the
  cachers. zarr and numcodecs ship wheels for Windows, Linux and macOS.
- The README's Generate section documents `--zarr` and what it writes.

## 10. Out of scope

- `synthetic-neck inspect` on zarr output: it keeps reading the folder
  layout only.
- A contract clause for label maps: `vessel_ids` stays an uncontracted root
  array.
- Resizing or any other consumer-side transform.
- Skip-if-exists or resume: regenerating a sample replaces its store, as
  folder mode replaces its directory.
