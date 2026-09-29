# Raw data (not in git)

`make data` downloads the two labelled splits of GLUE SST-2 into this
directory:

| file | rows | SHA-256 |
|---|---|---|
| `sst2_train.parquet` | 67,349 | `66a253e67968acfabcbe49dbe9da964b42ac1c851c40ab760e8c8942efdb3229` |
| `sst2_validation.parquet` | 872 | `a1371f3b3a7b0bcefa8388799a9359dc3ce76c349cc0079507a7991364fd2a9b` |

The files come from the Hugging Face dataset `nyu-mll/glue` at revision
`bcdcba79d07bc864c1c254ccfcedcce55bcc9a8c`, the revision the original
analysis used. A file is accepted only if its SHA-256 matches `config.toml`.
The data are public and need no account, but they are not redistributed
here because the dataset card gives the license as "other". The test split
is not used, because its labels are hidden.

Without a working download, fetch `sst2/train-00000-of-00001.parquet` and
`sst2/validation-00000-of-00001.parquet` in a browser from
<https://huggingface.co/datasets/nyu-mll/glue/tree/bcdcba79d07bc864c1c254ccfcedcce55bcc9a8c/sst2>,
save them here under the names above, and run `make data` to verify them.
