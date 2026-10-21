# Data

## 1. The streams

All text is tokenised with the Llama-3 tokenizer (`configs/1b_e2e`), without BOS, every document ending in EOS
(128001). Counts are from each stream's `meta.json`. The window columns give the tokens available as single-document
windows of that length, the only kind of sequence the runs train on (`doc_windows`); the 128K column is derived from
the 32K index (section 5).

| stream (`data/tokens/...`) | contents | documents | tokens | 8K windows | 32K windows | 128K windows | GiB |
|:--|:--|--:|--:|--:|--:|--:|--:|
| `cp_gutenberg_6d1ade79103fe750` | Project Gutenberg | 55,334 | 4.90B | 4.67B | 4.03B | 1.98B | 18.2 |
| `cp_pre1929_fdbe1eb1616b3ee7` | books published before 1929 (OCR) | 124,832 | 10.56B | 10.06B | 8.77B | 4.90B | 39.3 |
| `cp_loc_3c5e33209476779f` | Library of Congress public-domain books (OCR) | 129,005 | 8.06B | 7.55B | 6.31B | 3.20B | 30.0 |
| `cp_doab_81b9b60fb006df0b` | open-access academic books (DOAB) | 403,992 | 2.80B | 2.17B | 1.69B | 0.95B | 10.4 |
| `cp_arxiv_5752128469516ca7` | arXiv papers, Markdown converted from LaTeX | 295,424 | 6.11B | 4.90B | 1.74B | 0.13B | 22.8 |
| `cp_pes2o_19a022ef9825dc77` | peS2o open-access papers | 2,023,763 | 13.09B | 4.57B | 0.05B | 0 | 48.8 |
| `fineweb_edu_long_dc_1961668477d8a501` | FineWeb-Edu documents of 4K+ tokens | 435,791 | 3.64B | 1.58B | 0.22B | 0 | 13.6 |
| `dclm_long_dc_17baf11a547ae088` | DCLM documents of 4K+ tokens | 336,113 | 2.99B | 1.40B | 0.24B | 0.01B | 11.1 |
| `starcoder_repo_34497f38b4da24f3` | StarCoder files, by repository | 2,812,172 files (338,481 repositories) | 5.88B | 2.20B | 1.03B | 0.40B | 21.9 |
| `open_web_math_long_63b0e44605eca383` | OpenWebMath pages of 4K+ tokens | 49,198 | 0.50B | 0.27B | 0.07B | 0.02B | 1.9 |
| `finemath_long_09873175963dc8ee` | FineMath (4+) pages of 4K+ tokens | 38,792 | 0.34B | 0.16B | 0.04B | 0.01B | 1.3 |
| `wikipedia_long_1734a5e75620e284` | Wikipedia articles of 4K+ tokens | 31,264 | 0.23B | 0.08B | 0.01B | 0 | 0.9 |
| **training streams** | | | **59.1B** | **39.6B** | **24.2B** | **11.6B** | **220** |
| `pg19_test_a4dfccb3b3ac65f9` | PG-19 test set, evaluation only | 100 books | 9.9M | — | — | — | 0.04 |
| `unigram_llama3.i64` | token counts for the rarity probe | — | — | — | — | — | 0.001 |

In `starcoder_repo` a repository's files follow each other, but every file ends in EOS: a window lies inside one
file, and its 32K and 128K windows are single files of that length.

## 2. Mixes and passes

How often each stream's windows are read, with the weights of `runs/` and the steps of the chain of runs
([config.md](config.md)): 8K is the trunk and the branches' 8K part (187.7B tokens), then the 32K stage (8.4B) and
the 128K stage (4.0B).

| stream | 8K weight | 8K passes | 32K weight | 32K passes | 128K weight | 128K passes |
|:--|--:|--:|--:|--:|--:|--:|
| gutenberg | 0.10 | 4.0 | 0.12 | 0.25 | 0.17 | 0.35 |
| pre1929 | 0.11 | 2.1 | 0.22 | 0.21 | 0.30 | 0.25 |
| loc | 0.09 | 2.2 | 0.16 | 0.21 | 0.22 | 0.28 |
| doab | 0.06 | 5.2 | 0.12 | 0.60 | 0.17 | 0.72 |
| arxiv | 0.06 | 2.3 | 0.14 | 0.68 | 0.03 | 0.95 |
| pes2o | 0.04 | 1.6 | — | — | — | — |
| fineweb_edu_long | 0.20 | **23.8** | 0.04 | 1.6 | — | — |
| dclm_long | 0.16 | **21.4** | 0.04 | 1.4 | — | — |
| starcoder_repo | 0.12 | 10.2 | 0.13 | 1.1 | 0.11 | 1.1 |
| open_web_math_long | 0.03 | **20.9** | 0.02 | 2.3 | — | — |
| finemath_long | 0.02 | **23.8** | 0.01 | 2.0 | — | — |
| wikipedia_long | 0.01 | **23.2** | — | — | — | — |

- **8K is short of data.** With single-document windows only, the web, math and Wikipedia windows are read about
  20 times. Documents are not packed into sequences (decided 2026-10-17): more long documents are downloaded
  instead, until no 8K source is read more than about 3 times (section 7).
- **32K and 128K have enough tokens but little variety:** 86 % and 95 % of their windows are books, mostly pre-1929
  and OCR'd.

## 3. Where the streams come from

| stream | dataset | revision | files | processing |
|:--|:--|:--|:--|:--|
| cp_gutenberg | `common-pile/project_gutenberg_filtered` | `3cdf687` | all 15 | |
| cp_pre1929 | `common-pile/pre_1929_books_filtered` | `23f9d96` | all 26 | |
| cp_loc | `common-pile/library_of_congress_filtered` | `56725c7` | all 25 | |
| cp_doab | `common-pile/doab_filtered` | `defb24c` | all 5 | |
| cp_arxiv | `common-pile/arxiv_papers_filtered` | `033cf7f` | all 8 | |
| cp_pes2o | `common-pile/peS2o_filtered` | `2977475` | the first 30 of 91 | |
| fineweb_edu_long_dc | `HuggingFaceFW/fineweb-edu` | `87f0914` | `sample/100BT/`, the first 21 | long, PG-19 removed |
| dclm_long_dc | `mlfoundations/dclm-baseline-1.0-parquet` | `817d675` | `.../global-shard_01_of_10/local-shard_0_of_10/`, the first 77 | long, PG-19 removed |
| starcoder_repo | `bigcode/starcoderdata` (gated) | `9fc30b5` | python 8, javascript 5, java 5, c 3, cpp 3, go 2, rust 1, markdown 3 | grouped by repository |
| open_web_math_long | `open-web-math/open-web-math` | `fde8ef8` | `data/`, the first 8 | long |
| finemath_long | `HuggingFaceTB/finemath` | `e92b25a` | `finemath-4plus/`, the first 7 | long |
| wikipedia_long | `wikimedia/wikipedia` | `b04c8d1` | `20231101.en/`, the first 10 | long |
| pg19_test | `emozilla/pg19-test` (the `deepmind/pg19` test split) | `c5e39bf` | the one file | |

- **Revisions.** The Common Pile ones are those downloaded (2026-10-14). The other datasets were downloaded in
  2026-09 without recording one; the revision given holds exactly these files at these sizes (checked 2026-10-21),
  and [data_files.json](data_files.json) lists every file with its size and the md5 of the copy used. "The first N"
  are the first by name.
- **Common Pile** (public domain and openly licensed): `python tools/download_common_pile.py --sources <names>
  [--max_files N]`, then `python tools/convert_dolma.py --src <raw>/<name> --out <parquet>` (to parquet, without the
  PG-19 test books), then `python -m ttt.data prepare --name cp_<x> --files <parquet>/*.parquet --text_key text
  --tokenizer configs/1b_e2e`.
- **The others:** `python -m ttt.data prepare` per dataset. Long: `python -m ttt.data filter --src <stream> --name
  <name>_long --min_doc_tokens 4096`, with `--exclude <list>` for the PG-19 removal (`*_dc`). Grouped by repository:
  `python -m ttt.data prepare_grouped`, files grouped by `max_stars_repo_name` and ordered by `max_stars_repo_path`,
  repositories of 4096+ tokens.
- `unigram_llama3.i64` (`python -m ttt.data unigram`) and `pg19_test` are carried over from the TPU v4 runs, so that
  the probes and the evaluation set match every earlier result.

**PG-19 decontamination.** A document is taken from a PG-19 test book when more than 20 % of its word 13-grams are
found in the 100 test books. `tools/convert_dolma.py` removes them from the Common Pile sources, and the test books'
Gutenberg ids as well: 120 documents of Gutenberg (87 by id), 66 of pre-1929, 47 of Library of Congress, none of
DOAB, arXiv and peS2o. `tools/pg19_overlap.py` scores the documents of 8000+ characters of a prepared stream: 5 in
FineWeb-Edu and 8 in DCLM, excluded from the `*_dc` streams; none in the math and Wikipedia streams. The code stream
was not scanned. The PG-19 test set keeps all 100 books.

## 4. Files of a stream

| file | contents |
|:--|:--|
| `tokens.u32` | the token ids, little-endian uint32, every document ending in EOS |
| `meta.json` | `num_tokens`, `num_docs`, the tokenizer and the processing parameters |
| `windows_<seq_len>_<holdout>.npy` | start offsets of the single-document windows: non-overlapping, aligned at a document's first token, EOS excluded, before the held-out tail |

- The last 4,194,304 tokens of every stream (`holdout`) are never trained on; evaluation reads them from the first
  EOS after that point.
- A stream directory is named `<name>_<hash>`, the hash covering the absolute input paths and the parameters. A
  rebuild elsewhere gets other names, so match streams by name and token count. The data hash a run checks on
  resume covers names, weights, token counts and window indexes, not paths.

## 5. Window indexes

```bash
python -m ttt.data windows --src data/tokens/<stream> [...] --seq_len <T> --holdout 4194304
```

- 8K and 32K exist for every training stream.
- **128K is not built yet.** Build it before the 128K stage for the six streams of `runs/stage_128k.json`: Gutenberg,
  pre-1929, Library of Congress, DOAB, arXiv, StarCoder.
- The command reads every document length of a stream, so run it on a compute node.
- An index is part of the data hash: it must not change during a stage.

## 6. On the training machine

- `data_root` in `runs/` is `data/tokens`, relative to the repository: link or copy the streams there. Relative
  stream directories in `sources` and `eval_sources`, and a relative `unigram_path`, are read under it.
- Put a copy on every host's local disk, or on one read-only volume attached to all hosts. Do not read the streams
  through gcsfuse: the sampler memory-maps them, and on the TPU v4 pod a gcsfuse mount that died unnoticed made the
  training crash with SIGBUS.
- Check a copy against the table in section 1: `num_docs` and `num_tokens` in every `meta.json`, and the sizes.
  There are no md5 sums for the streams themselves.

## 7. Not prepared yet

| data | for | state |
|:--|:--|:--|
| 128K window indexes | the 128K stage | section 5 |
| more long documents | the 8K mix without packing: about +19.5B tokens of 8K windows of web (FineWeb-Edu, DCLM), +5.3B of code, +2.7B of math; Wikipedia cannot reach it and gets a smaller weight | candidates measured on 2026-10-16, nothing downloaded: Common Pile sources not yet used (government publications with 47 % of their tokens in 128K+ documents, UK parliamentary debates, Ubuntu IRC logs, YouTube transcripts, case law, patents), and open software-engineering agent trajectories (CC BY 4.0, 25–57K tokens each, almost none above 128K) |
