# Upload Notes

The upload package follows the image filenames currently referenced in `main.tex`.

Missing locally at packaging time:

- `alien.png`
- `prometheus.png`
- `mr_mrs_smith.png`
- `star_trek_into_darkness.png`
- `example.bib`

These files should be added before expecting the LaTeX source to compile completely.

Excluded intentionally:

- Large movie/video files under `data/`
- generated local movie snippets under `data/missing_files_v1-2_test/movie_segments/`
- `data/missing_files_v1-2_test/`
- duplicate folders `data/results saved/` and `data/summaries/`
- Python cache files
- old duplicate figure names not referenced by the current `main.tex`
