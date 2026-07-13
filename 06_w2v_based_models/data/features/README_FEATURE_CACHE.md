# Feature cache

Notebook 02 writes the full feature cache here:

`iemocap_full_emotion2vec_acoustic_cache.npz`

Expected arrays:

- `sample_ids`
- `emotion2vec`
- `acoustic`
- `acoustic_names`

Notebook 03 and 04 require this cache. If it is missing, they stop and ask you to run notebook 02 first.

On Kaggle, `/kaggle/input` is read-only. Therefore notebook 02 reads raw data from `/kaggle/input/.../data`, but writes the generated cache to:

`/kaggle/working/iemocap_full_multitask_data/features/iemocap_full_emotion2vec_acoustic_cache.npz`

Notebook 03 and 04 check `/kaggle/working/...` first, then `/kaggle/input/...`. If you run notebook 03/04 in a fresh Kaggle session, either rerun notebook 02 or save the generated `.npz` as a Kaggle Dataset and attach it.
