# LIBERO-PRO install

We use a lightly patched fork of
[upstream LIBERO-PRO](https://github.com/Zxy-MLlab/LIBERO-PRO). The fork's
URL is withheld for double-blind review and will be released with the
camera-ready version; its `FORK_NOTES.md` lists what differs from upstream
(segmentation utilities, refactored `perturbation.py`, eval/path configs).

```bash
# from the policy_abstraction repo root
cd third_party
git clone <anonymized-fork-url> LIBERO-PRO

# fetch bddl + init asset files (LFS)
apt-get install -y git-lfs && git lfs install
git clone https://huggingface.co/datasets/zhouxueyang/LIBERO-Pro LIBERO_PRO_assets
mv LIBERO_PRO_assets/bddl_files/* LIBERO-PRO/libero/libero/bddl_files/
mv LIBERO_PRO_assets/init_files/* LIBERO-PRO/libero/libero/init_files/
rm -rf LIBERO_PRO_assets
```

No per-machine config edits are needed: the LIBERO benchmark and evaluation
configs live inside `policy_abstraction` at
`src/policy_abstraction/envs/libero/configs/` and use `${LIBERO_PRO_PATH}/...`
placeholders that get expanded at runtime. See
[`docs/libero_pro_setup.md`](libero_pro_setup.md) for the complete walkthrough
(venv creation, dataset download, recollection).
