"""Common-set (S=1, 20 clips/dataset) ORACLE eval configs for Stage A arms. Template = the arm's own TRAIN yml (so every model-side key --
LoRA rank/targets, unfreeze_modules, fp32_trainable, blur_every_frame, vjepa_layers, context_key/context_layout, inference_flow_shift -- matches
the checkpoint), with the data/eval sections taken from the COMMON_ORACLE config but the manifest forced to the pm1 copies.
    python scripts/make_v2_eval_configs.py v2_base v2f_full v3_base ...     (FULL arm keys, i.e. the part after 'stageA_')
    DS="gopro adobe fmo_tbd" CKPT=checkpoint-30000 python scripts/make_v2_eval_configs.py v3_base
Outputs Wan22VJEPADynamics_eval_<arm>_<ds>.yml with experiment_key EVAL_<ARM>_<ds> (e.g. EVAL_V3_BASE_gopro)."""
import glob, json, os, sys
import yaml

D = "checkpoints/MotionBlur2Video"
M = "/scratch/gilbreth/wweligam/experiments/Motion"
CKPT = os.environ.get("CKPT", "checkpoint-30000")
DSS = os.environ.get("DS", "gopro adobe").split()
if not sys.argv[1:]:
    sys.exit(__doc__)
for arm in sys.argv[1:]:
    train_yml = f"{D}/Wan22VJEPADynamics_stageA_{arm}.yml"
    if not os.path.isfile(train_yml):
        sys.exit(f"no train yml {train_yml} -- pass the FULL arm key (e.g. v2_base, v3_base), not a suffix")
    # exact run-dir match: <M>/Wan22VJEPADynamics_stageA_<arm>_<timestamp> (the timestamp starts with '20'); never a different arm's prefix
    def ckpt_of(r):
        # CKPT=best -> the run's single best-<step> dir written by train.save_best
        if CKPT == "best":
            b = sorted(d for d in os.listdir(r) if d.startswith("best-"))
            return b[-1] if b else None
        return CKPT if os.path.isdir(f"{r}/{CKPT}") else None
    runs = [r for r in glob.glob(f"{M}/Wan22VJEPADynamics_stageA_{arm}_20*") if ckpt_of(r) and "_SMOKE" not in r]
    runs.sort(key=os.path.getmtime)
    if not runs:
        sys.exit(f"no run dir with {CKPT} for arm {arm}")
    run = runs[-1]
    if len(runs) > 1:
        print(f"WARNING {arm}: {len(runs)} run dirs have {CKPT}; using newest {os.path.basename(run)}")
    for ds in DSS:
        c = yaml.safe_load(open(train_yml))
        o = yaml.safe_load(open(f"{D}/Wan22VJEPADynamics_common_oracle_{ds}.yml"))
        c["experiment_key"] = f"EVAL_{arm.upper()}_{ds}"
        c["is_train"] = False
        c["datasets"], c["val"], c["sampler_seed"] = o["datasets"], o["val"], o["sampler_seed"]
        # The model assumes wan_frames in [-1,1]. findings/common_testset manifests carry wan_frames_range '01' (washed GT + V-JEPA oracle input).
        man = f"/scratch/gilbreth/wweligam/common_testset_pm1/manifest_{ds}.json"
        inner = json.load(open(man))["inner"]
        for i in (inner if isinstance(inner, list) else [inner]):
            assert i["wan_frames_range"] == "pm1", (man, i["wan_frames_range"])
        for split in ("train", "val"):
            c["datasets"][split]["manifest"] = man
        if c.get("t_sampling", "uniform") != "uniform" and c.get("inference_flow_shift") is None:
            # trained with a shifted t distribution but the arm yml predates inference_flow_shift: evaluating with the loader's default
            # shift=1 Euler grid is a train/inference mismatch, so sample with the shift the t distribution was warped by.
            c["inference_flow_shift"] = float(c.get("t_shift", 1.0))
            print(f"NOTE {arm}: trained with t_shift={c['inference_flow_shift']} -> eval uses inference_flow_shift={c['inference_flow_shift']}")
        c["train"]["max_train_steps"] = 1
        ck = ckpt_of(run)
        c["path"] = {"root": c["path"]["root"], "logging_dir": "logs", "resume_from_path": run, "resume_from_checkpoint": ck}
        if CKPT == "best":
            c["experiment_key"] += "_BEST"
        c.pop("xattn_max_tubelets", None)   # the oracle config forces 20; the arm's pos_t was allocated for its own tubelet count (4)
        out = f"{D}/Wan22VJEPADynamics_eval_{arm}_{ds}.yml"
        yaml.safe_dump(c, open(out, "w"), sort_keys=False)
        keep = {k: c.get(k) for k in ("fp32_trainable", "context_key", "deblur_prefix_frames", "inference_flow_shift", "unfreeze_modules")}
        print(arm, ds, os.path.basename(run), ck, keep)
