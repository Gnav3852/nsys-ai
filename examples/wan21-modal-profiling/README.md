# Wan2.1 Modal Profiling

Scaffold for profiling FastVideo Wan2.1 inference on Modal with Nsight Systems,
then analyzing the exported SQLite profile with nsys-ai.

The intended workflow is:

1. Run a Wan2.1 inference command on 1x L40S.
2. Capture an Nsight Systems profile (`.nsys-rep`) and export SQLite.
3. Download the profile locally.
4. Analyze it with nsys-ai.
5. Repeat after one optimization, and later validate on H100.

## Prerequisites

```bash
pip install modal
modal setup
```

Confirm the CLI is authenticated:

```bash
modal token whoami
```

## Smoke Test

This checks Modal can build the image and start a GPU container:

```bash
modal run examples/wan21-modal-profiling/profile_wan21_modal.py -- \
  --gpu L40S \
  --run-name smoke_l40s \
  --launch-cmd "python -c 'import torch; print(torch.cuda.get_device_name()); print(torch.randn(1, device=\"cuda\"))'" \
  --no-profile
```

## Cheap Wan2.1 Smoke Run

FastVideo Wan2.1 inference is launched through the config-first CLI. This short
run checks the FastVideo install and model path before spending time on a full
profile:

```bash
modal run examples/wan21-modal-profiling/profile_wan21_modal.py -- \
  --gpu L40S \
  --run-name wan21_1_3b_l40s_smoke \
  --repo-url https://github.com/hao-ai-lab/FastVideo.git \
  --repo-ref main \
  --setup-cmd "pip install -e . && pip install nsys-ai" \
  --launch-cmd "fastvideo generate --config scripts/inference/inference_wan.yaml --request.sampling.num_inference_steps 2 --request.sampling.num_frames 17 --request.sampling.height 256 --request.sampling.width 448 --request.output.output_path /profiles/wan21_1_3b_l40s_smoke/videos" \
  --no-profile
```

## Profile Wan2.1 1.3B on L40S

This is the baseline profile requested for 1x L40S:

```bash
modal run examples/wan21-modal-profiling/profile_wan21_modal.py -- \
  --gpu L40S \
  --run-name wan21_1_3b_l40s_baseline \
  --repo-url https://github.com/hao-ai-lab/FastVideo.git \
  --repo-ref main \
  --setup-cmd "pip install -e . && pip install nsys-ai" \
  --launch-cmd "fastvideo generate --config scripts/inference/inference_wan.yaml --request.output.output_path /profiles/wan21_1_3b_l40s_baseline/videos"
```

## Profile Wan2.1 14B on L40S

The 14B I2V config defaults to 2 GPUs in FastVideo. This command forces the run
onto 1 GPU for the requested L40S data point; if it OOMs, keep the failed log
and try a smaller frame/step smoke first.

```bash
modal run examples/wan21-modal-profiling/profile_wan21_modal.py -- \
  --gpu L40S \
  --run-name wan21_14b_i2v_l40s_baseline \
  --repo-url https://github.com/hao-ai-lab/FastVideo.git \
  --repo-ref main \
  --setup-cmd "pip install -e . && pip install nsys-ai" \
  --launch-cmd "fastvideo generate --config scripts/inference/inference_wan_i2v.yaml --generator.engine.num_gpus 1 --generator.engine.parallelism.tp_size 1 --generator.engine.parallelism.sp_size 1 --request.output.output_path /profiles/wan21_14b_i2v_l40s_baseline/videos"
```

The profile artifacts are written to the Modal volume:

```text
wan21-nsys-profiles:/profiles/<run-name>/
```

Expected artifacts:

```text
<run-name>.nsys-rep
<run-name>.sqlite
<run-name>_timing.json
```

## H100 Run

Use the same workload and change the GPU/run name. On H100, force the
FlashAttention backend and install the Hopper/FA3 package during setup.

```bash
modal run examples/wan21-modal-profiling/profile_wan21_modal.py -- \
  --gpu H100 \
  --run-name wan21_1_3b_h100_fa3 \
  --repo-url https://github.com/hao-ai-lab/FastVideo.git \
  --repo-ref main \
  --setup-cmd "pip install -e . && pip install nsys-ai && git clone https://github.com/Dao-AILab/flash-attention.git /tmp/flash-attention && cd /tmp/flash-attention/hopper && pip install ninja && python setup.py install" \
  --launch-cmd "FASTVIDEO_ATTENTION_BACKEND=FLASH_ATTN fastvideo generate --config scripts/inference/inference_wan.yaml --request.output.output_path /profiles/wan21_1_3b_h100_fa3/videos"
```

FastVideo selects the attention backend from `FASTVIDEO_ATTENTION_BACKEND`.
For H100, check the Modal logs for the selected Flash Attention backend and
inspect kernels in nsys-ai to confirm the profile used the intended path.

## Candidate Optimization Run

One Wan2.1-specific candidate already present in FastVideo is VMoBA attention
for the 1.3B model. Run it after the baseline and compare wall time plus nsys-ai
findings:

```bash
modal run examples/wan21-modal-profiling/profile_wan21_modal.py -- \
  --gpu L40S \
  --run-name wan21_1_3b_l40s_vmoba \
  --repo-url https://github.com/hao-ai-lab/FastVideo.git \
  --repo-ref main \
  --setup-cmd "pip install -e . && pip install nsys-ai" \
  --launch-cmd "FASTVIDEO_ATTENTION_BACKEND=VMOBA_ATTN fastvideo generate --config scripts/inference/inference_wan_1.3B_VMoba.yaml --request.output.output_path /profiles/wan21_1_3b_l40s_vmoba/videos"
```

## Analyze Locally

After downloading the `.sqlite` file:

```bash
python -m nsys_ai skill run profile_health_manifest path/to/profile.sqlite
python -m nsys_ai skill run gpu_idle_gaps path/to/profile.sqlite -p device=0
python -m nsys_ai skill run kernel_launch_overhead path/to/profile.sqlite -p device=0
python -m nsys_ai skill run top_kernels path/to/profile.sqlite -p device=0
python -m nsys_ai skill run tensor_core_usage path/to/profile.sqlite -p device=0
python -m nsys_ai skill run memory_transfers path/to/profile.sqlite -p device=0
```
