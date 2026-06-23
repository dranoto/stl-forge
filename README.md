# STL Forge

Image-in → 3D-printable `.stl` out, billed per request on RunPod Serverless.

[![RunPod](https://api.runpod.io/badge/user/thankfulcarp)](https://console.runpod.io/hub/user/thankfulcarp)

- **Model:** [Hunyuan3D 2.1](https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1) (shape-only — no Paint/texture for v1)
- **Pipeline:** image → Hunyuan3D DiT → trimesh cleanup + decimation → binary STL
- **GPU:** L4 / 3090 (24 GB) recommended; A4000 (16 GB) cost-optimized target
- **VRAM target:** 8–10 GB with `--low_vram_mode` (see [[STL Forge - Project Plan#GPU trim playbook]])

## Quick start

### Build the image

```bash
docker build --platform linux/amd64 -t thankfulcarp/stl-forge:runpod-latest .
docker push thankfulcarp/stl-forge:runpod-latest
```

### Test locally (no GPU — handler import will fail at model load, but Dockerfile is validated)

```bash
docker run --rm -it --gpus all \
    -e MODE_TO_RUN=serverless \
    -e HF_MODEL=tencent/Hunyuan3D-2.1 \
    -p 8000:8000 \
    thankfulcarp/stl-forge:runpod-latest \
    python -u src/handler.py --rp_serve_api --rp_api_port 8000
```

### Test with `runsync` against a real RunPod endpoint

Once deployed (see `## Deploy to RunPod` below):

```bash
curl -X POST https://api.runpod.ai/v2/$ENDPOINT_ID/runsync \
  -H "authorization: Bearer $RUNPOD_API_KEY" \
  -H "content-type: application/json" \
  -d '{
    "input": {
      "image": "https://upload.wikimedia.org/wikipedia/commons/thumb/0/05/Stanford_bunny.stl/256px-Stanford_bunny.stl.png",
      "target_faces": 100000
    }
  }'
```

The response is `{"stl_url": "...", "report": {...}}` by default (presigned HTTPS URL on S3-compatible storage, 24h TTL). Set `force_inline: true` in the input to receive `stl_b64` inline instead — only safe for ≤100k faces.

## API

### Input

| Field | Type | Default | Notes |
|---|---|---|---|
| `image` | str | — | **Required.** base64, `data:image/...;base64,...`, or `http(s)://` URL. |
| `target_faces` | int | 100000 | Decimation target. Range 10k–1M. 100k is a good default for FDM. |
| `mc_resolution` | int | 256 | Marching Cubes voxel grid. Lower = less VRAM, coarser mesh. |
| `quant` | str | `fp16` | One of `fp16`, `fp8`, `int8` — quantize the DiT to reduce VRAM. |
| `force_inline` | bool | false | Opt in to inline `stl_b64` response (only safe for ≤100k faces). |

### Output

```json
{
  "stl_url": "https://...",
  "stl_bytes": 6043212,
  "report": {
    "vertices": 50234,
    "faces": 100000,
    "watertight": true,
    "bbox_extents": [80.5, 91.2, 78.3],
    "generation_time_s": 38.4,
    "model_load_time_s": 92.1,
    "model": "tencent/Hunyuan3D-2.1"
  }
}
```

When `force_inline=true` and the mesh fits the 20 MB response cap, the response carries `stl_b64` instead of `stl_url`. Both forms include the same `report`.

## Cold start

First request to a fresh worker will time out at `/runsync`'s default 90-second wait — model load + first inference typically takes 90–120 seconds. Two options:

1. **One-shot:** append `?wait=300000` (5 min) to your first `runsync` URL.
2. **Keep warm:** set **Active workers: 1** in the RunPod endpoint config so one worker stays resident (incurs idle cost, but eliminates cold start).

Subsequent warm requests typically return in <60s for a 100k-face mesh.

## Deploy to RunPod

1. **Build & push the image** (above).
2. **Create a Serverless endpoint** in the RunPod console:
   - Container image: `docker.io/thankfulcarp/stl-forge:runpod-latest`
   - Type: **Queue**
   - **Model:** `tencent/Hunyuan3D-2.1` (enables HF cached model)
   - **GPU:** L4 (24 GB) or A4000 (16 GB)
   - Active workers: 0 (Flex) for dev, 1 once traffic justifies zero cold start
   - Max workers: 5
   - Idle timeout: 30s
   - Env vars: `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1` — and `LOW_VRAM=1` only when deploying on A4000 (16 GB); leave at default `LOW_VRAM=0` for L4/3090 (24 GB)
3. **Send a test request** with the curl example above.
4. **Iterate** — see `[[STL Forge - Project Plan]]` §6 for the phased build plan.

## Publish to RunPod Hub

This repo is published to [RunPod Hub](https://console.runpod.io/hub/user/thankfulcarp) as a serverless template. To ship a new release:

1. Update `.runpod/hub.json` and `.runpod/tests.json` if the API or smoke test has changed.
2. Tag a release on `main`: `git tag vX.Y.Z && git push origin vX.Y.Z`.
3. RunPod Hub picks up the tag, runs `tests.json` against a fresh worker, and promotes the release.

Before publishing:
- [ ] Code license locked (see `## License` below).
- [ ] Sample outputs in README reflect the current pipeline.
- [ ] First-request cold start within Hub's allowed budget (see `## Cold start`).

## Project layout

```
stl-forge/
├── Dockerfile                # CUDA 12.8 + PyTorch 2.7.0 (Blackwell-compatible) + Hunyuan3D 2.1 + our deps
├── requirements.txt          # serverless + cleanup deps
├── src/
│   ├── handler.py            # runpod.serverless handler
│   └── pipeline.py           # make_printable() + STL export (reusable, no runpod)
├── tests/
│   └── fixtures/             # sample input images (gitkeep)
├── .runpod/
│   ├── hub.json              # RunPod Hub template metadata
│   └── tests.json            # smoke test RunPod Hub runs before promoting
├── .dockerignore
├── .gitignore
└── README.md
```

## Project notes

The full design, phased build plan, GPU trim playbook, and cost analysis live in
the Obsidian vault at `10 - Projects/STL Forge/`:

- [[STL Forge MOC]] — top-level project note
- [[STL Forge - Project Plan]] — full design + handler skeleton + cost estimate
- [[RunPod Reference]] — condensed RunPod docs (serverless, model caching, deployment)

## License

Code: MIT.

Model: [Tencent Hunyuan3D Community License](https://huggingface.co/tencent/Hunyuan3D-2) — read it before commercial use.
