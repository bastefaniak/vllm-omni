# Cosmos3

Source <https://github.com/vllm-project/vllm-omni/tree/main/examples/online_serving/cosmos3>.


This example shows Cosmos3 online serving with `Cosmos3OmniDiffusersPipeline`.

Set `COSMOS3_MODEL` to a local Diffusers-format checkpoint before starting the server:

```bash
export COSMOS3_MODEL=/path/to/cosmos3-diffusers
cd examples/online_serving/cosmos3
bash run_server.sh
```

`run_server.sh` accepts these environment overrides:

- `MODEL`: checkpoint path, defaults to `COSMOS3_MODEL`
- `PORT`: server port, defaults to `8091`
- `CACHE_BACKEND`: set to `cache_dit` to enable Cache-DiT
- `ENABLE_LAYERWISE_OFFLOAD`: set to `1` to enable layerwise offload
- `CFG_PARALLEL_SIZE`, `TENSOR_PARALLEL_SIZE`, `ULYSSES_DEGREE`, `USE_HSDP`: parallel execution controls
- `ALLOWED_LOCAL_MEDIA_PATH`: local media access path, defaults to `/`

## Text-to-Image

```bash
bash run_curl_t2i.sh
```

The script calls `POST /v1/images/generations`, which selects Cosmos3 text-to-image through `modalities=["image"]` internally.

## Text-to-Video

```bash
bash run_curl_t2v.sh
```

## Image-to-Video

Download an example image or set `IMAGE_PATH` to your own image:

```bash
wget https://vllm-public-assets.s3.us-west-2.amazonaws.com/vision_model_images/cherry_blossom.jpg
IMAGE_PATH=cherry_blossom.jpg bash run_curl_i2v.sh
```

## Video With Sound

This path requires a sound-capable Cosmos3 checkpoint with `sound_gen` weights.

```bash
bash run_curl_t2v_sound.sh
```

The script passes `generate_sound=true` and `sound_duration` to the video endpoint.

## Action Policy

This path requires an action-capable Cosmos3 checkpoint with `action_gen` weights. Pass either `domain_name` or `domain_id` through `extra_params`.

```bash
IMAGE_PATH=cherry_blossom.jpg bash run_curl_action_policy.sh
```

The script uses the asynchronous `POST /v1/videos` job endpoint so it can download the MP4 and save the returned action metadata JSON.

## Example materials

??? abstract "run_curl_action_policy.sh"
    ``````sh
    --8<-- "examples/online_serving/cosmos3/run_curl_action_policy.sh"
    ``````
??? abstract "run_curl_i2v.sh"
    ``````sh
    --8<-- "examples/online_serving/cosmos3/run_curl_i2v.sh"
    ``````
??? abstract "run_curl_t2i.sh"
    ``````sh
    --8<-- "examples/online_serving/cosmos3/run_curl_t2i.sh"
    ``````
??? abstract "run_curl_t2v.sh"
    ``````sh
    --8<-- "examples/online_serving/cosmos3/run_curl_t2v.sh"
    ``````
??? abstract "run_curl_t2v_sound.sh"
    ``````sh
    --8<-- "examples/online_serving/cosmos3/run_curl_t2v_sound.sh"
    ``````
??? abstract "run_server.sh"
    ``````sh
    --8<-- "examples/online_serving/cosmos3/run_server.sh"
    ``````
