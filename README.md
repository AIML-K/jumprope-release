# jumprope-release
Jumprope project release

## Environment Setup
- Docker environment is supported for now -- also, it requires to install `nvidia-docker`. See [here](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) for installation.
- Now MUST either add the source code and model weights, or simply mount directory into container when creating it.

```
docker build -t foo:foo .
```

- If you want to build an environment from scratch, use PyTorch 2.0 or later. Except for PyTorch, other packages may be critical.

## Run
```
python main.py
```
- The output is available via stdout.