# Workshop 04 Apr 2026

## Prerequisites

Join [Discord](https://discord.gg/kdT3GfUd)

1. Creare VM in [Nebius cloud](https://console.nebius.com/)
Folositi link-ul de invitatie din mail-ul primit, dupa care se urmeaza pasii de mai jos:

1. Din meniul din stanga se alege `Compute > Virtual Machines`
2. Click pe `Create virtual machine`
3. Se completeaza urmatoarele:
   a. Project: `default-project-eu-north1`
   b. Name: ok cel generat
   c. Available platform: `NVIDIA H100 NVLink`
   d. Preset: `1 GPU`
   e. Public IP address: `Auto assign dynamic IP`
   f. Username and SSH key: `+ Create`. 
      
      Se completeaza `Username`: `workshop` si `Public key`:
```text
ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIBov86L2J1+2EI8dtDPlVZ8Youp4pbmupiEpO+GYSYRf densemax2@densemax2
```

6. Instalare [Surogate](https://surogate.ai).
```shell
curl -sSL https://surogate.ai/install.sh | bash
```

Evaluare model antrenat:
```shell
surogate vf-eval markdown-table-qa -m "workshop/model/" -b http://localhost:8000/v1 -n 50 --max-tokens 4096
```

## Supervised Fine-Tuning
Datasets: [https://huggingface.co/cetusian/datasets](https://huggingface.co/cetusian/datasets)

```shell
nvidia-smi
source .venv/bin/activate
cd gpuX
CUDA_VISIBLE_DEVICES=X surogate sft sft.yaml
```


## Reinforcement Learning (GRPO)
```shell
surogte grpo-infer ./infer.yaml
surogate grpo-orch ./orch.yaml
surogate grpo-train ./train.yaml
```
