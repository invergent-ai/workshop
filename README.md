# Workshop 04 Apr 2026

## Prerequisites

1. Creare VM in [Nebius cloud](https://console.nebius.com/)
Folositi link-ul de invitatie din mail-ul primit, dupa care se urmeaza pasii de mai jos:

1. Generare cheie SSH: `ssh-keygen -f nebius`
2. Din meniul din stanga se alege `Compute > Virtual Machines`
3. Click pe `Create virtual machine`
4. Se completeaza urmatoarele:
   a. Project: `default-project-eu-north1`
   b. Name: ok cel generat
   c. Available platform: `NVIDIA H100 NVLink`
   d. Preset: `1 GPU`
   e. Public IP address: `Auto assign dynamic IP`
   f. Username and SSH key: `+ Create`. Se completeaza nume utilizator `workshop` pt login si continutul `nebius.pub` de la pasul 1

6. Instalare [Surogate](https://surogate.ai).
```shell
curl -sSL https://surogate.ai/install.sh | bash
```

## Supervised Fine-Tuning



## Reinforcement Learning (GRPO)

