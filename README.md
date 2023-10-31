## Official codebase for Delta

---

### Overview
Private data are decompose into private and public parts.
The private part is fed to a small private model; 
the public part is perturbed and then fed to a large model in public environments (GPUs).

Therefore, privacy is protected in the private environment, 
while computing performance is achieved by leveraging fast GPUs.

<img src="figure/framework.png" alt="drawing" width="500" class="center" />

Note: Current release is only an implementation for testing the model performance using GPUs only.
Actual deployment with TEEs will be released in the future.

---

### Environment

```
python=3.9  
torch=1.10  
numpy=1.21
```

---

### Run

#### On CIFAR

```bash
# Training hparams are included in run.sh
bash run.sh
```

#### On ImageNet

```bash
# Training hparams are included in run_imagenet.sh
bash run_imagenet.sh
```

