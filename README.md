# adversarial-attack
Iterative least likely method from [Adversarial examples in the physical world](https://arxiv.org/pdf/1607.02533.pdf)

# Installation
clone repository  
    git clone https://github.com/mahkons/adversarial-attack
    cd adversarial-attack
create and activate conda env (optional)  
    conda env create -f environment.yml
    conda activate adversarial-attack

# Launch
    python attack.py [--show]
    --show flag determines whether adversarial images will be shown
