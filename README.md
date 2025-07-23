# CallAgent

## Installation
### Windows:
    GitBash:
        Open Powershell as Admin and:
            choco install just
        
        Open gitbash under folder where you want to clone:
            source /c/ProgramData/miniconda3/etc/profile.d/conda.sh 

        Create Python venv or Conda env:
            conda create -n ultravox python=3.11
            conda activate ultravox
        
        Setup project:
            pip install tensorflow-io-gcs-filesystem
            git clone https://github.com/fixie-ai/ultravox.git
            cd ultravox
            just install
        
    WSL:
        Create Python venv or Conda env:
            conda create -n ultravox python=3.11
            conda activate ultravox

        Clone and Setup:
            git clone https://github.com/fixie-ai/ultravox.git
            cd ultravox
            just install
        
### Linux:
    Create Python venv or Conda env:
        conda create -n ultravox python=3.11
        conda activate ultravox

    Clone and Setup:
        git clone https://github.com/fixie-ai/ultravox.git
        cd ultravox
        just install
