# a2c_football
Google Research football environment \
paper: Kurach et al., Google Research Football: A Novel Reinforcement Learning Environment, AAAI-2020
(https://arxiv.org/pdf/1907.11180.pdf)

## Pre-requisit on your server
- If you can deal with docker, use our Dockerfile and set your server easily.

1. Set your server (ONLY work for Ubuntu18.04)
```
sudo apt-get install git cmake build-essential libgl1-mesa-dev libsdl2-dev \
libsdl2-image-dev libsdl2-ttf-dev libsdl2-gfx-dev libboost-all-dev \
libdirectfb-dev libst-dev mesa-utils xvfb x11vnc
```

2. Install gfootball
```
git clone -b v2.3 https://github.com/google-research/football.git
cd football
pip3 install .
```

3. torch installation

4. Install requirements
