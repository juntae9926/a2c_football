# a2c_football
Using Advanced-Actor-Critic on Google Research football environment \
Environment paper: Kurach et al., Google Research Football: A Novel Reinforcement Learning Environment, AAAI-2020
(https://arxiv.org/pdf/1907.11180.pdf)

![football](https://user-images.githubusercontent.com/79918796/206921146-676bbc8e-bb0b-4331-a894-ae704fab02de.gif)

                                                                             
## Pre-requisit on your server![football]

- If you can deal with docker, use our Dockerfile and set your server easily.

1. Docker build
```
git clone https://github.com/juntae9926/a2c_football.git

vi docker-compose.yml
container_name: "your container name"
    ports:
      - "18100:22"    # your ssh port
      - "18101:6006"  # your tensorboard port
      - "18102:8000"  # your web port
      - "18103:8888"  # your jupyter-notebook port
:wq # save

docker-compose up -d
```

2. Install gfootball
```
docker attach "your container name"
git clone -b v2.3 https://github.com/google-research/football.git
cd football
pip3 install .
```

3. Install requirements
```
pip install -r requirements.txt
```

4. running code
```
# jupyter notebook
jupyter-notebook
your ip address:8888
password : root

# console
python main.py
```

5. model trained by a2c method
```
pip install gdown
gdown https://drive.google.com/uc?id=1KAVM8LDwZHzhU267TbnGd9wU4TuA4WeH
```

https://drive.google.com/file/d/1KAVM8LDwZHzhU267TbnGd9wU4TuA4WeH/view?usp=share_link

6. Run demo.ipynb jupyter notebook with your trained model
