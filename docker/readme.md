- build the docker image
- run the docker image with /bin/bash
- put .env in /workspace folder
- you will also need kaggle.json from kaggle to run competitions
- sometimes when running kaggle competition, you might get error related to accepting the rules. Go to kaggle and accept the rules for the competition in that case.

---

- by default it installs from `https://github.com/jayantjha/RD-Agent.git > develop` and bakes into the image
- to install latest while job submission, use `/workspace/startup.sh` followed by job command
    - set environment variables `RDAGENT_GH_REPO` / `RDAGENT_BRANCH` to appropriate repo / branch