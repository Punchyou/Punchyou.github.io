---
layout: post
title: Building Docker Images with GitLab CI/CD
author: Maria Pantsiou
date: '2020-07-03 14:35:23 +0530'
category: devops
summary: Building Docker Images with GitLab CI/CD
thumbnail:

---
## Building Docker Images with GitLab CI/CD

*I've recently writter getting-started articles on CI/CD with GitLab and Docker, so here I combine both. In this article I will have a go with creating a docker container as a part of a CI/CD pipeline to test my applications.*

The steps that this process follow are:

1. Create an application image
2. Run tests against the created image
3. Push image to a remote registry
4. Deploy to a server from the pushed image


After [configuring a gitlab pipeline](https://punchyou.github.io/devops/2020/07/02/ci_cd_gitlab/) and [installing docker](https://punchyou.github.io/devops/2020/07/03/dicker_get_started/#/) you can build or download a docker images within the CI/CD pipeline you've created on gitlab.

### Download a Docker image
Configure gitlab to download an existing image that containes most of the packages you want, by adding commands to the scripts of the `.gitlab-ci.yml`.

1. Choose in image from the [docker hub](https://hub.docker.com). I use the [python image](https://hub.docker.com/_/python)
2. Download the image:
```bash
docker pull python
```
3. Add the following under the `build` job in `.gitlab-ci.yml`:
```bash
image: python
```



## Steps

1. Add `gitlab-runner` user to the `docker` group (granting `gitlab-runner` full root permissions). On your terminal run:

```bash
$ sudo usermod -aG docker gitlab-runner
```

2. Verify that `gitllab-runner` has acces to Docker:

```bash
$ sudo -u gitlab-runner -H docker info


Client:
 Debug Mode: false

Server:
 Containers: 19
  Running: 0
  Paused: 0
  Stopped: 19
 Images: 3
...
```

3. Verify that everything works by adding `docker info` to `.gitlab-ci.yml` configuration file you'll find in your projects root directory:

```yml
before_script:
  - docker info

build_image:
  script:
    - docker build -t my-docker-image .
    - docker run my-docker-image /script/to/run/tests
```

Notes:

1. GitLab logs:
```bash
/var/log/syslog
```
2. When register a runner, choose `docker` as the `executor`