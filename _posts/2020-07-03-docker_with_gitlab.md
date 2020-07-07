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


