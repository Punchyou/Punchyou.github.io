---
layout: post
title: Get Started with Docker
author: Maria Pantsiou
date: '2020-07-03 14:35:23 +0530'
category: devops
summary: Get Started with Docker
thumbnail:

---

This is how-to-get-started article about docker. I fund the idea of containerization, but I haven't used it before, so the best way for me to learn something is to write about it!

### What is Docker?

Docker is an open platform for developing, shipping, and running applications. Docker enables you to separate your applications from your infrastructure so you can deliver software quickly. It provides the ability to package and run an application in a loosely isolated environment called a container. The isolation and security allow you to run many containers simultaneously on a given host. Containers are lightweight because they don’t need the extra load of a hypervisor, but run directly within the host machine’s kernel.

The lifecycle :
- Develop your application and its supporting components using containers.
- The container becomes the unit for distributing and testing your application.
- When you’re ready, deploy your application into your production environment, as a container or an orchestrated service. This works the same whether your production environment is a local data center, a cloud provider, or a hybrid of the two.

Containers are great for continuous integration and continuous delivery (CI/CD) workflows. Docker’s portability and lightweight nature also make it easy to dynamically manage workloads, scaling up or tearing down applications and services as business needs dictate, in near real time. It is perfect for **high density environments and for small and medium deployments** where you need to do more with fewer resources.


### Why Docker?
Docker is:
1. **Flexible**: Even the most complex applications can be containerized.
2. **Lightweight**: Containers leverage and share the host kernel, making them much more efficient in terms of system resources than virtual machines.
3. **Portable**: You can build locally, deploy to the cloud, and run anywhere.
4. **Loosely coupled**: Containers are highly self sufficient and encapsulated, allowing you to replace or upgrade one without disrupting others.
5. **Scalable**: You can increase and automatically distribute container replicas across a datacenter.
6. **Secure**: Containers apply aggressive constraints and isolations to processes without any configuration required on the part of the user.

### The Docker Engine

Docker Engine is a client-server application with these major components:

1. **A server** which is a type of long-running program called a **daemon process** (the `dockerd` command)

2. **A REST API** which specifies interfaces that programs can use to talk to the daemon and instruct it what to do

3. **A command line interface** (CLI) client (the `docker` command)


<div align="center">
<img src="/assets/img/posts/docker/engine-components-flow.png" alt="engine-components-flow">
</div>

The Docker client talks to the Docker daemon, which does the heavy lifting of building, running, and distributing your Docker containers.The Docker client and daemon can run on the same system, or you can connect a Docker client to a remote Docker daemon.

When you use docker client commands such as `docker run`, the client sends these commands to dockerd, which carries them out. The docker command uses the Docker API. The Docker client can communicate with more than one daemon.

#### The Archtecture

<div align="center">
<img src="/assets/img/posts/docker/architecture.svg" alt="engine-components-flow">
</div>

## Docker registries

A Docker registry stores Docker images. Docker Hub is a public registry that anyone can use, and Docker is configured to **look for images on Docker Hub** by default. You can even run your own private registry. If you use Docker Datacenter (DDC), it includes Docker Trusted Registry (DTR).

##### How it works
When you use the `docker pull` or `docker run commands`, the required images are pulled from your configured registry. When you use the `docker push` command, your image is pushed to your configured registry.

## Docker Objects

### Images
An image includes everything needed to run an application - the code or binary, runtimes, dependencies, and any other filesystem objects required. Is a read-only **template with instructions** for creating a Docker container. Often, an image is based on another image (like an Ubuntu image), with some additional customization. 

To build your own image, you create a Dockerfile with a simple syntax for defining the steps needed to create the image and run it. **Each instruction** in a Dockerfile creates **a layer** in the image. When you change the Dockerfile and **rebuild** the image, **only those layers which have changed** are rebuilt.

### Containers
Fundamentally, a container is nothing but a running process, with some added encapsulation features applied to it in order to keep it isolated from the host and from other containers. A container is a **runnable instance of an image**. You can create, start, stop, move, or delete a container using the Docker API or CLI. You can connect a container to one or more networks, attach storage to it, or even create a new image based on its current state.

A container is relatively well isolated from other containers and its host machine by default, but this is contolable by the user. A container is defined by its image as well as any configuration options you provide to it when you create or start it. **When a container is removed**, any changes to its state that are not stored in persistent storage **disappear**.

#### Differences from VMs:
A container runs natively on Linux and **shares the kernel of the host** machine with other containers .By contrast, a virtual machine (VM) runs a full-blown “guest” operating system with virtual access to host resources through a hypervisor. 

### What does the Docker do when create a container?
We create a container by running
```bash
docker run -i -t ubuntu /bin/bash
```
for an Ubuntu image.

The Docker:
1. Pulls it from your configured registry, if there isn't an image locally (`docker pull ubuntu` command)

2. Creates a new container (`docker container create`command)

3. Allocates a read-write filesystem to the container, as its final layer

4. Creates a network interface to connect the container to the default network. This includes assigning an IP address to the container. By default, containers can connect to external networks using the host machine’s network connection.

5. Starts the container and executes `/bin/bash`. Because the container is running interactively and attached to your terminal (due to the -i and -t flags), you can provide input using your keyboard while the output is logged to your terminal.

`exit` terminates the `/bin/bash` command, the container stops but is not removed. You can start it again or remove it.

### Behind the scenes
#### Namespaces
Docker uses a technology called **namespaces** to provide the isolated workspace called the container. When you run a container, Docker creates a set of namespaces for that container. Each aspect of a container runs in a separate namespace and its access is limited to that namespace.

#### Control groups (`cgroups`)
A cgroup limits an application to a specific set of resources. Control groups allow Docker Engine to share available hardware resources to containers and optionally enforce limits and constraints. For example, you can limit the memory available to a specific container.

#### Union file systems

Union file systems, or UnionFS, are file systems that operate by creating layers, making them very lightweight and fast. 

#### Container format

Docker Engine combines the namespaces, control groups, and UnionFS into a wrapper called a container format. The default container format is `libcontainer`.


# Getting Sterted with Docker
## Docker Set Up
I'm setting up docker for for a linux `Ubuntu 18.04 bionic` distribution VM, but you can find more information on how to download and install it on Windows OS and Mac OS [here](https://docs.docker.com/get-started/).

On linux Ubuntu you can uninstall previous version, if any, by running:

```bash
$ sudo apt-get remove docker docker-engine docker.io containerd runc
```

1. #### Set Up the Repository
Before you install Docker Engine for the first time on a new host machine, you need to set up the Docker repository:

```bash
$ sudo apt-get update
$ sudo apt-get install \
    apt-transport-https \
    ca-certificates \
    curl \
    gnupg-agent \
    software-properties-common
```

Add Add Docker’s official GPG key:
```bash
$ curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -

OK
```

Verify that you now have the key with the fingerprint `9DC8 5822 9FC7 DD38 854A  E2D8 8D81 803C 0EBF CD88`, by searching for the last 8 characters of the fingerprint.

```bash
$ sudo apt-key fingerprint 0EBFCD88

pub   rsa4096 2017-02-22 [SCEA]
      9DC8 5822 9FC7 DD38 854A  E2D8 8D81 803C 0EBF CD88
uid           [ unknown] Docker Release (CE deb) <docker@docker.com>
sub   rsa4096 2017-02-22 [S]
```
Now, set up the [stable repository](https://docs.docker.com/engine/install/) (for amd64 architecture):

```bash
$ sudo add-apt-repository \
    "deb [arch=amd64] https://download.docker.com/linux/ubuntu \
    $(lsb_release -cs) \
    stable"
```

2. #### Install the latest version of Docker Engine:
```bash
$ sudo apt-get update
$ sudo apt-get install docker-ce docker-ce-cli containerd.io
 ```

**Optional**: If you want to install or update a specific version, you can find all the versions on your repo by running:

```bash
$ apt-cache madison docker-ce

docker-ce | 5:19.03.12~3-0~ubuntu-bionic | https://download.docker.com/linux/ubuntu bionic/stable amd64 Packages
docker-ce | 5:19.03.11~3-0~ubuntu-bionic | https://download.docker.com/linux/ubuntu bionic/stable amd64 Packages
docker-ce | 5:19.03.10~3-0~ubuntu-bionic | https://download.docker.com/linux/ubuntu bionic/stable amd64 Packages
...
```
You can install a specific version by using the version string from the second column, for example:

```bash
sudo apt-get install docker-ce=5:19.03.12~3-0~ubuntu-bionic docker-ce-cli=5:19.03.12~3-0~ubuntu-bionic containerd.io
```

3. #### Verify that Docker Engine is installed

Run the hello-world image:

```bash
$ sudo docker run hello-world


Unable to find image 'hello-world:latest' locally
latest: Pulling from library/hello-world
...
Hello from Docker!
This message shows that your installation appears to be working correctly.
...
```
Docker Engine is installed and running. The docker group is created but no users are added to it. You need to use sudo to run Docker commands.

Find how you can install form package [here](https://docs.docker.com/engine/install/ubuntu/).

Now we can run an Ubuntu container by running:
```bash
$ docker run -it ubuntu bash
```

## Uninstall Docker Engine

Uninstall the Docker Engine, CLI, and Containerd packages:
```bash
$ sudo apt-get purge docker-ce docker-ce-cli containerd.io
```
Images, containers, volumes, or customized configuration files on your host are not automatically removed. To delete all images, containers, and volumes:
```bash
$ sudo rm -rf /var/lib/docker
```
You must delete any edited configuration files manually.


### Docker commands
To enter a running docker you must include your `user` in the `docker` group. To do that run:

```bash
$ sudo usermod -aG  docker $USER
$ sudo su -
$ su $USER
$ groups
```
The three last commands are for logging the user out and back in. You now shoud see "docker" in the list of groups there.


To see the list of the docker containers on your machine run:
```bash
$ docker container ls -all
```
and to see the running containers:
```bash
$ docker container ps
```

List of docker images:
```bash
$ docker images
```

Start a container from an image:
```bash
$ docker start [CONTAINER ID]
```

Run an image:
```bash
$ docker run -it ubuntu [COMMAND]
```

Exit a container, but keep it:
```bash
$ exit
```


*sources:*
1. [Docker overview](https://docs.docker.com/get-started/overview/)
2. [Orientation and setup](https://docs.docker.com/get-started/)