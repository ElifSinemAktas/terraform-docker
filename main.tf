terraform {
  required_providers {
    # We recommend pinning to the specific version of the Docker Provider you're using
    # since new versions are released frequently
    docker = {
      source  = "kreuzwerker/docker"
      version = "2.23.1"
    }
  }
}

# Configure the docker provider
provider "docker" {
}

# Create a docker image resource

resource "docker_image" "fastapi_res" {
  name = "fastapi"
  build {
    path = "."
    tag  = ["fastapi:develop"]
    build_arg = {
      name : "fastapi"
    }
    label = {
      author : "esa"
    }
  }
}

# Create a docker container resource

resource "docker_container" "fastapi" {
  name    = "fastapi"
  image   = docker_image.fastapi_res.image_id
  network_mode = "docker_mlops-net"

  ports {
    external = 8000
    internal = 8000
  }
}