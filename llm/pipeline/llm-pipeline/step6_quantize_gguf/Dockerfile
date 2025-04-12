# step6_quantize_gguf/Dockerfile
#FROM buildpack-deps:bullseye as builder  # include tools for build
#WORKDIR /app
#COPY quantize.sh .
#RUN chmod +x quantize.sh
#CMD ["./quantize.sh"]

# FROM buildpack-deps:bullseye AS builder

# WORKDIR /app
# COPY quantize.sh .

# RUN chmod +x quantize.sh

# CMD ["./quantize.sh"]
# # CMD ["ls", "-l"]

FROM buildpack-deps:bullseye

RUN apt-get update && apt-get install -y cmake build-essential git && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY quantize.sh .
RUN chmod +x quantize.sh

CMD ["./quantize.sh"]
