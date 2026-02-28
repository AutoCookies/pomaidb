# alpine image  
FROM alpine

# Install tools
RUN apk add build-base make cmake
RUN apk add git
RUN apk add vim

RUN mkdir -p  /home/dev
WORKDIR /home/dev

# Get palloc
RUN git clone https://github.com/microsoft/palloc -b dev2
RUN mkdir -p palloc/out/release
RUN mkdir -p palloc/out/debug

# Build palloc debug
WORKDIR /home/dev/palloc/out/debug
RUN cmake ../.. -DMI_DEBUG_FULL=ON
RUN make -j
RUN make test

CMD ["/bin/sh"]