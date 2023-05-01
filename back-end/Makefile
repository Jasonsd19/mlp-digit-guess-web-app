build:
	cmake -S . -B ./build
.PHONY: build

start:
	cd ./build && make && clear && ./main
.PHONY: start

train:
	cd ./build && cd ./train && make && clear && ./train
.PHONY: train