all: level_set_segmentation

level_set_segmentation:
	nvcc -arch=sm_61 -rdc=true level_set_segmentation.cu -o lss -lcudadevrt

clean:
	rm lss
