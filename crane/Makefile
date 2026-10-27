CC = xcrun clang
PYTHON ?= python3
CFLAGS = -O2 -Wall -Wno-deprecated-declarations -fobjc-arc -fPIC
FRAMEWORKS = -framework Foundation -framework IOSurface -framework Metal -ldl
BRIDGE_DIR = src
TARGET = $(BRIDGE_DIR)/libane_bridge.dylib

all: $(TARGET)

$(TARGET): $(BRIDGE_DIR)/ane_bridge.m $(BRIDGE_DIR)/ane_bridge.h
	$(CC) $(CFLAGS) -dynamiclib -o $@ $(BRIDGE_DIR)/ane_bridge.m $(FRAMEWORKS)

clean:
	rm -f $(TARGET)

test:
	cd .. && $(PYTHON) -m pytest -q \
		tests/test_coreml_surface.py \
		tests/test_surface_session.py \
		tests/test_ggml_transport_native_surface.py \
		tests/test_mac_fastvlm_coreml_native.py

.PHONY: all clean test
