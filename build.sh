#!/bin/bash
cd foreign/build && cmake .. && make && cd ../../
cp foreign/build/lib/libvideo_review.dylib macos/