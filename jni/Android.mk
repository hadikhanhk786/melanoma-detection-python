LOCAL_PATH := $(call my-dir)

include $(CLEAR_VARS)

#OPENCV_CAMERA_MODULES:=on
#OPENCV_INSTALL_MODULES:=on

include $(OPENCV_SDK)/sdk/native/jni/OpenCV.mk

LOCAL_MODULE    := active_contour
LOCAL_SRC_FILES := jni_part.cpp ac_withoutedges_yuv.cpp activecontour.cpp linked_list.cpp
LOCAL_LDLIBS +=  -llog -ldl

include $(PREBUIT_SHARED_LIBRARY)

LOCAL_SHARED_LIBRARY := active_contour 

include $(BUILD_SHARED_LIBRARY)
