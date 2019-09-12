package org.tensorflow.demo;

import android.graphics.Bitmap;

import org.tensorflow.demo.env.Logger;

import java.lang.annotation.Native;

public class MobileNetSSD {
    private static final Logger LOGGER = new Logger();
    public native boolean Init(byte[] param, byte[] bin);
    public native float[] Detect(Object bitmap);
    static  {
        System.loadLibrary("MobileNetSSD");
        LOGGER.e("loadLibrary libMobileNetSSD.so");
    }
}
