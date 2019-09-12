//
// Created by tong on 2019/9/6.
//
#include <android/bitmap.h>
#include <android/log.h>
#include <string>
#include <vector>

#include "ncnn/opencv.h"
#include "MobileNetSSD.id.h"
#include <sys/time.h>
#include <unistd.h>
#include "ncnn/net.h"
#include "MobileNetSSD.h"

static ncnn::UnlockedPoolAllocator g_blob_pool_allocator;
static ncnn::PoolAllocator g_workspace_pool_allocator;

static ncnn::Mat ncnn_param;
static ncnn::Mat ncnn_bin;
static ncnn::Net ncnn_net;

JNIEXPORT jboolean JNICALL Java_org_tensorflow_demo_MobileNetSSD_Init(JNIEnv *env, jobject obj, jbyteArray param, jbyteArray bin){
  __android_log_print(ANDROID_LOG_DEBUG, "MobileNetSSD", "enter the jni func");
  //init param
  {
      int len = env->GetArrayLength(param);
      ncnn_param.create(len, (size_t)1u);
      env->GetByteArrayRegion(param, 0, len, (jbyte *)ncnn_param);
      int ret = ncnn_net.load_param((const unsigned char *)ncnn_param);
      __android_log_print(ANDROID_LOG_DEBUG, "MobileNetSSD", "load_param %d %d", ret, len);
  }

  //init bin
  {
      int len = env->GetArrayLength(bin);
      ncnn_bin.create(len, (size_t)1u);
      env->GetByteArrayRegion(bin, 0, len, (jbyte *)ncnn_bin);
      int ret = ncnn_net.load_model((const unsigned char *)ncnn_bin);
      __android_log_print(ANDROID_LOG_DEBUG, "MobileNetSSD", "load_model %d %d", ret, len);
  }

  ncnn::Option opt;
  opt.lightmode = true;
  opt.num_threads = 4;
  opt.blob_allocator = &g_blob_pool_allocator;
  opt.workspace_allocator = &g_workspace_pool_allocator;

  ncnn_net.opt = opt;

  return JNI_TRUE;
}

JNIEXPORT jfloatArray JNICALL Java_org_tensorflow_demo_MobileNetSSD_Detect(JNIEnv* env, jobject thiz, jobject bitmap){
    ncnn::Mat in;
    {
        AndroidBitmapInfo info;
        AndroidBitmap_getInfo(env, bitmap, &info);
        int width = info.width;
        int height = info.height;
        if(info.format != ANDROID_BITMAP_FORMAT_RGBA_8888)
            return NULL;

        void* indata;
        AndroidBitmap_lockPixels(env, bitmap, &indata);
        in = ncnn::Mat::from_pixels((const unsigned char*)indata, ncnn::Mat::PIXEL_RGBA2RGB, width, height);

        __android_log_print(ANDROID_LOG_DEBUG, "MobileNetSSD", "MobileNetSSD_predict_has_input1, in.w: %d; in.h: %d", in.w, in.h);
        AndroidBitmap_unlockPixels(env, bitmap);
    }

    std::vector<float> cls_scores;
    {
        const float mean_vals[3] = {127.5f, 127.5f, 127.5f};
        const float scale[3] = {0.007843f, 0.007843f, 0.007843f};

        in.substract_mean_normalize(mean_vals, scale);

        ncnn::Extractor ex = ncnn_net.create_extractor();
        ex.input(MobileNetSSD_deploy_param_id::BLOB_data, in);

        ncnn::Mat out;
        ex.extract(MobileNetSSD_deploy_param_id::BLOB_detection_out, out);

        int output_wsize = out.w;
        int output_hsize = out.h;

        jfloat *output[output_wsize * output_hsize];
        for(int i = 0; i < out.h; i++){
            for(int j = 0; j < out.w; j++){
                output[i*output_wsize + j] = &out.row(i)[j];
            }
        }
        jfloatArray jOutputData = env->NewFloatArray(output_wsize * output_hsize);
        if(jOutputData == NULL)
            return NULL;
        env->SetFloatArrayRegion(jOutputData, 0, output_wsize * output_hsize, reinterpret_cast<const jfloat *>(*output));
        return jOutputData;
    }
}