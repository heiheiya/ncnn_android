/* DO NOT EDIT THIS FILE - it is machine generated */
#include <jni.h>
/* Header for class org_tensorflow_demo_MobileNetSSD */

#ifndef _Included_org_tensorflow_demo_MobileNetSSD
#define _Included_org_tensorflow_demo_MobileNetSSD
#ifdef __cplusplus
extern "C" {
#endif
/*
 * Class:     org_tensorflow_demo_MobileNetSSD
 * Method:    Init
 * Signature: ([B[B)Z
 */
JNIEXPORT jboolean JNICALL Java_org_tensorflow_demo_MobileNetSSD_Init
  (JNIEnv *, jobject, jbyteArray, jbyteArray);

/*
 * Class:     org_tensorflow_demo_MobileNetSSD
 * Method:    Detect
 * Signature: (Ljava/lang/Object;)[F
 */
JNIEXPORT jfloatArray JNICALL Java_org_tensorflow_demo_MobileNetSSD_Detect
  (JNIEnv *, jobject, jobject);

#ifdef __cplusplus
}
#endif
#endif
