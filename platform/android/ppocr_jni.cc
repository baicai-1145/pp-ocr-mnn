// pp-ocr-mnn — Android JNI wrapper for the C ABI.
//
// This file is the only piece of pp-ocr-mnn that uses #ifdef
// __ANDROID__ — the rest of the codebase is platform-agnostic.
// The thin C ABI (include/ppocr/ppocr.h) is the contract, and
// this file is a pure trampoline:
//
//   Java_ppocr_Ppocr_create   → ppocr_create
//   Java_ppocr_Ppocr_run      → ppocr_run
//   Java_ppocr_Ppocr_runFile  → ppocr_run_file
//   Java_ppocr_Ppocr_destroy  → ppocr_destroy
//
// Image data plumbing: the user passes a Bitmap and we extract
// the ARGB_8888 pixel buffer via GetIntArrayElements + Bitmap
// pixel copy, then convert ARGB_8888 → BGR (the order the
// ppocr_run() C ABI expects) into a JNI-allocated byte array.
// The result text is returned as a String[] (one per line); the
// poly / score fields are exposed as a jobjectArray of String[8]
// (the four TL,TR,BR,BL corner points) and a float[].
//
// Java side, see platform/android/README.md for the full class
// layout. The Java declarations are:
//
//   package ppocr;
//   public class Ppocr {
//     public Ppocr(String modelDir, String detName, String recName);
//     public Line[] run(Bitmap bmp);            // rec on, det+rec
//     public Line[] runFile(String path);       // file-based
//     public void close();                      // → ppocr_destroy
//     public static class Line { int[] poly; String text; float score; }
//   }
//
// The class is declared in the Java side stub (NOT shipped here —
// the user is expected to drop a `ppocr/Ppocr.java` into their
// Android project's `src/main/java/`). We expose only the native
// method signatures; the Java class itself is trivial.

#include <jni.h>
#include <android/log.h>
#include <stdint.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>

#include "ppocr/ppocr.h"

#define LOG_TAG "ppocr-jni"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO,  LOG_TAG, __VA_ARGS__)
#define LOGW(...) __android_log_print(ANDROID_LOG_WARN,  LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)

// ---- helpers --------------------------------------------------------------
// Convert an ARGB_8888 int buffer (Android Bitmap pixels) into a
// BGR byte buffer that the C ABI's ppocr_run(bgr, w, h) expects.
// The C ABI input is documented as "BGR interleaved uint8,
// w*h*3 bytes" (see include/ppocr/ppocr.h).
//
// We do this in-place on a freshly malloc'd buffer of size w*h*3.
// On big-endian we would need byte-swaps; Android is little-endian
// (arm64-v8a, x86, x86_64 are all LE), so a direct shift is safe.
static uint8_t* argb_to_bgr(const uint32_t* argb, int w, int h) {
  size_t n = (size_t)w * (size_t)h;
  uint8_t* bgr = (uint8_t*)malloc(n * 3);
  if (!bgr) return NULL;
  for (size_t i = 0; i < n; ++i) {
    uint32_t p = argb[i];
    bgr[3 * i + 0] = (uint8_t)(p & 0xFF);          // B
    bgr[3 * i + 1] = (uint8_t)((p >> 8) & 0xFF);   // G
    bgr[3 * i + 2] = (uint8_t)((p >> 16) & 0xFF);  // R
  }
  return bgr;
}

// Throw a Java RuntimeException with a C string message. The
// caller is responsible for returning (via a `return NULL;` or
// equivalent) right after the throw.
static void throw_runtime(JNIEnv* env, const char* msg) {
  jclass cls = env->FindClass("java/lang/RuntimeException");
  if (cls) env->ThrowNew(cls, msg ? msg : "ppocr error");
  LOGE("exception: %s", msg ? msg : "(null)");
}

// ---- native methods -------------------------------------------------------
//
// Java_ppocr_Ppocr_create: long create(String modelDir, String detName,
//                                      String recName, int backend, int numThreads)
//
// Returns a non-zero handle (the ppocr_engine* cast to jlong) on
// success, 0 on failure (and a RuntimeException is thrown).
JNIEXPORT jlong JNICALL
Java_ppocr_Ppocr_create(JNIEnv* env, jclass cls, jstring jModelDir,
                        jstring jDetName, jstring jRecName,
                        jint backend, jint numThreads) {
  (void)cls;
  const char* model_dir = env->GetStringUTFChars(jModelDir, NULL);
  const char* det_name  = env->GetStringUTFChars(jDetName, NULL);
  const char* rec_name  = env->GetStringUTFChars(jRecName, NULL);

  ppocr_config cfg;
  memset(&cfg, 0, sizeof(cfg));
  cfg.model_dir   = model_dir;
  cfg.det_name    = det_name;
  cfg.rec_name    = rec_name;
  // For an Android app the model_dir is typically the app's
  // getFilesDir() or getExternalFilesDir(); we never want auto-
  // download from a phone, so the user should pass
  // PPORC_MNN_MIRROR unset and rely on files already on disk.
  cfg.download    = 0;       // see platform/android/README.md
  cfg.offline     = 1;       // hard-fail on missing
  cfg.backend     = (ppocr_backend)backend;
  cfg.num_threads = (int)numThreads;
  cfg.rec_batch   = 8;

  ppocr_engine* e = NULL;
  char err_buf[256] = {0};
  ppocr_status st = ppocr_create(&cfg, &e, err_buf, sizeof(err_buf));

  env->ReleaseStringUTFChars(jModelDir, model_dir);
  env->ReleaseStringUTFChars(jDetName,  det_name);
  env->ReleaseStringUTFChars(jRecName,  rec_name);

  if (st != PPOCR_OK) {
    throw_runtime(env, err_buf[0] ? err_buf : ppocr_status_string(st));
    return 0;
  }
  return (jlong)(uintptr_t)e;
}

// Java_ppocr_Ppocr_destroy: void destroy(long handle)
JNIEXPORT void JNICALL
Java_ppocr_Ppocr_destroy(JNIEnv* env, jclass cls, jlong handle) {
  (void)env; (void)cls;
  if (handle == 0) return;
  ppocr_engine* e = (ppocr_engine*)(uintptr_t)handle;
  ppocr_destroy(e);
}

// Java_ppocr_Ppocr_run: Line[] run(long handle, Bitmap bmp)
//
// Reads the Bitmap's ARGB_8888 buffer, converts to BGR, calls
// ppocr_run, and packages the result as a jobjectArray of
// "ppocr/Line" objects (each with int[8] poly, String text,
// float score).
JNIEXPORT jobjectArray JNICALL
Java_ppocr_Ppocr_run(JNIEnv* env, jclass cls, jlong handle, jobject bitmap) {
  (void)cls;
  if (handle == 0 || !bitmap) {
    throw_runtime(env, "Ppocr.run: null handle or bitmap");
    return NULL;
  }
  ppocr_engine* e = (ppocr_engine*)(uintptr_t)handle;

  // Read the bitmap's dimensions and pixel buffer.
  // Bitmap.getWidth() / getHeight() are method IDs cached on the
  // class for the lifetime of the JVM. We resolve them per-call
  // here for clarity; the JNI overhead is negligible compared to
  // the inference.
  jclass bmp_cls = env->GetObjectClass(bitmap);
  jmethodID mid_getW = env->GetMethodID(bmp_cls, "getWidth", "()I");
  jmethodID mid_getH = env->GetMethodID(bmp_cls, "getHeight", "()I");
  if (!mid_getW || !mid_getH) {
    throw_runtime(env, "Ppocr.run: Bitmap.getWidth/getHeight not found");
    return NULL;
  }
  int w = (int)env->CallIntMethod(bitmap, mid_getW);
  int h = (int)env->CallIntMethod(bitmap, mid_getH);

  // Read ARGB_8888 into a jintArray (we use a temporary int[] and
  // let the JVM allocate it; or we can copy into a heap int*
  // directly via Bitmap's getPixels). Direct getPixels is the
  // fast path but requires a C-style int* — Android's NDK does
  // provide a `AndroidBitmap_lockPixels` path, but the JNI
  // approach keeps this file portable to any Java caller.
  jintArray jbuf = env->NewIntArray(w * h);
  if (!jbuf) {
    throw_runtime(env, "Ppocr.run: NewIntArray OOM");
    return NULL;
  }
  // Bitmap.getPixels(int[] pixels, int offset, int stride,
  //                   int x, int y, int width, int height)
  jmethodID mid_getPx = env->GetMethodID(bmp_cls, "getPixels",
      "([IIIIIII)V");
  if (!mid_getPx) {
    throw_runtime(env, "Ppocr.run: Bitmap.getPixels not found");
    return NULL;
  }
  env->CallVoidMethod(bitmap, mid_getPx, jbuf, 0, w, 0, 0, w, h);

  jint* argb = env->GetIntArrayElements(jbuf, NULL);
  if (!argb) {
    throw_runtime(env, "Ppocr.run: GetIntArrayElements failed");
    return NULL;
  }
  uint8_t* bgr = argb_to_bgr((const uint32_t*)argb, w, h);
  env->ReleaseIntArrayElements(jbuf, argb, JNI_ABORT);
  if (!bgr) {
    throw_runtime(env, "Ppocr.run: argb_to_bgr OOM");
    return NULL;
  }

  ppocr_result* res = NULL;
  ppocr_status st = ppocr_run(e, bgr, w, h, &res);
  free(bgr);

  if (st != PPOCR_OK) {
    throw_runtime(env, ppocr_status_string(st));
    return NULL;
  }

  // Build jobjectArray of ppocr/Line. We resolve the Line class
  // and constructor at call time; for hot paths this should be
  // cached in a static initializer (out of scope for the v1
  // trampoline).
  jclass line_cls = env->FindClass("ppocr/Line");
  if (!line_cls) {
    throw_runtime(env, "Ppocr.run: class ppocr/Line not found");
    return NULL;
  }
  jmethodID line_ctor = env->GetMethodID(line_cls, "<init>",
      "([ILjava/lang/String;F)V");
  if (!line_ctor) {
    throw_runtime(env, "Ppocr.run: Line.<init>([I,String,F)V not found");
    return NULL;
  }

  jobjectArray arr = env->NewObjectArray(res->n_lines, line_cls, NULL);
  for (int i = 0; i < res->n_lines; ++i) {
    const ppocr_line* ln = &res->lines[i];
    jintArray jpoly = env->NewIntArray(8);
    env->SetIntArrayRegion(jpoly, 0, 8, (const jint*)ln->poly);
    jstring jtext = env->NewStringUTF(ln->text ? ln->text : "");
    jobject line = env->NewObject(line_cls, line_ctor, jpoly, jtext,
                                  (jfloat)ln->score);
    env->SetObjectArrayElement(arr, i, line);
    // NewLocalRef objects are freed when the JNI frame returns;
    // explicit DeleteLocalRef keeps the JNI local frame from
    // overflowing on large results.
    env->DeleteLocalRef(jpoly);
    env->DeleteLocalRef(jtext);
    env->DeleteLocalRef(line);
  }
  env->DeleteLocalRef(line_cls);
  return arr;
}

// Java_ppocr_Ppocr_runFile: Line[] runFile(long handle, String path)
//
// File-based variant. The path should be readable by the app
// (e.g. /data/data/<pkg>/files/foo.jpg) — the C ABI calls
// stb_image internally, no Android-specific code.
JNIEXPORT jobjectArray JNICALL
Java_ppocr_Ppocr_runFile(JNIEnv* env, jclass cls, jlong handle, jstring jPath) {
  (void)cls;
  if (handle == 0 || !jPath) {
    throw_runtime(env, "Ppocr.runFile: null handle or path");
    return NULL;
  }
  const char* path = env->GetStringUTFChars(jPath, NULL);
  ppocr_engine* e = (ppocr_engine*)(uintptr_t)handle;

  ppocr_result* res = NULL;
  ppocr_status st = ppocr_run_file(e, path, &res);
  env->ReleaseStringUTFChars(jPath, path);

  if (st != PPOCR_OK) {
    throw_runtime(env, ppocr_status_string(st));
    return NULL;
  }

  // Same Line[] packing as Java_ppocr_Ppocr_run; copy-pasted
  // for clarity (a single helper would force an internal C struct
  // we don't need to expose).
  jclass line_cls = env->FindClass("ppocr/Line");
  jmethodID line_ctor = env->GetMethodID(line_cls, "<init>",
      "([ILjava/lang/String;F)V");
  jobjectArray arr = env->NewObjectArray(res->n_lines, line_cls, NULL);
  for (int i = 0; i < res->n_lines; ++i) {
    const ppocr_line* ln = &res->lines[i];
    jintArray jpoly = env->NewIntArray(8);
    env->SetIntArrayRegion(jpoly, 0, 8, (const jint*)ln->poly);
    jstring jtext = env->NewStringUTF(ln->text ? ln->text : "");
    jobject line = env->NewObject(line_cls, line_ctor, jpoly, jtext,
                                  (jfloat)ln->score);
    env->SetObjectArrayElement(arr, i, line);
    env->DeleteLocalRef(jpoly);
    env->DeleteLocalRef(jtext);
    env->DeleteLocalRef(line);
  }
  env->DeleteLocalRef(line_cls);
  return arr;
}
