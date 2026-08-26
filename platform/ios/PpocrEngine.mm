// ppocr-mnn — Objective-C wrapper around the public C ABI.
//
// This is the .mm file that the user adds to their iOS / macOS
// app project (or that gets compiled by the Swift Package
// described in platform/ios/Package.swift). The .mm extension
// is required because we mix Objective-C and C++ (the public
// C ABI is C; the file is .mm to get ARC / ObjC method-call
// syntax / block support).
//
// The wrapper is intentionally thin: it just hands the
// ppocr_engine handle around and converts the ppocr_result
// into an NSArray<NSDictionary*> for the Swift side. We do
// not implement Metal or CoreML wiring here — that lives in
// src/ppocr.cpp::pickBackend(), which the MNN build enables
// with MNN_METAL=ON / MNN_COREML=ON. This file is purely
// the Swift bridge.

#import <Foundation/Foundation.h>
#include "ppocr/ppocr.h"

@interface PpocrEngine : NSObject {
@public
  ppocr_engine* _handle;
}
- (instancetype)initWithModelDir:(NSString*)modelDir
                         detName:(NSString*)detName
                         recName:(NSString*)recName
                         backend:(NSInteger)backend
                     numThreads:(NSInteger)numThreads
                          error:(NSError**)error;
- (NSArray<NSDictionary*>*)runFile:(NSString*)path
                             error:(NSError**)error;
- (NSArray<NSDictionary*>*)runBGR:(NSData*)bgrData
                              width:(NSInteger)w
                             height:(NSInteger)h
                              error:(NSError**)error;
- (void)close;
@end

@implementation PpocrEngine

- (instancetype)initWithModelDir:(NSString*)modelDir
                         detName:(NSString*)detName
                         recName:(NSString*)recName
                         backend:(NSInteger)backend
                     numThreads:(NSInteger)numThreads
                          error:(NSError**)error {
  self = [super init];
  if (!self) return nil;
  _handle = NULL;

  ppocr_config cfg;
  memset(&cfg, 0, sizeof(cfg));
  cfg.model_dir   = [modelDir UTF8String];
  cfg.det_name    = detName ? [detName UTF8String] : NULL;
  cfg.rec_name    = recName ? [recName UTF8String] : NULL;
  cfg.backend     = (ppocr_backend)backend;
  cfg.num_threads = (int)numThreads;
  cfg.rec_batch   = 8;
  // iOS apps usually bundle models in the .app and read-only
  // load them — auto-download is off by default. Same rationale
  // as the Android JNI shim.
  cfg.download    = 0;
  cfg.offline     = 1;

  char err_buf[256] = {0};
  ppocr_status st = ppocr_create(&cfg, &_handle, err_buf, sizeof(err_buf));
  if (st != PPOCR_OK) {
    if (error) {
      *error = [NSError errorWithDomain:@"Ppocr"
                                   code:st
                               userInfo:@{NSLocalizedDescriptionKey:
                                          [NSString stringWithUTF8String:
                                            err_buf[0] ? err_buf
                                                       : ppocr_status_string(st)]}];
    }
    return nil;
  }
  return self;
}

- (NSArray<NSDictionary*>*)runFile:(NSString*)path
                             error:(NSError**)error {
  ppocr_result* res = NULL;
  ppocr_status st = ppocr_run_file(_handle, [path UTF8String], &res);
  if (st != PPOCR_OK) {
    if (error) {
      *error = [NSError errorWithDomain:@"Ppocr"
                                   code:st
                               userInfo:@{NSLocalizedDescriptionKey:
                                          [NSString stringWithUTF8String:
                                            ppocr_status_string(st)]}];
    }
    return nil;
  }
  return [PpocrEngine linesToArray:res];
}

- (NSArray<NSDictionary*>*)runBGR:(NSData*)bgrData
                              width:(NSInteger)w
                             height:(NSInteger)h
                              error:(NSError**)error {
  ppocr_result* res = NULL;
  ppocr_status st = ppocr_run(_handle,
                              (const uint8_t*)bgrData.bytes,
                              (int)w, (int)h, &res);
  if (st != PPOCR_OK) {
    if (error) {
      *error = [NSError errorWithDomain:@"Ppocr"
                                   code:st
                               userInfo:@{NSLocalizedDescriptionKey:
                                          [NSString stringWithUTF8String:
                                            ppocr_status_string(st)]}];
    }
    return nil;
  }
  return [PpocrEngine linesToArray:res];
}

- (void)close {
  if (_handle) {
    ppocr_destroy(_handle);
    _handle = NULL;
  }
}

- (void)dealloc {
  [self close];
}

// Convert ppocr_result -> NSArray<NSDictionary*>. Each line is
// { "poly": [NSNumber x8], "text": NSString, "score": NSNumber }.
// The dictionary keys match the JSON output of ppocr_cli so
// the Swift side can be data-driven (Codable from JSON).
+ (NSArray<NSDictionary*>*)linesToArray:(const ppocr_result*)res {
  NSMutableArray* out = [NSMutableArray arrayWithCapacity:res->n_lines];
  for (int i = 0; i < res->n_lines; ++i) {
    const ppocr_line* ln = &res->lines[i];
    NSMutableArray* poly = [NSMutableArray arrayWithCapacity:8];
    for (int j = 0; j < 8; ++j) {
      [poly addObject:@(ln->poly[j])];
    }
    [out addObject:@{
      @"poly"  : poly,
      @"text"  : ln->text ? [NSString stringWithUTF8String:ln->text] : @"",
      @"score" : @(ln->score),
      @"det_ms": @(res->det_ms),
      @"rec_ms": @(res->rec_ms),
    }];
  }
  return out;
}

@end
