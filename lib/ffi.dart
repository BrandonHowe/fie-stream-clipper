import 'dart:ffi';
import 'dart:io';

import 'package:ffi/ffi.dart';
import 'package:path/path.dart' as path;

base class StreamBoutSegment extends Struct {
  @Int32()
  external int startFrame;

  @Int32()
  external int endFrame;

  external Pointer<Utf8> nameLeft;

  external Pointer<Utf8> nameRight;
}

base class StreamAnalysis extends Struct {
  @Double()
  external double fps;

  @Int32()
  external int frameCount;

  @Uint8()
  external int boutCount;

  @Array<StreamBoutSegment>(64)
  external Array<StreamBoutSegment> bouts;
}

typedef CallbackFunc = Void Function(Int32);
typedef DartCallback = void Function(int);

typedef CutStreamC = Pointer<Void> Function(
    Pointer<Utf8>,
    Pointer<Utf8>,
    Pointer<Utf8>,
    Uint8,
    Pointer<Utf8>,
    Pointer<Utf8>,
    Pointer<NativeFunction<CallbackFunc>>);
typedef CutStreamDart = Pointer<Void> Function(
    Pointer<Utf8>,
    Pointer<Utf8>,
    Pointer<Utf8>,
    int,
    Pointer<Utf8>,
    Pointer<Utf8>,
    Pointer<NativeFunction<CallbackFunc>>);

class NativeLibrary {
  late final DynamicLibrary _lib;

  NativeLibrary() {
    print(
        "Path: ${Directory(Platform.resolvedExecutable).parent.parent.path}/Resources/libvideo_review.dylib");
    _lib = Platform.isMacOS
        ? DynamicLibrary.open(
            '${Directory(Platform.resolvedExecutable).parent.parent.path}/Resources/libvideo_review.dylib')
        : DynamicLibrary.open(
            '${Directory(Platform.resolvedExecutable).parent.path}\\video_review.dll');
  }

  late final CutStreamDart cutStream =
      _lib.lookup<NativeFunction<CutStreamC>>('cut_stream_async').asFunction();
}
