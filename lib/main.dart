import 'dart:ffi';
import 'dart:io';
import 'dart:isolate';

import 'package:ffi/ffi.dart';
import 'package:fie_stream_clipper/ffi.dart';
import 'package:file_picker/file_picker.dart';
import 'package:flutter/material.dart';
import 'package:path_provider/path_provider.dart';
import 'package:url_launcher/url_launcher.dart';

void main() {
  runApp(const MainApp());
}

class MainApp extends StatefulWidget {
  const MainApp({super.key});

  @override
  State<MainApp> createState() => _MainAppState();
}

class _MainAppState extends State<MainApp> {
  final _messengerKey = GlobalKey<ScaffoldMessengerState>();

  String? selectedFile;
  String? outputFolder;
  bool outputFolderIsDefault = true;

  bool showStreamPath = false;
  bool showOutputPath = false;

  bool converting = false;
  String? errorStr;

  @override
  void initState() {
    super.initState();
    Future.delayed(Duration.zero, () async {
      final folder = await getDefaultOutputFolder();
      setState(() {
        outputFolder = folder;
      });
    });
  }

  Future<String> getDefaultOutputFolder() async {
    final appDir = await getApplicationDocumentsDirectory(); // Get app's documents directory
    final streamClipsDir = Directory('${appDir.path}/stream_clips');

    if (!await streamClipsDir.exists()) {
      await streamClipsDir.create(recursive: true); // Create directory if it doesn't exist
    }

    return streamClipsDir.path;
  }

  void selectStreamFile() async {
    final result = await FilePicker.platform.pickFiles(
      dialogTitle: "Select stream file",
      type: FileType.video,
      initialDirectory: (await getDownloadsDirectory())?.path
    );
    if (result != null) {
      final filePath = result.files.single.path;
      print('Selected file: $filePath');
      setState(() {
        selectedFile = filePath;
      });
    }
  }
  
  void selectOutputFolder() async {
    final result = await FilePicker.platform.getDirectoryPath(
      dialogTitle: "Select output folder",
      initialDirectory: await getDefaultOutputFolder()
    );
    if (result != null) {
      final filePath = result;
      print('Selected output folder: $filePath');
      final bool isDefault = outputFolder == await getDefaultOutputFolder();
      setState(() {
        outputFolder = filePath;
        outputFolderIsDefault = isDefault;
      });
    }
  }

  void openOutputFolder() {
    if (outputFolder != null) {
      final folderPath = Uri.parse('file://$outputFolder');
      launchUrl(folderPath);
    }
  }

  void clipStream() {
    if (selectedFile == null || outputFolder == null) return;

    try {
      final receivePort = ReceivePort();
      Isolate.spawn(clipStreamImpl, receivePort.sendPort);

      // receivePort.listen((message) {
      //   _messengerKey.currentState!.showSnackBar(
      //     SnackBar(content: Text(message)),
      //   );
      //   receivePort.close();
      // });
    } catch (e, stackTrace) {
      print('Error: $e\n$stackTrace');
    }
  }

  void clipStreamImpl(SendPort sendPort) {
    try {
      if (selectedFile == null || outputFolder == null) return;
      final ffi = NativeLibrary();
      final ffmpegPtr = '${Directory(Platform.resolvedExecutable).parent.parent.path}/Resources/ffmpeg'.toNativeUtf8();
      final svmModelPtr = '${Directory(Platform.resolvedExecutable).parent.parent.path}/Resources/svm_model.xml'.toNativeUtf8();
      final tesseractPtr = '${Directory(Platform.resolvedExecutable).parent.parent.path}/Resources/'.toNativeUtf8();
      final selectedFilePtr = selectedFile!.toNativeUtf8();
      final outputFolderPtr = outputFolder!.toNativeUtf8();

      final resultPtr = ffi.cutStream(ffmpegPtr, tesseractPtr, svmModelPtr, selectedFilePtr, 1, outputFolderPtr);

      malloc.free(selectedFilePtr);
      malloc.free(outputFolderPtr);

      sendPort.send(resultPtr != nullptr ? 'Stream successfully clipped!' : 'Stream analysis failed');
    } catch (e) {
      sendPort.send("Error: $e");
    }
  }

  void clipStream2() {
    final ffi = NativeLibrary();
      final ffmpegPtr = '${Directory(Platform.resolvedExecutable).parent.parent.path}/Resources/ffmpeg'.toNativeUtf8();
      final svmModelPtr = '${Directory(Platform.resolvedExecutable).parent.parent.path}/Resources/svm_model.xml'.toNativeUtf8();
      final tesseractPtr = '${Directory(Platform.resolvedExecutable).parent.parent.path}/Resources/'.toNativeUtf8();
      final selectedFilePtr = selectedFile!.toNativeUtf8();
      final outputFolderPtr = outputFolder!.toNativeUtf8();

      final resultPtr = ffi.cutStream(ffmpegPtr, tesseractPtr, svmModelPtr, selectedFilePtr, 1, outputFolderPtr);

      malloc.free(selectedFilePtr);
      malloc.free(outputFolderPtr);

      _messengerKey.currentState!.showSnackBar(
          SnackBar(content: Text(resultPtr != nullptr ? 'Stream successfully clipped!' : 'Stream analysis failed')),
        );
  }
  
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      scaffoldMessengerKey: _messengerKey,
      home: Scaffold(
        appBar: AppBar(title: Text('Stream Clipper')),
        body: Padding(
          padding: const EdgeInsets.all(16.0),
          child: Center(
            child: Column(
              mainAxisAlignment: MainAxisAlignment.center,
              children: [
                Row(
                  mainAxisAlignment: MainAxisAlignment.center,
                  children: [
                    ElevatedButton(
                      onPressed: selectStreamFile,
                      child: Text("Select Stream File"),
                      style: ElevatedButton.styleFrom(
                        padding: EdgeInsets.symmetric(horizontal: 32, vertical: 12),
                        shape: RoundedRectangleBorder(
                          borderRadius: BorderRadius.circular(8),
                        ),
                      ),
                    ),
                    SizedBox(width: 6),
                    Tooltip(
                      message: "Show stream file path",
                      child: IconButton(
                        icon: Icon(
                          showStreamPath ? Icons.visibility_off : Icons.visibility,
                          size: 20,
                        ),
                        onPressed: () {
                          setState(() {
                            showStreamPath = !showStreamPath;
                          });
                        },
                      ),
                    )
                  ],
                ),
                if (showStreamPath) ...[
                  SizedBox(height: 8),
                  Text('File: ${selectedFile ?? "No file selected"}', style: TextStyle(fontSize: 14)),
                ],
                SizedBox(height: 8),
                Row(
                  mainAxisAlignment: MainAxisAlignment.center,
                  children: [
                    ElevatedButton(
                      onPressed: selectOutputFolder,
                      child: Text("Select Output Folder"),
                      style: ElevatedButton.styleFrom(
                        padding: EdgeInsets.symmetric(horizontal: 32, vertical: 12),
                        shape: RoundedRectangleBorder(
                          borderRadius: BorderRadius.circular(8),
                        ),
                      ),
                    ),
                    SizedBox(width: 6),
                    Tooltip(
                      message: "Show output folder path",
                      child: IconButton(
                        icon: Icon(
                          showOutputPath ? Icons.visibility_off : Icons.visibility,
                          size: 20,
                        ),
                        onPressed: () {
                          setState(() {
                            showOutputPath = !showOutputPath;
                          });
                        },
                      )
                    ),
                    Tooltip(
                      message: "Open output folder in finder",
                      child: IconButton(
                        onPressed: openOutputFolder,
                        icon: Icon(Icons.folder_open, size: 20),
                      ),
                    )
                  ]
                ),
                SizedBox(height: 8),
                if (showOutputPath) ...[
                  Text(
                    'Folder: ${outputFolderIsDefault ? "Default folder" : outputFolder == null ? "No folder selected" : outputFolder!.length > 40 ? outputFolder!.substring(0, 40) + '...' : outputFolder}',
                    style: TextStyle(fontSize: 14),
                    overflow: TextOverflow.ellipsis,
                  ),
                ],
                SizedBox(height: 16),
                ElevatedButton(
                  onPressed: clipStream2,
                  child: Text("Clip Stream!"),
                  style: ElevatedButton.styleFrom(
                    padding: EdgeInsets.symmetric(horizontal: 32, vertical: 12),
                    shape: RoundedRectangleBorder(
                      borderRadius: BorderRadius.circular(8),
                    ),
                  ),
                )
              ]
            ),
          ),
        ),
      )
    );
  }
}
