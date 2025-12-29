import 'dart:io';
import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';
import 'package:image_picker/image_picker.dart';
import 'package:tflite_flutter/tflite_flutter.dart';
import 'package:image/image.dart' as img;
import 'package:google_mlkit_face_detection/google_mlkit_face_detection.dart';
import 'detect_emotion.dart';

class EmotionPage extends StatefulWidget {
  const EmotionPage({super.key});

  @override
  State<EmotionPage> createState() => _EmotionPageState();
}

class _EmotionPageState extends State<EmotionPage> {
  final ImagePicker _picker = ImagePicker();
  Interpreter? _interpreter;
  final FaceDetector _faceDetector = FaceDetector(
    options: FaceDetectorOptions(
      enableContours: false,
      enableLandmarks: false,
      enableClassification: true,
      performanceMode: FaceDetectorMode.accurate,
    ),
  );
  final List<String> allModelLabels = ["Angry", "Fear", "Sad", "Disgust"];

  @override
  void initState() {
    super.initState();
    _loadModel();
  }

  Future<void> _loadModel() async {
    try {
      _interpreter =
          await Interpreter.fromAsset('assets/model/emotion_model.tflite');
      print("✅ Model loaded successfully!");
    } catch (e) {
      print("❌ Failed to load model: $e");
    }
  }

  Future<File?> captureImageAndDetect() async {
    final pickedFile = await _picker.pickImage(source: ImageSource.camera);
    if (pickedFile == null) return null;

    final emotion = await _detectEmotion(File(pickedFile.path));
    if (!mounted) return null;

    if (emotion != null) {

      await Navigator.push(
        context,
        MaterialPageRoute(
          builder: (_) => DetectResultPage(
            imageFile: File(pickedFile.path),
            detectedEmotion: emotion,
            onRetake: captureImageAndDetect,
          ),
        ),
      );
    } else {

      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(
          content: Text("No face detected, please try again."),
          duration: Duration(seconds: 2),
        ),
      );
    }

    return File(pickedFile.path);
  }

  Future<String?> _detectEmotion(File imageFile) async {
    if (_interpreter == null) return null;

    final bytes = await imageFile.readAsBytes();
    img.Image? image = img.decodeImage(bytes);
    if (image == null) return null;

    final inputImage = InputImage.fromFile(imageFile);
    final faces = await _faceDetector.processImage(inputImage);
    if (faces.isEmpty) return null;

    final resized = img.copyResize(image, width: 48, height: 48);

    var input = List.generate(
      1,
      (_) => List.generate(
        48,
        (_) => List.generate(
          48,
          (_) => List.filled(1, 0.0),
        ),
      ),
    );

    for (int y = 0; y < 48; y++) {
      for (int x = 0; x < 48; x++) {
        final pixel = resized.getPixel(x, y);
        input[0][y][x][0] = (pixel.r + pixel.g + pixel.b) / 3.0 / 255.0;
      }
    }

    var output = List.filled(4, 0.0).reshape([1, 4]);
    _interpreter!.run(input, output);

    int maxIndex = 0;
    double maxValue = output[0][0];
    for (int i = 1; i < allModelLabels.length; i++) {
      if (output[0][i] > maxValue) {
        maxValue = output[0][i];
        maxIndex = i;
      }
    }

    return allModelLabels[maxIndex];
  }

  @override
  void dispose() {
    _interpreter?.close();
    _faceDetector.close();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: Stack(
        children: [
          Positioned.fill(
            child: Image.asset(
              'assets/images/Cloud.jpg',
              fit: BoxFit.cover,
            ),
          ),
          Positioned.fill(
            child: Container(color: Colors.lightBlue.withOpacity(0.08)),
          ),
          SafeArea(
            child: Center(
              child: SingleChildScrollView(
                child: Container(
                  margin: const EdgeInsets.all(24),
                  padding: const EdgeInsets.all(24),
                  decoration: BoxDecoration(
                    color: Colors.white.withOpacity(0.75),
                    borderRadius: BorderRadius.circular(20),
                    boxShadow: const [
                      BoxShadow(
                        color: Colors.black26,
                        blurRadius: 10,
                        offset: Offset(0, 5),
                      ),
                    ],
                  ),
                  child: Column(
                    mainAxisSize: MainAxisSize.min,
                    children: [
                      Text(
                        "Start When You Get Ready！",
                        textAlign: TextAlign.center,
                        style: GoogleFonts.playpenSans(
                          fontSize: 24,
                          fontWeight: FontWeight.w600,
                        ),
                      ),
                      const SizedBox(height: 40),
                      ElevatedButton.icon(
                        onPressed: captureImageAndDetect,
                        icon: const Icon(Icons.camera_alt),
                        label: Text(
                          "Capture Image",
                          style: GoogleFonts.playpenSans(fontSize: 16),
                        ),
                        style: ElevatedButton.styleFrom(
                          padding: const EdgeInsets.symmetric(
                              vertical: 14, horizontal: 24),
                          backgroundColor: Colors.blueAccent,
                          foregroundColor: Colors.white,
                          shape: RoundedRectangleBorder(
                            borderRadius: BorderRadius.circular(12),
                          ),
                        ),
                      ),
                    ],
                  ),
                ),
              ),
            ),
          ),
        ],
      ),
    );
  }
}
