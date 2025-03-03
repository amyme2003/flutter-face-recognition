import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import 'dart:io';
import 'api_service.dart'; // Import the API service

void main() {
  runApp(FaceRecognitionApp());
}

class FaceRecognitionApp extends StatefulWidget {
  @override
  _FaceRecognitionAppState createState() => _FaceRecognitionAppState();
}

class _FaceRecognitionAppState extends State<FaceRecognitionApp> {
  File? _image;
  String _recognizedName = "No name detected";
  bool _isLoading = false;

  Future<void> pickImage() async {
    final pickedFile = await ImagePicker().pickImage(source: ImageSource.gallery);

    if (pickedFile != null) {
      setState(() {
        _image = File(pickedFile.path);
        _isLoading = true;
      });
      await detectFace();
    }
  }

  Future<void> detectFace() async {
    if (_image == null) return;
    String result = await ApiService.detectFace(_image!);
    setState(() {
      _recognizedName = result;
      _isLoading = false;
    });
  }

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
        home: Scaffold(
            appBar: AppBar(title: Text("Face Recognition")),
            body: Center(
              child: Column(
                mainAxisAlignment: MainAxisAlignment.center,
                children: [
                  _image == null
                      ? Text("No image selected")
                      : Image.file(_image!, height: 200),
                  SizedBox(height: 20),
                  _isLoading
                      ? CircularProgressIndicator()
                      : Text("Recognized: $_recognizedName", style: TextStyle(fontSize: 18)),
                  SizedBox(height: 20),
                  ElevatedButton(
                    onPressed: pickImage,
                    child: Text("Pick Image & Detect"),
                  ),
                ],
              ),
            ),
            ),
    );}
}