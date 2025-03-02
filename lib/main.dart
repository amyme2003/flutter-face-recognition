import 'dart:convert';
import 'dart:io';
import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import 'package:http/http.dart' as http;

void main() {
  runApp(MyApp());
}

class MyApp extends StatefulWidget {
  @override
  _MyAppState createState() => _MyAppState();
}

class _MyAppState extends State<MyApp> {
  File? _image;
  String _result = "No image uploaded yet";
  final picker = ImagePicker();
  final String baseUrl = " https://1ec1-2409-4073-2010-f1ec-f8c6-4a0c-c853-b169.ngrok-free.app"; // Update this

  // Function to pick an image from gallery
  Future<void> _pickImage() async {
    final pickedFile = await picker.pickImage(source: ImageSource.gallery);
    if (pickedFile != null) {
      setState(() {
        _image = File(pickedFile.path);
      });
    }
  }

  // Function to capture image from camera
  Future<void> _captureImage() async {
    final pickedFile = await picker.pickImage(source: ImageSource.camera);
    if (pickedFile != null) {
      setState(() {
        _image = File(pickedFile.path);
      });
    }
  }

  Future<void> _detectFace() async {
    if (_image == null) {
      setState(() {
        _result = "Please select an image first!";
      });
      return;
    }

    var request = http.MultipartRequest('POST', Uri.parse("$baseUrl/detect"));
    request.files.add(await http.MultipartFile.fromPath('image', _image!.path));

    try {
      var response = await request.send();
      var responseBody = await response.stream.bytesToString();
      var jsonResponse = json.decode(responseBody);

      setState(() {
        if (response.statusCode == 200) {
          _result = "Recognized: ${jsonResponse['recognized']}\nConfidence: ${jsonResponse['confidence']}";
        } else {
          _result = "Error: ${jsonResponse['error']}";
        }
      });
    } catch (e) {
      setState(() {
        _result = "Server error. Check Flask backend.";
      });
    }
  }


  // Function to trigger training
  Future<void> _trainModel() async {
    try {
      final response = await http.get(Uri.parse("$baseUrl/train"));

      setState(() {
        if (response.statusCode == 200) {
          _result = "Training Successful: ${response.body}";
        } else {
          _result = "Training Failed: ${response.body}";
        }
      });
    } catch (e) {
      setState(() {
        _result = "Failed to connect to server for training";
      });
    }
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
                  ? Text("No Image Selected")
                  : Image.file(_image!, width: 200, height: 200),
              SizedBox(height: 20),
              Text(_result, textAlign: TextAlign.center),
              SizedBox(height: 20),
              ElevatedButton(onPressed: _pickImage, child: Text("Select Image from Gallery")),
              ElevatedButton(onPressed: _captureImage, child: Text("Capture Image from Camera")),
              ElevatedButton(onPressed: _detectFace, child: Text("Detect Face")),
              ElevatedButton(onPressed: _trainModel, child: Text("Train Model")),
            ],
          ),
        ),
      ),
    );
  }
}
