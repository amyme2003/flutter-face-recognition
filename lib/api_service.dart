import 'dart:convert';
import 'dart:io';
import 'package:http/http.dart' as http;
import 'package:path/path.dart';

class ApiService {
  // 🛠 Replace with your Hugging Face API URL
  static const String apiUrl = "https://amyme2003-face-recognition-flask.hf.space/detect";

  static Future<String> detectFace(File imageFile) async {
    try {
      var request = http.MultipartRequest("POST", Uri.parse(apiUrl));
      request.files.add(
        await http.MultipartFile.fromPath(
          "image", imageFile.path,
          filename: basename(imageFile.path),
        ),
      );

      var response = await request.send();
      if (response.statusCode == 200) {
        var responseData = json.decode(await response.stream.bytesToString());
        return responseData["recognized"] ?? "Unknown";
      } else {
        return "Face not recognized!";
      }
    } catch (e) {
      return "Error: ${e.toString()}";
  }
  }
}