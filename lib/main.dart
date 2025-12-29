import 'package:flutter/material.dart';
import 'emotion_page.dart';

void main() {
  runApp(const EmotionMusicApp());
}

class EmotionMusicApp extends StatelessWidget {
  const EmotionMusicApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Emotion-Based Music Recommendation',
      debugShowCheckedModeBanner: false,
      theme: ThemeData(primarySwatch: Colors.deepPurple),
      home: const EmotionPage(), 
    );
  }
}
