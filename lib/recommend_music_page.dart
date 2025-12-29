import 'package:flutter/material.dart';
import 'package:audioplayers/audioplayers.dart';
import 'music_player.dart';
import 'package:google_fonts/google_fonts.dart';

class RecommendMusicPage extends StatelessWidget {
  final String emotion;

  const RecommendMusicPage({Key? key, required this.emotion}) : super(key: key);


  Color _emotionColor() {
    switch (emotion.toLowerCase()) {
      case 'happy':
        return Colors.orange.shade700;
      case 'angry':
        return Colors.red.shade700;
      case 'sad':
        return Colors.blue.shade700;
      case 'fear':
        return Colors.grey.shade900;
      case 'neutral':
        return Colors.lightGreen.shade600;
      case 'disgust':
        return Colors.purple.shade600;
      case 'surprise':
        return Colors.green.shade600;
      default:
        return Colors.teal;
    }
  }


  bool _noButtonEmotion(String e) {
    final emo = e.toLowerCase();
    return emo == 'disgust' || emo == 'fear' || emo == 'surprise' || emo == 'happy';
  }


  List<Map<String, String>> _getSongs(String emotion, [String? mood]) {
    final e = emotion.toLowerCase();
    List<Map<String, String>> songs = [];

    switch (e) {
      case 'happy':
        songs = [
          {'title': 'Calm 1', 'path': 'assets/music/disgust/Calm 1.mp3', 'background': 'assets/images/Forest.jpg'},
          {'title': 'Calm 2', 'path': 'assets/music/disgust/Calm 2.mp3', 'background': 'assets/images/Waterfall.jpg'},
          {'title': 'Calm 3', 'path': 'assets/music/disgust/Calm 3.mp3', 'background': 'assets/images/Bird.jpg'},
          {'title': 'Calm 4', 'path': 'assets/music/disgust/Calm 4.mp3', 'background': 'assets/images/Canon.jpg'},
          {'title': 'Calm 5', 'path': 'assets/music/disgust/Calm 5.mp3', 'background': 'assets/images/Butterfly.jpg'},
          {'title': 'Calm 6', 'path': 'assets/music/disgust/Calm 6.mp3', 'background': 'assets/images/Love.jpg'},
          {'title': 'Calm 7', 'path': 'assets/music/disgust/Calm 7.mp3', 'background': 'assets/images/Star.jpg'},
          {'title': 'Calm 8', 'path': 'assets/music/disgust/Calm 8.mp3', 'background': 'assets/images/Bird.jpg'},
        ];
        break;
      case 'angry':
        songs = [
          {'title': 'Calm 1', 'path': 'assets/music/disgust/Calm 1.mp3', 'background': 'assets/images/Forest.jpg'},
          {'title': 'Calm 2', 'path': 'assets/music/disgust/Calm 2.mp3', 'background': 'assets/images/Waterfall.jpg'},
          {'title': 'Calm 3', 'path': 'assets/music/disgust/Calm 3.mp3', 'background': 'assets/images/Bird.jpg'},
          {'title': 'Calm 4', 'path': 'assets/music/disgust/Calm 4.mp3', 'background': 'assets/images/Mushroom.jpg'},
          {'title': 'Calm 5', 'path': 'assets/music/disgust/Calm 5.mp3', 'background': 'assets/images/Butterfly.jpg'},
          {'title': 'Calm 6', 'path': 'assets/music/disgust/Calm 6.mp3', 'background': 'assets/images/River.jpg'},
          {'title': 'Calm 7', 'path': 'assets/music/disgust/Calm 7.mp3', 'background': 'assets/images/Forest.jpg'},
          {'title': 'Calm 8', 'path': 'assets/music/disgust/Calm 8.mp3', 'background': 'assets/images/Bird.jpg'},
          {'title': 'Rock 1', 'path': 'assets/music/angry/Rock 1.mp3', 'background': 'assets/images/Fire.jpg'},
          {'title': 'Rock 2', 'path': 'assets/music/angry/Rock 2.mp3', 'background': 'assets/images/Fire Guitar.jpg'},
          {'title': 'Rock 3', 'path': 'assets/music/angry/Rock 3.mp3', 'background': 'assets/images/Fire Guitar.jpg'},
          {'title': 'Rock 4', 'path': 'assets/music/angry/Rock 4.mp3', 'background': 'assets/images/Fire Bottle.jpg'},
          {'title': 'Rock 5', 'path': 'assets/music/angry/Rock 5.mp3', 'background': 'assets/images/Fire Bottle.jpg'},
          {'title': 'Rock 6', 'path': 'assets/music/angry/Rock 6.mp3', 'background': 'assets/images/Fire Guitar.jpg'},
          {'title': 'Rock 7', 'path': 'assets/music/angry/Rock 7.mp3', 'background': 'assets/images/Fire Guitar.jpg'},
          {'title': 'Rock 8', 'path': 'assets/music/angry/Rock 8.mp3', 'background': 'assets/images/Fire.jpg'},
          {'title': 'Happy 1', 'path': 'assets/music/angry/Happy 1.mp3', 'background': 'assets/images/Bird.jpg'},
          {'title': 'Happy 2', 'path': 'assets/music/angry/Happy 2.mp3', 'background': 'assets/images/Love.jpg'},
          {'title': 'Happy 3', 'path': 'assets/music/angry/Happy 3.mp3', 'background': 'assets/images/Waterfall.jpg'},
          {'title': 'Happy 4', 'path': 'assets/music/angry/Happy 4.mp3', 'background': 'assets/images/River.jpg'},
          {'title': 'Happy 5', 'path': 'assets/music/angry/Happy 5.mp3', 'background': 'assets/images/Waterfall.jpg'},
          {'title': 'Happy 6', 'path': 'assets/music/angry/Happy 6.mp3', 'background': 'assets/images/Forest.jpg'},
          {'title': 'Happy 7', 'path': 'assets/music/angry/Happy 7.mp3', 'background': 'assets/images/Love.jpg'},
          {'title': 'Happy 8', 'path': 'assets/music/angry/Happy 8.mp3', 'background': 'assets/images/River.jpg'},
        ];
        break;
      case 'disgust':
        songs = [
          {'title': 'Calm 1', 'path': 'assets/music/disgust/Calm 1.mp3', 'background': 'assets/images/Forest.jpg'},
          {'title': 'Calm 2', 'path': 'assets/music/disgust/Calm 2.mp3', 'background': 'assets/images/Waterfall.jpg'},
          {'title': 'Calm 3', 'path': 'assets/music/disgust/Calm 3.mp3', 'background': 'assets/images/Bird.jpg'},
          {'title': 'Calm 4', 'path': 'assets/music/disgust/Calm 4.mp3', 'background': 'assets/images/Mushroom.jpg'},
          {'title': 'Calm 5', 'path': 'assets/music/disgust/Calm 5.mp3', 'background': 'assets/images/Butterfly.jpg'},
          {'title': 'Calm 6', 'path': 'assets/music/disgust/Calm 6.mp3', 'background': 'assets/images/River.jpg'},
          {'title': 'Calm 7', 'path': 'assets/music/disgust/Calm 7.mp3', 'background': 'assets/images/Forest.jpg'},
          {'title': 'Calm 8', 'path': 'assets/music/disgust/Calm 8.mp3', 'background': 'assets/images/Bird.jpg'},
        ];
        break;
      case 'fear':
        songs = [
          {'title': 'Positive 1', 'path': 'assets/music/sad/Motivate 1.mp3', 'background': 'assets/images/Victory.jpg'},
          {'title': 'Positive 2', 'path': 'assets/music/sad/Motivate 2.mp3', 'background': 'assets/images/Love.jpg'},
          {'title': 'Positive 3', 'path': 'assets/music/sad/Motivate 3.mp3', 'background': 'assets/images/Like.jpg'},
          {'title': 'Positive 4', 'path': 'assets/music/sad/Motivate 4.mp3', 'background': 'assets/images/Victory.jpg'},
          {'title': 'Positive 5', 'path': 'assets/music/sad/Motivate 5.mp3', 'background': 'assets/images/Love.jpg'},
          {'title': 'Positive 6', 'path': 'assets/music/sad/Motivate 6.mp3', 'background': 'assets/images/Like.jpg'},
          {'title': 'Positive 7', 'path': 'assets/music/sad/Motivate 7.mp3', 'background': 'assets/images/Love.jpg'},
          {'title': 'Positive 8', 'path': 'assets/music/sad/Motivate 8.mp3', 'background': 'assets/images/Victory.jpg'},
        ];
        break;
      case 'neutral':
        songs = [
          {'title': 'Calm 1', 'path': 'assets/music/disgust/Calm 1.mp3', 'background': 'assets/images/Forest.jpg'},
          {'title': 'Calm 2', 'path': 'assets/music/disgust/Calm 2.mp3', 'background': 'assets/images/Waterfall.jpg'},
          {'title': 'Calm 3', 'path': 'assets/music/disgust/Calm 3.mp3', 'background': 'assets/images/Bird.jpg'},
          {'title': 'Calm 4', 'path': 'assets/music/disgust/Calm 4.mp3', 'background': 'assets/images/Mushroom.jpg'},
          {'title': 'Calm 5', 'path': 'assets/music/disgust/Calm 5.mp3', 'background': 'assets/images/Butterfly.jpg'},
          {'title': 'Calm 6', 'path': 'assets/music/disgust/Calm 6.mp3', 'background': 'assets/images/River.jpg'},
          {'title': 'Calm 7', 'path': 'assets/music/disgust/Calm 7.mp3', 'background': 'assets/images/Forest.jpg'},
          {'title': 'Calm 8', 'path': 'assets/music/disgust/Calm 8.mp3', 'background': 'assets/images/Bird.jpg'},
        ];
        break;
      case 'sad':
        songs = [
          {'title': 'Happy 1', 'path': 'assets/music/angry/Happy 1.mp3', 'background': 'assets/images/Bird.jpg'},
          {'title': 'Happy 2', 'path': 'assets/music/angry/Happy 2.mp3', 'background': 'assets/images/Love.jpg'},
          {'title': 'Happy 3', 'path': 'assets/music/angry/Happy 3.mp3', 'background': 'assets/images/Forest.jpg'},
          {'title': 'Happy 4', 'path': 'assets/music/angry/Happy 4.mp3', 'background': 'assets/images/Bird.jpg'},
          {'title': 'Happy 5', 'path': 'assets/music/angry/Happy 5.mp3', 'background': 'assets/images/Waterfall.jpg'},
          {'title': 'Happy 6', 'path': 'assets/music/angry/Happy 6.mp3', 'background': 'assets/images/Bird.jpg'},
          {'title': 'Happy 7', 'path': 'assets/music/angry/Happy 7.mp3', 'background': 'assets/images/Love.jpg'},
          {'title': 'Happy 8', 'path': 'assets/music/angry/Happy 8.mp3', 'background': 'assets/images/Forest.jpg'},
          {'title': 'Sad 1', 'path': 'assets/music/sad/Sad 1.mp3', 'background': 'assets/images/Light.jpg'},
          {'title': 'Sad 2', 'path': 'assets/music/sad/Sad 2.mp3', 'background': 'assets/images/Heaven.jpg'},
          {'title': 'Sad 3', 'path': 'assets/music/sad/Sad 3.mp3', 'background': 'assets/images/Heaven.jpg'},
          {'title': 'Sad 4', 'path': 'assets/music/sad/Sad 4.mp3', 'background': 'assets/images/Light.jpg'},
          {'title': 'Sad 5', 'path': 'assets/music/sad/Sad 5.mp3', 'background': 'assets/images/Fire.jpg'},
          {'title': 'Sad 6', 'path': 'assets/music/sad/Sad 6.mp3', 'background': 'assets/images/Fire.jpg'},
          {'title': 'Sad 7', 'path': 'assets/music/sad/Sad 7.mp3', 'background': 'assets/images/Light.jpg'},
          {'title': 'Sad 8', 'path': 'assets/music/sad/Sad 8.mp3', 'background': 'assets/images/Heaven.jpg'},
          {'title': 'Motivate 1', 'path': 'assets/music/sad/Motivate 1.mp3', 'background': 'assets/images/Victory.jpg'},
          {'title': 'Motivate 2', 'path': 'assets/music/sad/Motivate 2.mp3', 'background': 'assets/images/Love.jpg'},
          {'title': 'Motivate 3', 'path': 'assets/music/sad/Motivate 3.mp3', 'background': 'assets/images/Chess.jpg'},
          {'title': 'Motivate 4', 'path': 'assets/music/sad/Motivate 4.mp3', 'background': 'assets/images/Victory.jpg'},
          {'title': 'Motivate 5', 'path': 'assets/music/sad/Motivate 5.mp3', 'background': 'assets/images/Love.jpg'},
          {'title': 'Motivate 6', 'path': 'assets/music/sad/Motivate 6.mp3', 'background': 'assets/images/Chess.jpg'},
          {'title': 'Motivate 7', 'path': 'assets/music/sad/Motivate 7.mp3', 'background': 'assets/images/Love.jpg'},
          {'title': 'Motivate 8', 'path': 'assets/music/sad/Motivate 8.mp3', 'background': 'assets/images/Victory.jpg'},
        ];
        break;
      case 'surprise':
        songs = [
          {'title': 'Calm 1', 'path': 'assets/music/disgust/Calm 1.mp3', 'background': 'assets/images/Forest.jpg'},
          {'title': 'Calm 2', 'path': 'assets/music/disgust/Calm 2.mp3', 'background': 'assets/images/Waterfall.jpg'},
          {'title': 'Calm 3', 'path': 'assets/music/disgust/Calm 3.mp3', 'background': 'assets/images/Bird.jpg'},
          {'title': 'Calm 4', 'path': 'assets/music/disgust/Calm 4.mp3', 'background': 'assets/images/Canon.jpg'},
          {'title': 'Calm 5', 'path': 'assets/music/disgust/Calm 5.mp3', 'background': 'assets/images/Butterfly.jpg'},
          {'title': 'Calm 6', 'path': 'assets/music/disgust/Calm 6.mp3', 'background': 'assets/images/Love.jpg'},
          {'title': 'Calm 7', 'path': 'assets/music/disgust/Calm 7.mp3', 'background': 'assets/images/Star.jpg'},
          {'title': 'Calm 8', 'path': 'assets/music/disgust/Calm 8.mp3', 'background': 'assets/images/Bird.jpg'},
        ];
        break;
    }

    return songs;
  }


  List<Map<String, String>> _getSongsForMood(String emotion, String mood) {
    final e = emotion.toLowerCase();

    if (e == 'angry') {
      switch (mood) {
        case 'calm':
          return _getSongs(e).where((s) => 
          s['title']!.contains('Calm 1') || 
          s['title']!.contains('Calm 2') || 
          s['title']!.contains('Calm 3') || 
          s['title']!.contains('Calm 4') || 
          s['title']!.contains('Calm 5') || 
          s['title']!.contains('Calm 6') || 
          s['title']!.contains('Calm 7') || 
          s['title']!.contains('Calm 8'))
          .toList();
        case 'vent':
          return _getSongs(e).where((s) => 
          s['title']!.contains('Rock 1') ||
          s['title']!.contains('Rock 2')||
          s['title']!.contains('Rock 3')||
          s['title']!.contains('Rock 4')||
          s['title']!.contains('Rock 5')||
          s['title']!.contains('Rock 6')||
          s['title']!.contains('Rock 7')||
          s['title']!.contains('Rock 8')
          )
          .toList();
        case 'happy':
          return _getSongs(e).where((s) => 
          s['title']!.contains('Happy 1') || 
          s['title']!.contains('Happy 2')||
          s['title']!.contains('Happy 3')||
          s['title']!.contains('Happy 4')||
          s['title']!.contains('Happy 5')||
          s['title']!.contains('Happy 6')||
          s['title']!.contains('Happy 7')||
          s['title']!.contains('Happy 8')
          )
          .toList();
      }
    }

    if (e == 'sad') {
      switch (mood) {
        case 'happy':
          return _getSongs(e).where((s) => 
          s['title']!.contains('Happy 1') || 
          s['title']!.contains('Happy 2')||
          s['title']!.contains('Happy 3')||
          s['title']!.contains('Happy 4')||
          s['title']!.contains('Happy 5')||
          s['title']!.contains('Happy 6')||
          s['title']!.contains('Happy 7')||
          s['title']!.contains('Happy 8'))
          .toList();
        case 'vent':
          return _getSongs(e).where((s) => 
          s['title']!.contains('Sad 1')||
          s['title']!.contains('Sad 2')||
          s['title']!.contains('Sad 3')||
          s['title']!.contains('Sad 4')||
          s['title']!.contains('Sad 5')||
          s['title']!.contains('Sad 6')||
          s['title']!.contains('Sad 7')||
          s['title']!.contains('Sad 8')
          )

          .toList();
        case 'motivate':
          return _getSongs(e).where((s) => 
          s['title']!.contains('Motivate 1') || 
          s['title']!.contains('Motivate 2') || 
          s['title']!.contains('Motivate 3') || 
          s['title']!.contains('Motivate 4') || 
          s['title']!.contains('Motivate 5')||
          s['title']!.contains('Motivate 6')||
          s['title']!.contains('Motivate 7')||
          s['title']!.contains('Motivate 8')
          ).toList();
      }
    }

    return _getSongs(emotion);
  }


  Widget _buildMoodButtons(BuildContext context) {
    final e = emotion.toLowerCase();
    if (_noButtonEmotion(e)) return const SizedBox.shrink();

    final moodOptions = {
      'angry': [
        {'mood': 'calm', 'label': 'Calm'},
        {'mood': 'vent', 'label': 'Rock'},
        {'mood': 'happy', 'label': 'Happy'}
      ],
      'neutral': [
        {'mood': 'light', 'label': 'Light'},
        {'mood': 'soft', 'label': 'Soft'},
      ],
      'sad': [
        {'mood': 'happy', 'label': 'Happy'},
        {'mood': 'vent', 'label': 'Sad'},
        {'mood': 'motivate', 'label': 'Motivate'},
      ],
    };

    final btnList = moodOptions[e] ?? [];

    return Center(
      child: Column(
        mainAxisAlignment: MainAxisAlignment.center,
        children: btnList.map((b) {
          return Padding(
            padding: const EdgeInsets.symmetric(vertical: 12),
            child: SizedBox(
              width: MediaQuery.of(context).size.width * 0.8,
              height: 60,
              child: ElevatedButton(
                onPressed: () {
                  final songs = _getSongsForMood(emotion, b['mood']!);
                  Navigator.push(
                    context,
                    MaterialPageRoute(
                      builder: (_) => MusicGridPage(
                        songs: songs,
                        emotion: emotion,
                        color: _emotionColor(),
                      ),
                    ),
                  );
                },
                style: ElevatedButton.styleFrom(
                  backgroundColor: _emotionColor(),
                  foregroundColor: Colors.white,
                  shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
                ),
                child: Text(
                  b['label']!,
                  style: GoogleFonts.playpenSans(fontSize: 20, fontWeight: FontWeight.bold),
                ),
              ),
            ),
          );
        }).toList(),
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    final color = _emotionColor();
    if (_noButtonEmotion(emotion)) {
      final songs = _getSongs(emotion, null);
      return MusicGridPage(songs: songs, emotion: emotion, color: color);
    }

    return Scaffold(
      appBar: AppBar(
        title: Text(
          "Music for $emotion",
          style: GoogleFonts.playpenSans(fontWeight: FontWeight.bold),
        ),
        backgroundColor: color,
      ),
      body: Center(child: SingleChildScrollView(child: _buildMoodButtons(context))),
    );
  }
}

class MusicGridPage extends StatefulWidget {
  final List<Map<String, String>> songs;
  final String emotion;
  final Color color;

  const MusicGridPage({Key? key, required this.songs, required this.emotion, required this.color}) : super(key: key);

  @override
  _MusicGridPageState createState() => _MusicGridPageState();
}

class _MusicGridPageState extends State<MusicGridPage> with SingleTickerProviderStateMixin {
  final AudioPlayer _player = AudioPlayer();
  String? _currentSong;

  late AnimationController _controller;

  @override
  void initState() {
    super.initState();
    _controller = AnimationController(
      vsync: this,
      duration: const Duration(seconds: 8),
    )..repeat(); 
  }

  @override
  void dispose() {
    _player.dispose();
    _controller.dispose();
    super.dispose();
  }

  void _playSong(String path) async {
    await _player.stop();
    await _player.play(DeviceFileSource(path));
    setState(() {
      _currentSong = path;
    });
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text(
          "${widget.emotion} Songs",
          style: GoogleFonts.playpenSans(fontWeight: FontWeight.bold),
        ),
        backgroundColor: widget.color,
      ),
      body: GridView.builder(
        padding: const EdgeInsets.all(12),
        itemCount: widget.songs.length,
        gridDelegate: const SliverGridDelegateWithFixedCrossAxisCount(
          crossAxisCount: 2,
          childAspectRatio: 0.9,
          crossAxisSpacing: 12,
          mainAxisSpacing: 12,
        ),
        itemBuilder: (context, index) {
          final song = widget.songs[index];
          final isPlaying = _currentSong == song['path'];
          final bgImage = song['background'] ?? 'assets/images/default_bg.jpg';

          return GestureDetector(
            onTap: () {
              _playSong(song['path']!);
              Navigator.push(
                context,
                MaterialPageRoute(
                  builder: (_) => MusicPlayerPage(
                    songPath: song['path']!,
                    songTitle: song['title']!,
                    background: song['background']!,
                    color: widget.color,
                  ),
                ),
              );
            },
            child: Container(
              decoration: BoxDecoration(
                image: DecorationImage(image: AssetImage(bgImage), fit: BoxFit.cover),
                borderRadius: BorderRadius.circular(12),
                boxShadow: [BoxShadow(color: Colors.black26, blurRadius: 5, offset: Offset(0, 3))],
              ),
              child: Container(
                decoration: BoxDecoration(
                  color: Colors.black.withOpacity(0.4),
                  borderRadius: BorderRadius.circular(12),
                ),
                child: Column(
                  mainAxisAlignment: MainAxisAlignment.center,
                  children: [
                    RotationTransition(
                      turns: _controller,
                      child: Icon(
                        Icons.album,
                        color: isPlaying ? widget.color : Colors.white,
                        size: 50,
                      ),
                    ),
                    const SizedBox(height: 8),
                    Padding(
                      padding: const EdgeInsets.symmetric(horizontal: 6),
                      child: Text(
                        song['title'] ?? '',
                        textAlign: TextAlign.center,
                        style: GoogleFonts.playpenSans(color: Colors.white, fontWeight: FontWeight.bold),
                        maxLines: 2,
                        overflow: TextOverflow.ellipsis,
                      ),
                    ),
                  ],
                ),
              ),
            ),
          );
        },
      ),
    );
  }
}
