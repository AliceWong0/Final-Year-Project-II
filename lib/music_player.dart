import 'package:flutter/material.dart';
import 'package:audioplayers/audioplayers.dart';
import 'package:google_fonts/google_fonts.dart';

class MusicPlayerPage extends StatefulWidget {
  final String songPath;
  final String songTitle;
  final String background;
  final Color color;

  const MusicPlayerPage({
    Key? key,
    required this.songPath,
    required this.songTitle,
    required this.background,
    required this.color,
  }) : super(key: key);

  @override
  State<MusicPlayerPage> createState() => _MusicPlayerPageState();
}

class _MusicPlayerPageState extends State<MusicPlayerPage>
    with SingleTickerProviderStateMixin {
  late AudioPlayer _audioPlayer;
  late AnimationController _rotationController;

  bool isPlaying = false;
  Duration duration = Duration.zero;
  Duration position = Duration.zero;

  @override
  void initState() {
    super.initState();

    _audioPlayer = AudioPlayer();
    _rotationController =
        AnimationController(vsync: this, duration: const Duration(seconds: 12));


    _audioPlayer
        .setSource(AssetSource(widget.songPath.replaceFirst('assets/', '')))
        .then((_) async {
      final d = await _audioPlayer.getDuration();
      if (d != null) {
        setState(() => duration = d);
      }
    });

    _audioPlayer.onPositionChanged.listen((p) {
      setState(() => position = p);
    });

    _audioPlayer.onPlayerComplete.listen((_) {
      setState(() {
        isPlaying = false;
        position = Duration.zero;
        _rotationController.stop();
      });
    });
  }

  @override
  void dispose() {
    _audioPlayer.dispose();
    _rotationController.dispose();
    super.dispose();
  }

  void _playPause() async {
    if (isPlaying) {
      await _audioPlayer.pause();
      _rotationController.stop();
    } else {
      await _audioPlayer.resume();
      _rotationController.repeat();
    }
    setState(() => isPlaying = !isPlaying);
  }

  void _seekBy(int seconds) {
    Duration newPos = position + Duration(seconds: seconds);

    if (newPos < Duration.zero) {
      newPos = Duration.zero;
    } else if (newPos > duration) {
      newPos = duration;
    }

    _audioPlayer.seek(newPos);
  }

  String _fmt(Duration d) {
    final m = d.inMinutes.toString().padLeft(2, '0');
    final s = (d.inSeconds % 60).toString().padLeft(2, '0');
    return "$m:$s";
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: Stack(
        fit: StackFit.expand,
        children: [
         
          Image.asset(widget.background, fit: BoxFit.cover),
          Container(color: Colors.black.withOpacity(0.45)),

          SafeArea(
            child: Column(
              children: [
                
                Align(
                  alignment: Alignment.centerLeft,
                  child: IconButton(
                    icon: const Icon(Icons.arrow_back_ios_new_rounded),
                    color: Colors.white,
                    iconSize: 26,
                    onPressed: () => Navigator.pop(context),
                  ),
                ),

                const Spacer(),

               
                RotationTransition(
                  turns: _rotationController,
                  child: Container(
                    width: 230,
                    height: 230,
                    decoration: const BoxDecoration(
                      shape: BoxShape.circle,
                      image: DecorationImage(
                        image: AssetImage('assets/images/vinyl.png'),
                        fit: BoxFit.cover,
                      ),
                    ),
                  ),
                ),

                const SizedBox(height: 24),

                
                Text(
                  widget.songTitle,
                  textAlign: TextAlign.center,
                  style: GoogleFonts.playpenSans(
                    fontSize: 22,
                    fontWeight: FontWeight.w600,
                    color: Colors.white,
                  ),
                ),

                const SizedBox(height: 20),

                
                Padding(
                  padding: const EdgeInsets.symmetric(horizontal: 24),
                  child: Column(
                    children: [
                      Slider(
                        value: position.inSeconds
                        .clamp(0, duration.inSeconds)
                        .toDouble(),
                        max: duration.inSeconds.toDouble().clamp(1, double.infinity),
                        onChanged: (v) =>
                        _audioPlayer.seek(Duration(seconds: v.toInt())),
                         activeColor: Colors.grey.shade300,   
                         inactiveColor: Colors.grey.shade700, 
                         thumbColor: Colors.grey.shade200,    
                         ),
                      Row(
                        mainAxisAlignment: MainAxisAlignment.spaceBetween,
                        children: [
                          Text(
                            _fmt(position),
                            style: GoogleFonts.playpenSans(
                              color: Colors.white70,
                              fontSize: 14,
                            ),
                          ),
                          Text(
                            _fmt(duration),
                            style: GoogleFonts.playpenSans(
                              color: Colors.white70,
                              fontSize: 14,
                            ),
                          ),
                        ],
                      ),
                    ],
                  ),
                ),

                const SizedBox(height: 24),

                
                Row(
                  mainAxisAlignment: MainAxisAlignment.center,
                  children: [
                    IconButton(
                      icon: const Icon(Icons.replay_10_rounded),
                      color: Colors.white,
                      iconSize: 36,
                      onPressed: () => _seekBy(-10),
                    ),
                    const SizedBox(width: 24),
                    Container(
                      decoration: BoxDecoration(
                        shape: BoxShape.circle,
                        color: Colors.white.withOpacity(0.15),
                      ),
                      child: IconButton(
                        icon: Icon(
                          isPlaying
                              ? Icons.pause_rounded
                              : Icons.play_arrow_rounded,
                        ),
                        color: Colors.white,
                        iconSize: 56,
                        onPressed: _playPause,
                      ),
                    ),
                    const SizedBox(width: 24),
                    IconButton(
                      icon: const Icon(Icons.forward_10_rounded),
                      color: Colors.white,
                      iconSize: 36,
                      onPressed: () => _seekBy(10),
                    ),
                  ],
                ),

                const Spacer(),
              ],
            ),
          ),
        ],
      ),
    );
  }
}
