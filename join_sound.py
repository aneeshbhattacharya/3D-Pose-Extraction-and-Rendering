
import moviepy.editor as mpe
import sys

out_file_name = sys.argv[1]

my_clip = mpe.VideoFileClip('./rendered.mp4')
audio_background = mpe.AudioFileClip('./temp_images/song.wav')

final_audio = mpe.CompositeAudioClip([audio_background])

final_clip = my_clip.set_audio(final_audio)
final_clip.write_videofile(out_file_name+'.mp4', audio_codec='aac')