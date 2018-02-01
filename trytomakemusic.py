from pydub import AudioSegment

import re

def get_the_square(array, aver):
    pass

sec = 100000
song1 = AudioSegment.from_mp3('yueyu1.mp3')
song2 = AudioSegment.from_mp3('thexx.mp3')
arr_song1 = song1.get_array_of_samples()
# print(sum(arr_song1.tolist()) // arr_song1.__len__())

