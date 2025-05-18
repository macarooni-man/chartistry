from chart_parser import Chart
from transcriber import transcribe_drums



# Load Clone Hero chart to modify
chart_file = r'/Users/kaleb/Desktop/Periphery - Satellites/notes.chart'
chart = Chart(chart_file)
# Access song metadata
# print(chart.metadata)
# print(chart.metadata["Name"], chart.resolution)
# # Iterate notes in ExpertSingle
# for note in chart.tracks["ExpertSingle"].notes:
#     print(note.tick, note.fret, note.length)



# Convert audio to chart format
# demucs --two-stems=drums -n htdemucs_ft "/Users/kaleb/Desktop/Periphery - Satellites/song.mp3"
stem = r'/Users/kaleb/Documents/GitHub/clone-hero-drum-generator/separated/htdemucs_ft/song/drums.wav'
ts = transcribe_drums(chart, stem)
chart.add_chart(ts)
