# FIE Stream Clipper

This program takes a several hours long livestream of an FIE event with an overlay and automatically cuts it up into smaller videos, one per bout. Each bout is labeled with the fencers' names and optionally the current tableau, fencers' countries, and event name/date e.g. "Orleans GP 2024 Torre Pietro ITA vs Homer Daryl USA T32". Although the name is FIE Stream Clipper, this program also works for USA Fencing livestreams, including NACs and NCAA championships.

## Installation

Installing the program is quite straightforward. Use the following steps to run the executable on your Windows computer:

* Download the latest release on the [Releases page](https://github.com/BrandonHowe/fie-stream-clipper/releases).
* Unzip the downloaded .zip file.
* Navigate into the unzipped folder and run `fie_stream_clipper.exe`.

## Usage

Once you have completed the installation process, you can begin using the program to cut up streams. To clip a video file of an FIE event, follow these steps:

* Select the video file of the event livestream from your computer.
* If you want the bouts to be outputted to a specific folder, select it. The videos automatically go to a default folder if no folder is selected.
* Select which overlay is present in the video. If the selected overlay is different from the livestream's overlay, the clipper may not work.
* If you want the file names to have specific text at the beginning, enter it into the text box. This can be useful when clipping multiple streams e.g. entering "Orleans GP 2024" into the box will make all file names start with "Orleans GP 2024". Otherwise, no extra text will be added to the file names.
* Once all the configuration is set, press the "Clip Stream!" button to start the stream clipping process. A progress bar will appear while the program is analyzing the stream and when it is saving the bout videos to your computer. The duration of this process can vary based on stream length and computer performance, but a general estimate is under 60 seconds per stream.

There may be errors when outputting the file name - sometimes names may be slightly misspelled or the tableau/country may be incorrect or missing. Make sure to double check all file names to ensure everything is accurate.

## Contact

If you experience any issues or have any feedback, please reach out to me!
