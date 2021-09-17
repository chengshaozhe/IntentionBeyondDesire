cd ~/goalCommitment-project/commitmentSnake/demoImg/

ffmpeg -start_number 1 -r 5 -s 1920x1080 -i  %d.png -vcodec libx264 -crf 25  -pix_fmt yuv420p ~/goalCommitment-project/commitmentSnake/demo/demo.mp4

