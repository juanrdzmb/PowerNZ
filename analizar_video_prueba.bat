@echo off
python main.py --input video_prueba.mp4 --output outputs\analisis.mp4 --pose-backend yolo --object-model models\powerai_bar_detector.pt --plate-diameter-px 120 --view-mode auto --segmentation-backend auto --athlete-lock auto --measurement-requires-hub --show-unmeasured-anchor --plate-box-style full
pause
