# Training Datasets

## Required Datasets

### SoccerNet
- 500+ full broadcast games with 17 action classes
- Install: `pip install SoccerNet`
- Download: `from SoccerNet.Downloader import SoccerNetDownloader; d = SoccerNetDownloader(LocalDirectory="data/soccernet"); d.downloadGames(split=["train", "valid", "test"])`

### SportsMOT
- Multi-object tracking benchmark for sports
- Top result: SportsTrack (HOTA 76.264 at ECCV 2022)
- Download: https://github.com/MCG-NJU/SportsMOT

### COCO Keypoints
- Used for ViTPose++ fine-tuning
- 200K labeled person instances with 17 keypoints
- Download: https://cocodataset.org/#keypoints-2017

### Custom AthleteView Dataset (to create)
- Cricket: 50+ hours annotated IPL training footage
- Football: 50+ hours European league footage
- Kabaddi: 20+ hours Pro Kabaddi footage
- Annotations: player bounding boxes, pose keypoints, ball position, action labels
