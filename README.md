# object-tracker-gbr

**Object tracker based on template-matching developed for the [Great Barrier Reef Kaggle competition](https://www.kaggle.com/c/tensorflow-great-barrier-reef/).**

---

The baseline to beat was the [Norfair tracker](https://github.com/tryolabs/norfair), the one publicly shared during the competition.

The Norfair tracker works well with fixed cameras and moving tracked objects (i.e. cars or people captured by a street camera), but in our problem we encounter the opposite situation: we have fixed objects to track (the starfishes) and a moving camera/point-of-view (the diver or drone carrying the camera moves along the reefs).

The fact that the objects to track are immobile is what makes possible to use template-matching as the tracking method, since even with a moving point-of-view, objects in consecutive frames are almost identical.

To stop small differences in size between frames from accumulating, we implement as well a *dynamic* mode, by which the tracker will try to match the template at different scales, chosing the best match of all. This allows for bounding boxes to expand/shrink appropiately as the object gets closer/further, which is critical for performance, since as most of object detection tasks, the competition's metric is based on IoU.

---

https://user-images.githubusercontent.com/49324844/156448097-45c8918d-ed45-4679-bd36-600b84f38f19.mp4

```
video legend

green     ground truth bbox
orange    bbox predicted by model (includes IoU, confidence)
red       bbox added by tracker (includes IoU, confidence)
```

You can see in the video the main improvements of our custom tracker (to the right) vs the Norfair tracker (to the left):
- tracks much better the movements, specially changes of direction/speed (Norfair's tracker doesn't actually *see* anything, it just estimates the position from past movements)
- is able to adjust bounding box size

---

**How to integrate the tracker in your ML detector pipeline:**
```python
from tracker_custom2 import Tracker

# initialize tracker
tracker = Tracker(**kwargs)

frame_id = 0
for img in imgs:
  bboxes, scores = fn_inference(img)
  
  # update tracker with frame predictions
  tracker.update(bboxes, scores, frame_id, img)
  
  # add tracked undetected objects
  bboxes_tr, scores_tr = tracker.find()
  
  bboxes += bboxes_tr
  scores += scores_tr
  frame_id += 1
```

---
