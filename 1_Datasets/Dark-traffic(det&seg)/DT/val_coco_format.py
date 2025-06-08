from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

gt = COCO("val_bbox_1.json")  # Ground truth in RLE format
dt = gt.loadRes("pre_s/redetr-predictions.json")  # Your model's predictions, also in RLE format

evaluator = COCOeval(gt, dt, iouType='bbox')
evaluator.evaluate()
evaluator.accumulate()
evaluator.summarize()
