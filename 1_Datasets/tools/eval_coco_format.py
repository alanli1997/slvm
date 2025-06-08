'''
SLVM-tools:
This script will help you evaluate the accuracy of the model's prediction results in coco format.
(if there are differences from the model's built-in evaluation results, please do not panic, this is normal)
'''
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

gt = COCO("eval_dt_instances.json")  # Ground truth in RLE format (seg only)
dt = gt.loadRes("pre_s/slvm-predictions.json")  # Your model's predictions, also in RLE format (seg only)

evaluator = COCOeval(gt, dt, iouType='segm') # or bbox
evaluator.evaluate()
evaluator.accumulate()
evaluator.summarize()