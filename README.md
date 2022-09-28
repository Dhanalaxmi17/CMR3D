# CMR3D
Contextualized Multi-Stage Refinement for 3D Object Detection

# Abstract

Existing deep learning-based 3D object detectors typically rely
on the appearance of individual objects and do not explicitly pay
attention to the rich contextual information of the scene. In this
work, we propose Contextualized Multi-Stage Refinement for 3D
Object Detection (CMR3D) framework, which takes a 3D scene
as an input and strives to explicitly integrate useful contextual
information of the scene at multiple levels to predict a set of object
bounding-boxes along with their corresponding semantic labels. To
this end, we propose to utilize a context enhancement network that
captures the contextual information at different levels of granularity
followed by a multi-stage refinement module to progressively refine
the box positions and class predictions. Extensive experiments on
the large-scale ScanNetV2 benchmark reveals the benefits of our
proposed method, leading to an absolute improvement of 2.0% over
the baseline. In addition to 3D object detection, we investigate the
effectiveness of our CMR3D framework for the problem of 3D object
counting. Our source code will be publicly released
