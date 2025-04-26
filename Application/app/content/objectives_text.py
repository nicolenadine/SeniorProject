# content/objectives_text.py

intro_cnn_paragraph = """
Traditional malware classification approaches using Convolutional Neural Networks 
(CNNs) typically rely on training models using full-size grayscale images converted 
from malware binaries. This image-based method captures spatial patterns within
the binary data, allowing CNNs to distinguish between malicious and benign samples. 
Prior work by Kalash et al. demonstrated that this approach can achieve extremely 
high classification accuracy—up to 99.97% on the Microsoft malware dataset—by utilizing
CNN architectures on grayscale representations of the original binary files [1]. 
Similarly, Vu et al. proposed an hybrid image transformation 
technique using entropy encoding and Hilbert space-filling curves, achieving 
93.01% accuracy while incorporating semantic and syntactic features from binaries into color images [2]. 
These results demonstrate the potential of image-based representations for 
malware classification.
"""

intro_cnn_paragraph2 = """ However, training on full images remains 
computationally intensive, particularly as dataset sizes grow or when 
deploying models in constrained environments. This project explores potential 
alternative approaches that involve training CNNs on smaller image segments instead 
of full images. The goal is to explore if CNNs trained using only a portion 
of the image can achieve similar classification accuracy and profile to those trained 
on images containing the full original binary data. Research by Saponara and Elhanash 
has demonstrated that smaller input images require fewer computations, leading 
to faster training and inference times [3]. If shown to be viable, a segmented 
approach would be reduce computational cost while maintaining high classification 
performance.
"""

segmentation_goal_paragraph = """
In order to evaluate the segment-based approach, standard 
classification metrics—precision, recall, F1-score, and test accuracy—across 
5-fold cross-validation will be collected. Beyond these metrics,the 
McNemar test will be applied to statistically assess whether prediction errors 
between full-image and segment-based models differ significantly. 
Additionally, the Kolmogorov–Smirnov (KS) test will be used to compare the 
distributions of predicted probabilities, and assess model 
calibration to determine how well predicted confidence scores align with observed 
outcomes. These comprehensive evaluations aim to determine whether segment-level 
training offers a viable and efficient alternative to full-image CNN models in 
malware classification.
"""


malware_paragraph = """
This project was provided a subset of the VirusShare dataset comprised of 
approximately 131,000 malware samples representing 50 different malware 
families. The top 20 families (by count) are displayed in the bar plot to the left.
This specific version of the VirusShare collection consists of binary files 
that have been transformed into assembly opcodes. This was an important 
characteristics for the needs of this project since it will be working with 
opcode sequences opposed to raw binary stream.[4]
"""

benign_paragraph = """
The proprietary nature of benign executable files presents a challenge in 
acquiring a large enough sample set of benign files. This proved to be a 
significant limiting factor in this project due to the desire to have an 
appropriate balance of class representation. In a real-world setting malware 
occurrences are relatively rare compared to encounters with benign files. 
Ideally, the training data would reflect this ratio but only the benign file 
obtained was approximately 14,400. Benign samples were collected from the
publicly available Benign-NET GitHub repository hosted by user bourmaa [4](
https://github.com/bormaa/Benign-NET/tree/main).
"""

# content/objectives_text.py

citations = [
    "1. M. Kalash, M. Rochan, N. Mohammed, N. D. B. Bruce, Y. Wang and F. Iqbal, 'Malware Classification with Deep Convolutional Neural Networks,' 2018 9th IFIP International Conference on New Technologies, Mobility and Security (NTMS), Paris, France, 2018, pp. 1-5, doi: 10.1109/NTMS.2018.8328749",
    "2. Vu D. L. Nguyen T. K. Nguyen T. V. Nguyen T. N. Massacci F. Phung P. H. (2020). HIT4Mal: Hybrid image transformation for malware classification.Transactions on Emerging Telecommunications Technologies, 31(11), e3789. 10.1002/ett.3789",
    "3.  Saponara, S., & Elhanashi, A. (2022). Impact of Image Resizing on "
    "Deep Learning Detectors for Training Time and Model Performance. In Applications in Electronics Pervading Industry, Environment and Society (pp. 10–17). Springer International Publishing. https://doi.org/10.1007/978-3-030-95498-7_2​",
    "4. Malicia Dataset: https://malicia-project.com/dataset",
    "5. Benign-NET: https://github.com/bormaa/Benign-NET/tree/main"
]

