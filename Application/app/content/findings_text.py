# content/findings_text.py

overview_paragraph = """
This research project explored alternative approaches to CNN-based malware 
classification focused on using image segments in model training. The project 
compared the performance of CNNs trained on full malware images versus those
trained on carefully selected image segments, with the goal of reducing 
computational costs while maintaining classification effectiveness.
"""

key_findings_intro = """
The comprehensive evaluation of full-image versus segment-based malware 
classification models revealed several important insights into their relative
performances, computational efficiency, and potential applications.
"""

# 1. Full-Image vs. Segment-Based Model Performance
model_performance_title = "1. Full-Image vs. Segment-Based Model Performance"

model_performance_bullet1 = """
• <strong>Full-Image Model Superiority:</strong> The CNN model trained on full 
images consistently outperformed the segment-based approach across most evaluation 
metrics. It demonstrated superior weighted average F1-scores and smaller 
standard deviations in the 5-fold cross-validation, 
indicating both better performance and more consistent generalization.
"""

model_performance_bullet2 = """
• <strong>Segment Model Progress:</strong> While the initial segmentation approach 
(using nine segments with majority voting) failed to generalize properly, the 
next experimental approach selecting the segment containing the highest variance 
out of 4 segments, showed significant improvement, though still not matching 
full-image performance.
"""

model_performance_bullet3 = """
• <strong>Statistical Differences:</strong> The McNemar test (statistic: 24.653, 
p-value: 6.86e-7) confirmed that the models have significantly different error 
patterns. While the models only disagreed ~5.4% of the time, the full-image model 
performed notably better when disagreements occurred (correct in 78.02% of disagreement cases).
"""

# 2. Model Calibration Analysis
calibration_title = "2. Model Calibration Analysis"

calibration_bullet1 = """
• <strong>Calibration Comparison:</strong> The segmented model demonstrated better 
calibration with an Expected calibration Error (ECE) of 0.0644 versus 0.1195 
for the full-image model, suggesting it more accurately aligns 
confidence with actual performance.
"""

calibration_bullet2 = """
• <strong>Probability Distribution Differences:</strong> The Kolmogorov-Smirnov 
test revealed significant differences in probability distributions between the models 
for both benign samples (KS statistic: 0.1160, p-value: 6.34e-6) and 
malware samples (KS statistic: 0.5222, p-value: 1.92e-93).
"""

calibration_bullet3 = """
• <strong>Classification Tendencies:</strong> The full-image model showed more appropriate 
confidence when classifying benign samples, while the segment model demonstrated 
consistently higher confidence in malware classifications, potentially explaining 
its higher false positive rate. This is likely due to the fact that legitimate 
files contain greater diversity throughout the binary, whereas malware may show 
greater localized patterns due to obfuscation and malicious instruction patterns. 
This causes the segmented to perform well at identifying the malware samples but  
struggle at generalizing the benign samples.

"""

# 3. Segment Analysis Insights
segment_analysis_title = "3. Segment Analysis Insights"

segment_analysis_bullet1 = """
• <strong>Variance Distribution:</strong> Benign samples showed consistently higher variance (4725-5376) across all 
segments compared to most malware families, suggesting greater diversity in legitimate executables.
"""

segment_analysis_bullet2 = """
• <strong>Family-Specific Patterns:</strong> Some malware families showed distinct segment preference patterns. 
For example, the "Expiro.BK" family showed a strong preference (75%) for the top-left segment, while "Vobfus" 
displayed higher variance in the bottom segments.
"""

segment_analysis_bullet3 = """
• <strong>Segment Selection Uniformity:</strong> Most sample types showed relatively uniform segment selection 
frequency across the four quadrants, indicating features were often distributed throughout the binary rather than 
concentrated in specific regions.
"""

# 4. Computational Efficiency Analysis
computational_efficiency_title = "4. Computational Efficiency Analysis"


computational_efficiency_bullet1 = """
• <strong>Image Size Reduction Effects:</strong> The primary factor affecting 
computational efficiency is the reduction from 256×256 to 128×128 input size. Each
convolutional layer is approximately O(height × width × kernel_size² × in_channels × 
out_channels). The smaller image size of the segmented model reduces the 
computational cost of convolutional layers by approximately 75% and similarly 
reduces memory requirements by roughly 75%.
"""

computational_efficiency_bullet2 = """
• <strong>Variance Calculation Overhead:</strong> While the segmented approach 
introduces minimal overhead in the form of variance calculations, this overhead 
is negligible compared to the forward and backward passes of the neural network.
"""

computational_efficiency_bullet3 = """
• <strong>Time Complexity Comparison:</strong> The ratio of computational work 
for the convolutional operations is approximately, Full model : Segmented 
model = (256²) : (128²) = 4:1 in favor of the segmented model, resulting in 
significantly less computation overall even with the variance calculation overhead.
"""

computational_efficiency_bullet4 = """
• <strong>Practical Efficiency Benefits:</strong> During training, the segmented 
model offers several advantages including reduced memory usage allowing larger 
batch sizes, ~75% reduction in computation per forward/backward pass, 
and variance-based selection focusing on more informative regions.
"""

# Impact of Research
impact_title = "Research Impact"

impact_bullet1 = """
• <strong>Computational Efficiency Improvements:</strong> The segment-based approach 
demonstrated a potential 75% reduction in computational costs while still maintaining 
reasonable performance. The variance-guided selection method optimizes this trade-off 
by focusing on the most informative regions.
"""

impact_bullet2 = """
• <strong>Calibration Insights:</strong> The discovery that segment-based models 
achieve better calibration suggests they could complement full-image models in 
ensemble approaches where confidence calibration is critical.
"""

impact_bullet3 = """
• <strong>Malware Family Characterization:</strong> The segment variance analysis 
provided insights into how different malware families structure their code, which 
could inform both detection strategies and understanding of malware evolution.
"""

impact_bullet4 = """
• <strong>Methodological Framework:</strong> The comprehensive evaluation approach 
combining standard metrics with statistical tests provides a robust framework for 
comparing different malware classification methodologies.
"""

# Next Steps
next_steps_title = "Next Steps"

next_steps_bullet1 = """
• <strong>Hybrid Model Development:</strong> Explore ensemble approaches that 
combine the strengths of both full-image and segment-based models, potentially 
utilizing the better calibration of segment models with the higher accuracy of 
full-image models.
"""

next_steps_bullet2 = """
• <strong>Segment Selection Refinement:</strong> Further investigate optimal 
segment selection criteria beyond variance, potentially incorporating semantic 
understanding of code regions or entropy based segment analysis and selection.
"""

next_steps_bullet3 = """
• <strong>Computational Performance Analysis:</strong> Conduct detailed benchmarking 
of training and inference times to quantify the computational savings of segment-based 
approaches versus the accuracy trade-offs.
"""

next_steps_bullet4 = """
• <strong>Family-Specific Models:</strong> Develop specialized segment-based models 
for malware families that showed distinct segment preferences, potentially 
creating a hierarchical classification system.
"""

next_steps_bullet5 = """
• <strong>Interpretability Enhancement:</strong> Leverage the segment-based 
approach to improve model interpretability by identifying which regions of 
malware binaries contribute most to classification decisions.
"""

conclusion_paragraph = """
This research demonstrates that while full-image CNN models currently provide 
superior classification performance for malware detection, segment-based approaches 
offer promising directions for efficiency optimization and complementary analysis 
techniques. With further refinement, segment-based methods could become valuable components 
in comprehensive, resource-efficient malware defense systems.
"""