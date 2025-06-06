# content/analytics_text.py

cross_validation_paragraph = """
To evaluate the generalization and reliability of my malware classification 
models, a 5-fold cross-validation strategy was applied. This involved splitting 
the dataset into five folds, training the model on four of them, and validating on the 
remaining one—repeating this process across all folds. This allowed every sample to 
be used for both training and validation, producing better estimate of performance 
than a single train-test split.

For each fold, class-specific precision, recall, and F1 scores were tracked for 
both benign (Class 0) and malware (Class 1) samples, along with overall 
accuracy and weighted F1. These metrics are visualized (see right) using 
grouped bar charts for Class 0 and Class 1, and each was summarized with 
fold-level averages and standard deviations. 

Low variability across folds indicates consistent performance and strong 
generalization. The Full Image CNN model showed superior average F1-scores and smaller 
standard deviations compared to the segment-based model, suggesting it was more both 
more stable and more effective across varying data partitions.
"""

calibration_curve_paragraph = """
The graphs below show the calibration curves for the full-image model (left) 
and the segmented model (right), where the dashed line represents perfect 
calibration (confidence exactly matches accuracy). Areas above the diagonal 
line correspond to under-confidence in predictions and areas under the line 
to over-confidence. The expected calibration error (ECE) for the full model is 
0.1195 which is relatively high. This suggests the a larger area between the 
model's calibration curve and perfect calibration baseline. There is a sharp 
increase in accuracy as confidence rises above 0.3. At levels of high 
accuracy the model is under-confident meaning it performs better 
than it predicts. The ECE of the segment model is lower at 0.0644 which is 
observed in the calibration curve sitting closer to the diagonal line. The 
segmented model is more accurately calibrated across most confidence levels, 
showing a gradual increase in accuracy as confidence increases. 
"""


initial_segment_fail_paragraph = """
In the initial segmentation-based approach, each malware image was divided into 
nine equal segments, and a separate CNN was trained on each one. Final predictions 
were made using majority voting across the segment classifiers. However, this strategy
failed to generalize. As shown in the confusion matrix, the ensemble classified 
every sample as malware, achieving perfect recall but extremely low precision and 
overall accuracy. This indicates that while the model successfully identified all 
malware cases, it also misclassified every benign sample.. The average 
precision and accuracy were low enough that, in a binary classification 
setting, random guessing would likely have produced better-balanced results. This failure 
is likely due to the lack of global context within individual segments; each one 
captured only a limited portion of the full opcode structure, making it difficult 
for the models to learn distinguishing patterns or make confident, generalizable predictions.
"""

new_segment_paragraph = """
In the second segmentation-based experiment, the number of segments per 
image was reduced from nine to four to preserve more contextual information 
within each segment. For each sample, the variance of all four segments was 
calculated and the segment with the highest variance was selected, under the 
assumption that higher variance would correspond more complex patterns and therefore 
more informative content. A single CNN model was then trained using only these 
high-variance segments. This approach significantly outperformed the initial 
segmentation attempt, achieving much stronger accuracy and F1 scores across the 
5-fold cross-validation. While the performance didn’t surpass that of the full-image 
model, it demonstrated clear potential, especially considering that minimal 
hyperparameter tuning was done at this stage. These early results suggest that a 
variance-guided segmentation strategy is a viable direction for building more 
efficient or interpretable malware classification pipelines.
"""

segment_analysis_paragraph = """
The stacked bar chart below breaks down the frequency with which each segment was 
selected as the highest-variance segment among that class's samples. For the 
majority of sample types, there is relatively similar selection frequency among 
the four segments, indicating that the features for these types are likely more 
evenly distributed throughout the binary. In contrast, the "Expiro.BK" malware 
family shows a strong preference for the top-left segment (nearly 75% of selections), 
pointing to characteristics or code patterns concentrated in that region. The 
heatmap confirms these patterns while also showing the corresponding average 
variance measurements. Interestingly, the Benign samples show higher variance 
values (4725–5376) across all segments compared to most malware families, which 
implies greater diversity throughout legitimate executables. Meanwhile, the 
"Vobfus" malware family shows especially high variance in the bottom-left (4688) 
and bottom-right (4580) segments, hinting that these regions may contain more 
obfuscation to make detection more challenging. 
"""

mcnemar_paragraph_1 = """
To test whether the models make significantly different predictions, I applied 
the McNemar test. The McNemar test is a non-parametric statistical method used to 
compare the performance of two models on the same test dataset, particularly 
effective when analyzing matched pairs of binary outcomes. The test focuses 
specifically on the sample cases where one model is correct while the other 
is incorrect. The test statistic is calculated as:
"""

mcnemar_paragraph_2 = """
where b represents cases where the first model succeeded but the second failed 
(59 samples), and c represents cases where the first model failed but the 
second succeeded (16 samples). The McNemar test resulted in a test statistic of 
24.653 and a p-value of 0.000000686 which is significant ( < 0.05). This means 
a rejection of the null hypothesis that both models have the same error rates. 
The models only disagree on 5.41% of the test samples (Quadrants II and III in confusion matrix). 
However, when they disagree, the full image model is correct 78.02% of the time. 
The full image model significantly outperformed the segment-based model on benign 
samples, indicating a lower false positive rate. Across the top malware families, 
there was no statistically significant difference in accuracy, even in cases 
where the segment model had slightly higher accuracy. This means that while the 
segmented model may seem to perform better in some malware classes, these differences 
were not strong enough to be considered meaningful through statistical testing. 
Overall, the segmented model appears to have a higher false positive rate, which 
may limit its utility in settings where minimizing false alarms is critical. 
While overall less accurate, the segment model does correctly classify some samples 
the full model misses. These are often malware samples, suggesting that the segment 
model may focus more narrowly on local features.
"""

ks_paragraph_1 = """
The Kolomogorov-Smirnov (KS) test is an another non-parametric test used to 
determine if two probability distributions differ significantly. Unlike 
parametric tests, the KS test: Makes no assumption about the underlying shape of distributions, 
is sensitive to differences in both location and shape of distributions, 
and compares the entire probability distribution rather than just central 
tendencies. The KS test statistic is calculated as the maximum absolute 
difference between the cumulative distribution function (CDFs) of the two 
samples: 
"""

ks_paragraph_2 = """
The null hypothesis states that both samples come from the same 
distribution. Rejection threshold is set at 0.05. The KS test was performed 
separately on benign samples and malware samples. The benign sample 
distribution had a KS test statistic value of 0.1160 and p-value of 6.34e-06. 
The distribution comparison for benign samples reveals important behavioral 
differences between the models.
"""

ks_benign_bullet1 = """
The full-image model demonstrates more consistent and appropriate behavior by assigning 
lower malware probability scores to benign samples, showing higher confidence in correct
benign classification. This is desired behavior. 
"""

ks_benign_bullet2 = """
The segmented model shows a flater, wider distribution with a concerning long right tail,
indicating it frequently assigns higher malware probability to benign 
samples. This reduced confidence in benign classifications likely 
explains the higher false positive rates. 
"""

ks_paragraph_3 = """
The malware sample distribution had a KS test statistic of 0.5222 and p-value 1.92e-93 
which is an extremely small p-value. For malware, samples, the models show 
significantly different confidence profiles. The segmented model demonstrates a strong 
right-skewed distribution with most predictions clustered at high 
probabilities (closer to 1), indicating consistently higher confidence in 
malware classifications. 
"""

