opcode_extraction = """
The first step required extracting opcode sequences 
from the binary files. This required using  objdump, a binary analysis tool, to 
disassemble the executable files and identify the operation codes while excluding 
raw instruction data. Using a custom shell script each binary file was 
processed to filter out common non-instruction elements and isolate the pure 
opcode sequences. This approach preserves the essential execution behavior 
signatures while reducing noise from variable names, memory addresses, and other 
implementation-specific details. The resulting text files contain clean opcode sequences 
that represent the behavioral patterns of both benign software and malware samples, 
providing a solid foundation for input into word2vec.
"""

word2vec = """
For transforming the sequential opcode data into a meaningful numerical representation, 
a Word2Vec embedding model was implemented. This approach is particularly well-suited 
for this malware classification project as it captures the semantic 
relationships between opcodes based on their contextual usage patterns. Unlike 
simple frequency-based approaches, Word2Vec preserves the structural and functional 
relationships between instruction sequences. As shown in our t-SNE visualization, 
certain opcodes demonstrate clear clustering tendencies that align with their 
prevalence in either malware or benign software. The red points (opcodes more common in malware) 
form distinct clusters separate from blue points (opcodes more common in benign software), 
suggesting that these may in fact form patterns that can be used to
distinguish malware from legitimate software. 
"""

hilbert_mapping = """
Hilbert curve mapping was selected as the method for converting opcode 
embeddings into images in an attempt to preserve local relationships while mapping 
between dimensions. The Hilbert curve's locality-preserving nature ensures that 
opcodes that are sequential in the executable remain relatively close in the 2D space, 
maintaining important structural information about opcode execution sequence. 
This property is crucial for malware detection, as malicious patterns often manifest 
as specific opcode sequences or unusual instruction neighborhoods. 
By mapping the 1D sequence of opcodes to a 2D space using Hilbert curves, 
we preserve these neighborhood relationships while creating a representation 
suitable for convolutional neural network analysis, which excels at identifying 
spatial patterns and features.
"""

data_sampling = """
To address the significant class imbalance in the dataset (approximately 100,
000 malware samples versus 14,000 benign samples), a multi-level stratified 
sampling approach was utilized. First, a target number of malware samples was 
established and then distributed evenly across all malware families. This family-aware 
sampling ensured that all malware variants maintained appropriate representation 
while preventing overrepresented families from dominating the model training. 
For example, with a target of 8,500 malware samples, in the absence of 
information regarding a family's real-world frequency, samples were selected 
uniformly across families that met the minimum sample size threshold. A 
70/15/15 split for training, validation, and testing sets, respectively was 
chosen, while maintaining stratification at both the binary class level (
malware/benign) and the family level. This dual-level stratification aims to 
prevent data leakage between splits while ensuring each partition contained 
representative examples from every malware family, giving the model the best 
chance at generalization to new, unseen malware variants from known families. 
"""
