# Source of the code
 - https://github.com/AkiRusProd/numpy-transformer?tab=readme-ov-file
 - Refer to the original git repository
 - The code has been modified from original numpy-transformer

# Inference Example
Example №1
Input sentence: the gadget was heartwarming but i wouldn't try it again
Decoded sentence: possitive <eos>
Target sentence: possitive

Example №2
Input sentence: the weather was a disaster but i've had better
Decoded sentence: negative <eos>
Target sentence: negative

Example №3
Input sentence: the product was exciting and it changed my perspective
Decoded sentence: possitive <eos>
Target sentence: possitive

Example №4
Input sentence: the scenery made me happy for sure
Decoded sentence: possitive <eos>
Target sentence: possitive

Example №5
Input sentence: the match was amazing but i wouldn't try it again
Decoded sentence: possitive <eos>
Target sentence: possitive

Example №6
Input sentence: the concert was fun without a doubt
Decoded sentence: possitive <eos>
Target sentence: possitive

Example №7
Input sentence: my experience was flawless and it was just okay
Decoded sentence: possitive <eos>
Target sentence: possitive

Example №8
Input sentence: the trip was flawless but i've had better
Decoded sentence: possitive <eos>
Target sentence: possitive

Example №9
Input sentence: this program was subpar for sure
Decoded sentence: negative <eos>
Target sentence: negative

Example №10
Input sentence: the product was awkward
Decoded sentence: negative <eos>
Target sentence: negative

Input sentence: ['the', 'gadget', 'was', 'heartwarming', 'but', 'i', "wouldn't", 'try', 'it', 'again']
Decoded sentence: ['possitive', '<eos>']

### References:
 - https://arxiv.org/abs/1706.03762 - article "Attention is All You Need"

### TODO:
1) clean up and refactor code
