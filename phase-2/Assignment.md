### A. Total Number of Source Titles: 1100864
    
  Total Number of Tokenized Titles: 1023851
    

---

### B. If A and B are different, what have you done for that?
  Remove any missing columns

---

### C. Parameters of Doc2Vec Embedding Model
  1. **Total Number of Training Documents:** 1,023,851
  2. **Output Vector Size:** 64      
     **Min Count**: 2      
     **Epochs:** 100      
     **Workers:** 8      
        
  3. **First Self Similarity**: 75.70%      
     **Second Self Similarity:** 86.18%
        

---

### D. Parameters of Multi-Class Classification Model
 1. **Arrangement of Linear Layers:**  $64 \times 256 \times 128 \times 64 \times 32 \times 9$
 2.  **Activation Function for Hidden Layers:** Linear, ReLU
 3.  **Activation Function for Output Layers:** Softmax
 4. **Loss Function:** Categorical Cross Entropy
 5. **Algorithms for Back-Propagation:** AdamW
 6. **Total Number of Training Documents:** 819,080 (80% of  1,023,851)
 7. **Total Number of Testing Documents:** 102,386 (10% of 1,023,851)
 8. Epochs: 43/500 (early stop)
     
     Learning Rate: 0.001 
     
 9. First Match: 87.30%
     
     Second Match: 94.42%
     
 10. Any other parameters you think are important.
     - Batch Size: 128
     - Layers:
         - `Linear(64, 256)`
         - `ReLU()`
         - `Dropout(0.2)`
         - `Linear(256, 128)`
         - `ReLU()`
         - `Dropout(0.2)`
         - `Linear(128, 64)`
         - `ReLU()`
         - `Linear(64, 32)`
         - `ReLU()`
         - `Linear(32, 9)`
     - Split into training (80%), validation (10%), and test (10%) datasets
     - Loss/Learning curve
         
         ![0413-1213-learning-curve.png](https://github.com/louis-chiu/wehelp-deep-learning-bootcamp/blob/master/phase-2/images/0413-1213-learning-curve.png?raw=true)
            

---

### E. Parameters of Multi-Labels Classification Model. (If you have one)
1. **Arrangement of Linear Layers:** $64 \times 128 \times 64 \times 16\times 9$
2. **Activation Function for Hidden Layers:** Linear, ReLU
3. **Activation Function for Output Layers:** Sigmoid
4. **Loss Function:** Binary Cross Entropy
5. **Algorithms for Back-Propagation:** AdamW
6. **Total Number of Training Documents:** 819,080 (80% of  1,023,851)
7. **Total Number of Testing Documents:** 102,386 (10% of 1,023,851)
8. **Epochs:** 200  
   **Learning Rate:** 0.001
    
1. **Threshold for Positive Label:** None. My evaluation logic is the same as that of a multi-class classification model—I choose the output with the highest value as the predicted label.  
    **Accuracy Rate:**  
      - First Match: 86.41%
      - Second Match: 93.49%
2.  **Any other parameters you think are important.**
    - Only pick POS tags that start with `'N'` or `'V'`, or are equal to `'A'` or `'FW'`
    - Batch Size: 128
    - Model Layers:
        - `Linear(64, 128)`
        - `ReLU()`
        - `Dropout(0.3)`
        - `Linear(128, 64)`
        - `ReLU()`
        - `Dropout(0.2)`
        - `Linear(64, 16)`
        - `ReLU()`
        - `Dropout(0.1)`
        - `Linear(16, 9)`
        - `Sigmoid()`
    - Split into training (80%), validation (10%), and test (10%) datasets
    - Loss/Learning Curve
        
        ![0413-0303-learning-curve.png](https://github.com/louis-chiu/wehelp-deep-learning-bootcamp/blob/master/phase-2/images/0413-0303-learning-curve.png?raw=true)
        

---

### F.  Share your experience of optimization, including at least 2 change/result pairs.
  1.   
     - **Change:** Only pick POS tags that start with `'N'` or `'V'`, or are equal to `'A'` or `'FW'`
     - **Result:** No significant impact on classification accuracy.In fact, not applying this filter results in slightly higher accuracy—around 1%.
  2.      
     - **Change:** Enhanced prediction logic by handling batches using tensor manipulation.
      - **Result:** Training time per epoch was reduced significantly—approximately 5 times faster, from 150 seconds to less than 30 seconds.