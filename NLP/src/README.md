Currently it is a simple implementation for text classification. 
<br>performs k class CV
<br>All data loaed into memory at the moment and optimised wrt gpu memory usage. 
<br>Use it only to get a baseline on a small dataset
<br>Further work: 
<br>1. Create custom dataloaders to perform kclass CV and for proper pipeline
<br>2. Add class-wise accuracies in plots
<br>3. Implement on Attn LSTM to obtain important words
