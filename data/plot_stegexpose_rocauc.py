import pandas as pd
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
"""
Expects a StegExpose generated csv file with column names renamed. Modify path below as needed.

"""
df = pd.read_csv("./StegExpose/steganalysisOfTestFolder.csv")
df = df.drop(columns=['message_size', 'primary_sets', "chi_square", "sample_pairs", "rs_analysis"])
df['truth'] = False
df.loc[df['file_name'].str.startswith("stego_"),'truth'] = True
# above_threshold is basically predicted
print(df.to_string()) 


fpr, tpr, _ = roc_curve(df['truth'], df["above_threshold"])
roc_auc = auc(fpr, tpr)
plt.plot(fpr, tpr, label=f'StegExpose Fusion (mean), threshold=0.2 (AUC = {roc_auc:.2f})')

plt.plot([0, 1], [0, 1], 'r--', label='Random Guess')

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.savefig("out_roc_auc.png")